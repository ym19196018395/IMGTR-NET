import os
import sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import time
import argparse

# 将当前文件所在目录加入 Python 搜索路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets.triangulation import get_cdt_datas
from utils import batch_convert_to_tri_infos_new
from models.PlanePatchMatch import DensePlaneFitter, PlaneVisualizer

def read_whu_cam(filename):
    with open(filename, 'r') as f:
        # 过滤空行并去除两端空格，用于大图格式的行定位
        lines = [line.strip() for line in f.readlines() if line.strip()]

    # 1. 判定是否为大场景参数格式 (第一行为 'intrinsic')
    if len(lines) > 0 and lines[0] == 'intrinsic':
        # 读取内参 (f, x0, y0)
        int_params = list(map(float, lines[1].split()))
        f_val = int_params[0]
        x0 = int_params[1]
        y0 = int_params[2]
        
        intrinsics = np.array([
            [-f_val, 0, x0],
            [0, f_val, y0],
            [0, 0, 1]
        ], dtype=np.float32)

        # 查找 'extrinsic' 标识定位外参
        idx_ext = lines.index('extrinsic')
        extrinsics = []
        for i in range(idx_ext + 1, idx_ext + 5):
            extrinsics.append(list(map(float, lines[i].split())))
        extrinsics = np.array(extrinsics, dtype=np.float32)

        # 读取深度范围和步长 (最后一行)
        depth_line = list(map(float, lines[-1].split()))
        depth_min = depth_line[0]
        depth_max = depth_line[1]
        depth_interval = depth_line[2] if len(depth_line) > 2 else (depth_max - depth_min) / 192.0
        
        return intrinsics, extrinsics, depth_min, depth_max, depth_interval
    else:
        # 2. 普通 WHU 格式读取方式 (使用未过滤空行的 raw 索引防止因为空行导致行数偏移)
        with open(filename, 'r') as f:
            raw_lines = f.readlines()
            
        extrinsics = []
        for i in range(1, 5):
            extrinsics.append(list(map(float, raw_lines[i].strip().split())))
        extrinsics = np.array(extrinsics, dtype=np.float32)
        
        vals = list(map(float, raw_lines[6].strip().split()))
        f_val = vals[0]
        x0 = vals[1]
        y0 = vals[2]
        
        depth_line = list(map(float, raw_lines[8].strip().split()))
        depth_min = depth_line[0]
        depth_max = depth_line[1]
        depth_interval = depth_line[2]
        
        intrinsics = np.array([
            [-f_val, 0, x0],
            [0, f_val, y0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return intrinsics, extrinsics, depth_min, depth_max, depth_interval

def compute_gt_planar_soft_confidence(gt_depth_s0, tri_id_s0, intrinsics_s0, sigma=0.05, min_pixels=3):
    H, W = gt_depth_s0.shape
    planar_soft_conf_s0 = np.zeros((H, W), dtype=np.float32)
    tri_conf_dict = {}

    dx = np.zeros_like(gt_depth_s0)
    dy = np.zeros_like(gt_depth_s0)
    dx[:, :-1] = np.abs(gt_depth_s0[:, 1:] - gt_depth_s0[:, :-1])
    dy[:-1, :] = np.abs(gt_depth_s0[1:, :] - gt_depth_s0[:-1, :])
    depth_mutant = (dx > 0.25) | (dy > 0.25)

    dtri_x = np.zeros_like(tri_id_s0)
    dtri_y = np.zeros_like(tri_id_s0)
    dtri_x[:, :-1] = tri_id_s0[:, 1:] != tri_id_s0[:, :-1]
    dtri_y[:-1, :] = tri_id_s0[1:, :] != tri_id_s0[:-1, :]
    tri_boundary = (dtri_x != 0) | (dtri_y != 0)
    
    kernel = np.ones((3, 3), dtype=np.uint8)
    tri_boundary_expanded = cv2.dilate(tri_boundary.astype(np.uint8), kernel, iterations=1) > 0
    discard_mask = depth_mutant & tri_boundary_expanded
    
    valid_mask = (gt_depth_s0 > 0) & (tri_id_s0 >= 0) & (~discard_mask)
    if not np.any(valid_mask):
        return planar_soft_conf_s0, tri_conf_dict

    v_indices, u_indices = np.where(valid_mask)
    z = gt_depth_s0[v_indices, u_indices]
    
    fx = intrinsics_s0[0, 0]
    fy = intrinsics_s0[1, 1]
    cx = intrinsics_s0[0, 2]
    cy = intrinsics_s0[1, 2]
    
    x = (u_indices - cx) / fx * z
    y = (v_indices - cy) / fy * z
    points_3d = np.stack([x, y, z], axis=1)
    tri_ids_valid = tri_id_s0[v_indices, u_indices].astype(int)

    tri_counts = np.bincount(tri_ids_valid)
    valid_tri_ids = np.where(tri_counts >= min_pixels)[0]
    if len(valid_tri_ids) == 0:
        return planar_soft_conf_s0, tri_conf_dict

    sort_idx = np.argsort(tri_ids_valid)
    points_sorted = points_3d[sort_idx]
    tri_ids_sorted = tri_ids_valid[sort_idx]

    left_boundaries = np.searchsorted(tri_ids_sorted, valid_tri_ids, side='left')
    right_boundaries = np.searchsorted(tri_ids_sorted, valid_tri_ids, side='right')

    for tri_id, left, right in zip(valid_tri_ids, left_boundaries, right_boundaries):
        pts = points_sorted[left:right]
        N = len(pts)
        centroid = np.mean(pts, axis=0)
        pts_centered = pts - centroid
        cov = np.dot(pts_centered.T, pts_centered) / N
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            normal = eigenvectors[:, 0]
        except np.linalg.LinAlgError:
            tri_conf_dict[tri_id] = 0.0
            continue
        dists = np.abs(np.dot(pts_centered, normal))
        mean_dist = np.mean(dists)
        scale_compensator = 1.0 + 1.5 * np.exp(-N / 30.0)
        sigma_adapted = sigma * scale_compensator
        conf = 1.0 / (1.0 + (mean_dist / sigma_adapted) ** 2)
        conf = np.clip(conf, 0.0, 1.0)
        tri_conf_dict[tri_id] = float(conf)
    return planar_soft_conf_s0, tri_conf_dict

def main():
    parser = argparse.ArgumentParser(description="WHU MVS dataset offline preprocessing script")
    parser.add_argument("--datapath", type=str, default=r"E:\RemoteCodeEx\Datas\WHU_MVS_dataset", help="Dataset path")
    parser.add_argument("--listfile", type=str, default=r"lists\whu\train.txt", help="List of scans")
    parser.add_argument("--mode", type=str, default="train", help="Mode: train/val")
    parser.add_argument("--device", type=str, default="cuda:0", help="GPU device ID")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 读取 scan 列表
    if not os.path.exists(args.listfile):
        project_list = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.listfile)
        if os.path.exists(project_list):
            args.listfile = project_list
        else:
            raise FileNotFoundError(f"Listfile not found: {args.listfile}")

    with open(args.listfile, 'r') as f:
        scans = [line.strip() for line in f.readlines() if line.strip()]

    print(f"Loaded scans: {scans}")
    total_processed = 0

    for scan in scans:
        print(f"\nProcessing scan: {scan}")
        scan_dir = os.path.join(args.datapath, args.mode, "Images", scan)
        list_txt_path = os.path.join(scan_dir, "list.txt")
        
        file_ids = []
        if os.path.exists(list_txt_path):
            with open(list_txt_path, 'r') as f:
                file_ids = [line.strip().split('.')[0] for line in f.readlines() if line.strip()]
        else:
            # 自动容错自愈方案：扫描 urd 目录下的所有图片
            urd_dir = os.path.join(scan_dir, "urd")
            if os.path.exists(urd_dir):
                file_ids = [f.split('.')[0] for f in os.listdir(urd_dir) if f.lower().endswith('.png')]
                file_ids = sorted(list(set(file_ids)))
                print(f" ⚠️ Note: list.txt not found in {scan_dir}. Automatically detected {len(file_ids)} views from 'urd' directory.")
            else:
                # 兼容普通 WHU 格式，如果无 urd 目录，扫描子视角 1 目录下的图片
                sub_dir_1 = os.path.join(scan_dir, "1")
                if os.path.exists(sub_dir_1):
                    file_ids = [f.split('.')[0] for f in os.listdir(sub_dir_1) if f.lower().endswith('.png')]
                    file_ids = sorted(list(set(file_ids)))
                    print(f" ⚠️ Note: list.txt not found in {scan_dir}. Automatically detected {len(file_ids)} views from subdirectory '1'.")

        if len(file_ids) == 0:
            print(f" Warning: No view IDs could be resolved for {scan_dir}, skipping.")
            continue

        for file_idx, file_id in enumerate(file_ids):
            # 定义输入文件路径
            img_filename = os.path.join(args.datapath, args.mode, "Images", scan, "urd", f"{file_id}.png")
            triangulation_filename = os.path.join(args.datapath, args.mode, "Images", scan, "triangulation", "CDTinfo", f"CDT_info_vlf_{file_id}.txt")
            depth_filename_hr = os.path.join(args.datapath, args.mode, "Depths", scan, "1", f"{file_id}.png")
            proj_mat_filename = os.path.join(args.datapath, args.mode, "Cams", scan, "1", f"{file_id}.txt")

            # 定义输出文件路径
            out_dir = os.path.join(args.datapath, args.mode, "Images", scan, "triangulation")
            out_filename = os.path.join(out_dir, f"geom_svd_{file_id}.npz")

            if os.path.exists(out_filename) and os.path.getsize(out_filename) > 0:
                print(f" [{file_idx+1}/{len(file_ids)}] View {file_id} already processed. Skipping.")
                total_processed += 1
                continue

            print(f" [{file_idx+1}/{len(file_ids)}] Processing view {file_id} ...", end="", flush=True)

            t0 = time.time()

            # 1. 读取参考图像维度
            img = cv2.imread(img_filename)
            if img is None:
                print(f" Error loading image {img_filename}, skipped.")
                continue
            H, W, _ = img.shape

            # 2. 读取深度真值
            depth_png = cv2.imread(depth_filename_hr, cv2.IMREAD_UNCHANGED)
            if depth_png is None:
                print(f" Error loading depth {depth_filename_hr}, skipped.")
                continue
            depth_hr = depth_png.astype(np.float32) / 64.0
            depth_hr = np.nan_to_num(depth_hr, nan=0.0, posinf=0.0, neginf=0.0)

            # 3. 读取相机参数
            if not os.path.exists(proj_mat_filename):
                print(f" Error cam file {proj_mat_filename} missing, skipped.")
                continue
            intrinsics, extrinsics, depth_min_, depth_max_, depth_interval = read_whu_cam(proj_mat_filename)

            # 4. 读取三角网剖分数据
            if not os.path.exists(triangulation_filename):
                print(f" Error triangulation file {triangulation_filename} missing, skipped.")
                continue
            cdt_data = get_cdt_datas(triangulation_filename, H=H, W=W)
            vertexs = np.asarray(cdt_data.vertexs, dtype=np.float32)
            lines = np.asarray(cdt_data.lines, dtype=np.int64)
            triangles = []
            for t in cdt_data.triangles:
                tri_v = np.asarray(t.vertex_ids, dtype=np.int64)
                tri_l = np.asarray(t.line_ids, dtype=np.int64)
                tri_pts = np.asarray(t.valid_points, dtype=np.int64)
                triangles.append({'vertex_ids': tri_v, 'line_ids': tri_l, 'valid_points': tri_pts})

            # 5. 执行拓扑转换得到 tri_id_map_stage0 (原分辨率)
            v_batch = [torch.from_numpy(vertexs).to(device)]
            l_batch = [torch.from_numpy(lines).to(device)]
            t_batch = [[(torch.from_numpy(t['vertex_ids']).to(device),
                         torch.from_numpy(t['line_ids']).to(device),
                         torch.from_numpy(t['valid_points']).to(device)) for t in triangles]]

            # 调用已有的 batch_convert_to_tri_infos_new
            tri_infos = batch_convert_to_tri_infos_new(v_batch, l_batch, t_batch, H, W, device)
            tri_id_map_stage0 = np.squeeze(tri_infos[0]['tri_id_map_stage0'][0].cpu().numpy())

            # 6. 计算 OLS 初始置信度
            _, tri_conf_dict = compute_gt_planar_soft_confidence(depth_hr, tri_id_map_stage0, intrinsics, sigma=0.05, min_pixels=3)

            # 7. SVD 平面拟合 (GPU)
            max_tri_num = int(np.max(tri_id_map_stage0)) + 1
            depth_min_tensor = torch.tensor([depth_min_], device=device).float().view(1, 1)
            depth_max_tensor = torch.tensor([depth_max_], device=device).float().view(1, 1)

            fitter_svd = DensePlaneFitter(
                height_s1=H,
                width_s1=W,
                device=device,
                num_hypotheses=1,
                depth_min=depth_min_tensor,
                depth_max=depth_max_tensor
            )
            visualizer_svd = PlaneVisualizer(height=H, width=W, device=device)

            depth_tensor = torch.from_numpy(depth_hr).unsqueeze(0).unsqueeze(0).to(device).float()
            tri_id_tensor = torch.from_numpy(tri_id_map_stage0).unsqueeze(0).to(device).long()
            intrinsics_tensor = torch.from_numpy(intrinsics).unsqueeze(0).to(device).float()

            hypotheses, *rest = fitter_svd.get_plane_hypotheses(
                depth_stage2=depth_tensor,
                tri_id_map=tri_id_tensor,
                intrinsics_s1=intrinsics_tensor,
                max_num_triangles=max_tri_num
            )
            current_planes = hypotheses[:, :, 0, :].detach()

            depth_range = (depth_min_tensor, depth_max_tensor)
            depth_gt_svd_tensor, _ = visualizer_svd.render_from_planes(
                plane_params=current_planes,
                tri_id_map=tri_id_tensor,
                intrinsics=intrinsics_tensor,
                depth_range=depth_range
            )
            depth_gt_svd = np.squeeze(depth_gt_svd_tensor.cpu().numpy())

            # 8. 误差计算与 95% 分位数统计
            pixel_err_map = np.abs(depth_gt_svd - depth_hr)
            current_planes_np = current_planes[0].cpu().numpy()  # [max_tri_num, 4]

            tri_initial_conf = np.zeros(max_tri_num, dtype=np.float32)
            tri_svd_err95 = np.zeros(max_tri_num, dtype=np.float32)
            tri_svd_plane = np.zeros((max_tri_num, 4), dtype=np.float32)

            for tid in range(max_tri_num):
                tri_initial_conf[tid] = tri_conf_dict.get(tid, 0.0)
                px_mask = (tri_id_map_stage0 == tid) & (depth_gt_svd > 0) & (depth_hr > 0)
                if np.any(px_mask):
                    tri_svd_err95[tid] = np.percentile(pixel_err_map[px_mask], 95)
                    param = current_planes_np[tid].copy()
                    normal = param[:3]
                    d_val = param[3]
                    norm_val = np.linalg.norm(normal)
                    if norm_val > 1e-8:
                        unit_normal = normal / norm_val
                        unit_d = d_val / norm_val
                    else:
                        unit_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                        unit_d = 0.0
                    if unit_normal[2] < 0:
                        unit_normal = -unit_normal
                        unit_d = -unit_d
                    tri_svd_plane[tid] = np.array([unit_normal[0], unit_normal[1], unit_normal[2], unit_d], dtype=np.float32)
                else:
                    tri_svd_err95[tid] = 0.0
                    tri_svd_plane[tid] = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

            # 9. 压缩保存为 .npz 文件
            os.makedirs(out_dir, exist_ok=True)
            np.savez_compressed(
                out_filename,
                tri_initial_conf=tri_initial_conf,
                tri_svd_err95=tri_svd_err95,
                tri_svd_plane=tri_svd_plane
            )

            t_elapsed = time.time() - t0
            print(f" Done ({t_elapsed:.2f}s). File size: {os.path.getsize(out_filename)/1024.0:.1f} KB")
            total_processed += 1

    print(f"\nProcessing finished! Total processed views: {total_processed}")

if __name__ == '__main__':
    main()
