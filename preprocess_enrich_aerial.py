#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ENRICH-Aerial 航空遥感数据集专用的离线几何特征 (SVD 平面 + 初始置信度) GPU 预处理工具
================================================================================
本脚本针对 ENRICH-Aerial 裁剪切片数据集设计：
1. 自动从每个 Scan 的 pair.txt 中精确解析参考视角 ID (默认 00000002)；
2. 读取标准 MVS 格式相机参数 (*_cam.txt) 与高精度 32 位浮点物理深度 (*.pfm)；
3. 读取 CDT 三角网剖分数据，生成原分辨率三角形拓扑图 (tri_id_map_stage0)；
4. 结合深度不连续边缘计算 OLS 初始平面置信度 (tri_initial_conf)；
5. 在 GPU 上利用 DensePlaneFitter 批量执行 SVD 平面参数拟合，并通过 PlaneVisualizer 渲染 SVD 重构深度图；
6. 统计每个三角形面片的 95% 分位数绝对拟合残差 (tri_svd_err95) 与单位法向量平面方程 (tri_svd_plane)；
7. 压缩保存至各 Scan 的 triangulation/geom_svd_{ref_id}.npz，原生支持断点续跑。
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
import torch

# 确保脚本能够正确引入项目内模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 确保在 Windows 控制台下正确输出 UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from datasets.data_io import read_pfm
from datasets.triangulation import get_cdt_datas
from models.PlanePatchMatch import DensePlaneFitter, PlaneVisualizer
from utils import batch_convert_to_tri_infos_new


def read_cam_file(filename):
    """
    读取标准 MVS / DTU 格式相机参数文件 (*_cam.txt)
    格式:
      extrinsic
      r00 r01 r02 t0
      ...
      intrinsic
      fx  0   cx
      0   fy  cy
      0   0   1
      depth_min depth_interval depth_num depth_max
    """
    with open(filename, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    # 1. 读取外参 (行 1~4)
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))

    # 2. 读取内参 (行 6~8)
    intrinsics = np.fromstring(' '.join(lines[6:9]), dtype=np.float32, sep=' ').reshape((3, 3))

    # 3. 读取深度范围 (行 9)
    tokens = list(map(float, lines[9].split()))
    depth_min = tokens[0]
    if len(tokens) >= 4:
        depth_interval = tokens[1]
        depth_num = int(tokens[2])
        depth_max = tokens[3]
    elif len(tokens) == 2:
        depth_max = tokens[1]
        depth_interval = (depth_max - depth_min) / 192.0
        depth_num = 192
    else:
        depth_interval = tokens[1] if len(tokens) > 1 else 1.0
        depth_max = tokens[-1]
        depth_num = 192

    return intrinsics, extrinsics, depth_min, depth_max, depth_interval


def get_ref_view_id_from_pair(pair_path, default_id=2):
    """从 pair.txt 中自动提取参考视角编号 (行 2)"""
    if not os.path.exists(pair_path):
        return default_id
    try:
        with open(pair_path, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        if len(lines) >= 2:
            return int(lines[1])
    except Exception:
        pass
    return default_id


def compute_gt_planar_soft_confidence(gt_depth_s0, tri_id_s0, intrinsics_s0, sigma=0.05, min_pixels=3):
    """
    计算三角形的 OLS 几何平整度初始置信度
    - 利用深度梯度结合三角形边界剔除遮挡跳变杂质
    - 针对小三角形执行尺度补偿
    """
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
    fx, fy = intrinsics_s0[0, 0], intrinsics_s0[1, 1]
    cx, cy = intrinsics_s0[0, 2], intrinsics_s0[1, 2]

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
        tri_conf_dict[tri_id] = float(np.clip(conf, 0.0, 1.0))

    return planar_soft_conf_s0, tri_conf_dict


def parse_args():
    parser = argparse.ArgumentParser(description="ENRICH-Aerial 离线 SVD 几何特征 GPU 预处理工具")
    parser.add_argument("--datapath", type=str, default=r"E:\RemoteCodeEx\ENRICH-Aerial_Cropped_0.6x_Overlap50",
                        help="ENRICH-Aerial 数据集根目录")
    parser.add_argument("--listfile", type=str, default=None,
                        help="待处理的 Scan 列表文件路径 (默认优先使用 datapath 下的 scan_list.txt)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="运行设备 (如 cuda:0 或 cpu)")
    parser.add_argument("--force", action="store_true",
                        help="是否强制重新计算已存在的 geom_svd_*.npz 文件")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("  [START] 启动 ENRICH-Aerial 离线 SVD 平面拟合与几何特征预处理流水线")
    print("=" * 80)
    print(f" [DATA] 数据集根目录 : {args.datapath}")
    print(f" [DEVICE] 运算硬件设备 : {device}")

    # 1. 确定 Scan 列表
    if args.listfile and os.path.exists(args.listfile):
        list_path = args.listfile
    else:
        candidate_list = os.path.join(args.datapath, "scan_list.txt")
        if os.path.exists(candidate_list):
            list_path = candidate_list
        else:
            list_path = None

    if list_path and os.path.exists(list_path):
        with open(list_path, 'r', encoding='utf-8') as f:
            scans = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        print(f" [LIST] Scan 列表来源: {list_path} (共 {len(scans)} 个场景)")
    else:
        scans = sorted([d for d in os.listdir(args.datapath)
                        if os.path.isdir(os.path.join(args.datapath, d)) and d.startswith("scan_")])
        print(f" [LIST] 自动扫描目录发现: {len(scans)} 个 Scan 场景")

    if len(scans) == 0:
        print("[ERROR] 未在指定目录下检索到任何 Scan 场景，请核实路径！")
        return

    print("=" * 80)
    total_processed = 0
    total_skipped = 0
    t_start_all = time.time()

    for idx, scan in enumerate(scans):
        scan_dir = os.path.join(args.datapath, scan)
        pair_path = os.path.join(scan_dir, "pair.txt")
        ref_id_num = get_ref_view_id_from_pair(pair_path, default_id=2)
        ref_id_str = f"{ref_id_num:08d}"

        # 路径声明
        img_path = os.path.join(scan_dir, "images", f"{ref_id_str}.jpg")
        depth_path = os.path.join(scan_dir, "depths", f"{ref_id_str}.pfm")
        cam_path = os.path.join(scan_dir, "cams", f"{ref_id_str}_cam.txt")
        tri_dir = os.path.join(scan_dir, "triangulation")
        cdt_path = os.path.join(tri_dir, "CDTinfo", f"CDT_info_vlf_{ref_id_str}.txt")
        out_npz = os.path.join(tri_dir, f"geom_svd_{ref_id_str}.npz")

        # 断点续跑支持
        if not args.force and os.path.exists(out_npz) and os.path.getsize(out_npz) > 1024:
            print(f" [{idx+1:03d}/{len(scans):03d}] {scan} (视角 {ref_id_str}) 已就绪，跳过。")
            total_skipped += 1
            continue

        print(f" [{idx+1:03d}/{len(scans):03d}] 处理中: {scan} (参考视角 {ref_id_str}) ... ", end="", flush=True)
        t0 = time.time()

        # 1. 验证图像并获取分辨率
        img = cv2.imread(img_path)
        if img is None:
            print(f"[ERROR] 图像缺失: {img_path}")
            continue
        H, W = img.shape[:2]

        # 2. 读取 PFM 32位单通道真实物理深度 (单位: 米)
        if not os.path.exists(depth_path):
            print(f"[ERROR] 深度图缺失: {depth_path}")
            continue
        try:
            depth_hr, _ = read_pfm(depth_path)
            if depth_hr.ndim == 3:
                depth_hr = depth_hr.squeeze(2)
            depth_hr = np.nan_to_num(depth_hr, nan=0.0, posinf=0.0, neginf=0.0)
            depth_hr[depth_hr <= 0.0] = 0.0
        except Exception as e:
            print(f"[ERROR] 读取深度 PFM 异常: {e}")
            continue

        # 3. 读取标准 MVS 相机参数
        if not os.path.exists(cam_path):
            print(f"[ERROR] 相机文件缺失: {cam_path}")
            continue
        intrinsics, extrinsics, depth_min_, depth_max_, depth_interval = read_cam_file(cam_path)

        # 4. 读取 CDT 三角网剖分数据
        if not os.path.exists(cdt_path):
            print(f"[ERROR] CDT剖分文件缺失: {cdt_path}")
            continue
        try:
            cdt_data = get_cdt_datas(cdt_path, H=H, W=W)
            vertexs = np.asarray(cdt_data.vertexs, dtype=np.float32)
            lines = np.asarray(cdt_data.lines, dtype=np.int64)
            triangles = []
            for t in cdt_data.triangles:
                triangles.append({
                    'vertex_ids': np.asarray(t.vertex_ids, dtype=np.int64),
                    'line_ids': np.asarray(t.line_ids, dtype=np.int64),
                    'valid_points': np.asarray(t.valid_points, dtype=np.int64)
                })
        except Exception as e:
            print(f"[ERROR] 解析 CDT 剖分异常: {e}")
            continue

        # 5. 生成原分辨率三角形拓扑图 (tri_id_map_stage0)
        v_batch = [torch.from_numpy(vertexs).to(device)]
        l_batch = [torch.from_numpy(lines).to(device)]
        t_batch = [[(torch.from_numpy(t['vertex_ids']).to(device),
                     torch.from_numpy(t['line_ids']).to(device),
                     torch.from_numpy(t['valid_points']).to(device)) for t in triangles]]

        with torch.no_grad():
            tri_infos = batch_convert_to_tri_infos_new(v_batch, l_batch, t_batch, H, W, device)
            tri_id_map_stage0 = np.squeeze(tri_infos[0]['tri_id_map_stage0'][0].cpu().numpy())

            # 6. 计算 OLS 初始置信度
            _, tri_conf_dict = compute_gt_planar_soft_confidence(
                depth_hr, tri_id_map_stage0, intrinsics, sigma=0.05, min_pixels=3
            )

            # 7. GPU SVD 平面拟合与渲染
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

            depth_gt_svd_tensor, _ = visualizer_svd.render_from_planes(
                plane_params=current_planes,
                tri_id_map=tri_id_tensor,
                intrinsics=intrinsics_tensor,
                depth_range=(depth_min_tensor, depth_max_tensor)
            )
            depth_gt_svd = np.squeeze(depth_gt_svd_tensor.cpu().numpy())

        # 8. 误差计算与 95% 分位数统计
        pixel_err_map = np.abs(depth_gt_svd - depth_hr)
        current_planes_np = current_planes[0].cpu().numpy()

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

                # 规范化：确保法向 Z 朝向相机正向 (nz > 0)
                if unit_normal[2] < 0:
                    unit_normal = -unit_normal
                    unit_d = -unit_d

                tri_svd_plane[tid] = np.array([unit_normal[0], unit_normal[1], unit_normal[2], unit_d], dtype=np.float32)
            else:
                tri_svd_err95[tid] = 0.0
                tri_svd_plane[tid] = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

        # 9. 压缩保存至 Scan 内部的 triangulation 目录
        os.makedirs(tri_dir, exist_ok=True)
        np.savez_compressed(
            out_npz,
            tri_initial_conf=tri_initial_conf,
            tri_svd_err95=tri_svd_err95,
            tri_svd_plane=tri_svd_plane
        )

        t_cost = time.time() - t0
        sz_kb = os.path.getsize(out_npz) / 1024.0
        print(f"[OK] 完成 ({t_cost:.2f}s, {max_tri_num} 面片, {sz_kb:.1f} KB)")
        total_processed += 1

    t_all = time.time() - t_start_all
    print("=" * 80)
    print(f"[DONE] 全部处理完成！耗时: {t_all:.1f}s | 新生成: {total_processed} 个, 跳过已有: {total_skipped} 个")
    print(f"[OUTPUT] 几何特征文件已安全写入各 Scan 下的: triangulation/geom_svd_*.npz")
    print("=" * 80)


if __name__ == '__main__':
    main()
