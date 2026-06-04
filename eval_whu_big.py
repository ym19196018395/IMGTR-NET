import argparse
import os

from matplotlib import pyplot as plt
from tensorboard.plugins.hparams.metadata import NULL_TENSOR

os.environ["CUDA_VISIBLE_DEVICES"] = "4" #ym_add 要在torch之前因为要让服务器只看得见第二张卡
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
from datasets import find_dataset_def
from models import *
from utils import *
import sys
from datasets.data_io import read_pfm, save_pfm
import cv2
from plyfile import PlyData, PlyElement
from PIL import Image
from datasets.dtu_whu import collate_keep_list

import resource
import platform

# if platform.system() == 'Linux':
#     soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
#     # 将软限制（soft limit）提升到 65535，或者提升到系统允许的硬限制（hard limit）
#     resource.setrlimit(resource.RLIMIT_NOFILE, (min(65535, hard), hard))
#     print(f"设置文件句柄限制为: {min(65535, hard)}")

# torch.multiprocessing.set_sharing_strategy('file_system')

os.environ['CUDA_LAUNCH_BLOCKING'] = "0"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Predict depth, filter, and fuse')
parser.add_argument('--model', default='PatchmatchNet', help='select model')

parser.add_argument('--dataset', default='dtu_yao_eval', help='select dataset')
parser.add_argument('--testpath', help='testing data path')
parser.add_argument('--testlist', help='testing scan list')

parser.add_argument('--batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--n_views', type=int, default=5, help='num of view')


parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--outdir', default='./outputs', help='output dir')
parser.add_argument('--display', action='store_true', help='display depth images and masks')

parser.add_argument('--patchmatch_iteration', nargs='+', type=int, default=[1,2,2], 
        help='num of iteration of patchmatch on stages 1,2,3')
parser.add_argument('--patchmatch_num_sample', nargs='+', type=int, default=[8,8,16], 
        help='num of generated samples in local perturbation on stages 1,2,3')
parser.add_argument('--patchmatch_interval_scale', nargs='+', type=float, default=[0.005, 0.0125, 0.025], 
        help='normalized interval in inverse depth range to generate samples in local perturbation')
parser.add_argument('--patchmatch_range', nargs='+', type=int, default=[6,4,2], 
        help='fixed offset of sampling points for propogation of patchmatch on stages 1,2,3')
parser.add_argument('--propagate_neighbors', nargs='+', type=int, default=[0,8,16], 
        help='num of neighbors for adaptive propagation on stages 1,2,3')
parser.add_argument('--evaluate_neighbors', nargs='+', type=int, default=[9,9,9], 
        help='num of neighbors for adaptive matching cost aggregation of adaptive evaluation on stages 1,2,3')

parser.add_argument('--geo_pixel_thres', type=float, default=1, help='pixel threshold for geometric consistency filtering')
parser.add_argument('--geo_depth_thres', type=float, default=0.01, help='depth threshold for geometric consistency filtering')
parser.add_argument('--photo_thres', type=float, default=0.8, help='threshold for photometric consistency filtering')

# parse arguments and check
args = parser.parse_args()
print("argv:", sys.argv[1:])
print_args(args)


# read intrinsics and extrinsics
def read_camera_parameters(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # --- 保护措施 1: 预先检查行数 ---
    # 根据 whuMVS 格式，相机文件至少需要包含外参、内参、深度等信息，通常 > 9 行
    # 这里至少保证能读完外参 (前5行)
    if len(lines) < 5:
        raise IndexError(f"文件{filename}行数不足，仅有 {len(lines)} 行，无法解析外参。")

    # --- 1. 读取外参 (Lines 2-5) ---
    extrinsics = []
    for i in range(1, 5):
        # 去除首尾空格
        line_str = lines[i].strip()

        # --- 保护措施 2: 检查是不是空行 ---
        if not line_str:
            raise ValueError(f"文件{filename}第 {i + 1} 行内容为空，无法解析数值。")

        extrinsics.append(list(map(float, line_str.split())))

    extrinsics = np.array(extrinsics, dtype=np.float32)

    # --- 2. 读取原始内参 (Line 7) ---
    # f, x0, y0
    vals = list(map(float, lines[6].strip().split()))
    f_val = vals[0]
    x0 = vals[1]
    y0 = vals[2]

    # 构建内参矩阵 (注意 whuMVS 的 -f 定义)
    intrinsics = np.array([
        [-f_val, 0, x0],
        [0, f_val, y0],
        [0, 0, 1]
    ], dtype=np.float32)

    return intrinsics, extrinsics



# read an image
def read_img(filename, img_wh):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.float32) / 255.
    np_img = cv2.resize(np_img, img_wh, interpolation=cv2.INTER_LINEAR)
    return np_img


# save a binary mask
def save_mask(filename, mask):
    assert mask.dtype == bool
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)

def save_depth_img(filename, depth):
    # assert mask.dtype == np.bool
    depth = depth.astype(np.float32) * 255
    Image.fromarray(depth).save(filename)


def read_list_file(filename):
    data = []
    with open(filename) as f:
        for line in f:
            # 1. 去除行尾的换行符和空白
            clean_line = line.strip()
            # 2. 确保行不为空
            if clean_line:
                # 3. 去除后缀 (例如 .png)
                file_id = clean_line.split('.')[0]
                # 4. 加入集合,只能传一个参数可以传一个元组数
                data.append(file_id)
    return data

def read_pair_file(filename):
    data = []
    with open(filename) as pair:
        pair_lines = pair.readlines()
        # 过滤掉空行，防止报错
        pair_lines = [line.strip() for line in pair_lines if line.strip()]

        for pairline in pair_lines:
            values = pairline.split()
            # 1. 解析参考视图 (每行的第一个数)
            ref_view = int(values[0])
            # 2. 解析源视图 (每行剩下的数)
            # 你的文件格式只有ID没有分数，所以直接取 [1:] 即可
            src_views = [int(x) for x in values[1:]]
            data.append((ref_view, src_views))
    return data

# run MVS model to save depth maps
def save_depth():
    # dataset, dataloader
    MVSDataset = find_dataset_def(args.dataset)
    test_dataset = MVSDataset(args.testpath, args.testlist, "test", args.n_views)
    # 忘记加入collate_fn
    # TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4, drop_last=False)
    # todo:图片太大了无法多进程
    TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=collate_keep_list,
                               num_workers=0,
                               drop_last=False)

    # model
    model = PatchmatchNet(patchmatch_interval_scale=args.patchmatch_interval_scale,
                propagation_range = args.patchmatch_range, patchmatch_iteration=args.patchmatch_iteration, 
                patchmatch_num_sample = args.patchmatch_num_sample, 
                propagate_neighbors=args.propagate_neighbors, evaluate_neighbors=args.evaluate_neighbors)
    # ym_modify 取消并行处理
    # model = nn.DataParallel(model)
    # model.cuda()
    model.to(device)

    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])
    model.eval()
    
    with torch.no_grad():
        for batch_idx, sample in enumerate(TestImgLoader):
            start_time = time.time()

            vertexs_batch = [torch.from_numpy(v).to(device) for v in sample['vertexs']]
            lines_batch = [torch.from_numpy(v).to(device) for v in sample['lines']]
            triangles_batch = []
            for tri_list in sample['triangles']:  # tri_list 是一个 sample 的 triangles
                tri_processed = []
                for t in tri_list:
                    v_ids = torch.from_numpy(t['vertex_ids']).to(device)
                    l_ids = torch.from_numpy(t['line_ids']).to(device)
                    pts = torch.from_numpy(t['valid_points']).to(device)  # variable len
                    tri_processed.append((v_ids, l_ids, pts))
                triangles_batch.append(tri_processed)

            # ym-modify 重写了一下对于cdt—data数据进行了一个跳过
            skip = ["vertexs", "lines", "triangles"]
            sample_cuda = tocuda(sample, device=device, skip_keys=skip)

            # 获取可能存在的 Stage 1 真值深度（测试集如果没有则传 None）
            depth_gt = sample_cuda.get("depth", None)
            depth_stage_1 = depth_gt['stage_1'] if depth_gt is not None else None

            # max_lambda_c = 100.0
            # max_lambda_s = 3.0
            max_lambda_c = 0.0
            max_lambda_s = 0.0
            current_temp =0.55
            outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["intrinsics_mats"],
                            sample_cuda["depth_min"], sample_cuda["depth_max"],
                            vertexs_batch, lines_batch, triangles_batch, depth_stage_1,
                            max_lambda_c, max_lambda_s,current_temp)


            image_outputs_pre = generate_edge_alpha_overlays(
                ref_imgs=sample["imgs"]['stage_0'][:, 0],
                edge_alphas_list=outputs["edge_alphas"],
                edges_pixels_list=outputs["tri_infos"][0]['edges_pixels'],
                device=device, overlay_alpha=0.6, line_thickness=1
            )
            # 2. 提取 Tensor (假设 batch_size=1)
            # 结果形状为 [B, 3, H, W], float32, RGB
            res_tensor = image_outputs_pre["ref_img_edge_alpha"][0]

            # 3. 转换：Tensor -> CPU -> Numpy -> [H, W, C]
            res_np = res_tensor.detach().cpu().numpy()
            res_np = np.transpose(res_np, (1, 2, 0))

            # 4. 映射到 0-255 uint8
            res_img_uint8 = (res_np * 255.0).clip(0, 255).astype(np.uint8)

            # 5. 保存图片
            # 构造文件名，例如保存为 edge_overlay.png
            filename=sample["filename"][0]
            save_path = os.path.join(args.outdir, filename.format('edge_overlay', '.png'))
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            Image.fromarray(res_img_uint8).save(save_path)

            outputs = tensor2numpy(outputs)
            del sample_cuda
            print('Iter {}/{}, time = {:.3f}'.format(batch_idx, len(TestImgLoader), time.time() - start_time))
            filenames = sample["filename"]

            # ====================================================================
            # 💡 核心可视化逻辑: 提取 Stage 1 深度图和平面法向量图
            # ====================================================================
            #
            stage1_depths = outputs["depth_patchmatch"]['stage_1'][-1]
            stage1_normals = outputs["output_plane"]['normal_pro']
            tri_id_maps_np = outputs["output_plane"]['tri_id_map']

            # 🎯 架构师新增：提取像素级原生深度和 Stage 0 网格掩码
            stage1_depths_pixels = outputs["output_plane"]['depth_stage1_pixels']
            tri_id_maps_s0_np = outputs["output_plane"]['tri_id_map_stage0']

            # 统一转 numpy
            if isinstance(tri_id_maps_np, torch.Tensor):
                tri_id_maps_np = tri_id_maps_np.detach().cpu().numpy()
            if isinstance(tri_id_maps_s0_np, torch.Tensor):
                tri_id_maps_s0_np = tri_id_maps_s0_np.detach().cpu().numpy()

            if depth_stage_1 is not None:
                depth_gt_np = depth_stage_1.detach().cpu().numpy()
            else:
                depth_gt_np = None

            # stage1_depths = outputs["output_plane"]['depth_stage1_pixels']
            # stage1_normals = outputs["output_plane"]['normal_no_pro']

            for b_idx, (filename, depth_est, normal_est) in enumerate(zip(filenames, stage1_depths, stage1_normals)):
                # 1. 设定输出路径
                depth_filename = os.path.join(args.outdir, filename.format('depth_est_s1', '.pfm'))
                normal_filename = os.path.join(args.outdir, filename.format('normal_est_s1', '.png'))
                diff_filename = os.path.join(args.outdir, filename.format('depth_diff_s1', 'dif.png'))  # 差异图路径

                os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(normal_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(diff_filename.rsplit('/', 1)[0], exist_ok=True)

                # 去除多余的维度
                depth_est = np.squeeze(depth_est)

                # ==========================================
                # 【深度图可视化】
                # ==========================================
                valid_mask = depth_est > 0

                if valid_mask.any():
                    # 百分位数截断
                    d_min = np.percentile(depth_est[valid_mask], 1)
                    d_max = np.percentile(depth_est[valid_mask], 99)

                    # 线性归一化到 0.0 ~ 1.0 之间
                    depth_vis = (depth_est - d_min) / (d_max - d_min + 1e-8)
                    depth_vis = np.clip(depth_vis, 0, 1)

                    # 映射到 0 ~ 255 的 uint8 空间
                    depth_vis_uint8 = (depth_vis * 255).astype(np.uint8)

                    # 样式 1: JET 伪彩色图
                    depth_color = cv2.applyColorMap(depth_vis_uint8, cv2.COLORMAP_JET)
                    depth_color[~valid_mask] = 0
                    vis_color_filename = depth_filename.replace('.pfm', '_vis.png')
                    cv2.imwrite(vis_color_filename, depth_color)

                    # 样式 2: 类似 Tensorboard 的灰度图
                    depth_gray = depth_vis_uint8.copy()
                    depth_gray[~valid_mask] = 255
                    vis_gray_filename = depth_filename.replace('.pfm', '_black_vis.png')
                    cv2.imwrite(vis_gray_filename, depth_gray)


                # ==========================================
                # 【法向量图可视化】
                # ==========================================
                normal_est_sq = np.squeeze(normal_est)
                if normal_est_sq.ndim == 3 and normal_est_sq.shape[-1] == 3:
                    normal_est_sq = np.transpose(normal_est_sq, (2, 0, 1))

                if normal_est_sq.max() <= 2.0:
                    normal_vis_uint8 = (np.clip(normal_est_sq, 0.0, 1.0) * 255.0).astype(np.uint8)
                else:
                    normal_vis_uint8 = np.clip(normal_est_sq, 0, 255).astype(np.uint8)

                normal_vis_rgb = np.transpose(normal_vis_uint8, (1, 2, 0))
                if valid_mask is not None:
                    normal_vis_rgb[~valid_mask] = 0
                Image.fromarray(normal_vis_rgb).save(normal_filename)

                # ====================================================================
                # 提前提取平面置信度 (W_plane)，供平面 MAE 切分和最终保存使用
                # ====================================================================
                depth_est_sq = np.squeeze(depth_est)  # [H, W]
                w_plane_sq = []
                if 'W_plane_pixel' in outputs["output_plane"]:
                    w_plane_data = outputs["output_plane"]['W_plane_pixel'][b_idx]
                    w_plane_np = w_plane_data.detach().cpu().numpy() if isinstance(w_plane_data,
                                                                                   torch.Tensor) else w_plane_data
                    w_plane_sq = np.squeeze(w_plane_np)
                    if w_plane_sq.shape != depth_est_sq.shape:
                        w_plane_sq = cv2.resize(w_plane_sq, (depth_est_sq.shape[1], depth_est_sq.shape[0]),
                                                interpolation=cv2.INTER_LINEAR)
                else:
                    w_plane_sq = np.zeros_like(depth_est_sq)

                # ====================================================================
                # 🚨 3. 生成 Stage 1 预测与 GT 的差异热力图 (带全局与平面专属 MAE)
                # ====================================================================
                if depth_gt_np is not None:
                    gt_curr = np.squeeze(depth_gt_np[b_idx])
                    tri_id_curr = np.squeeze(tri_id_maps_np[b_idx])

                    if gt_curr.shape != depth_est_sq.shape:
                        gt_curr = cv2.resize(gt_curr, (depth_est_sq.shape[1], depth_est_sq.shape[0]),
                                             interpolation=cv2.INTER_NEAREST)
                    if tri_id_curr.shape != depth_est_sq.shape:
                        tri_id_curr = cv2.resize(tri_id_curr, (depth_est_sq.shape[1], depth_est_sq.shape[0]),
                                                 interpolation=cv2.INTER_NEAREST)

                    mask_diff = (depth_est_sq > 0) & (gt_curr > 0) & (tri_id_curr >= 0)

                    if mask_diff.any():
                        diff_map = np.zeros_like(depth_est_sq)
                        abs_error_array = np.abs(depth_est_sq[mask_diff] - gt_curr[mask_diff])
                        diff_map[mask_diff] = abs_error_array
                        mean_abs_error = np.mean(abs_error_array)

                        # 过滤提取高不确定性平面区域 (W_plane > 0.80)
                        mask_planar = mask_diff & (w_plane_sq > 0.80)
                        if mask_planar.any():
                            planar_mae = np.mean(np.abs(depth_est_sq[mask_planar] - gt_curr[mask_planar]))
                            planar_text = f"\nFlat Region MAE (W>0.80): {planar_mae:.4f}m"
                        else:
                            planar_mae = 0.0
                            planar_text = ""

                        # ── 图一：全局误差图（基于 replace 防爆设计） ────────────────────────
                        diff_max_plot = max(np.percentile(abs_error_array, 95), 0.5)
                        
                        plt.figure(figsize=(10, 8))
                        diff_map_masked = np.ma.masked_where(~mask_diff, diff_map)
                        cmap = plt.get_cmap('jet')
                        cmap.set_bad(color='black')
                        
                        im = plt.imshow(diff_map_masked, cmap=cmap, vmin=0, vmax=diff_max_plot)
                        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
                        cbar.set_label('Absolute Error (Meters)', size=14)
                        plt.title(f'Stage 1 MAE: {mean_abs_error:.4f}m {planar_text}\nMax Cutoff: {diff_max_plot:.2f}m',
                                  fontsize=14, fontweight='bold')
                        plt.axis('off')
                        
                        # 刚性落锁路径：先 format 拿到标准基准路径
                        diff_filename = os.path.join(args.outdir, filename.format('depth_diff_s1', '_dif.png'))
                        os.makedirs(os.path.dirname(diff_filename), exist_ok=True) # 🛡️ 刚性子目录防爆铁闸一
                        plt.savefig(diff_filename, dpi=150, bbox_inches='tight', pad_inches=0.1)
                        plt.close()

                        # ── 图二：仅平面区域误差图（W_plane > 0.80，满足用户只展现平面 MAE 的渴望） ──
                        if mask_planar.any():
                            planar_err_array = np.abs(depth_est_sq[mask_planar] - gt_curr[mask_planar])
                            planar_diff_map = np.zeros_like(depth_est_sq)
                            planar_diff_map[mask_planar] = planar_err_array

                            # 误差上限用平面区域自己的 95 百分位，使得色标分布针对平面高分辨率拉伸，拒绝曲面大误差污染
                            planar_diff_max = max(np.percentile(planar_err_array, 95), 0.1)

                            plt.figure(figsize=(10, 8))
                            # 刚性隔离：将平面区域外（曲面、虚空、背景）全部标记为 True 实施严格黑色放逐
                            planar_map_masked = np.ma.masked_where(~mask_planar, planar_diff_map)
                            cmap_planar = plt.get_cmap('jet')
                            cmap_planar.set_bad(color='black') # 外部曲面死死抹黑
                            
                            im2 = plt.imshow(planar_map_masked, cmap=cmap_planar, vmin=0, vmax=planar_diff_max)
                            cbar2 = plt.colorbar(im2, fraction=0.046, pad=0.04)
                            cbar2.set_label('Absolute Error (Meters)', size=14)
                            
                            plt.title(
                                f'Flat Region MAE (W_plane > 0.80): {planar_mae:.4f}m'
                                f'\nMax Cutoff: {planar_diff_max:.2f}m'
                                f'  |  Flat pixels: {mask_planar.sum()} / {mask_diff.sum()}',
                                fontsize=13, fontweight='bold'
                            )
                            plt.axis('off')
                            
                            # 🛡️ 核心修复：直接采用字符串级特征平替，在全局文件名基础上加上 _planar 后缀，
                            # 彻底规避原模板 filename 中由于多传参数炸出多层未建立文件夹的致命 Bug！
                            planar_diff_filename = diff_filename.replace('_dif.png', '_dif_planar.png')
                            os.makedirs(os.path.dirname(planar_diff_filename), exist_ok=True) # 🛡️ 刚性子目录防爆铁闸二
                            
                            plt.savefig(planar_diff_filename, dpi=150, bbox_inches='tight', pad_inches=0.1)
                            plt.close()

                # ====================================================================
                # 🚨 4. 生成平面置信度 (W_plane) 大图 (用于论文展示)
                # ====================================================================
                # 🚨 1. 必须修正：不要使用字符串硬编码，直接从已解析的物理场中获取
                # 确保你的上游数据字典里的 key 是 'W_plane_pixel' 且已经广播回了 [B, H, W]
                if 'W_plane_pixel' in outputs.get("output_plane", {}):
                    w_plane_data = outputs["output_plane"]['W_plane_pixel'][b_idx]
                    w_plane_np = w_plane_data.detach().cpu().numpy() if isinstance(w_plane_data,
                                                                                   torch.Tensor) else w_plane_data
                    w_plane_sq = np.squeeze(w_plane_np)
                    
                    # 强制尺寸闭环
                    if w_plane_sq.shape != depth_est_sq.shape:
                        w_plane_sq = cv2.resize(w_plane_sq, (depth_est_sq.shape[1], depth_est_sq.shape[0]), interpolation=cv2.INTER_LINEAR)
                        
                    # 🎯 【硬核修正】：直接使用你在上方计算好的有效 valid_mask，不要用 locals()!
                    # 如果 valid_mask 未定义，这里直接赋值为一个全 1 的掩码
                    current_mask = valid_mask if 'valid_mask' in locals() and valid_mask is not None else np.ones_like(depth_est_sq, dtype=bool)

                    conf_jet_filename = os.path.join(args.outdir, filename.format('confidence_s1', '_jet.png'))
                    os.makedirs(os.path.dirname(conf_jet_filename), exist_ok=True) # 强制建立路径

                    w_plane_uint8 = (np.clip(w_plane_sq, 0.0, 1.0) * 255.0).astype(np.uint8)
                    w_plane_uint8_inv = 255 - w_plane_uint8 

                    # 应用掩码，处理虚空与背景
                    w_plane_uint8_inv[~current_mask] = 0

                    w_plane_color = cv2.applyColorMap(w_plane_uint8_inv, cv2.COLORMAP_JET)
                    w_plane_color[~current_mask] = 0 # 虚空区涂黑

                    cv2.imwrite(conf_jet_filename, w_plane_color)
                    print(f"✅ 置信度图已成功写入: {conf_jet_filename}")

                    # =====================================================================
                    # 👑 【新增对比实验硬核资产】：熔炼并导出 W > 0.80 的 0/1 刚性流形掩码二值图
                    # =====================================================================
                    # 机制精剖：将连续概率空间一刀切。大于0.80的黄金平面记为 255（纯白），其余及背景统统归 0（纯黑）
                    binary_mask_np = np.zeros_like(w_plane_sq, dtype=np.uint8)
                    binary_mask_np[(w_plane_sq > 0.80) & current_mask] = 255

                    # 动态生成专属基准文件名，加上 _oracle_mask 后缀，防止混淆文件目录
                    oracle_mask_filename = os.path.join(args.outdir, filename.format('plane_mask_01', '_oracle_mask.png'))
                    os.makedirs(os.path.dirname(oracle_mask_filename), exist_ok=True)
                    
                    # 写入单通道灰度/二值图磁盘
                    cv2.imwrite(oracle_mask_filename, binary_mask_np)
                    print(f"👑 黄金实验 0/1 刚性掩码已导出至: {oracle_mask_filename}")
                else:
                    # 彻底告别沉默，报错提示你 key 没匹配上
                    print(f"⚠️ 警告: outputs['output_plane'] 中未找到 'W_plane_pixel'，跳过置信度可视化")

    

                
                # ====================================================================
                # 🚨 新增功能 3：生成 Stage 1 纯自由像素级 (Pixel-wise) 深度的差异热力图
                # 这是最原汁原味的 Baseline，用于和网格约束后的结果做对比
                # ====================================================================
                if depth_gt_np is not None:
                    depth_pixel_curr = stage1_depths_pixels[b_idx]
                    if isinstance(depth_pixel_curr, torch.Tensor):
                        depth_pixel_curr = np.squeeze(depth_pixel_curr.detach().cpu().numpy())
                    else:
                        depth_pixel_curr = np.squeeze(depth_pixel_curr)

                    # 尺寸对齐
                    if gt_curr.shape != depth_pixel_curr.shape:
                        gt_curr_pixel = cv2.resize(gt_curr, (depth_pixel_curr.shape[1], depth_pixel_curr.shape[0]),
                                                   interpolation=cv2.INTER_NEAREST)
                    else:
                        gt_curr_pixel = gt_curr

                    # 对于原生深度，只需要保证自身和GT都>0即可
                    mask_diff_pixel = (depth_pixel_curr > 0) & (gt_curr_pixel > 0)

                    if mask_diff_pixel.any():
                        diff_map_pixel = np.zeros_like(depth_pixel_curr)
                        abs_error_pixel = np.abs(depth_pixel_curr[mask_diff_pixel] - gt_curr_pixel[mask_diff_pixel])
                        diff_map_pixel[mask_diff_pixel] = abs_error_pixel

                        mean_error_pixel = np.mean(abs_error_pixel)
                        max_plot_pixel = max(np.percentile(abs_error_pixel, 95), 0.5)

                        plt.figure(figsize=(10, 8))
                        im_pixel = plt.imshow(np.ma.masked_where(~mask_diff_pixel, diff_map_pixel), cmap=cmap, vmin=0,
                                              vmax=max_plot_pixel)
                        cbar_pixel = plt.colorbar(im_pixel, fraction=0.046, pad=0.04)
                        cbar_pixel.set_label('Absolute Error (Meters)', size=14)

                        plt.title(
                            f'Baseline (Pixel-wise) MAE: {mean_error_pixel:.4f}m\nMax Cutoff: {max_plot_pixel:.2f}m',
                            fontsize=14, fontweight='bold')
                        plt.axis('off')

                        diff_filename_pixel = os.path.join(args.outdir,
                                                           filename.format('depth_diff_s1_pixel', '_dif.png'))
                        os.makedirs(os.path.dirname(diff_filename_pixel), exist_ok=True)
                        plt.savefig(diff_filename_pixel, dpi=150, bbox_inches='tight', pad_inches=0.1)
                        plt.close()

                # ====================================================================
                # 🚨 5. 终极更新版：生成 Stage 0 (Refined) 深度图与 GT 的热力图
                # 🎯 修复了你指出的漏洞：强制引入 tri_id_map_stage0 掩码，隔绝虚空噪点！
                # ====================================================================
                if "refined_depth" in outputs and "stage_0" in sample["depth"]:
                    depth_est_s0 = outputs["refined_depth"]["stage_0"]
                    if isinstance(depth_est_s0, torch.Tensor):
                        depth_est_s0_sq = np.squeeze(depth_est_s0.detach().cpu().numpy())
                    else:
                        depth_est_s0_sq = np.squeeze(depth_est_s0)

                    gt_curr_s0 = np.squeeze(sample["depth"]["stage_0"][b_idx].detach().cpu().numpy())
                    tri_id_curr_s0 = np.squeeze(tri_id_maps_s0_np[b_idx])  # 🎯 提取 Stage 0 专属网格掩码

                    # 尺寸保护
                    if gt_curr_s0.shape != depth_est_s0_sq.shape:
                        gt_curr_s0 = cv2.resize(gt_curr_s0, (depth_est_s0_sq.shape[1], depth_est_s0_sq.shape[0]),
                                                interpolation=cv2.INTER_NEAREST)
                    if tri_id_curr_s0.shape != depth_est_s0_sq.shape:
                        tri_id_curr_s0 = cv2.resize(tri_id_curr_s0,
                                                    (depth_est_s0_sq.shape[1], depth_est_s0_sq.shape[0]),
                                                    interpolation=cv2.INTER_NEAREST)

                    # 🔥🔥🔥 你的神级指正：加入 (tri_id_curr_s0 >= 0) 的安全锁！
                    mask_diff_s0 = (depth_est_s0_sq > 0) & (gt_curr_s0 > 0) & (tri_id_curr_s0 >= 0)

                    if mask_diff_s0.any():
                        diff_map_s0 = np.zeros_like(depth_est_s0_sq)
                        abs_error_array_s0 = np.abs(depth_est_s0_sq[mask_diff_s0] - gt_curr_s0[mask_diff_s0])
                        diff_map_s0[mask_diff_s0] = abs_error_array_s0

                        mean_abs_error_s0 = np.mean(abs_error_array_s0)

                        # 计算 Stage 0 阶段的纯平面区域 MAE
                        w_plane_sq_s0 = cv2.resize(w_plane_sq, (depth_est_s0_sq.shape[1], depth_est_s0_sq.shape[0]),
                                                   interpolation=cv2.INTER_LINEAR)
                        mask_planar_s0 = mask_diff_s0 & (w_plane_sq_s0 > 0.80)
                        if mask_planar_s0.any():
                            planar_mae_s0 = np.mean(
                                np.abs(depth_est_s0_sq[mask_planar_s0] - gt_curr_s0[mask_planar_s0]))
                            planar_text_s0 = f"\nFlat Region MAE (W>0.80): {planar_mae_s0:.4f}m"
                        else:
                            planar_text_s0 = ""

                        diff_max_plot_s0 = max(np.percentile(abs_error_array_s0, 95), 0.5)

                        plt.figure(figsize=(10, 8))
                        diff_map_masked_s0 = np.ma.masked_where(~mask_diff_s0, diff_map_s0)
                        cmap_s0 = plt.get_cmap('jet')
                        cmap_s0.set_bad(color='black')

                        im_s0 = plt.imshow(diff_map_masked_s0, cmap=cmap_s0, vmin=0, vmax=diff_max_plot_s0)
                        cbar_s0 = plt.colorbar(im_s0, fraction=0.046, pad=0.04)
                        cbar_s0.set_label('Absolute Error (Meters)', size=14)

                        plt.title(
                            f'Stage 0 Refined MAE: {mean_abs_error_s0:.4f}m {planar_text_s0}\nMax Cutoff: {diff_max_plot_s0:.2f}m',
                            fontsize=14, fontweight='bold')
                        plt.axis('off')

                        diff_filename_s0 = os.path.join(args.outdir, filename.format('depth_diff_s0', '_dif.png'))
                        os.makedirs(os.path.dirname(diff_filename_s0), exist_ok=True)
                        plt.savefig(diff_filename_s0, dpi=150, bbox_inches='tight', pad_inches=0.1)
                        plt.close()

                        # ====================================================================
                        # 🎯 【新增功能】Stage 0 (Refined) 密集深度图黑白灰度图可视化
                        # ====================================================================
                        # 提取 Stage 0 合法几何区域掩码
                        valid_mask_s0_vis = (depth_est_s0_sq > 0) & (tri_id_curr_s0 >= 0)

                        if valid_mask_s0_vis.any():
                            # 1% ~ 99% 百分位数动态截断，消除虚空飞点的噪声拉伸
                            d_min_s0 = np.percentile(depth_est_s0_sq[valid_mask_s0_vis], 1)
                            d_max_s0 = np.percentile(depth_est_s0_sq[valid_mask_s0_vis], 99)

                            # 线性映射至 0.0 ~ 1.0 晶格空间
                            depth_vis_s0 = (depth_est_s0_sq - d_min_s0) / (d_max_s0 - d_min_s0 + 1e-8)
                            depth_vis_s0 = np.clip(depth_vis_s0, 0, 1)
                            depth_vis_uint8_s0 = (depth_vis_s0 * 255).astype(np.uint8)

                            # 复制并建立 Tensorboard 风格灰度图（有效区变暗，背景区刷白）
                            depth_gray_s0 = depth_vis_uint8_s0.copy()
                            depth_gray_s0[~valid_mask_s0_vis] = 255

                            # 3. 动态构建输出路径并写入磁盘
                            vis_gray_filename_s0 = os.path.join(args.outdir,
                                                                filename.format('depth_est_s0', '_black_vis.png'))
                            os.makedirs(os.path.dirname(vis_gray_filename_s0), exist_ok=True)
                            cv2.imwrite(vis_gray_filename_s0, depth_gray_s0)



# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src,
                                geo_pixel_thres, geo_depth_thres):
    '''
    几何一致性过滤 (Geometric Consistency),核心部分
    '''
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref,
                                                     depth_src, intrinsics_src, extrinsics_src)
    # print(depth_ref.shape)
    # print(depth_reprojected.shape)
    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    # depth_ref = np.squeeze(depth_ref, 2)
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    mask = np.logical_and(dist < geo_pixel_thres, relative_depth_diff < geo_depth_thres)
    depth_reprojected[~mask] = 0

    return mask, depth_reprojected, x2d_src, y2d_src


def filter_depth(scan_folder, out_folder, plyfilename, geo_pixel_thres, geo_depth_thres, photo_thres, img_wh):
    # the pair file
    pair_file = os.path.join(scan_folder, "cams/pair.txt")
    # for the final point cloud
    vertexs = []
    vertex_colors = []

    pair_data = read_pair_file(pair_file)
    nviews = len(pair_data)
    original_w = 768
    original_h = 384
    

    # for each reference view and the corresponding source views
    for ref_view, src_views in pair_data:
        # load the camera parameters
        ref_intrinsics, ref_extrinsics = read_camera_parameters(
            os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(ref_view)))
        ref_intrinsics[0] *= img_wh[0]/original_w
        ref_intrinsics[1] *= img_wh[1]/original_h
        # load the reference image
        ref_img = read_img(os.path.join(scan_folder, 'blended_images/{:0>8}.jpg'.format(ref_view)), img_wh)
        # load the estimated depth of the reference view
        ref_depth_est = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(ref_view)))[0]
        ref_depth_est = np.squeeze(ref_depth_est, 2)
        # load the photometric mask of the reference view
        confidence = read_pfm(os.path.join(out_folder, 'confidence/{:0>8}.pfm'.format(ref_view)))[0]
        # 如果某点的置信度低于阈值 photo_thres,置信度掩码图
        photo_mask = confidence > photo_thres
        photo_mask = np.squeeze(photo_mask, 2)
        

        all_srcview_depth_ests = []
        

        # compute the geometric mask
        geo_mask_sum = 0
        for src_view in src_views:
            # camera parameters of the source view
            src_intrinsics, src_extrinsics = read_camera_parameters(
                os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(src_view)))
            src_intrinsics[0] *= img_wh[0]/original_w
            src_intrinsics[1] *= img_wh[1]/original_h
            # the estimated depth of the source view
            src_depth_est = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(src_view)))[0]
            

            geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(ref_depth_est, ref_intrinsics, ref_extrinsics,
                                                                      src_depth_est,
                                                                      src_intrinsics, src_extrinsics,
                                                                      geo_pixel_thres, geo_depth_thres)
            geo_mask_sum += geo_mask.astype(np.int32)
            all_srcview_depth_ests.append(depth_reprojected)
            

        depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)
        # at least 3 source views matched
        # large threshold, high accuracy, low completeness
        geo_mask = geo_mask_sum >= 1
        final_mask = np.logical_and(photo_mask, geo_mask)
        

        os.makedirs(os.path.join(out_folder, "mask"), exist_ok=True)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_photo.png".format(ref_view)), photo_mask)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_geo.png".format(ref_view)), geo_mask)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_final.png".format(ref_view)), final_mask)
        os.makedirs(os.path.join(out_folder, "depth_img"), exist_ok=True)


        print("processing {}, ref-view{:0>2}, geo_mask:{:3f} photo_mask:{:3f} final_mask: {:3f}".format(scan_folder, ref_view,
                                                                geo_mask.mean(), photo_mask.mean(), final_mask.mean()))

        if args.display:
            import cv2
            cv2.imshow('ref_img', ref_img[:, :, ::-1])
            cv2.imshow('ref_depth', ref_depth_est / 800)
            cv2.imshow('ref_depth * photo_mask', ref_depth_est * photo_mask.astype(np.float32) / 800)
            cv2.imshow('ref_depth * geo_mask', ref_depth_est * geo_mask.astype(np.float32) / 800)
            cv2.imshow('ref_depth * mask', ref_depth_est * final_mask.astype(np.float32) / 800)
            cv2.waitKey(1)

        height, width = depth_est_averaged.shape[:2]
        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
        
        valid_points = final_mask
        # print("valid_points", valid_points.mean())
        x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]
        
        color = ref_img[valid_points]
        xyz_ref = np.matmul(    np.linalg.inv(ref_intrinsics),
                            np.vstack((x, y, np.ones_like(x))) * depth)
        xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                              np.vstack((xyz_ref, np.ones_like(x))))[:3]
        vertexs.append(xyz_world.transpose((1, 0)))
        vertex_colors.append((color * 255).astype(np.uint8))

        
    vertexs = np.concatenate(vertexs, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)
    print("saving the final model to", plyfilename)


def filter_depth_new( scans ,plyfilename, geo_pixel_thres, geo_depth_thres, photo_thres, img_wh):
    '''
    没有用几何一致性过滤，只用置信度过滤
    '''

    # 用来存放所有子单元生成的点，最后再一次性写入 plyfilename
    all_vertexs = []
    all_colors = []

    for scan in scans:
        scan_folder = os.path.join(args.testpath, "test/Images/{}".format(scan))
        out_folder = os.path.join(args.outdir, scan)

        pair_file = os.path.join(scan_folder, "list.txt")
        # 进行了修改适配whu
        pair_data = read_pair_file(pair_file)

        # 提高光度阈值！因为没有几何校验了，必须把这个调高，否则噪点很多
        # 建议设为 0.8 或 0.9
        FINAL_CONFIDENCE_THRES = 0.8
        print(f"注意：几何一致性已关闭，仅使用置信度过滤 (阈值: {FINAL_CONFIDENCE_THRES})")

        for ref_view in pair_data:
            # ref_view 现在是类似 "006_8/images/1.png" 这样的路径
            # 我们需要解析出它是哪个子文件夹的

            # --- 1. 加载 Ref 数据 ---
            # 路径处理需要根据你的 pair.txt 格式微调
            # 假设 ref_view 是相对路径
            ref_img_path = os.path.join(scan_folder, "urd/{}.png".format(ref_view))
            # 对应的深度图路径 (PatchMatchNet 输出的)
            ref_depth_path = os.path.join(out_folder, "depth_est/{}.pfm".format(ref_view))
            # 对应的置信度路径
            ref_conf_path = os.path.join(out_folder, "confidence/{}.pfm".format(ref_view))
            # 对应的相机参数
            ref_cam_path = os.path.join(args.testpath, "test/Cams/{}/1/{}.txt".format(scan,ref_view))

            if not os.path.exists(ref_depth_path):
                print(f"跳过: 找不到深度图 {ref_depth_path}")
                continue

            ref_img = read_img(ref_img_path, img_wh)
            # 读取方式和dataloader保持一致
            ref_intrinsics, ref_extrinsics = read_camera_parameters(ref_cam_path)

            # 调整内参比例 (如果图片缩放过)
            original_w, original_h = 768, 384  # 假设原图尺寸
            ref_intrinsics[0] *= img_wh[0] / original_w
            ref_intrinsics[1] *= img_wh[1] / original_h

            ref_depth_est = read_pfm(ref_depth_path)[0]
            ref_depth_est = np.squeeze(ref_depth_est, 2)
            confidence = read_pfm(ref_conf_path)[0]
            confidence = np.squeeze(confidence, 2)

            # --- 2. 关键修改：只进行光度过滤 ---
            # 放弃 geometric consistency，因为我们没有 src_views 的深度图

            # 生成掩码：只保留置信度高的点
            final_mask = confidence > FINAL_CONFIDENCE_THRES

            # 保存一下掩码图片方便检查 (可选)
            # save_mask_path = os.path.join(out_folder, "mask", os.path.dirname(ref_view))
            # os.makedirs(save_mask_path, exist_ok=True)
            # save_mask(os.path.join(save_mask_path, "mask_final.png"), final_mask)

            print(f"Processing {ref_view}, valid points: {final_mask.mean():.4f}")

            # --- 3. 反投影生成点云 ---
            height, width = ref_depth_est.shape[:2]
            x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))

            # 应用掩码
            valid_points = final_mask
            x, y, depth = x[valid_points], y[valid_points], ref_depth_est[valid_points]
            color = ref_img[valid_points]

            # 2D -> 3D
            xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                                np.vstack((x, y, np.ones_like(x))) * depth)
            xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                                  np.vstack((xyz_ref, np.ones_like(x))))[:3]

            # 收集当前这块砖的点
            all_vertexs.append(xyz_world.transpose((1, 0)))
            all_colors.append((color * 255).astype(np.uint8))


    # --- 4. 合并保存所有点云 ---
    if not all_vertexs:
        print("没有生成任何有效点云！")
        return

    all_vertexs = np.concatenate(all_vertexs, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)

    # 转换为 PLY 格式所需的结构化数组
    vertexs_tuple = np.array([tuple(v) for v in all_vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    colors_tuple = np.array([tuple(v) for v in all_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs_tuple), vertexs_tuple.dtype.descr + colors_tuple.dtype.descr)
    for prop in vertexs_tuple.dtype.names:
        vertex_all[prop] = vertexs_tuple[prop]
    for prop in colors_tuple.dtype.names:
        vertex_all[prop] = colors_tuple[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)
    print(f"所有碎片已合并，最终模型保存至: {plyfilename}")


def filter_depth_new2( scans ,plyfilename, geo_pixel_thres, geo_depth_thres, photo_thres, img_wh):
    '''
    不计算几何一致性，将参考图和源图深度图直接转化为点云给一个极高的阈值0.9
    '''

    # 用来存放所有子单元生成的点，最后再一次性写入 plyfilename
    all_vertexs = []
    all_colors = []

    for scan in scans:
        # 图片路径
        scan_folder = os.path.join(args.testpath, "test/Images/{}".format(scan))
        # 模型输出结果的路径
        out_folder = os.path.join(args.outdir, scan)
        # 图片ids
        list_file = os.path.join(scan_folder, "list.txt")
        # 进行了修改适配whu
        list_data = read_pair_file(list_file)

        # 提高光度阈值！因为没有几何校验了，必须把这个调高，否则噪点很多
        FINAL_CONFIDENCE_THRES = 0.9
        print(f"注意：几何一致性已关闭，仅使用置信度过滤 (阈值: {FINAL_CONFIDENCE_THRES})")
        # 将各个视角的图片都加入进来
        view_ids=[0,1,2,3,4]
        for ref_view in view_ids:
            for list_id in list_data:

                # --- 1. 加载 Ref 数据 ---
                ref_img_path=[]
                # 因为参考图读取路径不一样
                if ref_view==1:
                    ref_img_path= os.path.join(scan_folder, "urd/{}.png".format(list_id))
                else:
                    ref_img_path = os.path.join(scan_folder, "{}/{}.png".format(ref_view,list_id))

                # 对应的深度图路径 (PatchMatchNet 输出的)
                ref_depth_path = os.path.join(out_folder, "{}/depth_est/{}.pfm".format(ref_view,list_id))
                # 对应的置信度路径
                ref_conf_path = os.path.join(out_folder, "{}/confidence/{}.pfm".format(ref_view,list_id))
                # 对应的相机参数
                ref_cam_path = os.path.join(args.testpath, "test/Cams/{}/{}/{}.txt".format(scan, ref_view,list_id))

                if not os.path.exists(ref_depth_path):
                    print(f"跳过: 找不到深度图 {ref_depth_path}")
                    continue

                ref_img = read_img(ref_img_path, img_wh)
                # 读取方式和dataloader保持一致
                ref_intrinsics, ref_extrinsics = read_camera_parameters(ref_cam_path)

                # 调整内参比例 (如果图片缩放过)
                original_w, original_h = 768, 384  # 假设原图尺寸
                ref_intrinsics[0] *= img_wh[0] / original_w
                ref_intrinsics[1] *= img_wh[1] / original_h

                ref_depth_est = read_pfm(ref_depth_path)[0]
                ref_depth_est = np.squeeze(ref_depth_est, 2)
                confidence = read_pfm(ref_conf_path)[0]
                confidence = np.squeeze(confidence, 2)

                # --- 2. 关键修改：只进行光度过滤 ---
                # 放弃 geometric consistency，因为我们没有 src_views 的深度图

                # 生成掩码：只保留置信度高的点
                final_mask = confidence > FINAL_CONFIDENCE_THRES

                # 保存一下掩码图片方便检查 (可选)
                save_mask_path = os.path.join(out_folder, "mask/{}".format(ref_view), os.path.dirname(list_id))
                os.makedirs(save_mask_path, exist_ok=True)
                save_mask(os.path.join(save_mask_path, "{}.png".format(list_id)), final_mask)

                print(f"Processing {ref_view}, valid points: {final_mask.mean():.4f}")

                # --- 3. 反投影生成点云 ---
                height, width = ref_depth_est.shape[:2]
                x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))

                # 应用掩码
                valid_points = final_mask
                x, y, depth = x[valid_points], y[valid_points], ref_depth_est[valid_points]
                color = ref_img[valid_points]

                # 2D -> 3D
                xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                                    np.vstack((x, y, np.ones_like(x))) * depth)
                xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                                      np.vstack((xyz_ref, np.ones_like(x))))[:3]

                # 收集当前这块砖的点
                all_vertexs.append(xyz_world.transpose((1, 0)))
                all_colors.append((color * 255).astype(np.uint8))


    # --- 4. 合并保存所有点云 ---
    if not all_vertexs:
        print("没有生成任何有效点云！")
        return

    all_vertexs = np.concatenate(all_vertexs, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)

    # 转换为 PLY 格式所需的结构化数组
    vertexs_tuple = np.array([tuple(v) for v in all_vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    colors_tuple = np.array([tuple(v) for v in all_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs_tuple), vertexs_tuple.dtype.descr + colors_tuple.dtype.descr)
    for prop in vertexs_tuple.dtype.names:
        vertex_all[prop] = vertexs_tuple[prop]
    for prop in colors_tuple.dtype.names:
        vertex_all[prop] = colors_tuple[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)
    print(f"所有碎片已合并，最终模型保存至: {plyfilename}")


def filter_depth_new3( scans ,plyfilename, geo_pixel_thres, geo_depth_thres, photo_thres, img_wh):
    '''
    不计算几何一致性，将参考图和源图深度图直接转化为点云给一个极高的阈值0.9
    '''

    # 用来存放所有子单元生成的点，最后再一次性写入 plyfilename
    all_vertexs = []
    all_colors = []

    # pair list
    pair_file = os.path.join(args.testpath, "test/pair.txt")
    pair_data = read_pair_file(pair_file)
    nviews = len(pair_data)

    for scan in scans:
        # 图片路径
        scan_folder = os.path.join(args.testpath, "test/Images/{}".format(scan))
        # 模型输出结果的路径
        out_folder = os.path.join(args.outdir, scan)
        # 图片ids
        list_file = os.path.join(scan_folder, "list.txt")
        # 进行了修改适配whu
        list_data = read_list_file(list_file)

        # 将各个视角的图片都加入进来
        for ref_view, src_views in pair_data:
            for list_id in list_data:

                # --- 1. 加载 Ref 数据 ---
                ref_img_path=[]
                # 因为参考图读取路径不一样
                if ref_view==1:
                    ref_img_path= os.path.join(scan_folder, "urd/{}.png".format(list_id))
                else:
                    ref_img_path = os.path.join(scan_folder, "{}/{}.png".format(ref_view,list_id))

                # 对应的深度图路径 (PatchMatchNet 输出的)
                ref_depth_path = os.path.join(out_folder, "{}/depth_est/{}.pfm".format(ref_view,list_id))
                # 对应的置信度路径
                ref_conf_path = os.path.join(out_folder, "{}/confidence/{}.pfm".format(ref_view,list_id))
                # 对应的相机参数
                ref_cam_path = os.path.join(args.testpath, "test/Cams/{}/{}/{}.txt".format(scan, ref_view,list_id))

                if not os.path.exists(ref_depth_path):
                    print(f"跳过: 找不到深度图 {ref_depth_path}")
                    continue

                ref_img = read_img(ref_img_path, img_wh)
                # 读取方式和dataloader保持一致
                ref_intrinsics, ref_extrinsics = read_camera_parameters(ref_cam_path)

                # 调整内参比例 (如果图片缩放过)
                original_w, original_h = 768, 384  # 假设原图尺寸
                ref_intrinsics[0] *= img_wh[0] / original_w
                ref_intrinsics[1] *= img_wh[1] / original_h

                ref_depth_est = read_pfm(ref_depth_path)[0]
                ref_depth_est = np.squeeze(ref_depth_est, 2)
                # 读取光度置信度
                confidence = read_pfm(ref_conf_path)[0]

                # --- 2. 获取source图,并计算几何和光度一致性 ---

                # 生成掩码：只保留置信度高的点
                photo_mask = confidence > photo_thres
                photo_mask = np.squeeze(photo_mask, 2)

                all_srcview_depth_ests = []

                # compute the geometric mask
                geo_mask_sum = 0

                for src_view in src_views:
                    # camera parameters of the source view
                    source_cam_path = os.path.join(args.testpath, "test/Cams/{}/{}/{}.txt".format(scan, src_view,list_id))
                    src_intrinsics, src_extrinsics = read_camera_parameters(source_cam_path)
                    src_intrinsics[0] *= img_wh[0] / original_w
                    src_intrinsics[1] *= img_wh[1] / original_h
                    # the estimated depth of the source view
                    source_depth_path=os.path.join(out_folder, "{}/depth_est/{}.pfm".format(src_view,list_id))
                    src_depth_est = read_pfm(source_depth_path)[0]

                    geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(ref_depth_est,
                                                                                                ref_intrinsics,
                                                                                                ref_extrinsics,
                                                                                                src_depth_est,
                                                                                                src_intrinsics,
                                                                                                src_extrinsics,
                                                                                                geo_pixel_thres,
                                                                                                geo_depth_thres)
                    geo_mask_sum += geo_mask.astype(np.int32)
                    all_srcview_depth_ests.append(depth_reprojected)

                depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)
                # at least 3 source views matched
                # large threshold, high accuracy, low completeness
                geo_mask = geo_mask_sum >= 1
                final_mask = np.logical_and(photo_mask, geo_mask)

                # 保存一下掩码图片方便检查 (可选)

                os.makedirs(os.path.join(out_folder, "mask/{}".format(ref_view)), exist_ok=True)
                save_mask(os.path.join(out_folder, "mask/{}/{}_photo.png".format(ref_view,list_id)), photo_mask)
                save_mask(os.path.join(out_folder, "mask/{}/{}_geo.png".format(ref_view,list_id)), geo_mask)
                save_mask(os.path.join(out_folder, "mask/{}/{}_final.png".format(ref_view,list_id)), final_mask)
                os.makedirs(os.path.join(out_folder, "depth_img"), exist_ok=True)

                print("processing {}, ref-view{:0>2}, geo_mask:{:3f} photo_mask:{:3f} final_mask: {:3f}".format(
                    scan_folder, ref_view,
                    geo_mask.mean(), photo_mask.mean(), final_mask.mean()))

                # --- 3. 反投影生成点云 ---
                height, width = depth_est_averaged.shape[:2]
                x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))

                # 应用掩码
                valid_points = final_mask
                x, y, depth = x[valid_points], y[valid_points], ref_depth_est[valid_points]
                color = ref_img[valid_points]

                # 2D -> 3D
                xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                                    np.vstack((x, y, np.ones_like(x))) * depth)
                xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                                      np.vstack((xyz_ref, np.ones_like(x))))[:3]

                # 收集当前这块砖的点
                all_vertexs.append(xyz_world.transpose((1, 0)))
                all_colors.append((color * 255).astype(np.uint8))


    # --- 4. 合并保存所有点云 ---
    if not all_vertexs:
        print("没有生成任何有效点云！")
        return

    all_vertexs = np.concatenate(all_vertexs, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)

    # 转换为 PLY 格式所需的结构化数组
    vertexs_tuple = np.array([tuple(v) for v in all_vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    colors_tuple = np.array([tuple(v) for v in all_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs_tuple), vertexs_tuple.dtype.descr + colors_tuple.dtype.descr)
    for prop in vertexs_tuple.dtype.names:
        vertex_all[prop] = vertexs_tuple[prop]
    for prop in colors_tuple.dtype.names:
        vertex_all[prop] = colors_tuple[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)
    print(f"所有碎片已合并，最终模型保存至: {plyfilename}")

if __name__ == '__main__':
    # step1. save all the depth maps and the masks in outputs directory
    save_depth()
    # img_wh=(768, 384)
    
    # with open(args.testlist) as f:
    #     scans = f.readlines()
    #     scans = [line.rstrip() for line in scans]
        
    # 将每一张图都作为参考图，然后经过光度一致性和几何一致性过滤
    # for scan in scans:
    #     scan_folder = os.path.join(args.testpath, scan)
    #     out_folder = os.path.join(args.outdir, scan)
    #     # step2. filter saved depth maps with geometric constraints
    #     filter_depth(scan_folder, out_folder, os.path.join(args.outdir, '{}.ply'.format(scan)),
    #                 args.geo_pixel_thres, args.geo_depth_thres, args.photo_thres, img_wh)

    # now-1 不计算几何一致性，简单的用参考图深度图转化为点云
    # filter_depth_new(scans, os.path.join(args.outdir, 'full.ply'),
    #                 args.geo_pixel_thres, args.geo_depth_thres, args.photo_thres, img_wh)

    # now-2 不计算几何一致性，将参考图和源图深度图直接转化为点云给一个极高的阈值0.9
    # filter_depth_new2(scans, os.path.join(args.outdir, 'full_2.ply'),
    #                 args.geo_pixel_thres, args.geo_depth_thres, args.photo_thres, img_wh)

    # now-3 全部都计算
    # filter_depth_new3(scans, os.path.join(args.outdir, 'full_3.ply'),
    #                 args.geo_pixel_thres, args.geo_depth_thres, args.photo_thres, img_wh)