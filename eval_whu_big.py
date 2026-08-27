import argparse

from eval_config import GT_PLANAR_CONF_THRESHOLD, PRED_PLANAR_CONF_THRESHOLD


import os

from matplotlib import pyplot as plt
from tensorboard.plugins.hparams.metadata import NULL_TENSOR

# 控制要暴露给进程的 GPU id：优先使用外部环境变量 GPU_ID，
# 否则使用已有的 CUDA_VISIBLE_DEVICES，最后回退到 '0'
_gpu_choice = os.environ.get('GPU_ID', os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
os.environ['CUDA_VISIBLE_DEVICES'] = str(_gpu_choice)
import torch
import torch.nn as nn
import scipy.ndimage as ndimage
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

error_num=0.2

def visualize_diagnostic_maps(outdir, filename, pixel_cost_min, view_weights_mean, pixel_cost_min_raw, cost_variance, W_plane_pixel, depth_gt, depth_est_2d):
    """
    自愈可视化诊断工具：生成并保存已加权/未加权代价图、视角方差图以及带符号误差图（Signed Error Map）。
    """
    # 延迟加载防止主线程开销
    import matplotlib.pyplot as plt
    import numpy as np

    def make_2d_or_rgb(arr):
        if arr is None:
            return None
        arr_s = np.squeeze(arr)
        if arr_s.ndim == 3:
            # 如果第一维是通道维，且大小不等于后两维（通道数较小）
            if arr_s.shape[0] < arr_s.shape[1] and arr_s.shape[0] < arr_s.shape[2]:
                arr_s = np.min(arr_s, axis=0)
            # 如果最后一维是通道维，且大小不是 3 或 4
            elif arr_s.shape[2] != 3 and arr_s.shape[2] != 4:
                if arr_s.shape[2] == 1:
                    arr_s = arr_s[:, :, 0]
                else:
                    arr_s = np.min(arr_s, axis=2)
        return arr_s

    # 0. 准备黄金平面观测范围：直接以映射完的离线真值平面置信度大于 0.80 的区域为掩码，
    w_s = make_2d_or_rgb(W_plane_pixel)
    gt_s = make_2d_or_rgb(depth_gt)
    
    if w_s is not None:
        flat_mask = (w_s > GT_PLANAR_CONF_THRESHOLD)
        if gt_s is not None:
            flat_mask = flat_mask & (gt_s > 0.0)
    else:
        flat_mask = None

    # 对比度自适应拉伸并在 RGB 三通道层面物理抹黑背景的函数
    def stretch_and_blacken_bg(arr, mask, cmap_name='jet'):
        if arr is None:
            return None
        arr_s = make_2d_or_rgb(arr)
        
        # 1. 仅在 mask 平面内部进行 Min-Max 对比度自适应拉伸
        if mask is not None and mask.shape == arr_s.shape:
            flat_vals = arr_s[mask]
            if flat_vals.size > 10:
                min_v = np.min(flat_vals)
                max_v = np.max(flat_vals)
                # 线性映射到 [0.0, 1.0]
                stretched = (arr_s - min_v) / (max_v - min_v + 1e-8)
                stretched = np.clip(stretched, 0.0, 1.0)
            else:
                stretched = np.zeros_like(arr_s)
        else:
            stretched = (arr_s - arr_s.min()) / (arr_s.max() - arr_s.min() + 1e-8)
            stretched = np.clip(stretched, 0.0, 1.0)
            
        # 2. 将拉伸后的 2D 矩阵通过 Colormap 转换为 RGB 图像 (数据范围 0.0~1.0)
        cmap = plt.get_cmap(cmap_name)
        rgb_img = cmap(stretched)[..., :3] # 去掉 Alpha 得到 [H, W, 3]
        
        # 3. 对非掩码背景像素，在三通道上直接赋予 [0.0, 0.0, 0.0] 物理置黑
        if mask is not None:
            rgb_img[~mask] = 0.0
            
        return rgb_img

    # 1. 可视化最小匹配代价图 (已加权，平地局部对比度自适应拉伸)
    if pixel_cost_min is not None:
        cost_rgb = stretch_and_blacken_bg(pixel_cost_min, flat_mask, cmap_name='jet')
        cost_img_filename = os.path.join(outdir, filename.format('diagnostic_cost_s1', '.png'))
        os.makedirs(os.path.dirname(cost_img_filename), exist_ok=True)
        plt.imsave(cost_img_filename, cost_rgb)

    # 2. 可视化多视平均可见性权重图 (平地局部遮挡)
    if view_weights_mean is not None:
        weights_rgb = stretch_and_blacken_bg(view_weights_mean, flat_mask, cmap_name='gray')
        weights_img_filename = os.path.join(outdir, filename.format('diagnostic_view_weights_s1', '.png'))
        os.makedirs(os.path.dirname(weights_img_filename), exist_ok=True)
        plt.imsave(weights_img_filename, weights_rgb)

    # 3. 可视化未加权匹配代价图 (Raw Cost, 平地局部对比度自适应拉伸)
    if pixel_cost_min_raw is not None:
        raw_rgb = stretch_and_blacken_bg(pixel_cost_min_raw, flat_mask, cmap_name='jet')
        raw_img_filename = os.path.join(outdir, filename.format('diagnostic_cost_raw_s1', '.png'))
        os.makedirs(os.path.dirname(raw_img_filename), exist_ok=True)
        plt.imsave(raw_img_filename, raw_rgb)

    # 4. 可视化视角相似度方差图 (Variance Map, 平地局部对比度自适应拉伸)
    if cost_variance is not None:
        var_rgb = stretch_and_blacken_bg(cost_variance, flat_mask, cmap_name='jet')
        var_img_filename = os.path.join(outdir, filename.format('diagnostic_cost_variance_s1', '.png'))
        os.makedirs(os.path.dirname(var_img_filename), exist_ok=True)
        plt.imsave(var_img_filename, var_rgb)

    # 5. 可视化带符号重建误差图 (Signed Error Map: pred - gt, [-0.20m, +0.20m] 绝对对称映射)
    if depth_est_2d is not None and depth_gt is not None:
        est_s = make_2d_or_rgb(depth_est_2d)
        gt_s = make_2d_or_rgb(depth_gt)
        if est_s is not None and gt_s is not None and est_s.shape == gt_s.shape:
            # 计算 Signed Error: pred - gt
            signed_err = est_s - gt_s
            # 物理截断范围对称锁定在 [-0.20m, +0.20m]
            v_max = 0.20
            # 对称归一化到 [0.0, 1.0]，使 0 误差严格对齐 0.5 (纯白色)
            stretched_err = (signed_err + v_max) / (2.0 * v_max)
            stretched_err = np.clip(stretched_err, 0.0, 1.0)
            
            # 使用 RdBu_r 发散型色表 (正值红色，负值蓝色，零值白色)
            cmap_div = plt.get_cmap('RdBu_r')
            rgb_err = cmap_div(stretched_err)[..., :3]
            
            # 非平面区（~flat_mask）强行赋予 [0.0, 0.0, 0.0] 物理置黑
            if flat_mask is not None and flat_mask.shape == rgb_err.shape[:2]:
                rgb_err[~flat_mask] = 0.0
                
            err_img_filename = os.path.join(outdir, filename.format('diagnostic_signed_error_s1', '.png'))
            os.makedirs(os.path.dirname(err_img_filename), exist_ok=True)
            plt.imsave(err_img_filename, rgb_err)

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

def compute_gt_planar_soft_confidence(gt_depth_s0, tri_id_s0, intrinsics_s0, sigma=0.05, min_pixels=10):
    """
    计算基于 3D 正交最小二乘平面拟合 (OLS/PCA) 与自适应小三角形温标补偿的连续软置信度真值 (W_GT)。
    极简高速版：剔除两阶段迭代与微观法向量运算，仅在 CPU 上利用 numpy 实对称特征分解快速拟合。
    """
    H, W = gt_depth_s0.shape
    planar_soft_conf_s0 = np.zeros((H, W), dtype=np.float32)
    tri_conf_dict = {}

    # 额外处理：找出处于三角形边界且发生深度值突变（悬空）的像素并进行 Dropout 剔除
    # 1. 计算深度图一阶差分以识别深度断裂线（突变阈值设为 0.25米）
    dx = np.zeros_like(gt_depth_s0)
    dy = np.zeros_like(gt_depth_s0)
    dx[:, :-1] = np.abs(gt_depth_s0[:, 1:] - gt_depth_s0[:, :-1])
    dy[:-1, :] = np.abs(gt_depth_s0[1:, :] - gt_depth_s0[:-1, :])
    depth_mutant = (dx > 0.25) | (dy > 0.25)

    # 2. 计算三角形 ID 的分界线（即边界线）
    dtri_x = np.zeros_like(tri_id_s0)
    dtri_y = np.zeros_like(tri_id_s0)
    dtri_x[:, :-1] = tri_id_s0[:, 1:] != tri_id_s0[:, :-1]
    dtri_y[:-1, :] = tri_id_s0[1:, :] != tri_id_s0[:-1, :]
    tri_boundary = (dtri_x != 0) | (dtri_y != 0)
    
    # 膨胀三角形边界，使其完全覆盖边界两侧
    kernel = np.ones((3, 3), dtype=np.uint8)
    tri_boundary_expanded = cv2.dilate(tri_boundary.astype(np.uint8), kernel, iterations=1) > 0

    # 3. 悬边噪点 Dropout
    discard_mask = depth_mutant & tri_boundary_expanded
    
    valid_mask = (gt_depth_s0 > 0) & (tri_id_s0 >= 0) & (~discard_mask)
    if not np.any(valid_mask):
        return planar_soft_conf_s0, tri_conf_dict

    # 1. 快速反投影 3D 点云
    v_indices, u_indices = np.where(valid_mask)
    z = gt_depth_s0[v_indices, u_indices]
    
    fx = intrinsics_s0[0, 0]
    fy = intrinsics_s0[1, 1]
    cx = intrinsics_s0[0, 2]
    cy = intrinsics_s0[1, 2]
    
    x = (u_indices - cx) / fx * z
    y = (v_indices - cy) / fy * z
    points_3d = np.stack([x, y, z], axis=1)  # [M, 3]
    tri_ids_valid = tri_id_s0[v_indices, u_indices].astype(int)

    # 2. 统计各三角形的像素点数
    tri_counts = np.bincount(tri_ids_valid)
    valid_tri_ids = np.where(tri_counts >= min_pixels)[0]
    if len(valid_tri_ids) == 0:
        return planar_soft_conf_s0, tri_conf_dict

    # 3. 按 tri_id 排序以实现高效的分段提取 (规避 mask 过滤开销)
    sort_idx = np.argsort(tri_ids_valid)
    points_sorted = points_3d[sort_idx]
    tri_ids_sorted = tri_ids_valid[sort_idx]

    left_boundaries = np.searchsorted(tri_ids_sorted, valid_tri_ids, side='left')
    right_boundaries = np.searchsorted(tri_ids_sorted, valid_tri_ids, side='right')

    for tri_id, left, right in zip(valid_tri_ids, left_boundaries, right_boundaries):
        pts = points_sorted[left:right]  # [N, 3]
        N = len(pts)
        
        # 4. 快速 OLS 平面拟合 (PCA)
        centroid = np.mean(pts, axis=0)
        pts_centered = pts - centroid
        cov = np.dot(pts_centered.T, pts_centered) / N
        
        try:
            # np.linalg.eigh 针对实对称矩阵比 svd 更加高速稳定
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            normal = eigenvectors[:, 0]  # 最小特征值对应的法向量
        except np.linalg.LinAlgError:
            tri_conf_dict[tri_id] = 0.0
            continue
            
        # 5. 计算平均绝对正交距离
        dists = np.abs(np.dot(pts_centered, normal))
        mean_dist = np.mean(dists)
        
        # 6. 小三角形像素自适应温标补偿 (指数衰减)
        scale_compensator = 1.0 + 1.5 * np.exp(-N / 30.0)
        sigma_adapted = sigma * scale_compensator
        
        # 7. 基于柯西核映射得到 [0, 1] 软概率
        conf = 1.0 / (1.0 + (mean_dist / sigma_adapted) ** 2)
        conf = np.clip(conf, 0.0, 1.0)
        
        tri_conf_dict[tri_id] = float(conf)

    # 8. 建立映射数组，以极快速度进行批量映射填充
    max_id = np.max(tri_id_s0)
    conf_lookup = np.zeros(max_id + 1, dtype=np.float32)
    for tid, val in tri_conf_dict.items():
        if tid <= max_id:
            conf_lookup[tid] = val
            
    planar_soft_conf_s0[valid_mask] = conf_lookup[tri_id_s0[valid_mask].astype(int)]
        
    return planar_soft_conf_s0, tri_conf_dict



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
    # 使用更鲁棒的加载：过滤掉 checkpoint 中当前模型不存在的 key（例如旧版的 cost_smoother）
    state_dict = torch.load(args.loadckpt)
    ckpt_model_state = state_dict.get('model', state_dict)
    current_model_state = model.state_dict()
    # 找出多余的 keys 和缺失的 keys，打印提示以便调试
    ckpt_keys = set(ckpt_model_state.keys())
    model_keys = set(current_model_state.keys())
    unexpected_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    if unexpected_keys:
        print(f"Warning: unexpected keys in checkpoint (will be ignored): {list(sorted(unexpected_keys))[:20]}")
    if missing_keys:
        print(f"Note: missing keys from checkpoint (will be randomly initialized): {list(sorted(missing_keys))[:20]}")

    # 过滤并加载已有的权重，非严格模式以允许部分缺失
    filtered_state = {k: v for k, v in ckpt_model_state.items() if k in model_keys}
    model.load_state_dict(filtered_state, strict=False)
    model.eval()
    
    with torch.no_grad():
        for batch_idx, sample in enumerate(TestImgLoader):
            start_time = time.time()

            # sample['vertexs'] structure: [B, V, ...] because __getitem__ returns a list of V items
            # In save_depth we only use view 0 as reference
            vertexs_batch = [torch.from_numpy(b_list[0]).to(device) for b_list in sample['vertexs']]
            lines_batch = [torch.from_numpy(b_list[0]).to(device) for b_list in sample['lines']]
            triangles_batch = []
            for b_tri_list in sample['triangles']:  # b_tri_list 是一个 batch 中某个 sample 的所有视图 triangles
                tri_list = b_tri_list[0] # 只取参考图 view 0
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

            # ====================================================================
            # 👑 架构师新增：使用特征网络 (FeatureNet) 提取的深度特征来诊断 DoH
            # ====================================================================
            import matplotlib.pyplot as plt
            ref_feat_s1 = outputs["output_plane"]["ref_feature_s1"] # [B, C, H1, W1]
            
            # 1. 弃用图像灰度，改用特征网络多通道能量的平方和，代表局部高维几何响应强度
            feat_energy = (ref_feat_s1 ** 2).sum(dim=1, keepdim=True) # [B, 1, H1, W1]
            
            # 2. 计算二阶差分 (Hessian 矩阵元素)
            f_xx = F.pad(feat_energy[..., 2:] - 2*feat_energy[..., 1:-1] + feat_energy[..., :-2], (1, 1, 0, 0), 'replicate')
            f_yy = F.pad(feat_energy[..., 2:, :] - 2*feat_energy[..., 1:-1, :] + feat_energy[..., :-2, :], (0, 0, 1, 1), 'replicate')
            
            # 交叉二阶导数的纯中心差分公式：(f(x+1,y+1) - f(x-1,y+1) - f(x+1,y-1) + f(x-1,y-1))/4
            f_xy_raw = (feat_energy[..., 2:, 2:] - feat_energy[..., 2:, :-2] - feat_energy[..., :-2, 2:] + feat_energy[..., :-2, :-2]) / 4.0
            f_xy = F.pad(f_xy_raw, (1, 1, 1, 1), 'replicate')
            
            # 3. 计算 DoH 行列式并截断负值（马鞍点）
            doh_map = (f_xx * f_yy - f_xy**2).clamp(min=0.0)
            
            # 解决高频噪声爆点导致的黑洞效应：不用 max，用 98% 分位数截断长尾
            doh_flat = doh_map.view(-1)
            p98 = torch.quantile(doh_flat.float(), 0.98).item()
            p98 = max(p98, 1e-6) # 保底防止除零
            doh_norm = (doh_map / p98).clamp(max=1.0)  # [B, 1, H1, W1]

            # 4. 可视化：将其转换为热力图 Tensor
            doh_np = doh_norm[0, 0].detach().cpu().numpy()
            doh_rgb = plt.get_cmap('jet')(doh_np)[..., :3]  # [H1, W1, 3]
            doh_rgb_tensor_s1 = torch.from_numpy(doh_rgb).permute(2, 0, 1).unsqueeze(0).to(device).float() # [B, 3, H1, W1]

            # 🚀 几何对齐校正：网格坐标已被上采样至 Stage 0 (H*2, W*2)，因此将热力图插值对齐
            doh_rgb_tensor_s0 = F.interpolate(doh_rgb_tensor_s1, scale_factor=2.0, mode='bilinear', align_corners=False)

            # [屏蔽：不需要这些可视化了]
            # 🚀 架构师新增：保存纯净的 DoH 热力图 (无网格线干扰)，以判断特征点是否在三角形内部！
            # doh_pure_np = np.transpose(doh_rgb_tensor_s0[0].detach().cpu().numpy(), (1, 2, 0))
            # doh_pure_uint8 = (doh_pure_np * 255.0).clip(0, 255).astype(np.uint8)
            # doh_pure_save_path = os.path.join(args.outdir, filename.format('diagnostic_doh_pure', '.png'))
            # os.makedirs(os.path.dirname(doh_pure_save_path), exist_ok=True)
            # Image.fromarray(doh_pure_uint8).save(doh_pure_save_path)

            # # 5. 叠加网格边缘 (强制使用绿色 [0, 255, 0])
            # doh_edge_overlays = generate_edge_alpha_overlays(
            #     ref_imgs=doh_rgb_tensor_s0,
            #     edge_alphas_list=outputs["edge_alphas"],
            #     edges_pixels_list=outputs["tri_infos"][0]['edges_pixels'],
            #     device=device, overlay_alpha=0.6, line_thickness=1,
            #     force_color=[0, 255, 0]
            # )

            # # 6. 保存带有网格覆盖的 DoH 诊断图
            # doh_res_tensor = doh_edge_overlays["ref_img_edge_alpha"][0]
            # doh_res_np = np.transpose(doh_res_tensor.detach().cpu().numpy(), (1, 2, 0))
            # doh_res_uint8 = (doh_res_np * 255.0).clip(0, 255).astype(np.uint8)
            # doh_save_path = os.path.join(args.outdir, filename.format('diagnostic_doh_edge', '.png'))
            # os.makedirs(os.path.dirname(doh_save_path), exist_ok=True)
            # Image.fromarray(doh_res_uint8).save(doh_save_path)
            # ====================================================================

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



                # ====================================================================
                # 👑 【新增静态软平面真值】：计算 Stage 0 物理平面软置信度，并利用拓扑对齐到 Stage 1
                # ====================================================================
                planar_soft_conf_s0 = None
                planar_soft_conf_s1 = None
                gt_normal_map_s1 = None

                # 🎯 极速查表模式：由于双重验证已证明离线预处理精度 100% 对齐一致，
                # 现彻底移除耗时的在线 3D OLS 拟合与 GPU SVD 拟合计算，直接采用离线读取的数据进行极速查表映射！
                if "tri_conf_cleaned" in sample and "tri_normal_cleaned" in sample:
                    tri_conf_pre = sample["tri_conf_cleaned"][b_idx]
                    tri_normal_pre = sample["tri_normal_cleaned"][b_idx]
                    
                    tri_conf_pre = tri_conf_pre.detach().cpu().numpy() if isinstance(tri_conf_pre, torch.Tensor) else tri_conf_pre
                    tri_normal_pre = tri_normal_pre.detach().cpu().numpy() if isinstance(tri_normal_pre, torch.Tensor) else tri_normal_pre

                    # 1. 映射得到 Stage 1 尺寸的软置信度与法向量真值图
                    tri_id_curr = np.squeeze(tri_id_maps_np[b_idx])
                    if isinstance(tri_id_curr, torch.Tensor):
                        tri_id_curr = tri_id_curr.detach().cpu().numpy()
                    
                    H_s1, W_s1 = depth_est.shape
                    if tri_id_curr.shape != (H_s1, W_s1):
                        tri_id_curr = cv2.resize(tri_id_curr, (W_s1, H_s1), interpolation=cv2.INTER_NEAREST)

                    planar_soft_conf_s1 = np.zeros_like(tri_id_curr, dtype=np.float32)
                    valid_id_mask = (tri_id_curr >= 0) & (tri_id_curr < len(tri_conf_pre))
                    planar_soft_conf_s1[valid_id_mask] = tri_conf_pre[tri_id_curr[valid_id_mask]]

                    gt_normal_map_s1 = np.zeros(tri_id_curr.shape + (3,), dtype=np.float32)
                    gt_normal_map_s1[valid_id_mask] = tri_normal_pre[tri_id_curr[valid_id_mask]]

                    # 2. 映射得到 Stage 0 尺寸的置信度真值图
                    if tri_id_maps_s0_np is not None:
                        tri_id_curr_s0 = np.squeeze(tri_id_maps_s0_np[b_idx])
                        if isinstance(tri_id_curr_s0, torch.Tensor):
                            tri_id_curr_s0 = tri_id_curr_s0.detach().cpu().numpy()
                        if "depth" in sample and "stage_0" in sample["depth"]:
                            gt_curr_s0_raw = sample["depth"]["stage_0"][b_idx]
                            gt_curr_s0 = np.squeeze(gt_curr_s0_raw.detach().cpu().numpy() if isinstance(gt_curr_s0_raw, torch.Tensor) else gt_curr_s0_raw)
                            if tri_id_curr_s0.shape != gt_curr_s0.shape:
                                tri_id_curr_s0 = cv2.resize(tri_id_curr_s0, (gt_curr_s0.shape[1], gt_curr_s0.shape[0]), interpolation=cv2.INTER_NEAREST)
                        
                        planar_soft_conf_s0 = np.zeros_like(tri_id_curr_s0, dtype=np.float32)
                        valid_id_mask_s0 = (tri_id_curr_s0 >= 0) & (tri_id_curr_s0 < len(tri_conf_pre))
                        planar_soft_conf_s0[valid_id_mask_s0] = tri_conf_pre[tri_id_curr_s0[valid_id_mask_s0]]
                else:
                    print("⚠️ Warning: sample 中未找到 'tri_conf_cleaned' 或 'tri_normal_cleaned'，且在线计算已移除，无法获取平面参数。")

                # 生成 Ground-Truth Planar Normal Map (normal_gt_s1.png)
                # 用户要求生成全部的平面的法向量，因此移除了原本 (planar_soft_conf_s1 > GT_PLANAR_CONF_THRESHOLD) 的优质平面置信度限制
                
                # ====================================================================
                # 👑 【自愈改造·光度 Ambiguity 可视化与 Pearson 诊断】并网
                # ====================================================================
                if planar_soft_conf_s1 is not None:
                    pixel_cost_val = outputs["output_plane"]["pixel_cost_min"][b_idx] if "pixel_cost_min" in outputs["output_plane"] else None
                    view_weights_val = outputs["output_plane"]["view_weights_mean"][b_idx] if "view_weights_mean" in outputs["output_plane"] else None
                    pixel_cost_raw_val = outputs["output_plane"]["pixel_cost_min_raw"][b_idx] if "pixel_cost_min_raw" in outputs["output_plane"] else None
                    cost_variance_val = outputs["output_plane"]["cost_variance"][b_idx] if "cost_variance" in outputs["output_plane"] else None

                    # [屏蔽：不需要这些可视化了]
                    # visualize_diagnostic_maps(
                    #     outdir=args.outdir,
                    #     filename=filename,
                    #     pixel_cost_min=pixel_cost_val.detach().cpu().numpy().squeeze() if hasattr(pixel_cost_val, 'detach') else (pixel_cost_val.squeeze() if pixel_cost_val is not None else None),
                    #     view_weights_mean=view_weights_val.detach().cpu().numpy().squeeze() if hasattr(view_weights_val, 'detach') else (view_weights_val.squeeze() if view_weights_val is not None else None),
                    #     pixel_cost_min_raw=pixel_cost_raw_val.detach().cpu().numpy().squeeze() if hasattr(pixel_cost_raw_val, 'detach') else (pixel_cost_raw_val.squeeze() if pixel_cost_raw_val is not None else None),
                    #     cost_variance=cost_variance_val.detach().cpu().numpy().squeeze() if hasattr(cost_variance_val, 'detach') else (cost_variance_val.squeeze() if cost_variance_val is not None else None),
                    #     W_plane_pixel=planar_soft_conf_s1,  # 🎯 物理核心：以映射完的离线真值平面置信度为唯一平面界定标准
                    #     depth_gt=depth_gt_np[b_idx] if depth_gt_np is not None else None,
                    #     depth_est_2d=depth_est
                    # )

                    # 在线计算基于离线真值平面置信度过滤的 3 个 Pearson 相关系数
                    with torch.no_grad():
                        if pixel_cost_val is not None and depth_gt_np is not None:
                            depth_error = np.abs(depth_est - np.squeeze(depth_gt_np[b_idx]))
                            
                            cost_min_b = np.squeeze(pixel_cost_val.detach().cpu().numpy() if hasattr(pixel_cost_val, 'detach') else pixel_cost_val.squeeze())
                            gt_depth_b = np.squeeze(depth_gt_np[b_idx])
                            
                            # 建立平面真值置信度（> PLANAR_CONF_THRESHOLD）且含有深度 GT 的掩码
                            flat_mask = (planar_soft_conf_s1 > GT_PLANAR_CONF_THRESHOLD) & (gt_depth_b > 0.0)
                            
                            cost_flat = cost_min_b[flat_mask]
                            error_flat = depth_error[flat_mask]
                            
                            if cost_flat.size > 50:
                                correlation = np.corrcoef(cost_flat.reshape(-1), error_flat.reshape(-1))[0, 1]
                                
                                # 未加权 Pearson 相关系数
                                raw_corr = 0.0
                                if pixel_cost_raw_val is not None:
                                    raw_np = np.squeeze(pixel_cost_raw_val.detach().cpu().numpy() if hasattr(pixel_cost_raw_val, 'detach') else pixel_cost_raw_val.squeeze())
                                    if raw_np.ndim == 3 and raw_np.shape[0] < raw_np.shape[1]:
                                        raw_np = np.min(raw_np, axis=0)
                                    raw_flat = raw_np[flat_mask]
                                    raw_corr = np.corrcoef(raw_flat.reshape(-1), error_flat.reshape(-1))[0, 1]
                                
                                # 视角方差 Pearson 相关系数
                                var_corr = 0.0
                                if cost_variance_val is not None:
                                    var_np = np.squeeze(cost_variance_val.detach().cpu().numpy() if hasattr(cost_variance_val, 'detach') else cost_variance_val.squeeze())
                                    if var_np.ndim == 3 and var_np.shape[0] < var_np.shape[1]:
                                        var_np = np.min(var_np, axis=0)
                                    var_flat = var_np[flat_mask]
                                    var_corr = np.corrcoef(var_flat.reshape(-1), error_flat.reshape(-1))[0, 1]
                                
                                # 👑 计算 Signed Error 的空间分布特征与高误差区正负占比
                                signed_error_flat = (depth_est - gt_depth_b)[flat_mask]
                                signed_error_mean = np.mean(signed_error_flat)
                                
                                # 统计高误差区 (abs_error > 0.10m) 内部的正负一致性
                                high_err_mask = np.abs(signed_error_flat) > 0.10
                                high_err_flat = signed_error_flat[high_err_mask]
                                positive_ratio = 0.0
                                if high_err_flat.size > 0:
                                    positive_ratio = np.sum(high_err_flat > 0) / high_err_flat.size

                                clean_filename = filename.format('', '').replace('/', '_').replace('\\', '_')
                                print(f"\n==================================================================================")
                                print(f"👑 ==> [Pearson - {clean_filename}] Weighted Cost-Error Corr: {correlation:.4f}")
                                print(f"👑 ==> [Pearson - {clean_filename}] Raw Cost-Error Corr: {raw_corr:.4f}")
                                print(f"👑 ==> [Pearson - {clean_filename}] Variance Cost-Error Corr: {var_corr:.4f}")
                                print(f"👑 ==> [Signed Error Stats - {clean_filename}] Flat Region Mean Error: {signed_error_mean:.4f}m")
                                print(f"👑 ==> [Signed Error Stats - {clean_filename}] High Error (>0.1m) Positive Ratio: {positive_ratio*100:.2f}% (0% or 100% means Systematic Bias)")
                                print(f"==================================================================================\n")
                if gt_normal_map_s1 is not None:
                    normal_gt_mask = np.linalg.norm(gt_normal_map_s1, axis=-1) > 0.1
                    normal_gt_vis = np.zeros_like(gt_normal_map_s1, dtype=np.uint8)
                    normal_gt_vis[normal_gt_mask] = ((gt_normal_map_s1[normal_gt_mask] + 1.0) / 2.0 * 255.0).astype(np.uint8)
                    
                    normal_gt_filename = os.path.join(args.outdir, filename.format('normal_gt_s1', '.png'))
                    os.makedirs(os.path.dirname(normal_gt_filename), exist_ok=True)
                    Image.fromarray(normal_gt_vis).save(normal_gt_filename)

                # 评估预测法向量与 SVD 真值法向量的角度偏差并生成 Normal Angular Deviation Heatmap (normal_diff_s1_dif.png)
                if 'normal_pro_pure' in outputs.get("output_plane", {}):
                    pred_normal_s1_tensor = outputs["output_plane"]['normal_pro_pure'][b_idx]
                    pred_normal_s1 = pred_normal_s1_tensor.detach().cpu().numpy() if isinstance(pred_normal_s1_tensor, torch.Tensor) else pred_normal_s1_tensor
                    pred_normal_s1 = np.transpose(pred_normal_s1, (1, 2, 0)) # [H_s1, W_s1, 3]

                    pred_norm_mag = np.linalg.norm(pred_normal_s1, axis=-1, keepdims=True)
                    pred_normal_s1_unit = pred_normal_s1 / (pred_norm_mag + 1e-8)

                    gt_norm_mag = np.linalg.norm(gt_normal_map_s1, axis=-1, keepdims=True)
                    gt_normal_s1_unit = gt_normal_map_s1 / (gt_norm_mag + 1e-8)

                    normal_eval_mask = (
                        (planar_soft_conf_s1 > GT_PLANAR_CONF_THRESHOLD) & 
                        (np.squeeze(gt_norm_mag) > 0.1) & 
                        (depth_est > 0)
                    )
                    if depth_gt_np is not None:
                        gt_depth_s1 = np.squeeze(depth_gt_np[b_idx])
                        normal_eval_mask = normal_eval_mask & (gt_depth_s1 > 0)

                    if normal_eval_mask.any():
                        cos_theta = np.abs(np.sum(pred_normal_s1_unit * gt_normal_s1_unit, axis=-1))
                        cos_theta = np.clip(cos_theta, 0.0, 1.0)
                        angle_diff = np.arccos(cos_theta) * (180.0 / np.pi)
                        mean_angle_err = np.mean(angle_diff[normal_eval_mask])
                        print(f"👑 [SVD NORMAL EVAL] 优质平面法向量 MAE 夹角偏差: {mean_angle_err:.4f}° (评估有效像素数: {np.sum(normal_eval_mask)})")

                        # 绘制夹角偏差热力图 (最大 15 度，背景黑色)
                        plt.figure(figsize=(10, 8))
                        angle_diff_masked = np.ma.masked_where(~normal_eval_mask, angle_diff)
                        cmap_normal_diff = plt.get_cmap('jet')
                        cmap_normal_diff.set_bad(color='black')

                        im_normal_diff = plt.imshow(angle_diff_masked, cmap=cmap_normal_diff, vmin=0, vmax=15.0)
                        cbar_normal_diff = plt.colorbar(im_normal_diff, fraction=0.046, pad=0.04)
                        cbar_normal_diff.set_label('Angular Deviation (Degrees)', size=14)
                        plt.title(f'Normal Angular Deviation (Conf > 0.8) MAE: {mean_angle_err:.4f}°', fontsize=14, fontweight='bold')
                        plt.axis('off')
                        
                        normal_diff_filename = os.path.join(args.outdir, filename.format('normal_diff_s1', '_dif.png'))
                        os.makedirs(os.path.dirname(normal_diff_filename), exist_ok=True)
                        plt.savefig(normal_diff_filename, dpi=150, bbox_inches='tight', pad_inches=0.1)
                        plt.close()
                    else:
                        print("👑 [SVD NORMAL EVAL] 无足够优质平面像素以进行法向量夹角偏差评估")

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
                    # 【初始平面拟合 (未传播) 深度图可视化】
                    # ==========================================
                    if 'depth_no_pro' in outputs.get('output_plane', {}):
                        depth_no_pro = outputs['output_plane']['depth_no_pro'][b_idx]
                        if isinstance(depth_no_pro, torch.Tensor):
                            depth_no_pro = depth_no_pro.detach().cpu().numpy()
                        depth_no_pro = np.squeeze(depth_no_pro)

                        valid_no_pro = depth_no_pro > 0 
                        if valid_no_pro.any():
                            d_min_np = np.percentile(depth_no_pro[valid_no_pro], 1)
                            d_max_np = np.percentile(depth_no_pro[valid_no_pro], 99)
                            depth_no_pro_vis = (depth_no_pro - d_min_np) / (d_max_np - d_min_np + 1e-8)
                            depth_no_pro_vis = np.clip(depth_no_pro_vis, 0, 1)
                            depth_no_pro_uint8 = (depth_no_pro_vis * 255).astype(np.uint8)

                            no_pro_color = cv2.applyColorMap(depth_no_pro_uint8, cv2.COLORMAP_JET)
                            no_pro_color[~valid_no_pro] = 0
                            vis_no_pro_color_filename = depth_filename.replace('.pfm', '_noprop_vis.png')
                            cv2.imwrite(vis_no_pro_color_filename, no_pro_color)

                            depth_no_pro_gray = depth_no_pro_uint8.copy()
                            depth_no_pro_gray[~valid_no_pro] = 255
                            vis_no_pro_gray_filename = depth_filename.replace('.pfm', '_noprop_black_vis.png')
                            cv2.imwrite(vis_no_pro_gray_filename, depth_no_pro_gray)

                        # 额外保存原始未传播深度 PFM，方便后续对比分析
                        no_pro_depth_filename = depth_filename.replace('.pfm', '_noprop.pfm')
                        save_pfm(no_pro_depth_filename, depth_no_pro.astype(np.float32))


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

                # ==========================================
                # 【修正：从 Stage1 像素级深度直接计算法向量并可视化】
                # ==========================================
                if stage1_depths_pixels is not None and "intrinsics_mats" in sample:
                    # 1. 提取参考视角的 Stage1 内参矩阵
                    K_ref_s1 = sample["intrinsics_mats"]['stage_1'][b_idx, 0].cpu().numpy()
                    
                    # 2. 提取当前 batch 的像素级原生深度
                    depth_pix_s1 = stage1_depths_pixels[b_idx]
                    if isinstance(depth_pix_s1, torch.Tensor):
                        depth_pix_s1 = depth_pix_s1.detach().cpu().numpy()
                    depth_pix_s1 = np.squeeze(depth_pix_s1)
                    
                    if depth_pix_s1.ndim == 2:
                        H_s1, W_s1 = depth_pix_s1.shape
                        
                        # 3. 反投影到 3D 坐标
                        x, y = np.meshgrid(np.arange(W_s1), np.arange(H_s1))
                        X = (x - K_ref_s1[0, 2]) * depth_pix_s1 / K_ref_s1[0, 0]
                        Y = (y - K_ref_s1[1, 2]) * depth_pix_s1 / K_ref_s1[1, 1]
                        pts_3d = np.stack([X, Y, depth_pix_s1], axis=-1)
                        
                        # 4. 利用差分求切向量
                        pts_x = np.zeros_like(pts_3d)
                        pts_y = np.zeros_like(pts_3d)
                        
                        # X方向差分
                        pts_x[:, 1:-1, :] = (pts_3d[:, 2:, :] - pts_3d[:, :-2, :]) / 2.0
                        pts_x[:, 0, :] = pts_3d[:, 1, :] - pts_3d[:, 0, :]
                        pts_x[:, -1, :] = pts_3d[:, -1, :] - pts_3d[:, -2, :]
                        
                        # Y方向差分
                        pts_y[1:-1, :, :] = (pts_3d[2:, :, :] - pts_3d[:-2, :, :]) / 2.0
                        pts_y[0, :, :] = pts_3d[1, :, :] - pts_3d[0, :, :]
                        pts_y[-1, :, :] = pts_3d[-1, :, :] - pts_3d[-2, :, :]
                        
                        # 5. 叉乘求法向 (并指向相机)
                        pixel_normal = np.cross(pts_x, pts_y)
                        pixel_normal_norm = np.linalg.norm(pixel_normal, axis=-1, keepdims=True)
                        pixel_normal = -pixel_normal / (pixel_normal_norm + 1e-8)
                        
                        # 6. 映射到 RGB (0-255) 并保存
                        pixel_normal_vis = ((np.clip(pixel_normal, -1.0, 1.0) + 1.0) / 2.0 * 255.0).astype(np.uint8)
                        
                        valid_mask_pix = depth_pix_s1 > 0
                        pixel_normal_vis[~valid_mask_pix] = 0
                        
                        pixel_normal_filename = normal_filename.replace('.png', '_pixel_raw.png')
                        Image.fromarray(pixel_normal_vis).save(pixel_normal_filename)

                # ====================================================================
                # 👑 【新增功能】评估预测法向量与 SVD 真值法向量的角度偏差 (MAE 角度)
                # ====================================================================
                # 预测法向量与 SVD 真值法向量的角度偏差评估以及偏差热力图绘制已在 SVD 拟合及置信度清洗阶段提前完成

                # ====================================================================
                # 提前提取平面置信度 (W_plane)，供平面 MAE 切分和最终保存使用
                # ====================================================================
                depth_est_sq = np.squeeze(depth_est)  # [H, W]
                w_plane_sq = []
                dev_gt = None  # 提前初始化，用于后续导出静态二值图
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

                        # 🪐 锁定大平面固定指标 (使用 3D OLS 拟合的三角几何平面，实现 100% 对齐评测)
                        if planar_soft_conf_s1 is not None:
                            mask_planar = mask_diff & (planar_soft_conf_s1 > GT_PLANAR_CONF_THRESHOLD)
                        else:
                            # 兜底
                            mask_planar = np.zeros_like(depth_est_sq, dtype=bool)

                        if mask_planar.any():
                            planar_mae = np.mean(np.abs(depth_est_sq[mask_planar] - gt_curr[mask_planar]))
                            planar_text = f"\nFlat Region MAE (GT Planar): {planar_mae:.4f}m"
                        else:
                            planar_mae = 0.0
                            planar_text = ""

                        # ── 图一：全局误差图（基于 replace 防爆设计） ────────────────────────
                        diff_max_plot = error_num
                        
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
 
                        # ── 图二：仅平面区域误差图（W_plane > GT_PLANAR_CONF_THRESHOLD，满足用户只展现平面 MAE 的渴望） ──
                        if mask_planar.any():
                            planar_err_array = np.abs(depth_est_sq[mask_planar] - gt_curr[mask_planar])
                            planar_diff_map = np.zeros_like(depth_est_sq)
                            planar_diff_map[mask_planar] = planar_err_array
 
                            # 误差上限用平面区域自己的 95 百分位，使得色标分布针对平面高分辨率拉伸，拒绝曲面大误差污染
                            planar_diff_max = error_num
 
                            plt.figure(figsize=(10, 8))
                            # 刚性隔离：将平面区域外（曲面、虚空、背景）全部标记为 True 实施严格黑色放逐
                            planar_map_masked = np.ma.masked_where(~mask_planar, planar_diff_map)
                            cmap_planar = plt.get_cmap('jet')
                            cmap_planar.set_bad(color='black') # 外部曲面死死抹黑
                            
                            im2 = plt.imshow(planar_map_masked, cmap=cmap_planar, vmin=0, vmax=planar_diff_max)
                            cbar2 = plt.colorbar(im2, fraction=0.046, pad=0.04)
                            cbar2.set_label('Absolute Error (Meters)', size=14)
                            
                            plt.title(
                                f'Flat Region MAE (W_plane > {GT_PLANAR_CONF_THRESHOLD}): {planar_mae:.4f}m'
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

                        # ── 图三：预测的平面区域误差图（Pred Planar W_plane > PRED_PLANAR_CONF_THRESHOLD） ──
                        mask_pred_planar = mask_diff & (w_plane_sq > PRED_PLANAR_CONF_THRESHOLD)
                        if mask_pred_planar.any():
                            pred_planar_err_array = np.abs(depth_est_sq[mask_pred_planar] - gt_curr[mask_pred_planar])
                            pred_planar_diff_map = np.zeros_like(depth_est_sq)
                            pred_planar_diff_map[mask_pred_planar] = pred_planar_err_array
                            
                            pred_planar_mae = np.mean(pred_planar_err_array)
                            pred_planar_diff_max = error_num

                            plt.figure(figsize=(10, 8))
                            pred_planar_map_masked = np.ma.masked_where(~mask_pred_planar, pred_planar_diff_map)
                            cmap_pred_planar = plt.get_cmap('jet')
                            cmap_pred_planar.set_bad(color='black')
                            
                            im3 = plt.imshow(pred_planar_map_masked, cmap=cmap_pred_planar, vmin=0, vmax=pred_planar_diff_max)
                            cbar3 = plt.colorbar(im3, fraction=0.046, pad=0.04)
                            cbar3.set_label('Absolute Error (Meters)', size=14)
                            
                            plt.title(
                                f'Pred Flat Region MAE (Pred W_plane > {PRED_PLANAR_CONF_THRESHOLD}): {pred_planar_mae:.4f}m'
                                f'\nMax Cutoff: {pred_planar_diff_max:.2f}m'
                                f'  |  Pred Flat pixels: {mask_pred_planar.sum()} / {mask_diff.sum()}',
                                fontsize=13, fontweight='bold'
                            )
                            plt.axis('off')
                            
                            pred_planar_diff_filename = diff_filename.replace('_dif.png', '_dif_pred_planar.png')
                            os.makedirs(os.path.dirname(pred_planar_diff_filename), exist_ok=True)
                            
                            plt.savefig(pred_planar_diff_filename, dpi=150, bbox_inches='tight', pad_inches=0.1)
                            plt.close()

                    # ====================================================================
                    # 🚨 3.b 生成 Stage 1 未传播预测与 GT 的差异热力图 (基于 W_plane_pixel_init)
                    # ====================================================================
                    if 'depth_no_pro' in outputs.get("output_plane", {}) and 'W_plane_pixel_init' in outputs.get("output_plane", {}):
                        depth_no_pro_data = outputs["output_plane"]['depth_no_pro'][b_idx]
                        if isinstance(depth_no_pro_data, torch.Tensor):
                            depth_no_pro_data = depth_no_pro_data.detach().cpu().numpy()
                        depth_no_pro_sq = np.squeeze(depth_no_pro_data)

                        w_before_data = outputs["output_plane"]['W_plane_pixel_init'][b_idx]
                        w_before_np = w_before_data.detach().cpu().numpy() if isinstance(w_before_data, torch.Tensor) else w_before_data
                        w_before_sq = np.squeeze(w_before_np)
                        if w_before_sq.shape != depth_no_pro_sq.shape:
                            w_before_sq = cv2.resize(w_before_sq, (depth_no_pro_sq.shape[1], depth_no_pro_sq.shape[0]), interpolation=cv2.INTER_LINEAR)

                        if gt_curr.shape != depth_no_pro_sq.shape:
                            gt_no_pro = cv2.resize(gt_curr, (depth_no_pro_sq.shape[1], depth_no_pro_sq.shape[0]), interpolation=cv2.INTER_NEAREST)
                            tri_id_no_pro = cv2.resize(tri_id_curr, (depth_no_pro_sq.shape[1], depth_no_pro_sq.shape[0]), interpolation=cv2.INTER_NEAREST)
                        else:
                            gt_no_pro = gt_curr
                            tri_id_no_pro = tri_id_curr

                        mask_diff_no_pro = (depth_no_pro_sq > 0) & (gt_no_pro > 0) & (tri_id_no_pro >= 0)
                        if mask_diff_no_pro.any():
                            diff_map_no_pro = np.zeros_like(depth_no_pro_sq)
                            abs_error_array_no_pro = np.abs(depth_no_pro_sq[mask_diff_no_pro] - gt_no_pro[mask_diff_no_pro])
                            diff_map_no_pro[mask_diff_no_pro] = abs_error_array_no_pro
                            mean_abs_error_no_pro = np.mean(abs_error_array_no_pro)

                            if planar_soft_conf_s1 is not None:
                                if planar_soft_conf_s1.shape != mask_diff_no_pro.shape:
                                    planar_soft_conf_no_pro_resized = cv2.resize(planar_soft_conf_s1, (mask_diff_no_pro.shape[1], mask_diff_no_pro.shape[0]), interpolation=cv2.INTER_NEAREST)
                                else:
                                    planar_soft_conf_no_pro_resized = planar_soft_conf_s1
                                mask_planar_no_pro = mask_diff_no_pro & (planar_soft_conf_no_pro_resized > GT_PLANAR_CONF_THRESHOLD)
                            else:
                                mask_planar_no_pro = mask_diff_no_pro & (w_before_sq > GT_PLANAR_CONF_THRESHOLD)

                            if mask_planar_no_pro.any():
                                planar_mae_no_pro = np.mean(np.abs(depth_no_pro_sq[mask_planar_no_pro] - gt_no_pro[mask_planar_no_pro]))
                                planar_no_pro_text = f"\nFlat Region MAE (GT Planar): {planar_mae_no_pro:.4f}m"
                            else:
                                planar_mae_no_pro = 0.0
                                planar_no_pro_text = ""

                            diff_max_plot_no_pro = error_num
                            plt.figure(figsize=(10, 8))
                            diff_map_masked_no_pro = np.ma.masked_where(~mask_diff_no_pro, diff_map_no_pro)
                            cmap_no_pro = plt.get_cmap('jet')
                            cmap_no_pro.set_bad(color='black')
                            im_no_pro = plt.imshow(diff_map_masked_no_pro, cmap=cmap_no_pro, vmin=0, vmax=diff_max_plot_no_pro)
                            cbar_no_pro = plt.colorbar(im_no_pro, fraction=0.046, pad=0.04)
                            cbar_no_pro.set_label('Absolute Error (Meters)', size=14)
                            plt.title(f'Stage 1 No-Propagation MAE: {mean_abs_error_no_pro:.4f}m {planar_no_pro_text}\nMax Cutoff: {diff_max_plot_no_pro:.2f}m',
                                      fontsize=14, fontweight='bold')
                            plt.axis('off')
                            diff_filename_no_pro = os.path.join(args.outdir, filename.format('depth_diff_s1_noprop', '_dif.png'))
                            os.makedirs(os.path.dirname(diff_filename_no_pro), exist_ok=True)
                            plt.savefig(diff_filename_no_pro, dpi=150, bbox_inches='tight', pad_inches=0.1)
                            plt.close()

                            if mask_planar_no_pro.any():
                                planar_diff_map_no_pro = np.zeros_like(depth_no_pro_sq)
                                planar_diff_map_no_pro[mask_planar_no_pro] = np.abs(depth_no_pro_sq[mask_planar_no_pro] - gt_no_pro[mask_planar_no_pro])
                                planar_diff_max_no_pro = error_num
                                plt.figure(figsize=(10, 8))
                                planar_map_masked_no_pro = np.ma.masked_where(~mask_planar_no_pro, planar_diff_map_no_pro)
                                cmap_planar_no_pro = plt.get_cmap('jet')
                                cmap_planar_no_pro.set_bad(color='black')
                                im_planar_no_pro = plt.imshow(planar_map_masked_no_pro, cmap=cmap_planar_no_pro, vmin=0, vmax=planar_diff_max_no_pro)
                                cbar_planar_no_pro = plt.colorbar(im_planar_no_pro, fraction=0.046, pad=0.04)
                                cbar_planar_no_pro.set_label('Absolute Error (Meters)', size=14)
                                plt.title(f'Flat Region No-Propagation MAE (GT Planar): {planar_mae_no_pro:.4f}m\nMax Cutoff: {planar_diff_max_no_pro:.2f}m',
                                          fontsize=13, fontweight='bold')
                                plt.axis('off')
                                planar_diff_filename_no_pro = diff_filename_no_pro.replace('_dif.png', '_dif_planar.png')
                                os.makedirs(os.path.dirname(planar_diff_filename_no_pro), exist_ok=True)
                                plt.savefig(planar_diff_filename_no_pro, dpi=150, bbox_inches='tight', pad_inches=0.1)
                                plt.close()

                    # ====================================================================
                    # 👑 【自愈改造·误差相变诊断】生成传播前、传播后的 signed_error 及其差值图
                    # ====================================================================
                    if 'depth_no_pro' in outputs.get("output_plane", {}):
                        # 获取公共有效且为平面区域的掩码
                        if planar_soft_conf_s1 is not None:
                            mask_signed = mask_diff & mask_diff_no_pro & (planar_soft_conf_s1 > GT_PLANAR_CONF_THRESHOLD)
                        else:
                            mask_signed = mask_diff & mask_diff_no_pro & (w_plane_sq > GT_PLANAR_CONF_THRESHOLD)
                        if mask_signed.any():
                            # 1. 计算 signed_error (预测 - 真值)
                            signed_err_before = np.zeros_like(depth_no_pro_sq)
                            signed_err_before[mask_signed] = depth_no_pro_sq[mask_signed] - gt_no_pro[mask_signed]

                            signed_err_after = np.zeros_like(depth_est_sq)
                            signed_err_after[mask_signed] = depth_est_sq[mask_signed] - gt_curr[mask_signed]

                            signed_err_diff = np.zeros_like(depth_est_sq)
                            signed_err_diff[mask_signed] = signed_err_after[mask_signed] - signed_err_before[mask_signed]

                            # 2. 确定可视化最大/最小对称截断边界
                            limit_before = np.percentile(np.abs(signed_err_before[mask_signed]), 95)
                            limit_after = np.percentile(np.abs(signed_err_after[mask_signed]), 95)
                            v_limit = max(limit_before, limit_after, 0.2)

                            # 定义一个专门的 signed_error 可视化辅助函数
                            def save_signed_error_map(err_map, mask, save_path, title_text, val_lim):
                                plt.figure(figsize=(10, 8))
                                err_masked = np.ma.masked_where(~mask, err_map)
                                cmap_bwr = plt.get_cmap('bwr')
                                cmap_bwr.set_bad(color='black')  # 无效区域和背景全涂黑
                                
                                im_se = plt.imshow(err_masked, cmap=cmap_bwr, vmin=-val_lim, vmax=val_lim)
                                cbar_se = plt.colorbar(im_se, fraction=0.046, pad=0.04)
                                cbar_se.set_label('Signed Error (Meters)', size=14)
                                plt.title(title_text, fontsize=14, fontweight='bold')
                                plt.axis('off')
                                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                                plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
                                plt.close()

                            # 3. 保存三张 signed_error 诊断图
                            # 图 1：传播前 signed_error
                            before_se_filename = os.path.join(args.outdir, filename.format('depth_diff_s1_noprop_signed_error', '.png'))
                            save_signed_error_map(
                                err_map=signed_err_before,
                                mask=mask_signed,
                                save_path=before_se_filename,
                                title_text=f'Signed Error Before Propagation (SVD Init)\n(Range: -{v_limit:.2f}m to {v_limit:.2f}m)',
                                val_lim=v_limit
                            )

                            # 图 2：传播后 signed_error
                            after_se_filename = os.path.join(args.outdir, filename.format('diagnostic_signed_error_s1', '.png'))
                            save_signed_error_map(
                                err_map=signed_err_after,
                                mask=mask_signed,
                                save_path=after_se_filename,
                                title_text=f'Signed Error After Propagation (GNN Refined)\n(Range: -{v_limit:.2f}m to {v_limit:.2f}m)',
                                val_lim=v_limit
                            )

                            # 图 3：两者差值 (After - Before)
                            limit_diff = max(np.percentile(np.abs(signed_err_diff[mask_signed]), 95), 0.1)
                            diff_se_filename = os.path.join(args.outdir, filename.format('signed_error_diff_s1', '.png'))
                            save_signed_error_map(
                                err_map=signed_err_diff,
                                mask=mask_signed,
                                save_path=diff_se_filename,
                                title_text=f'Signed Error Difference (After - Before)\n(Range: -{limit_diff:.2f}m to {limit_diff:.2f}m)',
                                val_lim=limit_diff
                            )

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

                    # 🪐 👑 模仿预测平面置信度，导出 GT 软置信度的可视化图 (0为红，1为蓝)
                    if planar_soft_conf_s1 is not None:
                        gt_conf_filename = os.path.join(args.outdir, filename.format('soft_planar_gt_jet', '.png'))
                        os.makedirs(os.path.dirname(gt_conf_filename), exist_ok=True)
                        
                        gt_conf_uint8 = (np.clip(planar_soft_conf_s1, 0.0, 1.0) * 255.0).astype(np.uint8)
                        gt_conf_uint8_inv = 255 - gt_conf_uint8
                        gt_conf_uint8_inv[~current_mask] = 0
                        
                        gt_conf_color = cv2.applyColorMap(gt_conf_uint8_inv, cv2.COLORMAP_JET)
                        gt_conf_color[~current_mask] = 0
                        
                        cv2.imwrite(gt_conf_filename, gt_conf_color)
                        print(f"✅ 软置信度真值图已成功写入: {gt_conf_filename}")

                    # =====================================================================
                    # 👑 【新增对比实验硬核资产】：熔炼并导出几何真平面掩码二值图
                    # =====================================================================
                    # 机制精剖：使用计算得到的 3D 几何真平面软置信度真值 (W_GT > PLANAR_CONF_THRESHOLD)
                    binary_mask_np = np.zeros_like(depth_est_sq, dtype=np.uint8)
                    if planar_soft_conf_s1 is not None:
                        binary_mask_np[(planar_soft_conf_s1 > GT_PLANAR_CONF_THRESHOLD) & current_mask] = 255
                    else:
                        if dev_gt is not None:
                            binary_mask_np[(dev_gt < 0.15) & current_mask] = 255
                        else:
                            binary_mask_np[(w_plane_sq > GT_PLANAR_CONF_THRESHOLD) & current_mask] = 255

                    # 动态生成专属基准文件名，加上 _oracle_mask 后缀，防止混淆文件目录
                    oracle_mask_filename = os.path.join(args.outdir, filename.format('plane_mask_01', '_oracle_mask.png'))
                    os.makedirs(os.path.dirname(oracle_mask_filename), exist_ok=True)
                    cv2.imwrite(oracle_mask_filename, binary_mask_np)
                    
                    # 👑 新增：预测的平面掩码二值图 (Prediction W_plane > PRED_PLANAR_CONF_THRESHOLD)
                    pred_binary_mask_np = np.zeros_like(depth_est_sq, dtype=np.uint8)
                    pred_binary_mask_np[(w_plane_sq > PRED_PLANAR_CONF_THRESHOLD) & current_mask] = 255
                    pred_mask_filename = os.path.join(args.outdir, filename.format('pred_plane_mask_01', '_pred_mask.png'))
                    os.makedirs(os.path.dirname(pred_mask_filename), exist_ok=True)
                    cv2.imwrite(pred_mask_filename, pred_binary_mask_np)
                    print(f"✅ 预测平面置信度(>{PRED_PLANAR_CONF_THRESHOLD})二值图已成功写入: {pred_mask_filename}")
                    
                    # 写入单通道灰度/二值图磁盘
                    cv2.imwrite(oracle_mask_filename, binary_mask_np)
                    print(f"👑 黄金实验 0/1 刚性掩码已导出至: {oracle_mask_filename}")
                else:
                    # 彻底告别沉默，报错提示你 key 没匹配上
                    print(f"⚠️ 警告: outputs['output_plane'] 中未找到 'W_plane_pixel'，跳过置信度可视化")

    

                # ==========================================
                # 4.2 👑 可视化传播前（Before Propagation/Init）的平面置信度
                # ==========================================
                if 'W_plane_pixel_init' in outputs.get("output_plane", {}):
                    w_before_data = outputs["output_plane"]['W_plane_pixel_init'][b_idx]
                    w_before_np = w_before_data.detach().cpu().numpy() if isinstance(w_before_data, torch.Tensor) else w_before_data
                    w_before_sq = np.squeeze(w_before_np)
                    
                    if w_before_sq.shape != depth_est_sq.shape:
                        w_before_sq = cv2.resize(w_before_sq, (depth_est_sq.shape[1], depth_est_sq.shape[0]), interpolation=cv2.INTER_LINEAR)

                    # 1) 传播前 Jet 伪彩图落盘（与传播后形成最完美的 Ablation 视觉对比！）
                    conf_before_filename = os.path.join(args.outdir, filename.format('confidence_s1_init', '_jet_before.png'))
                    os.makedirs(os.path.dirname(conf_before_filename), exist_ok=True) 
                    
                    w_before_uint8 = (np.clip(w_before_sq, 0.0, 1.0) * 255.0).astype(np.uint8)
                    w_before_uint8_inv = 255 - w_before_uint8 
                    w_before_uint8_inv[~current_mask] = 0

                    w_before_color = cv2.applyColorMap(w_before_uint8_inv, cv2.COLORMAP_JET)
                    w_before_color[~current_mask] = 0 
                    cv2.imwrite(conf_before_filename, w_before_color)
                    print(f"✅ 传播前初始置信度图已写入: {conf_before_filename}")
                else:
                    print(f"⚠️ 提示: 未找到 'W_plane_pixel_init'，无法生成传播前对比图")
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
                        max_plot_pixel = error_num
                        plt.figure(figsize=(10, 8))
                        cmap = plt.get_cmap('jet')
                        im_pixel = plt.imshow(np.ma.masked_where(~mask_diff_pixel, diff_map_pixel), cmap=cmap, vmin=0,
                                              vmax=max_plot_pixel)
                        cbar_pixel = plt.colorbar(im_pixel, fraction=0.046, pad=0.04)
                        cbar_pixel.set_label('Absolute Error (Meters)', size=14)

                        # 🪐 详解 s1 与 s0 掩码对齐 (由架构师根据您的指正优化)：
                        # 因为 depth_pixel_curr 是 Stage 1 尺寸的深度，因此这里采用 Stage 1 平面置信度地图 (planar_soft_conf_s1) 进行对齐判断。
                        # 若其形状与当前差异图不一致，使用最近邻插值缩放，提取出置信度 > PLANAR_CONF_THRESHOLD 且有有效深度的像素作为 Stage 1 优质平面掩码 (mask_planar_s1)。
                        if planar_soft_conf_s1 is not None:
                            if planar_soft_conf_s1.shape != mask_diff_pixel.shape:
                                planar_soft_conf_pixel_resized = cv2.resize(planar_soft_conf_s1, (mask_diff_pixel.shape[1], mask_diff_pixel.shape[0]), interpolation=cv2.INTER_NEAREST)
                            else:
                                planar_soft_conf_pixel_resized = planar_soft_conf_s1
                            mask_planar_s1 = mask_diff_pixel & (planar_soft_conf_pixel_resized > GT_PLANAR_CONF_THRESHOLD)
                        else:
                            # 兜底
                            mask_planar_s1 = np.zeros_like(depth_pixel_curr, dtype=bool)

                        plt.title(
                            f'Baseline (Pixel-wise) MAE: {mean_error_pixel:.4f}m\nMax Cutoff: {max_plot_pixel:.2f}m',
                            fontsize=14, fontweight='bold')
                        plt.axis('off')

                        diff_filename_pixel = os.path.join(args.outdir,
                                                           filename.format('depth_diff_s1_pixel', '_dif.png'))
                        os.makedirs(os.path.dirname(diff_filename_pixel), exist_ok=True)
                        plt.savefig(diff_filename_pixel, dpi=150, bbox_inches='tight', pad_inches=0.1)
                        plt.close()

                        # 🪐 计算原生像素级平面区域 (Conf > 0.8) 上的 MAE 误差并可视化 (采用 Stage 1 尺寸平面掩码 mask_planar_s1)
                        if mask_planar_s1.any():
                            abs_error_pixel_plane = np.abs(depth_pixel_curr[mask_planar_s1] - gt_curr_pixel[mask_planar_s1])
                            mean_error_pixel_plane = np.mean(abs_error_pixel_plane)
                            print(f"👑 [PIXEL PLANE EVAL] 原生像素级优质平面区域 MAE: {mean_error_pixel_plane:.4f}m (评估有效像素数: {np.sum(mask_planar_s1)})")

                            plt.figure(figsize=(10, 8))
                            diff_map_pixel_plane = np.zeros_like(depth_pixel_curr)
                            diff_map_pixel_plane[mask_planar_s1] = abs_error_pixel_plane
                            
                            cmap_plane = plt.get_cmap('jet')
                            cmap_plane.set_bad(color='black')
                            
                            im_pixel_plane = plt.imshow(np.ma.masked_where(~mask_planar_s1, diff_map_pixel_plane), cmap=cmap_plane, vmin=0, vmax=max_plot_pixel)
                            cbar_pixel_plane = plt.colorbar(im_pixel_plane, fraction=0.046, pad=0.04)
                            cbar_pixel_plane.set_label('Absolute Error (Meters)', size=14)
                            
                            plt.title(f'Baseline (Pixel-wise) Planar MAE: {mean_error_pixel_plane:.4f}m\nMax Cutoff: {max_plot_pixel:.2f}m', fontsize=14, fontweight='bold')
                            plt.axis('off')
                            
                            diff_filename_pixel_plane = os.path.join(args.outdir, filename.format('depth_diff_s1_pixel_plane', '_dif.png'))
                            os.makedirs(os.path.dirname(diff_filename_pixel_plane), exist_ok=True)
                            plt.savefig(diff_filename_pixel_plane, dpi=150, bbox_inches='tight', pad_inches=0.1)
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

                        # 计算 Stage 0 阶段的纯平面区域 MAE，使用静态几何平面
                        if planar_soft_conf_s0 is not None:
                            if planar_soft_conf_s0.shape != mask_diff_s0.shape:
                                planar_soft_conf_s0_resized = cv2.resize(planar_soft_conf_s0, (mask_diff_s0.shape[1], mask_diff_s0.shape[0]), interpolation=cv2.INTER_NEAREST)
                            else:
                                planar_soft_conf_s0_resized = planar_soft_conf_s0
                            mask_planar_s0 = mask_diff_s0 & (planar_soft_conf_s0_resized > GT_PLANAR_CONF_THRESHOLD)
                        else:
                            w_plane_sq_s0 = cv2.resize(w_plane_sq, (depth_est_s0_sq.shape[1], depth_est_s0_sq.shape[0]),
                                                       interpolation=cv2.INTER_LINEAR)
                            mask_planar_s0 = mask_diff_s0 & (w_plane_sq_s0 > GT_PLANAR_CONF_THRESHOLD)

                        if mask_planar_s0.any():
                            planar_mae_s0 = np.mean(
                                np.abs(depth_est_s0_sq[mask_planar_s0] - gt_curr_s0[mask_planar_s0]))
                            planar_text_s0 = f"\nFlat Region MAE (GT Planar): {planar_mae_s0:.4f}m"
                        else:
                            planar_text_s0 = ""

                        diff_max_plot_s0 = error_num

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

                        # SVD 拟合实验与降级处理已在 Stage 1 各种图生成前提前执行完成，此处跳过以避免重复计算

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


def swap_views_in_dict(d, v1, v2):
    if d is None:
        return None
    new_d = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            v_new = v.clone()
            v_new[:, [v1, v2]] = v_new[:, [v2, v1]]
            new_d[k] = v_new
    return new_d

def save_depth_cross_check():
    # dataset, dataloader
    MVSDataset = find_dataset_def(args.dataset)
    test_dataset = MVSDataset(args.testpath, args.testlist, "test", args.n_views)
    TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=collate_keep_list,
                               num_workers=0, drop_last=False)

    # model
    model = PatchmatchNet(patchmatch_interval_scale=args.patchmatch_interval_scale,
                propagation_range = args.patchmatch_range, patchmatch_iteration=args.patchmatch_iteration, 
                patchmatch_num_sample = args.patchmatch_num_sample, 
                propagate_neighbors=args.propagate_neighbors, evaluate_neighbors=args.evaluate_neighbors)
    model.to(device)

    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    ckpt_model_state = state_dict.get('model', state_dict)
    current_model_state = model.state_dict()
    filtered_state = {k: v for k, v in ckpt_model_state.items() if k in current_model_state}
    model.load_state_dict(filtered_state, strict=False)
    model.eval()
    
    with torch.no_grad():
        for batch_idx, sample in enumerate(TestImgLoader):
            print(f"\n[{batch_idx}/{len(TestImgLoader)}] Cross-checking view {sample['filename'][0]}...")
            num_views = args.n_views
            
            stage1_depths = []
            stage1_depths_pixels = []
            
            ref_stage0_conf = None
            ref_stage0_depth = None
            
            intrinsics_s1 = sample["intrinsics_mats"]['stage_1'][0].cpu().numpy() # [V, 3, 3]
            extrinsics = sample["proj_matrices"]['stage_0'][0].cpu().numpy() # [V, 4, 4], extrinsics are in proj_matrices initially
            # Wait! eval_whu_big.py line 361: proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            # The orig extrinsics are overwritten! 
            # We must parse extrinsics from proj_matrices by multiplying with inv(intrinsics).
            
            # Helper to get raw extrinsics:
            extrinsics_raw = []
            for v_idx in range(num_views):
                P_0 = sample["proj_matrices"]['stage_0'][0, v_idx].cpu().numpy() # 4x4
                K_0 = sample["intrinsics_mats"]['stage_0'][0, v_idx].cpu().numpy() # 3x3
                E = np.eye(4)
                E[:3, :4] = np.matmul(np.linalg.inv(K_0), P_0[:3, :4])
                extrinsics_raw.append(E)
            extrinsics_raw = np.stack(extrinsics_raw)
            
            for v in range(num_views):
                # Swap data
                imgs_v = tocuda(swap_views_in_dict(sample["imgs"], v, 0), device=device)
                proj_v = tocuda(swap_views_in_dict(sample["proj_matrices"], v, 0), device=device)
                intr_v = tocuda(swap_views_in_dict(sample["intrinsics_mats"], v, 0), device=device)
                
                # Swap vertexs, lines, triangles
                b_triangles = sample['triangles'][0] # length V
                triangles_v_list = list(b_triangles)
                triangles_v_list[0], triangles_v_list[v] = triangles_v_list[v], triangles_v_list[0]
                tri_processed = []
                for t in triangles_v_list[0]: # only pass view 0 to model
                    v_ids = torch.from_numpy(t['vertex_ids']).to(device)
                    l_ids = torch.from_numpy(t['line_ids']).to(device)
                    pts = torch.from_numpy(t['valid_points']).to(device)
                    tri_processed.append((v_ids, l_ids, pts))
                triangles_batch = [tri_processed]
                
                v_list = list(sample['vertexs'][0])
                v_list[0], v_list[v] = v_list[v], v_list[0]
                vertexs_batch = [torch.from_numpy(v_list[0]).to(device)]
                
                l_list = list(sample['lines'][0])
                l_list[0], l_list[v] = l_list[v], l_list[0]
                lines_batch = [torch.from_numpy(l_list[0]).to(device)]

                skip = ["vertexs", "lines", "triangles", "imgs", "proj_matrices", "intrinsics_mats"]
                sample_cuda = tocuda(sample, device=device, skip_keys=skip)
                
                outputs = model(imgs_v, proj_v, intr_v,
                                sample_cuda["depth_min"], sample_cuda["depth_max"],
                                vertexs_batch, lines_batch, triangles_batch, None,
                                0.0, 0.0, 0.55)
                                
                d1 = outputs["depth_patchmatch"]["stage_1"][-1].detach().cpu().numpy()[0]
                d1 = np.squeeze(d1)
                stage1_depths.append(d1)
                
                # 🎯 提取并保存各源视角的纯像素级原生深度 (depth_stage1_pixels)
                d1_pixel = outputs["output_plane"]["depth_stage1_pixels"].detach().cpu().numpy()[0]
                d1_pixel = np.squeeze(d1_pixel)
                stage1_depths_pixels.append(d1_pixel)
                
                if v == 0:
                    ref_stage1_conf = np.squeeze(outputs["output_plane"]["W_plane_pixel"].detach().cpu().numpy()[0])
                    ref_stage0_depth = np.squeeze(outputs["refined_depth"]["stage_0"].detach().cpu().numpy()[0])
                    
                    # 🎯 提取 Stage 1 的两路原生深度用于物理融合：
                    # 1. 像素级原生深度 (原 PatchmatchNet 像素级光度代价预测深度)
                    ref_stage1_pixel_depth = d1_pixel
                    # 2. 平面几何解析深度 (经过三角网格拟合/传播的平面几何深度)
                    ref_stage1_plane_depth = d1
                    
                    ref_tri_id_map = None
                    if "tri_id_map" in outputs.get("output_plane", {}):
                        tri_map_data = outputs["output_plane"]["tri_id_map"]
                        if isinstance(tri_map_data, torch.Tensor):
                            tri_map_data = tri_map_data.detach().cpu().numpy()
                        ref_tri_id_map = np.squeeze(tri_map_data[0])
                    
            # --- CROSS CHECK ---
            print("Running Stage 1 Geometric Cross-Check...")
            depth_ref_1 = stage1_depths[0]
            K_ref_1 = intrinsics_s1[0]
            E_ref = extrinsics_raw[0]
            
            geo_pixel_thres = 1.0 # 1 pixel in stage 1
            geo_abs_depth_thres = 0.08 # 8cm strict absolute depth error for cross-check (收紧几何偏差门槛)
            
            valid_src_count = np.zeros_like(depth_ref_1)
            
            for src_idx in range(1, num_views):
                # 👑 核心破局：源视角采用纯像素级深度 (stage1_depths_pixels)，作为不受平面先验污染的客观几何裁判！
                depth_src_1 = stage1_depths_pixels[src_idx]
                K_src_1 = intrinsics_s1[src_idx]
                E_src = extrinsics_raw[src_idx]
                
                # We need depth_reprojected to compute MAE
                depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref_1, K_ref_1, E_ref, depth_src_1, K_src_1, E_src)
                
                dist = np.sqrt((x2d_reprojected - np.arange(0, depth_ref_1.shape[1]).reshape(1,-1)) ** 2 + 
                               (y2d_reprojected - np.arange(0, depth_ref_1.shape[0]).reshape(-1,1)) ** 2)
                depth_diff = np.abs(depth_reprojected - depth_ref_1)
                
                # Check valid pixels (where depth_reprojected > 0)
                valid_reproj = depth_reprojected > 0
                if valid_reproj.sum() > 0:
                    mae = np.mean(depth_diff[valid_reproj])
                    print(f"   [View {src_idx}] Reproject to Pixel-Depth MAE: {mae:.4f} m, valid pixels: {valid_reproj.sum()}")
                else:
                    print(f"   [View {src_idx}] No valid reprojected pixels!")
                
                # Use Absolute Depth Difference!
                mask = np.logical_and(valid_reproj, np.logical_and(dist < geo_pixel_thres, depth_diff < geo_abs_depth_thres))
                valid_src_count += mask.astype(np.int32)
                
            # 👑 视角支持度要求：若视角数充裕(>=3)，要求至少 2 个源视角一致；否则至少 1 个源视角一致
            min_views_needed = 2 if num_views >= 3 else 1
            is_consistent_s1 = valid_src_count >= min_views_needed
            
            # 👑 依用户指令：基于三角形级别的几何一致性聚合降级 (仅分支 A)
            ref_stage1_conf_degraded = ref_stage1_conf.copy()
            tri_ratio_threshold = 0.60  # 三角形内部至少有 60% 像素通过多视角一致性，否则认定整体偏差过大
            degraded_target_val = 0.30  # 统一降级至固定低值 0.30 (< 0.8)

            if ref_tri_id_map is not None:
                unique_tris = np.unique(ref_tri_id_map)
                for tri_id in unique_tris:
                    # 背景或未覆盖区域 (-1 或 255) 保持原始置信度不变
                    if tri_id < 0 or tri_id == 255:
                        continue
                    
                    mask_tri = (ref_tri_id_map == tri_id)
                    if not mask_tri.any():
                        continue
                    
                    # 统计该三角形内部像素的多视角几何一致性率
                    consistency_ratio = is_consistent_s1[mask_tri].mean()
                    
                    # 分支 A：若几何一致性比例过低，且内部包含高置信度像素 (>0.8)，则统一降级为 0.30
                    if consistency_ratio < tri_ratio_threshold:
                        degrade_mask = mask_tri & (ref_stage1_conf > PRED_PLANAR_CONF_THRESHOLD)
                        ref_stage1_conf_degraded[degrade_mask] = degraded_target_val
            else:
                # 若无 tri_id_map 则退化为逐像素降级
                ref_stage1_conf_degraded[~is_consistent_s1 & (ref_stage1_conf > PRED_PLANAR_CONF_THRESHOLD)] = degraded_target_val
            
            # Upsample everything to Stage 0 for Evaluation and Saving
            H0, W0 = ref_stage0_depth.shape[-2:]
            def upsample_to_s0(arr):
                arr_t = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
                return torch.nn.functional.interpolate(arr_t, size=(H0, W0), mode='nearest').squeeze().numpy()
                
            ref_stage0_conf = upsample_to_s0(ref_stage1_conf)
            ref_stage0_conf_degraded = upsample_to_s0(ref_stage1_conf_degraded)
            is_consistent_s0 = upsample_to_s0(is_consistent_s1.astype(np.float32)) > 0.5
            
            # ---> 👑 Stage 1 物理深度融合与全方位 MAE 评估
            depth_gt_dict = sample.get("depth", {})
            if 'stage_1' in depth_gt_dict and depth_gt_dict['stage_1'] is not None:
                depth_gt_s1 = depth_gt_dict['stage_1'][0].cpu().numpy()
                if depth_gt_s1.ndim == 3: depth_gt_s1 = np.squeeze(depth_gt_s1)
                
                valid_gt_mask_s1 = depth_gt_s1 > 0.0
                
                # 1. 构建平面掩码
                mask_planar_orig = ref_stage1_conf > PRED_PLANAR_CONF_THRESHOLD
                mask_planar_deg = ref_stage1_conf_degraded > PRED_PLANAR_CONF_THRESHOLD
                
                # 2. 👑 核心融合流向 (Mask-Gated Fusion): 
                # 平面区域采用高精度几何平面深度 ref_stage1_plane_depth，非平面/剔除伪平面区域回退原 PatchmatchNet 原生像素深度 ref_stage1_pixel_depth
                depth_fused_orig = np.where(mask_planar_orig, ref_stage1_plane_depth, ref_stage1_pixel_depth)
                depth_fused_deg = np.where(mask_planar_deg, ref_stage1_plane_depth, ref_stage1_pixel_depth)
                
                # 3. 统计多维度深度与区域消融 MAE
                valid_orig_plane = valid_gt_mask_s1 & mask_planar_orig   # 未过滤时的初始预测平面区
                valid_deg_plane = valid_gt_mask_s1 & mask_planar_deg     # 经 Cross-Check 过滤后的真平面区
                
                # 【1. 原版 PatchmatchNet 纯像素深度 ref_stage1_pixel_depth】
                mae_pixel_global = np.mean(np.abs(ref_stage1_pixel_depth[valid_gt_mask_s1] - depth_gt_s1[valid_gt_mask_s1]))
                mae_pixel_orig_plane = np.mean(np.abs(ref_stage1_pixel_depth[valid_orig_plane] - depth_gt_s1[valid_orig_plane])) if valid_orig_plane.sum() > 0 else 0.0
                mae_pixel_deg_plane = np.mean(np.abs(ref_stage1_pixel_depth[valid_deg_plane] - depth_gt_s1[valid_deg_plane])) if valid_deg_plane.sum() > 0 else 0.0

                # 【2. 三角几何平面解析深度 ref_stage1_plane_depth】
                mae_plane_orig_plane = np.mean(np.abs(ref_stage1_plane_depth[valid_orig_plane] - depth_gt_s1[valid_orig_plane])) if valid_orig_plane.sum() > 0 else 0.0
                mae_plane_deg_plane = np.mean(np.abs(ref_stage1_plane_depth[valid_deg_plane] - depth_gt_s1[valid_deg_plane])) if valid_deg_plane.sum() > 0 else 0.0

                # 【3. 全图物理融合深度 (平面走 plane, 其余走 pixel)】
                mae_fused_orig_global = np.mean(np.abs(depth_fused_orig[valid_gt_mask_s1] - depth_gt_s1[valid_gt_mask_s1]))
                mae_fused_deg_global = np.mean(np.abs(depth_fused_deg[valid_gt_mask_s1] - depth_gt_s1[valid_gt_mask_s1]))

                print(f"\n   ======================= 👑 Stage 1 多维度深度与区域消融评估 👑 =======================")
                print(f"   【1. 原版 PatchmatchNet 纯像素深度 (Z_pixel)】")
                print(f"      • 全图有效区域 MAE:               {mae_pixel_global:.4f} m")
                print(f"      • 未 Cross-Check 原始平面区 MAE:  {mae_pixel_orig_plane:.4f} m (像素数: {valid_orig_plane.sum()})")
                print(f"      • 经 Cross-Check 真实平面区 MAE:  {mae_pixel_deg_plane:.4f} m (像素数: {valid_deg_plane.sum()})")
                print(f"   【2. 三角几何平面解析深度 (Z_plane)】")
                print(f"      • 未 Cross-Check 原始平面区 MAE:  {mae_plane_orig_plane:.4f} m (包含误判伪平面)")
                print(f"      • 🌟 经 Cross-Check 真实平面区 MAE: {mae_plane_deg_plane:.4f} m (剔除伪平面后的真平面！)")
                print(f"   【3. 全图物理融合深度 (Z_fused = 平面用 Z_plane, 其余回退 Z_pixel)】")
                print(f"      • 未 Cross-Check 原始融合全图 MAE: {mae_fused_orig_global:.4f} m")
                print(f"      • 🌟 Cross-Checked 最终融合全图 MAE: {mae_fused_deg_global:.4f} m")
                print(f"   ====================================================================================")

                # 4. 生成融合深度绝对误差热力图
                import matplotlib.pyplot as plt
                diff_map_fused = np.zeros_like(depth_fused_deg)
                diff_map_fused[valid_gt_mask_s1] = np.abs(depth_fused_deg[valid_gt_mask_s1] - depth_gt_s1[valid_gt_mask_s1])
                
                plt.figure(figsize=(10, 8))
                map_masked_fused = np.ma.masked_where(~valid_gt_mask_s1, diff_map_fused)
                cmap_err = plt.get_cmap('jet')
                cmap_err.set_bad(color='black')
                
                diff_max = 1.0 # 截断 1.0 米
                im = plt.imshow(map_masked_fused, cmap=cmap_err, vmin=0, vmax=diff_max)
                cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
                cbar.set_label('Absolute Error (Meters)', size=14)
                
                plt.title(f'Stage 1 Cross-Checked Fused Depth Error Map\n'
                          f'Global MAE: {mae_fused_deg_global:.4f}m | Flat MAE: {mae_plane_deg_plane:.4f}m (Flat px: {valid_deg_plane.sum()})',
                          fontsize=13, fontweight='bold')
                plt.axis('off')
                
                fused_err_filename = os.path.join(args.outdir, sample["filename"][0].format('s1_err_fused', '.png'))
                os.makedirs(os.path.dirname(fused_err_filename), exist_ok=True)
                plt.savefig(fused_err_filename, dpi=150, bbox_inches='tight', pad_inches=0.1)
                plt.close()
                print(f"   [Visualization] Fused depth error map saved to: {fused_err_filename}")

                # =========================================================================
                # 5. 核心：Cross-Check 剔除对比与伪平面标记可视化图
                # =========================================================================
                import matplotlib.patches as mpatches

                H_s1, W_s1 = depth_gt_s1.shape
                retained_mask = valid_orig_plane & valid_deg_plane
                pruned_mask = valid_orig_plane & (~valid_deg_plane)
                non_plane_mask = valid_gt_mask_s1 & (~valid_orig_plane)

                pruned_count = pruned_mask.sum()
                retained_count = retained_mask.sum()
                prune_rate = (pruned_count / max(valid_orig_plane.sum(), 1)) * 100.0

                rgb_vis = np.zeros((H_s1, W_s1, 3), dtype=np.uint8)
                rgb_vis[non_plane_mask] = [30, 80, 200]    # 蓝色：原始非平面
                rgb_vis[retained_mask] = [0, 230, 70]       # 绿色：一致保留的高质量平面
                rgb_vis[pruned_mask] = [235, 40, 40]       # 红色：被 Cross-Check 剔除的伪平面！

                plt.figure(figsize=(11, 8))
                plt.imshow(rgb_vis)
                plt.axis('off')

                # 构造图例与标题
                patch_green = mpatches.Patch(color='#00E646', label=f'Retained Plane: {retained_count} px')
                patch_red = mpatches.Patch(color='#EB2828', label=f'Pruned False-Plane: {pruned_count} px ({prune_rate:.1f}%)')
                patch_blue = mpatches.Patch(color='#1E50C8', label=f'Non-Planar Region: {non_plane_mask.sum()} px')
                plt.legend(handles=[patch_green, patch_red, patch_blue], loc='upper right', framealpha=0.85, fontsize=12)

                plt.title(f'Cross-Check Planar Pruning Analysis (Conf > {PRED_PLANAR_CONF_THRESHOLD})\n'
                          f'Fused Global MAE: {mae_fused_deg_global:.4f}m | Pruned False-Planar Area: {pruned_count} px ({prune_rate:.1f}%)',
                          fontsize=13, fontweight='bold')

                prune_vis_filename = os.path.join(args.outdir, sample["filename"][0].format('cross_check_pruned_regions', '.png'))
                os.makedirs(os.path.dirname(prune_vis_filename), exist_ok=True)
                plt.savefig(prune_vis_filename, dpi=150, bbox_inches='tight', pad_inches=0.1)
                plt.close()
                print(f"   [Visualization] Pruned comparison map saved to: {prune_vis_filename}")
            
            # Save results
            print("Saving results...")
            filenames = sample["filename"]
            outdir = args.outdir
            for i, filename in enumerate(filenames):
                depth_filename = os.path.join(outdir, filename.format('depth_est', '.pfm'))
                mask_filename = os.path.join(outdir, filename.format('cross_check_mask', '.png'))
                
                # Make colorized images for confidence
                import matplotlib.pyplot as plt
                
                def save_color_conf(fpath, conf_map):
                    # using jet_r colormap for confidence (0 to 1), where 1.0 is blue, 0.0 is red
                    cmap = plt.get_cmap('jet_r')
                    conf_rgb = cmap(np.clip(conf_map, 0, 1))[..., :3]
                    # black out regions where confidence is exactly 0 (or very close)
                    conf_rgb[conf_map < 1e-3] = 0.0
                    plt.imsave(fpath, conf_rgb)
                    
                conf_img_filename = os.path.join(outdir, filename.format('confidence_color', '.png'))
                conf_deg_img_filename = os.path.join(outdir, filename.format('confidence_degraded_color', '.png'))
                
                for fpath in [depth_filename, mask_filename, conf_img_filename, conf_deg_img_filename]:
                    os.makedirs(os.path.dirname(fpath), exist_ok=True)
                
                # 仅保存 depth_est.pfm，不再保存庞大的 confidence pfm
                save_pfm(depth_filename, ref_stage0_depth)
                
                save_color_conf(conf_img_filename, ref_stage0_conf)
                save_color_conf(conf_deg_img_filename, ref_stage0_conf_degraded)
                
                mask_img = (is_consistent_s0 * 255).astype(np.uint8)
                cv2.imwrite(mask_filename, mask_img)

if __name__ == '__main__':
    # step1. save all the depth maps and the masks in outputs directory
    # save_depth()
    save_depth_cross_check()
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