import argparse
import os
import sys
import time
import json
import math
import cv2

# 控制要暴露给进程的 GPU id：优先使用外部环境变量 GPU_ID，
# 否则使用已有的 CUDA_VISIBLE_DEVICES，最后回退到 '0'
_gpu_choice = os.environ.get('GPU_ID', os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
os.environ['CUDA_VISIBLE_DEVICES'] = str(_gpu_choice)

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np

from datasets import find_dataset_def
from datasets.dtu_whu import collate_keep_list
from datasets.data_io import save_pfm
from models import PatchmatchNet
from utils import print_args, tocuda, make_nograd_func, DictAverageMeter, AbsDepthError_metrics, Thres_metrics, map_tri_to_pixel_single

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(
        description='WHU MVS 独立深度精度与平面区域评测脚本 (生成平面掩码供源模型基线对比)'
    )
    # 模型与数据集
    parser.add_argument('--model', default='PatchmatchNet', help='select model')
    parser.add_argument('--dataset', default='dtu_whu', help='select dataset (default: dtu_whu)')
    parser.add_argument('--testpath', default='/home/ym/Experiment/Datas/WHU_MVS_dataset',
                        help='testing data path')
    parser.add_argument('--testlist', default='lists/whu/minitest.txt',
                        help='testing scan list file (default: lists/whu/minitest.txt)')
    parser.add_argument('--loadckpt', required=True, help='path to the checkpoint (.ckpt) to evaluate')
    parser.add_argument('--outdir', default='./outputs_test',
                        help='output directory to save masks, reports and depths')

    # 批次与设备设置
    parser.add_argument('--batch_size', type=int, default=1, help='testing batch size (recommend 1)')
    parser.add_argument('--n_views', type=int, default=5, help='number of views per sample')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers for DataLoader')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    # 评测模式解耦开关
    parser.add_argument('--eval_mode', type=str, default='planar_comp',
                        choices=['planar_comp', 'all', 'global'],
                        help='评测模式: planar_comp (仅执行Stage0平面三方对照实验, 极速轻量, 默认), all (执行全部完整多尺度与GNN评估), global (仅执行全局多尺度精度)')

    # 掩码导出与深度保存
    parser.add_argument('--save_masks', action='store_true', default=False,
                        help='save stage0 and stage1 planar mask images (default: False)')
    parser.add_argument('--no_masks', '--no_save_masks', dest='save_masks', action='store_false',
                        help='fast evaluation mode: do not save planar mask images (default: enabled)')
    parser.add_argument('--save_depth', action='store_true', default=False,
                        help='save predicted depth maps in .pfm and 16-bit .png format')

    # PatchMatch 超参数 (与 train_whu.py 严格对齐)
    parser.add_argument('--patchmatch_iteration', nargs='+', type=int, default=[1, 2, 2],
                        help='num of iteration of patchmatch on stages 1,2,3')
    parser.add_argument('--patchmatch_num_sample', nargs='+', type=int, default=[8, 8, 16],
                        help='num of generated samples in local perturbation on stages 1,2,3')
    parser.add_argument('--patchmatch_interval_scale', nargs='+', type=float, default=[0.005, 0.0125, 0.025],
                        help='normalized interval in inverse depth range to generate samples in local perturbation')
    parser.add_argument('--patchmatch_range', nargs='+', type=int, default=[6, 4, 2],
                        help='fixed offset of sampling points for propogation of patchmatch on stages 1,2,3')
    parser.add_argument('--propagate_neighbors', nargs='+', type=int, default=[0, 8, 16],
                        help='num of neighbors for adaptive propagation on stages 1,2,3')
    parser.add_argument('--evaluate_neighbors', nargs='+', type=int, default=[9, 9, 9],
                        help='num of neighbors for adaptive matching cost aggregation on stages 1,2,3')

    return parser.parse_args()


def load_checkpoint_with_channel_adaptation(model, state_dict_model):
    """
    自适应权重迁移与平滑加载：
    针对 Stage 0 Refinement 从 3 通道 (RGB) 升级至 4 通道 (RGB + W_plane_s0) 实行前3通道继承、第4通道置零初始化，
    支持剥离 DataParallel 的 'module.' 前缀，确保任意架构状态平滑加载。
    """
    cleaned_state = {}
    for k, v in state_dict_model.items():
        clean_k = k[7:] if k.startswith('module.') else k
        cleaned_state[clean_k] = v

    model_state = model.state_dict()
    key = 'upsample_net.conv0.conv.weight'
    if key in cleaned_state and key in model_state:
        ckpt_w = cleaned_state[key]
        tgt_w = model_state[key]
        if ckpt_w.shape[1] == 4 and tgt_w.shape[1] == 3:
            print(f"[Adaptive Loading] Adapting {key} from [8, 4, 3, 3] -> [8, 3, 3, 3] (discarding 4th channel)...")
            cleaned_state[key] = ckpt_w[:, :3, :, :]
        elif ckpt_w.shape[1] == 3 and tgt_w.shape[1] == 4:
            print(f"[Adaptive Loading] Adapting {key} from [8, 3, 3, 3] -> [8, 4, 3, 3] (Channel 4 set to 0.0 for zero-shock start)...")
            adapted_w = torch.zeros_like(tgt_w)
            adapted_w[:, :3, :, :] = ckpt_w
            cleaned_state[key] = adapted_w

    try:
        model.load_state_dict(cleaned_state, strict=True)
        print("✓ Model checkpoint loaded successfully with strict=True.")
    except RuntimeError as e:
        print(f"Warning: Exact load failed ({e}), loading with strict=False...")
        model.load_state_dict(cleaned_state, strict=False)
        print("✓ Model checkpoint loaded with strict=False.")


@make_nograd_func
def evaluate_single_sample(model, sample, device):
    """
    执行单个测试 Batch 的模型前向推断与各项指标严密提取。
    """
    # 1. 准备 CDT 三角网与几何数据 (与 train_whu.py 严格一致)
    vertexs_batch = sample['vertexs']
    lines_batch = sample['lines']
    triangles_batch = []
    for tri_list in sample['triangles']:
        tri_processed = []
        for t in tri_list:
            v_ids = t['vertex_ids']
            l_ids = t['line_ids']
            pts = torch.from_numpy(t['valid_points']).to(device)
            tri_processed.append((v_ids, l_ids, pts))
        triangles_batch.append(tri_processed)

    skip = ["vertexs", "lines", "triangles", "tri_conf_cleaned", "tri_normal_cleaned", "is_gt_planar", "tri_err95", "tri_weight_normal", "tri_weight_conf"]
    sample_cuda = tocuda(sample, device=device, skip_keys=skip)

    sample_cuda['tri_conf_cleaned'] = [torch.from_numpy(v).to(device).float() for v in sample['tri_conf_cleaned']]
    sample_cuda['tri_normal_cleaned'] = [torch.from_numpy(v).to(device).float() for v in sample['tri_normal_cleaned']]
    if 'is_gt_planar' in sample:
        sample_cuda['is_gt_planar'] = [torch.from_numpy(v).to(device).bool() for v in sample['is_gt_planar']]

    depth_gt = sample_cuda["depth"]
    mask = sample_cuda["mask"]
    intrinsics_s0 = sample_cuda["intrinsics_mats"]['stage_0'][:, 0]

    # 测试期参数：关闭约束反向梯度，使用收敛期确定性温度 0.55
    max_lambda_c = 0.0
    max_lambda_s = 0.0
    eval_temperature = 0.55

    outputs = model(
        sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["intrinsics_mats"],
        sample_cuda["depth_min"], sample_cuda["depth_max"],
        vertexs_batch, lines_batch, triangles_batch, depth_gt['stage_1'],
        max_lambda_c, max_lambda_s, eval_temperature, compute_edge_pixels=False
    )

    return outputs, depth_gt, mask, intrinsics_s0


def safe_mae(depth_est, depth_gt, mask):
    """安全计算 MAE，若有效像素数为 0 则返回 NaN。"""
    if mask is None or not mask.any():
        return float('nan')
    est_valid = depth_est[mask]
    gt_valid = depth_gt[mask]
    return torch.mean((est_valid - gt_valid).abs()).item()


def safe_thres_error(depth_est, depth_gt, mask, thres):
    """安全计算误差超过阈值的像素比例 (Error Rate: error > thres)。"""
    if mask is None or not mask.any():
        return float('nan')
    est_valid = depth_est[mask]
    gt_valid = depth_gt[mask]
    err = (est_valid - gt_valid).abs()
    return (err > thres).float().mean().item()


def evaluate_global_accuracy(d_est_s0_b, depth_pm, depth_gt, mask, b):
    """【全图精度算子】计算 Stage 0~3 多尺度全局 MAE 与阈值误差率。"""
    m_s0_b = (mask['stage_0'][b:b+1] > 0.5)
    d_gt_s0_b = depth_gt['stage_0'][b:b+1]

    s0_mae = safe_mae(d_est_s0_b, d_gt_s0_b, m_s0_b)
    t1_err = safe_thres_error(d_est_s0_b, d_gt_s0_b, m_s0_b, 1.0)
    t2_err = safe_thres_error(d_est_s0_b, d_gt_s0_b, m_s0_b, 2.0)
    t4_err = safe_thres_error(d_est_s0_b, d_gt_s0_b, m_s0_b, 4.0)
    t8_err = safe_thres_error(d_est_s0_b, d_gt_s0_b, m_s0_b, 8.0)

    m_s1_b = (mask['stage_1'][b:b+1] > 0.5)
    s1_mae = safe_mae(depth_pm['stage_1'][-1][b:b+1], depth_gt['stage_1'][b:b+1], m_s1_b)
    s2_mae = safe_mae(depth_pm['stage_2'][-1][b:b+1], depth_gt['stage_2'][b:b+1], mask['stage_2'][b:b+1] > 0.5)
    s3_mae = safe_mae(depth_pm['stage_3'][-1][b:b+1], depth_gt['stage_3'][b:b+1], mask['stage_3'][b:b+1] > 0.5)

    return {
        "stage0_mae": s0_mae,
        "stage0_thres1mm_err": t1_err,
        "stage0_thres2mm_err": t2_err,
        "stage0_thres4mm_err": t4_err,
        "stage0_thres8mm_err": t8_err,
        "stage1_fused_mae": s1_mae,
        "stage2_mae": s2_mae,
        "stage3_mae": s3_mae,
    }


def evaluate_stage1_gnn_gain(depth_pm, depth_gt, mask_gt_s1, eval_mask_s1_planar, b):
    """【GNN 平面增益算子】对比 Stage 1 纯像素深度 vs GNN 传播深度在平面区域的表现。"""
    m_s1_b = mask_gt_s1[b:b+1]
    m_s1_planar_b = eval_mask_s1_planar[b:b+1]
    d_gt_s1_b = depth_gt['stage_1'][b:b+1]
    d_s1_prop_b = depth_pm['stage_1'][-1][b:b+1]
    d_s1_pixel_b = depth_pm['stage_1'][0][b:b+1]

    s1_prop_mae = safe_mae(d_s1_prop_b, d_gt_s1_b, m_s1_b)
    s1_pixel_mae = safe_mae(d_s1_pixel_b, d_gt_s1_b, m_s1_b)
    s1_planar_prop_mae = safe_mae(d_s1_prop_b, d_gt_s1_b, m_s1_planar_b)
    s1_planar_pixel_mae = safe_mae(d_s1_pixel_b, d_gt_s1_b, m_s1_planar_b)

    s1_valid_cnt = m_s1_b.sum().item()
    s1_planar_cnt = m_s1_planar_b.sum().item()
    s1_planar_ratio = (s1_planar_cnt / s1_valid_cnt * 100.0) if s1_valid_cnt > 0 else 0.0

    return {
        "stage1_prop_mae": s1_prop_mae,
        "stage1_pixel_mae": s1_pixel_mae,
        "stage1_planar_prop_mae": s1_planar_prop_mae,
        "stage1_planar_pixel_mae": s1_planar_pixel_mae,
        "s1_valid_pixels": s1_valid_cnt,
        "s1_planar_pixels": s1_planar_cnt,
        "s1_planar_ratio_pct": s1_planar_ratio,
    }


def compute_analytic_planar_depth_stage0(final_planes, tri_id_map_s0, intrinsics_s0, H0, W0):
    """
    【几何解析算子】根据 Stage 1 优化后的三维平面方程 (n, d) 与 Stage 0 像素视线射线做纯数学解析求交，
    生成 Stage 0 尺度的静态无损几何平面深度图 (Analytic Ray-Plane Intersection: Z = -d / (n · r))。

    Args:
        final_planes:    [B, N_tri, 4] GNN 优化后的三维平面参数 (nx, ny, nz, d)
        tri_id_map_s0:   [B, H0, W0] Stage 0 三角形索引地图 (-1 表示非平面区域)
        intrinsics_s0:   [B, 3, 3] Stage 0 主视角内参矩阵
        H0, W0:          int, Stage 0 分辨率高宽

    Returns:
        depth_analytic:  [B, 1, H0, W0] 静态无损几何平面深度图
        valid_ray_mask:  [B, 1, H0, W0] 射线与平面有效相交掩码 (点积非零且处于有效三角形内)
    """
    B = final_planes.shape[0]
    device = final_planes.device

    # 1. 将稀疏平面方程映射为 Stage 0 密集场 [B, 4, H0, W0]
    pixel_planes_s0 = map_tri_to_pixel_single(final_planes, tri_id_map_s0, H0, W0)
    nx = pixel_planes_s0[:, 0:1, :, :]
    ny = pixel_planes_s0[:, 1:2, :, :]
    nz = pixel_planes_s0[:, 2:3, :, :]
    d = pixel_planes_s0[:, 3:4, :, :]

    # 2. 构造 Stage 0 视线射线方向 r = [(u - cx)/fx, (v - cy)/fy, 1.0]^T
    fx = intrinsics_s0[:, 0, 0].view(B, 1, 1, 1)
    fy = intrinsics_s0[:, 1, 1].view(B, 1, 1, 1)
    cx = intrinsics_s0[:, 0, 2].view(B, 1, 1, 1)
    cy = intrinsics_s0[:, 1, 2].view(B, 1, 1, 1)

    y_grid, x_grid = torch.meshgrid(
        torch.arange(H0, device=device, dtype=torch.float32),
        torch.arange(W0, device=device, dtype=torch.float32),
        indexing='ij'
    )
    x_grid = x_grid.view(1, 1, H0, W0).expand(B, 1, -1, -1)
    y_grid = y_grid.view(1, 1, H0, W0).expand(B, 1, -1, -1)

    ray_x = (x_grid - cx) / fx
    ray_y = (y_grid - cy) / fy

    # 3. 解析点积 n · r = nx * rx + ny * ry + nz * 1.0
    dot_product = nx * ray_x + ny * ray_y + nz

    # 4. 几何有效性过滤：dot 不能太接近 0 (避免视线平行于平面)，且必须落在有效三角形内
    valid_dot = torch.abs(dot_product) > 1e-4
    valid_tri = (tri_id_map_s0.unsqueeze(1) >= 0)
    valid_ray_mask = valid_dot & valid_tri

    dot_safe = torch.where(valid_dot, dot_product, torch.sign(dot_product + 1e-10) * 1e-4)

    # 5. 平面深度解析解 Z = -d / (n · r)
    depth_analytic = (-d / dot_safe).abs()
    depth_analytic = torch.where(valid_ray_mask, depth_analytic, torch.zeros_like(depth_analytic))

    return depth_analytic, valid_ray_mask


def compare_stage0_planar_depths(depth_refined_s0, depth_analytic_s0, depth_bilinear_s0,
                                 depth_gt_s0, eval_mask_s0_planar, valid_ray_mask):
    """
    【三方严密对照算子】在 Stage 0 真实平面区域内量化对比三级深度状态：
    1. 双线性插值基底 (Bilinear Interpolation): Stage 1 融合深度直接上采样
    2. 静态无损几何 (Analytic Ray-Plane): GNN 平面方程纯几何解析求交 Z = -d / (n · r)
    3. 残差细化深度 (Residual Refinement): CNN 残差网络最终输出

    Args:
        depth_refined_s0:     [B, 1, H0, W0] Stage 0 残差网络预测深度
        depth_analytic_s0:    [B, 1, H0, W0] 静态无损几何放大深度
        depth_bilinear_s0:    [B, 1, H0, W0] Stage 1 深度双线性放大深度
        depth_gt_s0:          [B, 1, H0, W0] Stage 0 真实深度 (Ground Truth)
        eval_mask_s0_planar:  [B, 1, H0, W0] Stage 0 核心评估掩码 (GT有效且为平面)
        valid_ray_mask:       [B, 1, H0, W0] 射线求交有效掩码

    Returns:
        metrics: dict 包含三方的 MAE、两两差值 (Δ mm)、成对及最优胜率 (%) 与两两分歧 (mm)
    """
    # 联合有效掩码
    joint_mask = eval_mask_s0_planar & valid_ray_mask
    valid_cnt = joint_mask.sum().item()

    default_res = {
        "mae_bilinear": float('nan'),
        "mae_analytic": float('nan'),
        "mae_residual": float('nan'),
        "delta_ana_vs_bil_mm": float('nan'),
        "delta_res_vs_bil_mm": float('nan'),
        "delta_res_vs_ana_mm": float('nan'),
        "win_ana_over_bil_pct": float('nan'),
        "win_res_over_bil_pct": float('nan'),
        "win_res_over_ana_pct": float('nan'),
        "best_bil_pct": float('nan'),
        "best_ana_pct": float('nan'),
        "best_res_pct": float('nan'),
        "disagree_ana_bil_mm": float('nan'),
        "disagree_res_bil_mm": float('nan'),
        "disagree_res_ana_mm": float('nan'),
        "valid_pixels": 0,
    }

    if valid_cnt == 0:
        return default_res

    d_bil = depth_bilinear_s0[joint_mask]
    d_ana = depth_analytic_s0[joint_mask]
    d_res = depth_refined_s0[joint_mask]
    d_gt = depth_gt_s0[joint_mask]

    err_bil = (d_bil - d_gt).abs()
    err_ana = (d_ana - d_gt).abs()
    err_res = (d_res - d_gt).abs()

    mae_bil = err_bil.mean().item()
    mae_ana = err_ana.mean().item()
    mae_res = err_res.mean().item()

    # 差值 (单位：mm，负数代表前者比后者误差小/优)
    delta_ana_vs_bil_mm = (mae_ana - mae_bil) * 1000.0  # <0 说明无损放大优于双线性插值
    delta_res_vs_bil_mm = (mae_res - mae_bil) * 1000.0  # <0 说明残差网络优于双线性插值
    delta_res_vs_ana_mm = (mae_res - mae_ana) * 1000.0  # <0 说明残差网络优于无损放大

    # 成对胜率 (A < B 的比例)
    win_ana_over_bil = (err_ana < err_bil).float().mean().item() * 100.0
    win_res_over_bil = (err_res < err_bil).float().mean().item() * 100.0
    win_res_over_ana = (err_res < err_ana).float().mean().item() * 100.0

    # 三方最优胜率 (谁的绝对误差最小)
    best_bil = ((err_bil <= err_ana) & (err_bil <= err_res)).float().mean().item() * 100.0
    best_ana = ((err_ana < err_bil) & (err_ana <= err_res)).float().mean().item() * 100.0
    best_res = ((err_res < err_bil) & (err_res < err_ana)).float().mean().item() * 100.0

    # 两两之间的直接分歧 (Disagreement mm)
    disagree_ana_bil = (d_ana - d_bil).abs().mean().item() * 1000.0
    disagree_res_bil = (d_res - d_bil).abs().mean().item() * 1000.0
    disagree_res_ana = (d_res - d_ana).abs().mean().item() * 1000.0

    return {
        "mae_bilinear": mae_bil,
        "mae_analytic": mae_ana,
        "mae_residual": mae_res,
        "delta_ana_vs_bil_mm": delta_ana_vs_bil_mm,
        "delta_res_vs_bil_mm": delta_res_vs_bil_mm,
        "delta_res_vs_ana_mm": delta_res_vs_ana_mm,
        "win_ana_over_bil_pct": win_ana_over_bil,
        "win_res_over_bil_pct": win_res_over_bil,
        "win_res_over_ana_pct": win_res_over_ana,
        "best_bil_pct": best_bil,
        "best_ana_pct": best_ana,
        "best_res_pct": best_res,
        "disagree_ana_bil_mm": disagree_ana_bil,
        "disagree_res_bil_mm": disagree_res_bil,
        "disagree_res_ana_mm": disagree_res_ana,
        "valid_pixels": int(valid_cnt),
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print("=" * 80)
    print("WHU MVS 独立精度评测与平面掩码导出系统 (test_whu.py)")
    print("=" * 80)
    print_args(args)

    os.makedirs(args.outdir, exist_ok=True)

    # 1. 构建测试数据集与 DataLoader
    MVSDataset = find_dataset_def(args.dataset)
    print(f"\n[Dataset] 加载测试数据集: path={args.testpath}, listfile={args.testlist}")
    test_dataset = MVSDataset(args.testpath, args.testlist, "test", nviews=args.n_views, robust_train=False)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_keep_list, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=(args.num_workers > 0),
        drop_last=False
    )
    total_samples = len(test_dataset)
    print(f"[Dataset] 测试样本总量: {total_samples} 张图像 (共 {len(test_loader)} 个批次)\n")

    # 2. 构建模型并加载权重
    model = PatchmatchNet(
        patchmatch_interval_scale=args.patchmatch_interval_scale,
        propagation_range=args.patchmatch_range,
        patchmatch_iteration=args.patchmatch_iteration,
        patchmatch_num_sample=args.patchmatch_num_sample,
        propagate_neighbors=args.propagate_neighbors,
        evaluate_neighbors=args.evaluate_neighbors
    )
    model.to(device)

    print(f"[Model] 载入 Checkpoint 权重: {args.loadckpt}")
    checkpoint = torch.load(args.loadckpt, map_location=device)
    state_dict_model = checkpoint.get('model', checkpoint)
    load_checkpoint_with_channel_adaptation(model, state_dict_model)
    model.eval()

    # 3. 统计容器初始化与模式分流
    per_scan_records = {}
    all_sample_records = []

    do_planar_comp = args.eval_mode in ['planar_comp', 'all']
    do_global = args.eval_mode in ['global', 'all']
    do_stage1_gnn = args.eval_mode in ['all']

    print("\n" + "=" * 80)
    print(f"开始逐样本前向推断与评测... 模式: [{args.eval_mode}] "
          f"(平面三方对照: {'开启' if do_planar_comp else '关闭'}, "
          f"全局精度: {'开启' if do_global else '关闭'}, "
          f"Stage1 GNN增益: {'开启' if do_stage1_gnn else '关闭'})")
    print("=" * 80)

    start_eval_time = time.time()

    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            step_start = time.time()
            B = sample['imgs']['stage_0'].shape[0]

            outputs, depth_gt, mask, intrinsics_s0 = evaluate_single_sample(model, sample, device)

            depth_est_s0 = outputs["refined_depth"]['stage_0']           # [B, 1, H0, W0]
            depth_pm = outputs["depth_patchmatch"]                       # dict of stages
            output_plane = outputs.get("output_plane", {})

            # ----------------------------------------------------
            # 掩码准备 (Stage 0 / Stage 1)
            # ----------------------------------------------------
            mask_gt_s0 = (mask['stage_0'] > 0.5)                         # [B, 1, H0, W0]
            is_planar_s0 = output_plane.get("is_planar_s0", torch.zeros_like(mask_gt_s0)) # [B, 1, H0, W0] bool
            eval_mask_s0_planar = mask_gt_s0 & is_planar_s0
            eval_mask_s0_curved = mask_gt_s0 & (~is_planar_s0)

            mask_gt_s1 = (mask['stage_1'] > 0.5)                         # [B, 1, H1, W1]
            w_pixel_s1 = output_plane.get("W_plane_pixel", None)
            tri_id_map_s1 = output_plane.get("tri_id_map", None)

            if w_pixel_s1 is not None and tri_id_map_s1 is not None:
                valid_tri_s1 = (tri_id_map_s1.unsqueeze(1) >= 0)
                is_planar_s1 = (w_pixel_s1 >= 0.80) & valid_tri_s1
                eval_mask_s1_planar = mask_gt_s1 & is_planar_s1
                depth_s1_fused = torch.where(is_planar_s1, depth_pm['stage_1'][-1], depth_pm['stage_1'][0])
            else:
                is_planar_s1 = torch.zeros_like(mask_gt_s1)
                eval_mask_s1_planar = torch.zeros_like(mask_gt_s1)
                depth_s1_fused = depth_pm['stage_1'][-1]

            # ----------------------------------------------------
            # 接口 1: Stage 0 平面三方对照基底 (按需激活，节省算力显存)
            # ----------------------------------------------------
            if do_planar_comp:
                final_planes = output_plane.get("final_plane", None)
                tri_id_map_s0 = output_plane.get("tri_id_map_stage0", None)
                H0, W0 = depth_est_s0.shape[2], depth_est_s0.shape[3]

                if final_planes is not None and tri_id_map_s0 is not None:
                    depth_analytic_s0, valid_ray_mask_s0 = compute_analytic_planar_depth_stage0(
                        final_planes, tri_id_map_s0, intrinsics_s0, H0, W0
                    )
                else:
                    depth_analytic_s0 = None
                    valid_ray_mask_s0 = None

                depth_bilinear_s0 = F.interpolate(depth_s1_fused, scale_factor=2, mode='bilinear', align_corners=True)
            else:
                depth_analytic_s0 = None
                valid_ray_mask_s0 = None
                depth_bilinear_s0 = None

            # ----------------------------------------------------
            # 逐 Batch 内样本分别计算与保存
            # ----------------------------------------------------
            for b in range(B):
                meta_idx = batch_idx * args.batch_size + b
                scan, file_id = test_dataset.metas[meta_idx]

                if scan not in per_scan_records:
                    per_scan_records[scan] = []

                record = {
                    "scan": scan,
                    "file_id": file_id,
                }

                # --- 接口 1: 执行 Stage 0 平面区域三方对照 ---
                if do_planar_comp:
                    if depth_analytic_s0 is not None and valid_ray_mask_s0 is not None:
                        comp_metrics = compare_stage0_planar_depths(
                            depth_est_s0[b:b+1],
                            depth_analytic_s0[b:b+1],
                            depth_bilinear_s0[b:b+1],
                            depth_gt['stage_0'][b:b+1],
                            eval_mask_s0_planar[b:b+1],
                            valid_ray_mask_s0[b:b+1]
                        )
                    else:
                        comp_metrics = {
                            "mae_bilinear": float('nan'),
                            "mae_analytic": float('nan'),
                            "mae_residual": float('nan'),
                            "delta_ana_vs_bil_mm": float('nan'),
                            "delta_res_vs_bil_mm": float('nan'),
                            "delta_res_vs_ana_mm": float('nan'),
                            "win_ana_over_bil_pct": float('nan'),
                            "win_res_over_bil_pct": float('nan'),
                            "win_res_over_ana_pct": float('nan'),
                            "best_bil_pct": float('nan'),
                            "best_ana_pct": float('nan'),
                            "best_res_pct": float('nan'),
                            "disagree_ana_bil_mm": float('nan'),
                            "disagree_res_bil_mm": float('nan'),
                            "disagree_res_ana_mm": float('nan'),
                            "valid_pixels": 0,
                        }
                    record.update({
                        "comp_mae_bil": comp_metrics["mae_bilinear"],
                        "comp_mae_ana": comp_metrics["mae_analytic"],
                        "comp_mae_res": comp_metrics["mae_residual"],
                        "comp_delta_ana_vs_bil_mm": comp_metrics["delta_ana_vs_bil_mm"],
                        "comp_delta_res_vs_bil_mm": comp_metrics["delta_res_vs_bil_mm"],
                        "comp_delta_res_vs_ana_mm": comp_metrics["delta_res_vs_ana_mm"],
                        "comp_win_ana_over_bil": comp_metrics["win_ana_over_bil_pct"],
                        "comp_win_res_over_bil": comp_metrics["win_res_over_bil_pct"],
                        "comp_win_res_over_ana": comp_metrics["win_res_over_ana_pct"],
                        "comp_best_bil_pct": comp_metrics["best_bil_pct"],
                        "comp_best_ana_pct": comp_metrics["best_ana_pct"],
                        "comp_best_res_pct": comp_metrics["best_res_pct"],
                        "comp_disagree_ana_bil_mm": comp_metrics["disagree_ana_bil_mm"],
                        "comp_disagree_res_bil_mm": comp_metrics["disagree_res_bil_mm"],
                        "comp_disagree_res_ana_mm": comp_metrics["disagree_res_ana_mm"],
                        "comp_valid_pixels": comp_metrics["valid_pixels"],
                    })

                # --- 接口 2: 执行全图多尺度全局精度 (Table 1) ---
                if do_global:
                    global_metrics = evaluate_global_accuracy(
                        depth_est_s0[b:b+1], depth_pm, depth_gt, mask, b
                    )
                    s0_planar_mae = safe_mae(depth_est_s0[b:b+1], depth_gt['stage_0'][b:b+1], eval_mask_s0_planar[b:b+1])
                    s0_curved_mae = safe_mae(depth_est_s0[b:b+1], depth_gt['stage_0'][b:b+1], eval_mask_s0_curved[b:b+1])
                    s0_valid_cnt = (mask['stage_0'][b:b+1] > 0.5).sum().item()
                    s0_planar_cnt = eval_mask_s0_planar[b:b+1].sum().item()
                    s0_planar_ratio = (s0_planar_cnt / s0_valid_cnt * 100.0) if s0_valid_cnt > 0 else 0.0

                    record.update(global_metrics)
                    record.update({
                        "stage0_planar_mae": s0_planar_mae,
                        "stage0_curved_mae": s0_curved_mae,
                        "s0_valid_pixels": s0_valid_cnt,
                        "s0_planar_pixels": s0_planar_cnt,
                        "s0_planar_ratio_pct": s0_planar_ratio,
                    })

                # --- 接口 3: 执行 Stage 1 GNN 平面增益 (Table 2) ---
                if do_stage1_gnn:
                    gnn_metrics = evaluate_stage1_gnn_gain(
                        depth_pm, depth_gt, mask_gt_s1, eval_mask_s1_planar, b
                    )
                    record.update(gnn_metrics)

                per_scan_records[scan].append(record)
                all_sample_records.append(record)

                # --- 导出平面掩码 (供源版 PatchmatchNet 对比) ---
                if args.save_masks:
                    mask_save_dir = os.path.join(args.outdir, scan, "masks")
                    os.makedirs(mask_save_dir, exist_ok=True)
                    s0_eval_mask_np = (eval_mask_s0_planar[b, 0].cpu().numpy().astype(np.uint8)) * 255
                    cv2.imwrite(os.path.join(mask_save_dir, f"stage0_planar_mask_{file_id}.png"), s0_eval_mask_np)
                    s1_eval_mask_np = (eval_mask_s1_planar[b, 0].cpu().numpy().astype(np.uint8)) * 255
                    cv2.imwrite(os.path.join(mask_save_dir, f"stage1_planar_mask_{file_id}.png"), s1_eval_mask_np)

                # --- 导出预测深度图 ---
                if args.save_depth:
                    depth_save_dir = os.path.join(args.outdir, scan, "depths")
                    os.makedirs(depth_save_dir, exist_ok=True)
                    d_s0_np = depth_est_s0[b, 0].cpu().numpy()
                    save_pfm(os.path.join(depth_save_dir, f"stage0_refined_depth_{file_id}.pfm"), d_s0_np)
                    d_s1_fused_np = depth_s1_fused[b, 0].cpu().numpy()
                    save_pfm(os.path.join(depth_save_dir, f"stage1_fused_depth_{file_id}.pfm"), d_s1_fused_np)

            # 终端简明打印进度 (根据模式自适应输出)
            cur_idx = batch_idx + 1
            step_time = time.time() - step_start
            if args.eval_mode == 'planar_comp':
                d_ana_bil = f"{comp_metrics['delta_ana_vs_bil_mm']:+.2f}mm" if not math.isnan(comp_metrics['delta_ana_vs_bil_mm']) else "N/A"
                d_res_bil = f"{comp_metrics['delta_res_vs_bil_mm']:+.2f}mm" if not math.isnan(comp_metrics['delta_res_vs_bil_mm']) else "N/A"
                print(f"[{cur_idx:03d}/{len(test_loader):03d}] Scan: {scan} | Img: {file_id} | "
                      f"Comp: Bil={comp_metrics['mae_bilinear']:.4f}m | Ana={comp_metrics['mae_analytic']:.4f}m | Res={comp_metrics['mae_residual']:.4f}m "
                      f"(ΔAna-Bil={d_ana_bil}, ΔRes-Bil={d_res_bil}) | Time: {step_time:.2f}s")
            elif args.eval_mode == 'global':
                print(f"[{cur_idx:03d}/{len(test_loader):03d}] Scan: {scan} | Img: {file_id} | "
                      f"S0 MAE: {global_metrics['stage0_mae']:.4f}m | S1 MAE: {global_metrics['stage1_fused_mae']:.4f}m | "
                      f">1mm Err: {global_metrics['stage0_thres1mm_err']*100:.1f}% | Time: {step_time:.2f}s")
            else:
                print(f"[{cur_idx:03d}/{len(test_loader):03d}] Scan: {scan} | Img: {file_id} | "
                      f"S0 MAE: {global_metrics['stage0_mae']:.4f}m | S0 Planar: {s0_planar_mae:.4f}m | "
                      f"Comp: Bil={comp_metrics['mae_bilinear']:.4f}m | Ana={comp_metrics['mae_analytic']:.4f}m | Res={comp_metrics['mae_residual']:.4f}m | "
                      f"Time: {step_time:.2f}s")

    eval_duration = time.time() - start_eval_time
    print("\n" + "=" * 80)
    print(f"测试完成！耗时: {eval_duration:.2f} 秒，共评估 {len(all_sample_records)} 张图像。")
    print("=" * 80)

    # =========================================================================
    # 4. 数据汇总与论文级 Markdown 表格生成
    # =========================================================================
    def compute_mean_ignore_nan(records, key):
        vals = [r[key] for r in records if key in r and not math.isnan(r[key])]
        return float(np.mean(vals)) if len(vals) > 0 else float('nan')

    # 按 Scan 汇总 (按需聚合开启的模块指标)
    summary_by_scan = {}
    for scan, recs in per_scan_records.items():
        s_dict = {"count": len(recs)}
        if do_global:
            for k in ["stage0_mae", "stage0_planar_mae", "stage0_curved_mae", "stage0_thres1mm_err",
                      "stage0_thres2mm_err", "stage0_thres4mm_err", "stage0_thres8mm_err",
                      "stage1_fused_mae", "stage2_mae", "stage3_mae", "s0_planar_ratio_pct"]:
                s_dict[k] = compute_mean_ignore_nan(recs, k)
        if do_stage1_gnn:
            for k in ["stage1_prop_mae", "stage1_pixel_mae", "stage1_planar_prop_mae",
                      "stage1_planar_pixel_mae", "s1_planar_ratio_pct"]:
                s_dict[k] = compute_mean_ignore_nan(recs, k)
        if do_planar_comp:
            for k in ["comp_mae_bil", "comp_mae_ana", "comp_mae_res", "comp_delta_ana_vs_bil_mm",
                      "comp_delta_res_vs_bil_mm", "comp_delta_res_vs_ana_mm", "comp_win_ana_over_bil",
                      "comp_win_res_over_bil", "comp_win_res_over_ana", "comp_best_bil_pct",
                      "comp_best_ana_pct", "comp_best_res_pct", "comp_disagree_ana_bil_mm",
                      "comp_disagree_res_bil_mm", "comp_disagree_res_ana_mm"]:
                s_dict[k] = compute_mean_ignore_nan(recs, k)
        summary_by_scan[scan] = s_dict

    # 全局总计 (按需聚合)
    overall_summary = {"count": len(all_sample_records)}
    if do_global:
        for k in ["stage0_mae", "stage0_planar_mae", "stage0_curved_mae", "stage0_thres1mm_err",
                  "stage0_thres2mm_err", "stage0_thres4mm_err", "stage0_thres8mm_err",
                  "stage1_fused_mae", "stage2_mae", "stage3_mae", "s0_planar_ratio_pct"]:
            overall_summary[k] = compute_mean_ignore_nan(all_sample_records, k)
    if do_stage1_gnn:
        for k in ["stage1_prop_mae", "stage1_pixel_mae", "stage1_planar_prop_mae",
                  "stage1_planar_pixel_mae", "s1_planar_ratio_pct"]:
            overall_summary[k] = compute_mean_ignore_nan(all_sample_records, k)
        gnn_gain_overall = (
            (overall_summary["stage1_planar_pixel_mae"] - overall_summary["stage1_planar_prop_mae"])
            / overall_summary["stage1_planar_pixel_mae"] * 100.0
        ) if not math.isnan(overall_summary["stage1_planar_pixel_mae"]) and overall_summary["stage1_planar_pixel_mae"] > 0 else 0.0
    if do_planar_comp:
        for k in ["comp_mae_bil", "comp_mae_ana", "comp_mae_res", "comp_delta_ana_vs_bil_mm",
                  "comp_delta_res_vs_bil_mm", "comp_delta_res_vs_ana_mm", "comp_win_ana_over_bil",
                  "comp_win_res_over_bil", "comp_win_res_over_ana", "comp_best_bil_pct",
                  "comp_best_ana_pct", "comp_best_res_pct", "comp_disagree_ana_bil_mm",
                  "comp_disagree_res_bil_mm", "comp_disagree_res_ana_mm"]:
            overall_summary[k] = compute_mean_ignore_nan(all_sample_records, k)

    # 构造 Markdown 报告
    report_lines = []
    report_lines.append("# WHU MVS 独立评测深度精度与平面分析报告\n")
    report_lines.append(f"- **评测模式**: `{args.eval_mode}`")
    report_lines.append(f"- **Checkpoint**: `{args.loadckpt}`")
    report_lines.append(f"- **Testlist**: `{args.testlist}`")
    report_lines.append(f"- **测试样本数**: {overall_summary['count']} 张\n")

    # Table 1 (按需输出)
    if do_global:
        report_lines.append("## 1. 全图多尺度深度精度与误差分布 (Table 1: Global Depth Accuracy)")
        report_lines.append("| Scan | 样本数 | Stage 0 MAE (m) | Stage 1 融合 MAE (m) | Stage 2 MAE (m) | Stage 3 MAE (m) | >1mm 误差率 (%) | >2mm 误差率 (%) | >4mm 误差率 (%) | >8mm 误差率 (%) |")
        report_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

        for scan, s in summary_by_scan.items():
            report_lines.append(
                f"| `{scan}` | {s['count']} | **{s['stage0_mae']:.4f}** | {s['stage1_fused_mae']:.4f} | "
                f"{s['stage2_mae']:.4f} | {s['stage3_mae']:.4f} | "
                f"{s['stage0_thres1mm_err']*100:.2f}% | {s['stage0_thres2mm_err']*100:.2f}% | "
                f"{s['stage0_thres4mm_err']*100:.2f}% | {s['stage0_thres8mm_err']*100:.2f}% |"
            )

        report_lines.append(
            f"| **Overall (平均)** | **{overall_summary['count']}** | **{overall_summary['stage0_mae']:.4f}** | "
            f"**{overall_summary['stage1_fused_mae']:.4f}** | {overall_summary['stage2_mae']:.4f} | {overall_summary['stage3_mae']:.4f} | "
            f"**{overall_summary['stage0_thres1mm_err']*100:.2f}%** | **{overall_summary['stage0_thres2mm_err']*100:.2f}%** | "
            f"**{overall_summary['stage0_thres4mm_err']*100:.2f}%** | **{overall_summary['stage0_thres8mm_err']*100:.2f}%** |\n"
        )

    # Table 2 (按需输出)
    if do_stage1_gnn:
        report_lines.append("## 2. 平面区域 vs 曲面区域精度分析与 GNN 增益 (Table 2: Regional & GNN Performance)")
        report_lines.append("| Scan | S0 平面 MAE (m) | S0 曲面 MAE (m) | S0 平面占比 (%) | S1 平面传播 MAE (m) | S1 纯像素 MAE (m) | GNN 平面提升 (%) | S1 平面占比 (%) |")
        report_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

        for scan, s in summary_by_scan.items():
            gain = ((s['stage1_planar_pixel_mae'] - s['stage1_planar_prop_mae']) / s['stage1_planar_pixel_mae'] * 100.0) \
                if (not math.isnan(s['stage1_planar_pixel_mae']) and s['stage1_planar_pixel_mae'] > 0) else 0.0
            report_lines.append(
                f"| `{scan}` | **{s['stage0_planar_mae']:.4f}** | {s['stage0_curved_mae']:.4f} | "
                f"{s['s0_planar_ratio_pct']:.1f}% | **{s['stage1_planar_prop_mae']:.4f}** | "
                f"{s['stage1_planar_pixel_mae']:.4f} | **+{gain:.2f}%** | {s['s1_planar_ratio_pct']:.1f}% |"
            )

        report_lines.append(
            f"| **Overall (平均)** | **{overall_summary['stage0_planar_mae']:.4f}** | "
            f"{overall_summary['stage0_curved_mae']:.4f} | **{overall_summary['s0_planar_ratio_pct']:.1f}%** | "
            f"**{overall_summary['stage1_planar_prop_mae']:.4f}** | {overall_summary['stage1_planar_pixel_mae']:.4f} | "
            f"**+{gnn_gain_overall:.2f}%** | **{overall_summary['s1_planar_ratio_pct']:.1f}%** |\n"
        )

    # Table 3 (按需输出)
    if do_planar_comp:
        report_lines.append("## 3. Stage 0 平面区域：双线性插值 vs 静态无损几何 vs 残差细化 三方终极对照 (Table 3: Three-Way Planar Comparison)")
        report_lines.append("> **物理机理说明**：")
        report_lines.append("> - **双线性插值基底 (Bilinear)**: 将 Stage 1 深度直接双线性插值放大 2 倍至 Stage 0 分辨率（纯粹插值上采样）。")
        report_lines.append("> - **静态无损几何 (Analytic Ray-Plane)**: 基于 Stage 1 GNN 优化后的三维平面参数 $(n, d)$ 与 Stage 0 视线射线解析求交 $Z = -d / (\\mathbf{n} \\cdot \\mathbf{r})$。")
        report_lines.append("> - **残差细化深度 (Residual Refined)**: 双线性放大后输入 CNN 残差网络预测全图精细深度。")
        report_lines.append("> - **Δ(Ana - Bil) (mm)**: 无损几何相比双线性插值的误差差值。负数说明无损几何优于插值，正数说明插值更优。")
        report_lines.append("> - **Δ(Res - Bil) (mm)**: 残差细化相比双线性插值的误差差值。负数说明 CNN 残差起到了正向微调，正数说明 CNN 残差引入了噪声。")
        report_lines.append("> - **最优胜率 (Best %)**: 三者中该方法绝对误差最小的像素占比 (插值 / 无损 / 残差)。\n")
        report_lines.append("| Scan | 双线性插值 MAE (m) | 静态无损几何 MAE (m) | 残差细化 MAE (m) | Δ(Ana-Bil) (mm) | Δ(Res-Bil) (mm) | Δ(Res-Ana) (mm) | 最优胜率 (插值 / 无损 / 残差 %) | 两两分歧 (无损-插值 / 残差-无损 mm) |")
        report_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

        for scan, s in summary_by_scan.items():
            d_ana_bil = f"{s['comp_delta_ana_vs_bil_mm']:+.2f}" if not math.isnan(s['comp_delta_ana_vs_bil_mm']) else "N/A"
            d_res_bil = f"{s['comp_delta_res_vs_bil_mm']:+.2f}" if not math.isnan(s['comp_delta_res_vs_bil_mm']) else "N/A"
            d_res_ana = f"{s['comp_delta_res_vs_ana_mm']:+.2f}" if not math.isnan(s['comp_delta_res_vs_ana_mm']) else "N/A"
            best_str = f"{s['comp_best_bil_pct']:.1f}% / {s['comp_best_ana_pct']:.1f}% / {s['comp_best_res_pct']:.1f}%"
            dis_str = f"{s['comp_disagree_ana_bil_mm']:.1f} / {s['comp_disagree_res_ana_mm']:.1f}"
            report_lines.append(
                f"| `{scan}` | {s['comp_mae_bil']:.4f} | {s['comp_mae_ana']:.4f} | {s['comp_mae_res']:.4f} | "
                f"**{d_ana_bil}** | **{d_res_bil}** | {d_res_ana} | {best_str} | {dis_str} |"
            )

        o_d_ana_bil = f"{overall_summary['comp_delta_ana_vs_bil_mm']:+.2f}" if not math.isnan(overall_summary['comp_delta_ana_vs_bil_mm']) else "N/A"
        o_d_res_bil = f"{overall_summary['comp_delta_res_vs_bil_mm']:+.2f}" if not math.isnan(overall_summary['comp_delta_res_vs_bil_mm']) else "N/A"
        o_d_res_ana = f"{overall_summary['comp_delta_res_vs_ana_mm']:+.2f}" if not math.isnan(overall_summary['comp_delta_res_vs_ana_mm']) else "N/A"
        o_best_str = f"{overall_summary['comp_best_bil_pct']:.1f}% / {overall_summary['comp_best_ana_pct']:.1f}% / {overall_summary['comp_best_res_pct']:.1f}%"
        o_dis_str = f"{overall_summary['comp_disagree_ana_bil_mm']:.1f} / {overall_summary['comp_disagree_res_ana_mm']:.1f}"
        report_lines.append(
            f"| **Overall (平均)** | **{overall_summary['comp_mae_bil']:.4f}** | **{overall_summary['comp_mae_ana']:.4f}** | **{overall_summary['comp_mae_res']:.4f}** | "
            f"**{o_d_ana_bil}** | **{o_d_res_bil}** | **{o_d_res_ana}** | **{o_best_str}** | **{o_dis_str}** |\n"
        )

    # 附加信息
    if args.save_masks:
        report_lines.append("## 4. 平面掩码分布说明 (Mask Distribution Summary)")
        report_lines.append(f"- **说明**: 所有掩码图像已按场景保存至 `{args.outdir}/<scan>/masks/` 目录下。")

    full_report = "\n".join(report_lines)
    print("\n" + full_report + "\n")

    # 保存报告至文件
    summary_txt_path = os.path.join(args.outdir, "test_summary.md")
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write(full_report)
    print(f"✓ 评测报告已保存至: {summary_txt_path}")

    # 保存机器可读的 JSON 指标
    json_path = os.path.join(args.outdir, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "overall": overall_summary,
            "by_scan": summary_by_scan,
            "samples": all_sample_records
        }, f, indent=2)
    print(f"✓ 结构化评测指标已保存至: {json_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
