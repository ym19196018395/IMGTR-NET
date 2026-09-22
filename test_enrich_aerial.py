import argparse
import os
import sys
import time
import json
import math
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 确保项目根目录强制位于 sys.path[0]
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
while PROJECT_ROOT in sys.path:
    sys.path.remove(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

# 控制 GPU ID：优先使用外部环境变量 GPU_ID，其次 CUDA_VISIBLE_DEVICES，默认 '0'
_gpu_choice = os.environ.get('GPU_ID', os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
os.environ['CUDA_VISIBLE_DEVICES'] = str(_gpu_choice)

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from datasets import find_dataset_def
from datasets.enrich_aerial import collate_keep_list
from datasets.data_io import save_pfm
from models.net import PatchmatchNet
from utils import print_args, tocuda, make_nograd_func

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True


def compute_surface_normals_3d(depth, intrinsics=None, depth_jump_threshold=1.0):
    """
    基于 3D 点云切向量差分的高精度表面法向量估计 (Discontinuity-Aware One-Sided Difference)
    """
    H, W = depth.shape
    valid_mask = depth > 1e-4

    if intrinsics is not None:
        fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
        cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    else:
        fx, fy = max(H, W) * 1.5, max(H, W) * 1.5
        cx, cy = W / 2.0, H / 2.0

    u_grid, v_grid = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    X = (u_grid - cx) * depth / fx
    Y = (v_grid - cy) * depth / fy
    pts = np.stack([X, Y, depth], axis=-1)

    pts_left = np.zeros_like(pts)
    pts_right = np.zeros_like(pts)
    pts_left[:, 1:, :] = pts[:, 1:, :] - pts[:, :-1, :]
    pts_left[:, 0, :] = pts_left[:, 1, :]
    pts_right[:, :-1, :] = pts[:, 1:, :] - pts[:, :-1, :]
    pts_right[:, -1, :] = pts_right[:, -2, :]

    dz_left = np.abs(pts_left[:, :, 2])
    dz_right = np.abs(pts_right[:, :, 2])
    use_left = (dz_left < dz_right) | (dz_right > depth_jump_threshold)
    tangent_x = np.where(use_left[:, :, np.newaxis], pts_left, pts_right)

    pts_up = np.zeros_like(pts)
    pts_down = np.zeros_like(pts)
    pts_up[1:, :, :] = pts[1:, :, :] - pts[:-1, :, :]
    pts_up[0, :, :] = pts_up[1, :, :]
    pts_down[:-1, :, :] = pts[1:, :, :] - pts[:-1, :, :]
    pts_down[-1, :, :] = pts_down[-2, :, :]

    dz_up = np.abs(pts_up[:, :, 2])
    dz_down = np.abs(pts_down[:, :, 2])
    use_up = (dz_up < dz_down) | (dz_down > depth_jump_threshold)
    tangent_y = np.where(use_up[:, :, np.newaxis], pts_up, pts_down)

    raw_normal = np.cross(tangent_x, tangent_y)
    norm_mag = np.linalg.norm(raw_normal, axis=-1, keepdims=True)
    normal_unit = -raw_normal / np.maximum(norm_mag, 1e-8)

    flip_mask = normal_unit[:, :, 2] < 0
    normal_unit[flip_mask] = -normal_unit[flip_mask]
    normal_unit[~valid_mask] = 0.0

    rgb = ((normal_unit + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    rgb[~valid_mask] = 0
    return rgb


def render_mae_heatmap(depth_pred, depth_gt, mask=None, vmax=1.0, title="Depth Absolute Error (MAE)", out_path=None):
    """
    渲染高清 MAE 误差热力图并保存
    """
    H, W = depth_pred.shape
    if depth_gt.shape != (H, W):
        depth_gt = cv2.resize(depth_gt, (W, H), interpolation=cv2.INTER_NEAREST)

    if mask is None:
        eval_mask = (depth_pred > 0) & (depth_gt > 0)
    else:
        eval_mask = mask & (depth_pred > 0) & (depth_gt > 0)

    abs_diff = np.abs(depth_pred - depth_gt)
    valid_err = abs_diff[eval_mask]
    if len(valid_err) == 0:
        return

    mae_val = float(np.mean(valid_err))
    med_ae = float(np.median(valid_err))
    t10 = float(np.mean(valid_err < 0.10) * 100.0)
    t20 = float(np.mean(valid_err < 0.20) * 100.0)
    t50 = float(np.mean(valid_err < 0.50) * 100.0)

    fig, ax = plt.subplots(figsize=(11, 7), dpi=140)
    masked_diff = np.ma.masked_where(~eval_mask, abs_diff)

    try:
        cmap = plt.get_cmap('turbo').copy()
    except Exception:
        cmap = plt.get_cmap('jet').copy()
    cmap.set_bad(color='#111111')

    im = ax.imshow(masked_diff, cmap=cmap, vmin=0.0, vmax=vmax, interpolation='nearest')
    ax.axis('off')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.15)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Absolute Error (Meters)', fontsize=12, fontweight='bold')
    ticks = [0.0, 0.10, 0.20, 0.50, vmax]
    ticks = sorted(list(set([t for t in ticks if t <= vmax])))
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.2f}m" for t in ticks])

    ax.set_title(f"{title} | MAE: {mae_val:.4f}m | Median: {med_ae:.4f}m", fontsize=14, fontweight='bold')

    stats_text = (
        f"📊 Error Stats:\n"
        f" • MAE:    {mae_val:.4f}m\n"
        f" • Median: {med_ae:.4f}m\n"
        f" • <10cm:  {t10:5.1f}%\n"
        f" • <20cm:  {t20:5.1f}%\n"
        f" • <50cm:  {t50:5.1f}%"
    )
    ax.text(
        0.02, 0.03, stats_text, transform=ax.transAxes, fontsize=10.5, fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.75, edgecolor='cyan', lw=1.2),
        color='white'
    )
    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        plt.savefig(out_path, dpi=180, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def render_signed_bias_map(depth_pred, depth_gt, mask=None, vrange=1.0, title="Signed Bias (Pred - GT)", out_path=None):
    """
    渲染带符号高程偏差图 (-1m 浅蓝 ~ 0m 白 ~ +1m 深红)
    """
    H, W = depth_pred.shape
    if depth_gt.shape != (H, W):
        depth_gt = cv2.resize(depth_gt, (W, H), interpolation=cv2.INTER_NEAREST)

    if mask is None:
        eval_mask = (depth_pred > 0) & (depth_gt > 0)
    else:
        eval_mask = mask & (depth_pred > 0) & (depth_gt > 0)

    signed_diff = depth_pred - depth_gt
    valid_diff = signed_diff[eval_mask]
    if len(valid_diff) == 0:
        return

    mean_bias = float(np.mean(valid_diff))
    pos_ratio = float(np.mean(valid_diff > 0.0) * 100.0)

    fig, ax = plt.subplots(figsize=(11, 7), dpi=140)
    masked_diff = np.ma.masked_where(~eval_mask, signed_diff)
    cmap = plt.get_cmap('coolwarm').copy()
    cmap.set_bad(color='#222222')

    im = ax.imshow(masked_diff, cmap=cmap, vmin=-vrange, vmax=vrange, interpolation='nearest')
    ax.axis('off')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.15)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Signed Error (Pred - GT, Meters)', fontsize=12, fontweight='bold')
    cbar.set_ticks([-vrange, -vrange/2, 0.0, vrange/2, vrange])
    cbar.set_ticklabels([f"-{vrange:.2f}m", f"-{vrange/2:.2f}m", "0.0m", f"+{vrange/2:.2f}m", f"+{vrange:.2f}m"])

    ax.set_title(f"{title} | Mean Bias: {mean_bias:+.4f}m", fontsize=14, fontweight='bold')

    diag_text = f"🎯 Bias:\n • Mean: {mean_bias:+.4f}m\n • Deeper (+): {pos_ratio:.1f}%\n • Shallower (-): {100-pos_ratio:.1f}%"
    ax.text(
        0.02, 0.03, diag_text, transform=ax.transAxes, fontsize=10.5, fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.75, edgecolor='orange', lw=1.2),
        color='white'
    )
    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        plt.savefig(out_path, dpi=180, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="ENRICH-Aerial_Data 航空遥感数据集独立评测系统")

    parser.add_argument('--dataset', default='enrich_aerial', type=str, help='数据集定义名称')
    parser.add_argument('--testpath', default=r"E:\Code\PythonCode\Pytorch\ENRICH-Aerial_Data", type=str, help='ENRICH-Aerial_Data 数据集根目录')
    parser.add_argument('--testlist', default=r"E:\Code\PythonCode\Pytorch\ENRICH-Aerial_Data\scan_list.txt", type=str, help='测试场景列表文件')
    parser.add_argument('--loadckpt', default="./checkpoints/params.ckpt", type=str, help='待评测模型 Checkpoint 权重路径')
    parser.add_argument('--outdir', default="./outputs_enrich_aerial", type=str, help='评测指标与预测深度图保存目录')

    parser.add_argument('--batch_size', type=int, default=1, help='测试批次大小 (默认 1)')
    parser.add_argument('--n_views', type=int, default=5, help='测试视角数 (1 参 4 源，推荐 5 视角对齐 WHU)')
    parser.add_argument('--num_workers', type=int, default=2, help='DataLoader 线程数')
    parser.add_argument('--seed', type=int, default=123, help='随机种子')

    parser.add_argument('--eval_mode', type=str, default='all', choices=['all', 'global'],
                        help='评测模式: all(全图多尺度 + 细分平面/GNN增益), global(仅全图多尺度)')

    parser.add_argument('--save_depth', action='store_true', default=True, help='保存预测深度图 (PFM 与 PNG)')
    parser.add_argument('--no_depth', dest='save_depth', action='store_false', help='不保存预测深度图')

    parser.add_argument('--save_masks', action='store_true', default=True, help='保存同源平面掩码 (供基线模型对齐评测)')
    parser.add_argument('--no_masks', dest='save_masks', action='store_false', help='不保存同源平面掩码')

    parser.add_argument('--save_vis', action='store_true', default=True, help='保存高清 MAE 热力图、Signed 误差图与法向量图')
    parser.add_argument('--no_vis', dest='save_vis', action='store_false', help='不保存可视化图')
    parser.add_argument('--vmax', type=float, default=1.0, help='MAE 热力图误差上限 (米，默认 1.0m)')

    return parser.parse_args()


def load_checkpoint(model, ckpt_path):
    """自适应加载模型权重并兼容首层通道扩展"""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"🚨 Checkpoint 文件不存在: {ckpt_path}")

    print(f"[Model] 载入 Checkpoint 权重: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))

    cleaned_state = {}
    for k, v in state_dict.items():
        clean_k = k[7:] if k.startswith('module.') else k
        cleaned_state[clean_k] = v

    # 自适应处理第 0 层通道权重 (若原权重通道为 3 但当前模型为 4)
    for key in ['feature.conv0_0.0.weight', 'feature.conv0.0.weight']:
        if key in cleaned_state and hasattr(model, 'feature'):
            ckpt_w = cleaned_state[key]
            try:
                tgt_w = dict(model.named_parameters())[key]
                if ckpt_w.shape[1] == 3 and tgt_w.shape[1] == 4:
                    adapted_w = torch.zeros_like(tgt_w)
                    adapted_w[:, :3, :, :] = ckpt_w
                    cleaned_state[key] = adapted_w
            except KeyError:
                pass

    missing, unexpected = model.load_state_dict(cleaned_state, strict=False)
    if len(missing) > 0:
        print(f"[Model Warning] 缺失参数 (前5个): {missing[:5]}")
    if len(unexpected) > 0:
        print(f"[Model Warning] 冗余参数 (前5个): {unexpected[:5]}")
    print("✓ 模型权重载入成功！")


def safe_mae(depth_est, depth_gt, mask):
    """安全计算掩码区域内的绝对深度误差 MAE (m)"""
    if mask is None or not mask.any():
        return float('nan')
    est_valid = depth_est[mask]
    gt_valid = depth_gt[mask]
    return torch.mean((est_valid - gt_valid).abs()).item()


def safe_thres_error(depth_est, depth_gt, mask, thres):
    """安全计算误差大于特定阈值 (m) 的离群点比例"""
    if mask is None or not mask.any():
        return float('nan')
    est_valid = depth_est[mask]
    gt_valid = depth_gt[mask]
    err = (est_valid - gt_valid).abs()
    return (err > thres).float().mean().item()


@make_nograd_func
def evaluate_single_sample(model, sample, device):
    """执行单样本前向推断与几何张量解包"""
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

    skip = ["vertexs", "lines", "triangles", "scan", "file_id"]
    sample_cuda = tocuda(sample, device=device, skip_keys=skip)

    depth_gt = sample_cuda["depth"]
    mask = sample_cuda["mask"]

    # 纯推理模式：无训练约束梯度，温度参数设为 0.55
    outputs = model(
        sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["intrinsics_mats"],
        sample_cuda["depth_min"], sample_cuda["depth_max"],
        vertexs_batch, lines_batch, triangles_batch, depth_gt['stage_1'],
        lambda_c=0.0, lambda_s=0.0, current_temp=0.55, compute_edge_pixels=False
    )

    return outputs, depth_gt, mask


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print("=" * 85)
    print("ENRICH-Aerial_Data 航空遥感基准评测系统: 1 参 2 源 (严格 3 视角) 纯推理模式")
    print("=" * 85)
    print_args(args)

    os.makedirs(args.outdir, exist_ok=True)

    # 1. 构建 Dataset 与 DataLoader
    MVSDataset = find_dataset_def(args.dataset)
    test_dataset = MVSDataset(args.testpath, args.testlist, mode="test", nviews=args.n_views)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_keep_list, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=(args.num_workers > 0), drop_last=False
    )
    print(f"[Dataset] 测试集样本总量: {len(test_dataset)} (共 {len(test_loader)} 个批次)\n")

    # 2. 实例化模型并载入权重
    model = PatchmatchNet(
        patchmatch_interval_scale=[0.005, 0.0125, 0.025],
        propagation_range=[6, 4, 2],
        patchmatch_iteration=[1, 2, 2],
        patchmatch_num_sample=[8, 8, 16],
        propagate_neighbors=[0, 8, 16],
        evaluate_neighbors=[9, 9, 9]
    ).to(device)

    load_checkpoint(model, args.loadckpt)
    model.eval()

    do_global = args.eval_mode in ['all', 'global']
    do_stage1_gnn = args.eval_mode in ['all']

    per_scan_records = {}
    all_sample_records = []
    start_eval_time = time.time()

    print("\n" + "=" * 85)
    print("开始执行 ENRICH-Aerial_Data 逐场景推断与 Table 1 / Table 2 精度评测...")
    print("=" * 85)

    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            step_start = time.time()
            B = sample['imgs']['stage_0'].shape[0]

            outputs, depth_gt, mask = evaluate_single_sample(model, sample, device)

            depth_est_s0 = outputs["refined_depth"]['stage_0']  # [B, 1, H0, W0]
            depth_pm = outputs["depth_patchmatch"]              # dict of stages
            output_plane = outputs.get("output_plane", {})

            # 掩码准备 (Stage 0 与 Stage 1)
            mask_gt_s0 = (mask['stage_0'] > 0.5)
            is_planar_s0 = output_plane.get("is_planar_s0", torch.zeros_like(mask_gt_s0))
            eval_mask_s0_planar = mask_gt_s0 & is_planar_s0
            eval_mask_s0_curved = mask_gt_s0 & (~is_planar_s0)

            mask_gt_s1 = (mask['stage_1'] > 0.5)
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

            for b in range(B):
                scan = sample['scan'][b]
                file_id = sample['file_id'][b]

                if scan not in per_scan_records:
                    per_scan_records[scan] = []

                record = {"scan": scan, "file_id": file_id}

                # --- Table 1: 全图多尺度全局深度精度 ---
                if do_global:
                    s0_mae = safe_mae(depth_est_s0[b:b+1], depth_gt['stage_0'][b:b+1], mask_gt_s0[b:b+1])
                    s1_fused_mae = safe_mae(depth_s1_fused[b:b+1], depth_gt['stage_1'][b:b+1], mask_gt_s1[b:b+1])
                    s2_mae = safe_mae(depth_pm['stage_2'][-1][b:b+1], depth_gt['stage_2'][b:b+1], mask['stage_2'][b:b+1] > 0.5)
                    s3_mae = safe_mae(depth_pm['stage_3'][-1][b:b+1], depth_gt['stage_3'][b:b+1], mask['stage_3'][b:b+1] > 0.5)

                    # 航测场景多尺度误差阈值 (0.05m, 0.10m, 0.20m, 0.50m)
                    t05_err = safe_thres_error(depth_est_s0[b:b+1], depth_gt['stage_0'][b:b+1], mask_gt_s0[b:b+1], 0.05)
                    t10_err = safe_thres_error(depth_est_s0[b:b+1], depth_gt['stage_0'][b:b+1], mask_gt_s0[b:b+1], 0.10)
                    t20_err = safe_thres_error(depth_est_s0[b:b+1], depth_gt['stage_0'][b:b+1], mask_gt_s0[b:b+1], 0.20)
                    t50_err = safe_thres_error(depth_est_s0[b:b+1], depth_gt['stage_0'][b:b+1], mask_gt_s0[b:b+1], 0.50)

                    s0_planar_mae = safe_mae(depth_est_s0[b:b+1], depth_gt['stage_0'][b:b+1], eval_mask_s0_planar[b:b+1])
                    s0_curved_mae = safe_mae(depth_est_s0[b:b+1], depth_gt['stage_0'][b:b+1], eval_mask_s0_curved[b:b+1])

                    valid_cnt_s0 = mask_gt_s0[b:b+1].sum().item()
                    planar_cnt_s0 = eval_mask_s0_planar[b:b+1].sum().item()
                    planar_ratio_s0 = (planar_cnt_s0 / valid_cnt_s0 * 100.0) if valid_cnt_s0 > 0 else 0.0

                    record.update({
                        "stage0_mae": s0_mae,
                        "stage1_fused_mae": s1_fused_mae,
                        "stage2_mae": s2_mae,
                        "stage3_mae": s3_mae,
                        "stage0_thres05cm_err": t05_err,
                        "stage0_thres10cm_err": t10_err,
                        "stage0_thres20cm_err": t20_err,
                        "stage0_thres50cm_err": t50_err,
                        "stage0_planar_mae": s0_planar_mae,
                        "stage0_curved_mae": s0_curved_mae,
                        "s0_valid_pixels": valid_cnt_s0,
                        "s0_planar_pixels": planar_cnt_s0,
                        "s0_planar_ratio_pct": planar_ratio_s0,
                    })

                # --- Table 2: 细分区域与 Stage 1 GNN 增益 ---
                if do_stage1_gnn:
                    s1_prop_mae = safe_mae(depth_pm['stage_1'][-1][b:b+1], depth_gt['stage_1'][b:b+1], mask_gt_s1[b:b+1])
                    s1_pixel_mae = safe_mae(depth_pm['stage_1'][0][b:b+1], depth_gt['stage_1'][b:b+1], mask_gt_s1[b:b+1])
                    s1_planar_prop_mae = safe_mae(depth_pm['stage_1'][-1][b:b+1], depth_gt['stage_1'][b:b+1], eval_mask_s1_planar[b:b+1])
                    s1_planar_pixel_mae = safe_mae(depth_pm['stage_1'][0][b:b+1], depth_gt['stage_1'][b:b+1], eval_mask_s1_planar[b:b+1])

                    valid_cnt_s1 = mask_gt_s1[b:b+1].sum().item()
                    planar_cnt_s1 = eval_mask_s1_planar[b:b+1].sum().item()
                    planar_ratio_s1 = (planar_cnt_s1 / valid_cnt_s1 * 100.0) if valid_cnt_s1 > 0 else 0.0

                    record.update({
                        "stage1_prop_mae": s1_prop_mae,
                        "stage1_pixel_mae": s1_pixel_mae,
                        "stage1_planar_prop_mae": s1_planar_prop_mae,
                        "stage1_planar_pixel_mae": s1_planar_pixel_mae,
                        "s1_valid_pixels": valid_cnt_s1,
                        "s1_planar_pixels": planar_cnt_s1,
                        "s1_planar_ratio_pct": planar_ratio_s1,
                    })

                per_scan_records[scan].append(record)
                all_sample_records.append(record)

                # 保存预测深度图
                if args.save_depth:
                    depth_save_dir = os.path.join(args.outdir, scan, "depths")
                    os.makedirs(depth_save_dir, exist_ok=True)
                    d_s0_np = depth_est_s0[b, 0].cpu().numpy()
                    save_pfm(os.path.join(depth_save_dir, f"depth_{file_id}.pfm"), d_s0_np)
                    depth_png_s0 = np.clip(d_s0_np * 64.0, 0, 65535).astype(np.uint16)
                    cv2.imwrite(os.path.join(depth_save_dir, f"depth_{file_id}.png"), depth_png_s0)

                    # 同时保存 Stage 1 混合融合深度 (depth_s1_fused)
                    d_s1_fused_np = depth_s1_fused[b, 0].cpu().numpy()
                    save_pfm(os.path.join(depth_save_dir, f"depth_s1_fused_{file_id}.pfm"), d_s1_fused_np)
                    depth_png_s1 = np.clip(d_s1_fused_np * 64.0, 0, 65535).astype(np.uint16)
                    cv2.imwrite(os.path.join(depth_save_dir, f"depth_s1_fused_{file_id}.png"), depth_png_s1)

                # 保存同源平面掩码 (供 CasMVSNet / DiffMVS 直接读取对比)
                if args.save_masks:
                    mask_save_dir = os.path.join(args.outdir, scan, "masks")
                    os.makedirs(mask_save_dir, exist_ok=True)
                    s0_mask_np = (eval_mask_s0_planar[b, 0].cpu().numpy().astype(np.uint8)) * 255
                    cv2.imwrite(os.path.join(mask_save_dir, f"stage0_planar_mask_{file_id}.png"), s0_mask_np)
                    s1_mask_np = (eval_mask_s1_planar[b, 0].cpu().numpy().astype(np.uint8)) * 255
                    cv2.imwrite(os.path.join(mask_save_dir, f"stage1_planar_mask_{file_id}.png"), s1_mask_np)

                # 💡 高清可视化保存：MAE 绝对误差热力图与 Stage 1 Fuse 表面法向量图
                if args.save_vis:
                    vis_save_dir = os.path.join(args.outdir, scan, "visualizations")
                    os.makedirs(vis_save_dir, exist_ok=True)

                    d_s0_np = depth_est_s0[b, 0].cpu().numpy()
                    d_s1_fused_np = depth_s1_fused[b, 0].cpu().numpy()
                    gt_s0_np = depth_gt['stage_0'][b, 0].cpu().numpy()
                    gt_s1_np = depth_gt['stage_1'][b, 0].cpu().numpy()
                    mask_s0_np = (mask['stage_0'][b, 0].cpu().numpy() > 0.5)
                    mask_s1_np = (mask['stage_1'][b, 0].cpu().numpy() > 0.5)

                    K_s0 = sample['intrinsics_mats']['stage_0'][b, 0].cpu().numpy() if 'intrinsics_mats' in sample else None
                    K_s1 = sample['intrinsics_mats']['stage_1'][b, 0].cpu().numpy() if 'intrinsics_mats' in sample else None

                    # 1. 深度图 MAE 误差热力图 (Stage 0 全分辨率)
                    render_mae_heatmap(
                        d_s0_np, gt_s0_np, mask=mask_s0_np, vmax=args.vmax,
                        title=f"Stage 0 Depth MAE ({scan}_{file_id})",
                        out_path=os.path.join(vis_save_dir, f"stage0_mae_map_{file_id}.png")
                    )

                    # 2. 深度图 Signed Error 偏差图 (Stage 0)
                    render_signed_bias_map(
                        d_s0_np, gt_s0_np, mask=mask_s0_np, vrange=args.vmax,
                        title=f"Stage 0 Signed Bias ({scan}_{file_id})",
                        out_path=os.path.join(vis_save_dir, f"stage0_signed_bias_{file_id}.png")
                    )

                    # 3. Stage 1 Fused 深度 MAE 热力图
                    render_mae_heatmap(
                        d_s1_fused_np, gt_s1_np, mask=mask_s1_np, vmax=args.vmax,
                        title=f"Stage 1 Fused MAE ({scan}_{file_id})",
                        out_path=os.path.join(vis_save_dir, f"stage1_fused_mae_map_{file_id}.png")
                    )

                    # 4. 表面法向量图 (Normal Map): Stage 1 Fused (基于内参 3D 反投影几何法向量)
                    normal_s1_fused_rgb = compute_surface_normals_3d(d_s1_fused_np, intrinsics=K_s1)
                    cv2.imwrite(os.path.join(vis_save_dir, f"normal_s1_fused_{file_id}.png"), cv2.cvtColor(normal_s1_fused_rgb, cv2.COLOR_RGB2BGR))

                    # 5. 表面法向量图: Stage 1 GT (真值法向对比)
                    normal_s1_gt_rgb = compute_surface_normals_3d(gt_s1_np, intrinsics=K_s1)
                    cv2.imwrite(os.path.join(vis_save_dir, f"normal_s1_gt_{file_id}.png"), cv2.cvtColor(normal_s1_gt_rgb, cv2.COLOR_RGB2BGR))

                    # 6. Stage 1 网络预测的平面法向量 (若存在 normal_pro)
                    if "normal_pro" in output_plane and len(output_plane["normal_pro"]) > 0:
                        try:
                            norm_pro_b = output_plane["normal_pro"][b]
                            if isinstance(norm_pro_b, torch.Tensor):
                                norm_pro_np = norm_pro_b.detach().cpu().numpy()
                            else:
                                norm_pro_np = np.array(norm_pro_b)
                            if norm_pro_np.ndim == 3 and norm_pro_np.shape[0] == 3:
                                norm_pro_np = np.transpose(norm_pro_np, (1, 2, 0))
                            norm_pro_vis = ((np.clip(norm_pro_np, -1.0, 1.0) + 1.0) / 2.0 * 255.0).astype(np.uint8)
                            cv2.imwrite(os.path.join(vis_save_dir, f"normal_s1_pro_{file_id}.png"), cv2.cvtColor(norm_pro_vis, cv2.COLOR_RGB2BGR))
                        except Exception:
                            pass

            cur_idx = batch_idx + 1
            step_time = time.time() - step_start
            p0_str = f"{record.get('stage0_planar_mae', float('nan')):.4f}m" if not math.isnan(record.get('stage0_planar_mae', float('nan'))) else "N/A"
            s0_mae_str = f"{record.get('stage0_mae', float('nan')):.4f}m" if not math.isnan(record.get('stage0_mae', float('nan'))) else "N/A"
            print(f"[{cur_idx:03d}/{len(test_loader):03d}] Scan: {scan} | Img: {file_id} | "
                  f"S0 MAE: {s0_mae_str} | S0 Planar: {p0_str} | Time: {step_time:.2f}s")

    eval_duration = time.time() - start_eval_time
    print("\n" + "=" * 85)
    print(f"评测完成！共评估 {len(all_sample_records)} 个场景，总耗时: {eval_duration:.2f} 秒。")
    print("=" * 85)

    # 4. 汇总生成 Markdown 双表
    def compute_mean_ignore_nan(records, key):
        vals = [r[key] for r in records if key in r and not math.isnan(r[key])]
        return float(np.mean(vals)) if len(vals) > 0 else float('nan')

    summary_by_scan = {}
    for scan, recs in per_scan_records.items():
        s_dict = {"count": len(recs)}
        if do_global:
            for k in ["stage0_mae", "stage0_planar_mae", "stage0_curved_mae",
                      "stage0_thres05cm_err", "stage0_thres10cm_err", "stage0_thres20cm_err", "stage0_thres50cm_err",
                      "stage1_fused_mae", "stage2_mae", "stage3_mae", "s0_planar_ratio_pct"]:
                s_dict[k] = compute_mean_ignore_nan(recs, k)
        if do_stage1_gnn:
            for k in ["stage1_prop_mae", "stage1_pixel_mae", "stage1_planar_prop_mae", "stage1_planar_pixel_mae", "s1_planar_ratio_pct"]:
                s_dict[k] = compute_mean_ignore_nan(recs, k)
        summary_by_scan[scan] = s_dict

    overall_summary = {"count": len(all_sample_records)}
    if do_global:
        for k in ["stage0_mae", "stage0_planar_mae", "stage0_curved_mae",
                  "stage0_thres05cm_err", "stage0_thres10cm_err", "stage0_thres20cm_err", "stage0_thres50cm_err",
                  "stage1_fused_mae", "stage2_mae", "stage3_mae", "s0_planar_ratio_pct"]:
            overall_summary[k] = compute_mean_ignore_nan(all_sample_records, k)
    if do_stage1_gnn:
        for k in ["stage1_prop_mae", "stage1_pixel_mae", "stage1_planar_prop_mae", "stage1_planar_pixel_mae", "s1_planar_ratio_pct"]:
            overall_summary[k] = compute_mean_ignore_nan(all_sample_records, k)
        gnn_gain_overall = (
            (overall_summary["stage1_planar_pixel_mae"] - overall_summary["stage1_planar_prop_mae"])
            / overall_summary["stage1_planar_pixel_mae"] * 100.0
        ) if not math.isnan(overall_summary["stage1_planar_pixel_mae"]) and overall_summary["stage1_planar_pixel_mae"] > 0 else 0.0

    report_lines = []
    report_lines.append("# ENRICH-Aerial_Data 航空遥感大场景深度精度与平面特性评测报告\n")
    report_lines.append(f"- **模型权重**: `{args.loadckpt}`")
    report_lines.append(f"- **数据集路径**: `{args.testpath}`")
    report_lines.append(f"- **测试场景数**: {overall_summary['count']} 个")
    report_lines.append(f"- **分辨率**: `2112 × 1408` | **视角配置**: `1 参 2 源 (3 视角)`\n")

    if do_global:
        report_lines.append("## Table 1: 全图多尺度全局深度精度与误差分布 (Global Depth Accuracy)")
        report_lines.append("| Scan | 样本数 | S0 MAE (m) | S1 融合 MAE (m) | S2 MAE (m) | S3 MAE (m) | >5cm 误差率 (%) | >10cm 误差率 (%) | >20cm 误差率 (%) | >50cm 误差率 (%) |")
        report_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for scan, s in summary_by_scan.items():
            report_lines.append(
                f"| `{scan}` | {s['count']} | **{s['stage0_mae']:.4f}** | {s['stage1_fused_mae']:.4f} | "
                f"{s['stage2_mae']:.4f} | {s['stage3_mae']:.4f} | "
                f"{s['stage0_thres05cm_err']*100:.2f}% | {s['stage0_thres10cm_err']*100:.2f}% | "
                f"{s['stage0_thres20cm_err']*100:.2f}% | {s['stage0_thres50cm_err']*100:.2f}% |"
            )
        report_lines.append(
            f"| **Overall (平均)** | **{overall_summary['count']}** | **{overall_summary['stage0_mae']:.4f}** | "
            f"**{overall_summary['stage1_fused_mae']:.4f}** | {overall_summary['stage2_mae']:.4f} | {overall_summary['stage3_mae']:.4f} | "
            f"**{overall_summary['stage0_thres05cm_err']*100:.2f}%** | **{overall_summary['stage0_thres10cm_err']*100:.2f}%** | "
            f"**{overall_summary['stage0_thres20cm_err']*100:.2f}%** | **{overall_summary['stage0_thres50cm_err']*100:.2f}%** |\n"
        )

    if do_stage1_gnn:
        report_lines.append("## Table 2: 细分区域精度与平面特性分析 (Regional & Planar Analysis)")
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

    full_report = "\n".join(report_lines)
    print("\n" + full_report + "\n")

    summary_md_path = os.path.join(args.outdir, "test_summary.md")
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(full_report)
    print(f"✓ 完整评测报告已生成: {summary_md_path}")

    json_path = os.path.join(args.outdir, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"overall": overall_summary, "by_scan": summary_by_scan, "records": all_sample_records}, f, indent=2)
    print(f"✓ 结构化指标已保存至: {json_path}")
    print("=" * 85)


if __name__ == '__main__':
    main()
