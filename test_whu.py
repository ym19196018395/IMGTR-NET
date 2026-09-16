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
from utils import print_args, tocuda, make_nograd_func, DictAverageMeter, AbsDepthError_metrics, Thres_metrics

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

    return outputs, depth_gt, mask


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

    # 3. 统计容器初始化
    per_scan_records = {}
    all_sample_records = []

    print("\n" + "=" * 80)
    print("开始逐样本前向推断与指标计算...")
    print("=" * 80)

    start_eval_time = time.time()

    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            step_start = time.time()
            B = sample['imgs']['stage_0'].shape[0]

            outputs, depth_gt, mask = evaluate_single_sample(model, sample, device)

            depth_est_s0 = outputs["refined_depth"]['stage_0']           # [B, 1, H0, W0]
            depth_pm = outputs["depth_patchmatch"]                       # dict of stages
            output_plane = outputs.get("output_plane", {})

            # ----------------------------------------------------
            # Stage 0 掩码与深度
            # ----------------------------------------------------
            mask_gt_s0 = (mask['stage_0'] > 0.5)                         # [B, 1, H0, W0]
            is_planar_s0 = output_plane.get("is_planar_s0", torch.zeros_like(mask_gt_s0)) # [B, 1, H0, W0] bool
            eval_mask_s0_planar = mask_gt_s0 & is_planar_s0
            eval_mask_s0_curved = mask_gt_s0 & (~is_planar_s0)

            # ----------------------------------------------------
            # Stage 1 掩码与深度
            # ----------------------------------------------------
            mask_gt_s1 = (mask['stage_1'] > 0.5)                         # [B, 1, H1, W1]
            w_pixel_s1 = output_plane.get("W_plane_pixel", None)
            tri_id_map_s1 = output_plane.get("tri_id_map", None)

            if w_pixel_s1 is not None and tri_id_map_s1 is not None:
                valid_tri_s1 = (tri_id_map_s1.unsqueeze(1) >= 0)
                is_planar_s1 = (w_pixel_s1 >= 0.80) & valid_tri_s1
                eval_mask_s1_planar = mask_gt_s1 & is_planar_s1
                # 物理融合深度：平面区走最终平面深度 [-1]，非平面区回退到纯像素深度 [0]
                depth_s1_fused = torch.where(is_planar_s1, depth_pm['stage_1'][-1], depth_pm['stage_1'][0])
            else:
                is_planar_s1 = torch.zeros_like(mask_gt_s1)
                eval_mask_s1_planar = torch.zeros_like(mask_gt_s1)
                depth_s1_fused = depth_pm['stage_1'][-1]

            depth_s1_prop = depth_pm['stage_1'][-1]
            depth_s1_pixel = depth_pm['stage_1'][0]

            # ----------------------------------------------------
            # 逐 Batch 内样本分别计算与保存
            # ----------------------------------------------------
            for b in range(B):
                meta_idx = batch_idx * args.batch_size + b
                scan, file_id = test_dataset.metas[meta_idx]

                if scan not in per_scan_records:
                    per_scan_records[scan] = []

                # --- 1. 计算 Stage 0 深度误差 ---
                m_s0_b = mask_gt_s0[b:b+1]
                m_s0_planar_b = eval_mask_s0_planar[b:b+1]
                m_s0_curved_b = eval_mask_s0_curved[b:b+1]
                d_est_s0_b = depth_est_s0[b:b+1]
                d_gt_s0_b = depth_gt['stage_0'][b:b+1]

                s0_mae = safe_mae(d_est_s0_b, d_gt_s0_b, m_s0_b)
                s0_planar_mae = safe_mae(d_est_s0_b, d_gt_s0_b, m_s0_planar_b)
                s0_curved_mae = safe_mae(d_est_s0_b, d_gt_s0_b, m_s0_curved_b)

                # 阈值误差比例 (Error Rate: error > thres)
                t1_err = safe_thres_error(d_est_s0_b, d_gt_s0_b, m_s0_b, 1.0)
                t2_err = safe_thres_error(d_est_s0_b, d_gt_s0_b, m_s0_b, 2.0)
                t4_err = safe_thres_error(d_est_s0_b, d_gt_s0_b, m_s0_b, 4.0)
                t8_err = safe_thres_error(d_est_s0_b, d_gt_s0_b, m_s0_b, 8.0)

                # --- 2. 计算 Stage 1 深度误差 ---
                m_s1_b = mask_gt_s1[b:b+1]
                m_s1_planar_b = eval_mask_s1_planar[b:b+1]
                d_gt_s1_b = depth_gt['stage_1'][b:b+1]
                d_s1_fused_b = depth_s1_fused[b:b+1]
                d_s1_prop_b = depth_s1_prop[b:b+1]
                d_s1_pixel_b = depth_s1_pixel[b:b+1]

                s1_fused_mae = safe_mae(d_s1_fused_b, d_gt_s1_b, m_s1_b)
                s1_prop_mae = safe_mae(d_s1_prop_b, d_gt_s1_b, m_s1_b)
                s1_pixel_mae = safe_mae(d_s1_pixel_b, d_gt_s1_b, m_s1_b)

                # Stage 1 平面核心区域对比 (GNN 平面化深度 vs 纯像素深度)
                s1_planar_prop_mae = safe_mae(d_s1_prop_b, d_gt_s1_b, m_s1_planar_b)
                s1_planar_pixel_mae = safe_mae(d_s1_pixel_b, d_gt_s1_b, m_s1_planar_b)

                # --- 3. 计算 Stage 2 / 3 深度误差 ---
                s2_mae = safe_mae(depth_pm['stage_2'][-1][b:b+1], depth_gt['stage_2'][b:b+1], mask['stage_2'][b:b+1] > 0.5)
                s3_mae = safe_mae(depth_pm['stage_3'][-1][b:b+1], depth_gt['stage_3'][b:b+1], mask['stage_3'][b:b+1] > 0.5)

                # --- 4. 平面掩码分布统计 (Planar Distribution) ---
                s0_valid_cnt = m_s0_b.sum().item()
                s0_planar_cnt = m_s0_planar_b.sum().item()
                s0_planar_ratio = (s0_planar_cnt / s0_valid_cnt * 100.0) if s0_valid_cnt > 0 else 0.0

                s1_valid_cnt = m_s1_b.sum().item()
                s1_planar_cnt = m_s1_planar_b.sum().item()
                s1_planar_ratio = (s1_planar_cnt / s1_valid_cnt * 100.0) if s1_valid_cnt > 0 else 0.0

                # 记录单张样本指标
                record = {
                    "scan": scan,
                    "file_id": file_id,
                    "stage0_mae": s0_mae,
                    "stage0_planar_mae": s0_planar_mae,
                    "stage0_curved_mae": s0_curved_mae,
                    "stage0_thres1mm_err": t1_err,
                    "stage0_thres2mm_err": t2_err,
                    "stage0_thres4mm_err": t4_err,
                    "stage0_thres8mm_err": t8_err,
                    "stage1_fused_mae": s1_fused_mae,
                    "stage1_prop_mae": s1_prop_mae,
                    "stage1_pixel_mae": s1_pixel_mae,
                    "stage1_planar_prop_mae": s1_planar_prop_mae,
                    "stage1_planar_pixel_mae": s1_planar_pixel_mae,
                    "stage2_mae": s2_mae,
                    "stage3_mae": s3_mae,
                    "s0_valid_pixels": s0_valid_cnt,
                    "s0_planar_pixels": s0_planar_cnt,
                    "s0_planar_ratio_pct": s0_planar_ratio,
                    "s1_valid_pixels": s1_valid_cnt,
                    "s1_planar_pixels": s1_planar_cnt,
                    "s1_planar_ratio_pct": s1_planar_ratio,
                }
                per_scan_records[scan].append(record)
                all_sample_records.append(record)

                # --- 5. 导出平面掩码 (供后续源版 PatchmatchNet 跑相同区域精度) ---
                if args.save_masks:
                    mask_save_dir = os.path.join(args.outdir, scan, "masks")
                    os.makedirs(mask_save_dir, exist_ok=True)

                    # (1) Stage 0 核心评估掩码: GT 有效且被判定为平面 (uint8, 255/0)
                    s0_eval_mask_np = (m_s0_planar_b[0, 0].cpu().numpy().astype(np.uint8)) * 255
                    cv2.imwrite(os.path.join(mask_save_dir, f"stage0_planar_mask_{file_id}.png"), s0_eval_mask_np)

                    # (2) Stage 1 核心评估掩码: GT 有效且被判定为平面 (uint8, 255/0)
                    s1_eval_mask_np = (m_s1_planar_b[0, 0].cpu().numpy().astype(np.uint8)) * 255
                    cv2.imwrite(os.path.join(mask_save_dir, f"stage1_planar_mask_{file_id}.png"), s1_eval_mask_np)

                    # (3) 纯预测平面掩码 (不依赖 GT，方便可视化网络自身平面分割表现)
                    s0_pred_mask_np = (is_planar_s0[b, 0].cpu().numpy().astype(np.uint8)) * 255
                    cv2.imwrite(os.path.join(mask_save_dir, f"stage0_pred_planar_{file_id}.png"), s0_pred_mask_np)

                    s1_pred_mask_np = (is_planar_s1[b, 0].cpu().numpy().astype(np.uint8)) * 255
                    cv2.imwrite(os.path.join(mask_save_dir, f"stage1_pred_planar_{file_id}.png"), s1_pred_mask_np)

                # --- 6. (可选) 导出预测深度图 ---
                if args.save_depth:
                    depth_save_dir = os.path.join(args.outdir, scan, "depths")
                    os.makedirs(depth_save_dir, exist_ok=True)

                    d_s0_np = d_est_s0_b[0, 0].cpu().numpy()
                    save_pfm(os.path.join(depth_save_dir, f"stage0_refined_depth_{file_id}.pfm"), d_s0_np)

                    d_s1_fused_np = d_s1_fused_b[0, 0].cpu().numpy()
                    save_pfm(os.path.join(depth_save_dir, f"stage1_fused_depth_{file_id}.pfm"), d_s1_fused_np)

                    # 保存 16-bit 深度用于快速看图 (WHU depth: depth * 64 -> uint16)
                    depth_png_s0 = np.clip(d_s0_np * 64.0, 0, 65535).astype(np.uint16)
                    cv2.imwrite(os.path.join(depth_save_dir, f"stage0_refined_depth_{file_id}.png"), depth_png_s0)

            # 终端简明打印进度
            cur_idx = batch_idx + 1
            step_time = time.time() - step_start
            print(f"[{cur_idx:03d}/{len(test_loader):03d}] Scan: {scan} | Img: {file_id} | "
                  f"S0 MAE: {s0_mae:.4f}m | S0 Planar MAE: {s0_planar_mae:.4f}m (Cover: {s0_planar_ratio:.1f}%) | "
                  f"S1 Planar MAE: {s1_planar_prop_mae:.4f}m (Pixel: {s1_planar_pixel_mae:.4f}m) | Time: {step_time:.2f}s")

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

    # 按 Scan 汇总
    summary_by_scan = {}
    for scan, recs in per_scan_records.items():
        summary_by_scan[scan] = {
            "count": len(recs),
            "stage0_mae": compute_mean_ignore_nan(recs, "stage0_mae"),
            "stage0_planar_mae": compute_mean_ignore_nan(recs, "stage0_planar_mae"),
            "stage0_curved_mae": compute_mean_ignore_nan(recs, "stage0_curved_mae"),
            "stage0_thres1mm_err": compute_mean_ignore_nan(recs, "stage0_thres1mm_err"),
            "stage0_thres2mm_err": compute_mean_ignore_nan(recs, "stage0_thres2mm_err"),
            "stage0_thres4mm_err": compute_mean_ignore_nan(recs, "stage0_thres4mm_err"),
            "stage0_thres8mm_err": compute_mean_ignore_nan(recs, "stage0_thres8mm_err"),
            "stage1_fused_mae": compute_mean_ignore_nan(recs, "stage1_fused_mae"),
            "stage1_prop_mae": compute_mean_ignore_nan(recs, "stage1_prop_mae"),
            "stage1_pixel_mae": compute_mean_ignore_nan(recs, "stage1_pixel_mae"),
            "stage1_planar_prop_mae": compute_mean_ignore_nan(recs, "stage1_planar_prop_mae"),
            "stage1_planar_pixel_mae": compute_mean_ignore_nan(recs, "stage1_planar_pixel_mae"),
            "stage2_mae": compute_mean_ignore_nan(recs, "stage2_mae"),
            "stage3_mae": compute_mean_ignore_nan(recs, "stage3_mae"),
            "s0_planar_ratio_pct": compute_mean_ignore_nan(recs, "s0_planar_ratio_pct"),
            "s1_planar_ratio_pct": compute_mean_ignore_nan(recs, "s1_planar_ratio_pct"),
        }

    # 全局总计
    overall_summary = {
        "count": len(all_sample_records),
        "stage0_mae": compute_mean_ignore_nan(all_sample_records, "stage0_mae"),
        "stage0_planar_mae": compute_mean_ignore_nan(all_sample_records, "stage0_planar_mae"),
        "stage0_curved_mae": compute_mean_ignore_nan(all_sample_records, "stage0_curved_mae"),
        "stage0_thres1mm_err": compute_mean_ignore_nan(all_sample_records, "stage0_thres1mm_err"),
        "stage0_thres2mm_err": compute_mean_ignore_nan(all_sample_records, "stage0_thres2mm_err"),
        "stage0_thres4mm_err": compute_mean_ignore_nan(all_sample_records, "stage0_thres4mm_err"),
        "stage0_thres8mm_err": compute_mean_ignore_nan(all_sample_records, "stage0_thres8mm_err"),
        "stage1_fused_mae": compute_mean_ignore_nan(all_sample_records, "stage1_fused_mae"),
        "stage1_prop_mae": compute_mean_ignore_nan(all_sample_records, "stage1_prop_mae"),
        "stage1_pixel_mae": compute_mean_ignore_nan(all_sample_records, "stage1_pixel_mae"),
        "stage1_planar_prop_mae": compute_mean_ignore_nan(all_sample_records, "stage1_planar_prop_mae"),
        "stage1_planar_pixel_mae": compute_mean_ignore_nan(all_sample_records, "stage1_planar_pixel_mae"),
        "stage2_mae": compute_mean_ignore_nan(all_sample_records, "stage2_mae"),
        "stage3_mae": compute_mean_ignore_nan(all_sample_records, "stage3_mae"),
        "s0_planar_ratio_pct": compute_mean_ignore_nan(all_sample_records, "s0_planar_ratio_pct"),
        "s1_planar_ratio_pct": compute_mean_ignore_nan(all_sample_records, "s1_planar_ratio_pct"),
    }

    # 计算 GNN 在 Stage 1 平面区相对纯像素深度的提升百分比
    gnn_gain_overall = (
        (overall_summary["stage1_planar_pixel_mae"] - overall_summary["stage1_planar_prop_mae"])
        / overall_summary["stage1_planar_pixel_mae"] * 100.0
    ) if not math.isnan(overall_summary["stage1_planar_pixel_mae"]) and overall_summary["stage1_planar_pixel_mae"] > 0 else 0.0

    # 构造 Markdown 报告
    report_lines = []
    report_lines.append("# WHU MVS 独立评测深度精度与平面分析报告\n")
    report_lines.append(f"- **Checkpoint**: `{args.loadckpt}`")
    report_lines.append(f"- **Testlist**: `{args.testlist}`")
    report_lines.append(f"- **测试样本数**: {overall_summary['count']} 张")
    if args.save_masks:
        report_lines.append(f"- **掩码导出目录**: `{args.outdir}` (含各视角 `stage0_planar_mask` 和 `stage1_planar_mask`)\n")
    else:
        report_lines.append("- **掩码导出**: 未开启（快速评测模式，不生成/保存掩码 PNG 文件）\n")

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
        f"**{overall_summary['stage0_thres4mm_err']*100:.2f}%** | **{overall_summary['stage0_thres8mm_err']*100:.2f}%** |"
    )

    report_lines.append("\n## 2. 平面区域 vs 曲面区域精度分析与 GNN 增益 (Table 2: Regional & GNN Performance)")
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
        f"**+{gnn_gain_overall:.2f}%** | **{overall_summary['s1_planar_ratio_pct']:.1f}%** |"
    )

    report_lines.append("\n## 3. 平面掩码分布统计 (Mask Distribution Summary)")
    report_lines.append(f"- **Stage 0 平面区域平均像素占比**: `{overall_summary['s0_planar_ratio_pct']:.2f}%`")
    report_lines.append(f"- **Stage 1 平面区域平均像素占比**: `{overall_summary['s1_planar_ratio_pct']:.2f}%`")
    if args.save_masks:
        report_lines.append(f"- **说明**: 所有掩码图像已按场景保存至 `{args.outdir}/<scan>/masks/` 目录下。")
        report_lines.append("  后续运行原版 PatchmatchNet 时，直接加载 `stage0_planar_mask_<id>.png` 即可对原版输出施加相同的区域切片，进行完全严密的同源对比。")
    else:
        report_lines.append("- **说明**: 当前为快速评测模式（未向磁盘写入掩码 PNG 图像），各尺度指标均通过 GPU 显存即时张量统计得出。")

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
