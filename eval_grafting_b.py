import argparse
import os
import sys
import time
import json
import math
import cv2

# 控制要暴露给进程的 GPU id
_gpu_choice = os.environ.get('GPU_ID', os.environ.get('CUDA_VISIBLE_DEVICES', '3'))
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
from utils import (
    print_args, tocuda, make_nograd_func, DictAverageMeter,
    AbsDepthError_metrics, Thres_metrics, map_tri_to_pixel_single
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True


# ==============================================================================
# 1. 参数解析与模型拼接加载 (Train 24 主干 + Train 35 FOV 多视证据平面分支)
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='WHU MVS 权重拼接评测 (Train 24 主干 + Train 35 FOV 多视证据平面分支)'
    )
    # 模型与数据集
    parser.add_argument('--model', default='PatchmatchNet', help='select model')
    parser.add_argument('--dataset', default='dtu_whu', help='select dataset (default: dtu_whu)')
    parser.add_argument('--testpath', default='/home/ym/Experiment/Datas/WHU_MVS_dataset',
                        help='testing data path')
    parser.add_argument('--testlist', default='lists/whu/newtest.txt',
                        help='testing scan list file (default: lists/whu/newtest.txt)')
    parser.add_argument('--ckpt_backbone', default='./checkpoints/tensorboard_train24/model_000033.ckpt',
                        help='path to Train 24 checkpoint (for Backbone and Refinement)')
    parser.add_argument('--ckpt_plane', default='./checkpoints/tensorboard_train35/model_000033.ckpt',
                        help='path to Train 35 checkpoint (for FOV PlanePatchMatch branch)')
    parser.add_argument('--outdir', default='./outputs_grafting_fov',
                        help='output directory to save masks, reports and depths')

    # 批次与设备设置
    parser.add_argument('--batch_size', type=int, default=1, help='testing batch size (recommend 1)')
    parser.add_argument('--n_views', type=int, default=5, help='number of views per sample')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers for DataLoader')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    # 评测模式解耦开关
    parser.add_argument('--eval_mode', type=str, default='all',
                        choices=['all', 'global'],
                        help='评测模式: all (执行完整多尺度Table 1与GNN平面Table 2评估, 默认), global (仅执行全局多尺度精度Table 1)')

    # 掩码导出与深度保存
    parser.add_argument('--save_masks', action='store_true', default=False,
                        help='save stage0 and stage1 planar mask images (default: False)')
    parser.add_argument('--save_depth', action='store_true', default=False,
                        help='save predicted depth maps in .pfm and 16-bit .png format')

    # PatchMatch 超参数
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


def load_grafted_model(args, device):
    """
    构建拼接模型：
    1. 主干 (FeatureNet + PatchMatch 3/2/1 + Refinement) 来自 Train 24 Checkpoint
    2. 平面分支 (PlanePatchMatch with FOV / 5D Evidence, 2H + 19 门控) 来自 Train 35 Checkpoint
    """
    print("\n" + "=" * 80)
    print("构建实验组 B 权重拼接模型 (Train 24 主干 + Train 35 FOV 多视证据平面分支)")
    print("=" * 80)

    # 此时 models/PlanePatchMatch.py 与 models/net.py 原生为 Train 35 架构 (2H + 19 门控)
    model = PatchmatchNet(
        patchmatch_interval_scale=args.patchmatch_interval_scale,
        propagation_range=args.patchmatch_range,
        patchmatch_iteration=args.patchmatch_iteration,
        patchmatch_num_sample=args.patchmatch_num_sample,
        propagate_neighbors=args.propagate_neighbors,
        evaluate_neighbors=args.evaluate_neighbors
    )
    model.to(device)

    # 1. 加载 Train 24 主干与 Refinement 权重
    print(f"\n[1/2] 正在载入 Train 24 主干 Checkpoint: {args.ckpt_backbone}")
    ckpt_bb = torch.load(args.ckpt_backbone, map_location=device)
    state_bb = ckpt_bb.get('model', ckpt_bb)
    cleaned_bb = {k[7:] if k.startswith('module.') else k: v for k, v in state_bb.items()}

    # 自适应 Refinement 通道 (3通道 vs 4通道)
    key = 'upsample_net.conv0.conv.weight'
    model_state = model.state_dict()
    if key in cleaned_bb and key in model_state:
        ckpt_w = cleaned_bb[key]
        tgt_w = model_state[key]
        if ckpt_w.shape[1] == 4 and tgt_w.shape[1] == 3:
            print(f"[Adaptive Loading] Adapting {key} from [8, 4, 3, 3] -> [8, 3, 3, 3] (discarding 4th channel)...")
            cleaned_bb[key] = ckpt_w[:, :3, :, :]
        elif ckpt_w.shape[1] == 3 and tgt_w.shape[1] == 4:
            print(f"[Adaptive Loading] Adapting {key} from [8, 3, 3, 3] -> [8, 4, 3, 3] (Channel 4 set to 0.0)...")
            adapted_w = torch.zeros_like(tgt_w)
            adapted_w[:, :3, :, :] = ckpt_w
            cleaned_bb[key] = adapted_w

    # 过滤掉 Train 24 的平面分支权重，彻底阻断 137 维与 147 维门控的 size mismatch
    backbone_state = {
        k: v for k, v in cleaned_bb.items()
        if not k.startswith('plane_patchmatch_agent.')
    }
    model.load_state_dict(backbone_state, strict=False)
    print(f"✓ Train 24 主干与 Refinement 载入完成 (已排除平面分支，避免尺寸冲突)")

    # 2. 严格加载 Train 35 平面分支权重 (strict=True 确保 0 遗漏、0 错位)
    print(f"\n[2/2] 正在载入 Train 35 平面分支 Checkpoint: {args.ckpt_plane}")
    ckpt_plane = torch.load(args.ckpt_plane, map_location=device)
    state_plane = ckpt_plane.get('model', ckpt_plane)
    cleaned_plane = {k[7:] if k.startswith('module.') else k: v for k, v in state_plane.items()}

    prefix = 'plane_patchmatch_agent.'
    plane_state = {k[len(prefix):]: v for k, v in cleaned_plane.items() if k.startswith(prefix)}

    model.plane_patchmatch_agent.load_state_dict(plane_state, strict=True)
    print("✓ Train 35 FOV 多视证据平面分支以 strict=True 成功载入 (0 Missing, 0 Unexpected)!")

    model.eval()
    return model


# ==============================================================================
# 2. 评测指标算子 (与 test_whu.py 严格一致)
# ==============================================================================

@make_nograd_func
def evaluate_single_sample(model, sample, device):
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
    if mask is None or not mask.any():
        return float('nan')
    est_valid = depth_est[mask]
    gt_valid = depth_gt[mask]
    return torch.mean((est_valid - gt_valid).abs()).item()


def safe_thres_error(depth_est, depth_gt, mask, thres):
    if mask is None or not mask.any():
        return float('nan')
    est_valid = depth_est[mask]
    gt_valid = depth_gt[mask]
    err = (est_valid - gt_valid).abs()
    return (err > thres).float().mean().item()


def evaluate_global_accuracy(d_est_s0_b, depth_pm, depth_gt, mask, b, depth_s1_fused_b=None):
    m_s0_b = (mask['stage_0'][b:b+1] > 0.5)
    d_gt_s0_b = depth_gt['stage_0'][b:b+1]

    s0_mae = safe_mae(d_est_s0_b, d_gt_s0_b, m_s0_b)
    t1_err = safe_thres_error(d_est_s0_b, d_gt_s0_b, m_s0_b, 1.0)
    t2_err = safe_thres_error(d_est_s0_b, d_gt_s0_b, m_s0_b, 2.0)
    t4_err = safe_thres_error(d_est_s0_b, d_gt_s0_b, m_s0_b, 4.0)
    t8_err = safe_thres_error(d_est_s0_b, d_gt_s0_b, m_s0_b, 8.0)

    m_s1_b = (mask['stage_1'][b:b+1] > 0.5)
    d_s1_eval = depth_s1_fused_b if depth_s1_fused_b is not None else depth_pm['stage_1'][-1][b:b+1]
    s1_mae = safe_mae(d_s1_eval, depth_gt['stage_1'][b:b+1], m_s1_b)
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


# ==============================================================================
# 3. 主程序入口
# ==============================================================================

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print("=" * 80)
    print("WHU MVS 权重拼接评测系统: Train 24 主干 + Train 35 FOV 多视证据平面分支")
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

    # 2. 构建拼接模型
    model = load_grafted_model(args, device)

    # 3. 统计容器初始化
    per_scan_records = {}
    all_sample_records = []

    do_global = True
    do_stage1_gnn = (args.eval_mode == 'all')

    print("\n" + "=" * 80)
    print(f"开始逐样本前向推断与评测... 模式: [{args.eval_mode}] "
          f"(全局精度 Table 1: 开启, "
          f"Stage 1 GNN平面增益 Table 2: {'开启' if do_stage1_gnn else '关闭'})")
    print("=" * 80)

    start_eval_time = time.time()

    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            step_start = time.time()
            B = sample['imgs']['stage_0'].shape[0]

            outputs, depth_gt, mask, intrinsics_s0 = evaluate_single_sample(model, sample, device)

            depth_est_s0 = outputs["refined_depth"]['stage_0']
            depth_pm = outputs["depth_patchmatch"]
            output_plane = outputs.get("output_plane", {})

            # 掩码准备
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
                meta_idx = batch_idx * args.batch_size + b
                scan, file_id = test_dataset.metas[meta_idx]

                if scan not in per_scan_records:
                    per_scan_records[scan] = []

                record = {
                    "scan": scan,
                    "file_id": file_id,
                }

                if do_global:
                    global_metrics = evaluate_global_accuracy(
                        depth_est_s0[b:b+1], depth_pm, depth_gt, mask, b, depth_s1_fused[b:b+1]
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

                if do_stage1_gnn:
                    gnn_metrics = evaluate_stage1_gnn_gain(
                        depth_pm, depth_gt, mask_gt_s1, eval_mask_s1_planar, b
                    )
                    record.update(gnn_metrics)

                per_scan_records[scan].append(record)
                all_sample_records.append(record)

                if args.save_masks:
                    mask_save_dir = os.path.join(args.outdir, scan, "masks")
                    os.makedirs(mask_save_dir, exist_ok=True)
                    s0_eval_mask_np = (eval_mask_s0_planar[b, 0].cpu().numpy().astype(np.uint8)) * 255
                    cv2.imwrite(os.path.join(mask_save_dir, f"stage0_planar_mask_{file_id}.png"), s0_eval_mask_np)
                    s1_eval_mask_np = (eval_mask_s1_planar[b, 0].cpu().numpy().astype(np.uint8)) * 255
                    cv2.imwrite(os.path.join(mask_save_dir, f"stage1_planar_mask_{file_id}.png"), s1_eval_mask_np)

                if args.save_depth:
                    depth_save_dir = os.path.join(args.outdir, scan, "depths")
                    os.makedirs(depth_save_dir, exist_ok=True)
                    d_s0_np = depth_est_s0[b, 0].cpu().numpy()
                    save_pfm(os.path.join(depth_save_dir, f"stage0_refined_depth_{file_id}.pfm"), d_s0_np)
                    d_s1_fused_np = depth_s1_fused[b, 0].cpu().numpy()
                    save_pfm(os.path.join(depth_save_dir, f"stage1_fused_depth_{file_id}.pfm"), d_s1_fused_np)

            cur_idx = batch_idx + 1
            step_time = time.time() - step_start
            p_str = f"{s0_planar_mae:.4f}m" if not math.isnan(s0_planar_mae) else "N/A"
            if do_stage1_gnn:
                gnn_str = f"{gnn_metrics['stage1_planar_prop_mae']:.4f}m" if not math.isnan(gnn_metrics['stage1_planar_prop_mae']) else "N/A"
                print(f"[{cur_idx:03d}/{len(test_loader):03d}] Scan: {scan} | Img: {file_id} | "
                      f"S0 MAE: {global_metrics['stage0_mae']:.4f}m | S0 Planar: {p_str} | "
                      f"S1 GNN: {gnn_str} | Time: {step_time:.2f}s")
            else:
                print(f"[{cur_idx:03d}/{len(test_loader):03d}] Scan: {scan} | Img: {file_id} | "
                      f"S0 MAE: {global_metrics['stage0_mae']:.4f}m | S0 Planar: {p_str} | Time: {step_time:.2f}s")

    eval_duration = time.time() - start_eval_time
    print("\n" + "=" * 80)
    print(f"测试完成！耗时: {eval_duration:.2f} 秒，共评估 {len(all_sample_records)} 张图像。")
    print("=" * 80)

    # 汇总
    def compute_mean_ignore_nan(records, key):
        vals = [r[key] for r in records if key in r and not math.isnan(r[key])]
        return float(np.mean(vals)) if len(vals) > 0 else float('nan')

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
        summary_by_scan[scan] = s_dict

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

    # 生成与 test_whu.py 完全相同的 Markdown 报告
    report_lines = []
    report_lines.append("# WHU MVS 权重拼接评测报告 (Train 24 主干 + Train 35 FOV 多视证据平面分支)\n")
    report_lines.append(f"- **评测模式**: `{args.eval_mode}`")
    report_lines.append(f"- **Backbone Checkpoint (Train 24)**: `{args.ckpt_backbone}`")
    report_lines.append(f"- **Plane Checkpoint (Train 35)**: `{args.ckpt_plane}`")
    report_lines.append(f"- **Testlist**: `{args.testlist}`")
    report_lines.append(f"- **测试样本数**: {overall_summary['count']} 张\n")

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

    full_report = "\n".join(report_lines)
    print("\n" + full_report + "\n")

    summary_txt_path = os.path.join(args.outdir, "test_summary.md")
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write(full_report)
    print(f"✓ 评测报告已保存至: {summary_txt_path}")

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
