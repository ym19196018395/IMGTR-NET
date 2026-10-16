#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
========================================================================================
ENRICH-Aerial 航空遥感大场景测试集 A / B 对照与精确门控融合评估系统
========================================================================================
【实验目标】：
  1. 修复 Stage 1 融合深度指标：严格使用平面置信度物理门控
     depth_s1_fused = torch.where(is_planar_s1, prop_depth, pixel_depth)
     在平面区域吸收 GNN 几何平滑优势，在曲面/非平面区域完整保留原始 PatchMatch 像素深度。
  2. 聚焦模式 A (标准 eval - 原始全局统计) vs 模式 B (eval + 单图动态 BN 统计) 双列对比。
  3. 保留对全新未见测试集（36 张切片，涵盖 scan_031~scan_036）的完整 Table 1 与 Table 2 诊断报表。
========================================================================================
"""

import argparse
import os
import sys
import copy
import time
import json
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import numpy as np

# 确保项目根目录在 sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
while PROJECT_ROOT in sys.path:
    sys.path.remove(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from datasets import find_dataset_def
from datasets.enrich_aerial import collate_keep_list
from models.net import PatchmatchNet
from utils import tocuda, print_args

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def extract_region_name(scan_name):
    """从切片名称中提取拍摄组/测区名称 (例如 scan_031_p00 -> scan_031)"""
    if "_p" in scan_name:
        parts = scan_name.split("_p")
        if len(parts) > 1 and parts[-1].isdigit():
            return "_p".join(parts[:-1])
    return scan_name


def safe_mae(depth_est, depth_gt, mask):
    """安全计算有效掩码内的绝对误差 MAE (米)"""
    if mask is None or not mask.any():
        return float('nan')
    est_valid = depth_est[mask]
    gt_valid = depth_gt[mask]
    return torch.mean((est_valid - gt_valid).abs()).item()


def safe_thres_error(depth_est, depth_gt, mask, thres):
    """计算误差大于特定阈值 (m) 的离群比例 [0.0 ~ 1.0]"""
    if mask is None or not mask.any():
        return float('nan')
    est_valid = depth_est[mask]
    gt_valid = depth_gt[mask]
    err = (est_valid - gt_valid).abs()
    return (err > thres).float().mean().item()


def forward_single_sample(model, sample_cuda, device, args=None):
    """执行单次单样本前向传播，并精准计算多尺度与门控融合指标"""
    vertexs_batch = sample_cuda['vertexs']
    lines_batch = sample_cuda['lines']
    triangles_batch = []
    for tri_list in sample_cuda['triangles']:
        tri_processed = []
        for t in tri_list:
            v_ids = t['vertex_ids']
            l_ids = t['line_ids']
            pts = torch.from_numpy(t['valid_points']).to(device)
            tri_processed.append((v_ids, l_ids, pts))
        triangles_batch.append(tri_processed)

    depth_gt = sample_cuda["depth"]
    mask = sample_cuda["mask"]

    kwargs = {}
    if args is not None:
        kwargs['pre_base_depth_thresh'] = getattr(args, 'pre_base_depth_thresh', 0.20)
        kwargs['pre_pixel_dist_thresh'] = getattr(args, 'pre_pixel_dist_thresh', 1.2)
        kwargs['post_base_depth_thresh'] = getattr(args, 'post_base_depth_thresh', 0.16)
        kwargs['post_pixel_dist_thresh'] = getattr(args, 'post_pixel_dist_thresh', 1.2)

    outputs = model(
        sample_cuda["imgs"],
        sample_cuda["proj_matrices"],
        sample_cuda["intrinsics_mats"],
        sample_cuda["depth_min"],
        sample_cuda["depth_max"],
        vertexs_batch,
        lines_batch,
        triangles_batch,
        depth_gt['stage_1'],
        lambda_c=0.0,
        lambda_s=0.0,
        progress=1.0,
        compute_edge_pixels=False,
        **kwargs
    )

    depth_est = outputs["refined_depth"]
    depth_patchmatch = outputs["depth_patchmatch"]
    output_plane = outputs.get("output_plane", {})

    mask_s0 = mask['stage_0'] > 0.5
    mask_s1 = mask['stage_1'] > 0.5
    mask_s2 = mask['stage_2'] > 0.5
    mask_s3 = mask['stage_3'] > 0.5

    # =========================================================================
    # 👑 核心修复：Stage 1 真正物理门控融合深度
    # =========================================================================
    w_pixel_s1 = output_plane.get("W_plane_pixel", None)
    tri_id_map_s1 = output_plane.get("tri_id_map", None)

    if w_pixel_s1 is not None and tri_id_map_s1 is not None:
        valid_tri_s1 = (tri_id_map_s1.unsqueeze(1) >= 0)
        is_planar_s1 = (w_pixel_s1 >= 0.80) & valid_tri_s1
        eval_mask_s1_planar = mask_s1 & is_planar_s1
        eval_mask_s1_curved = mask_s1 & (~is_planar_s1)
        # 门控融合：平面区采用 GNN 传播深度，曲面/非平面区保留 PatchMatch 原始像素深度
        depth_s1_fused = torch.where(is_planar_s1, depth_patchmatch['stage_1'][-1], depth_patchmatch['stage_1'][0])
    else:
        is_planar_s1 = torch.zeros_like(mask_s1)
        eval_mask_s1_planar = torch.zeros_like(mask_s1)
        eval_mask_s1_curved = mask_s1
        depth_s1_fused = depth_patchmatch['stage_1'][0]

    # 全图各阶段 MAE
    mae_s0 = safe_mae(depth_est['stage_0'], depth_gt['stage_0'], mask_s0)
    mae_s1_fused = safe_mae(depth_s1_fused, depth_gt['stage_1'], mask_s1)
    mae_s1_raw = safe_mae(depth_patchmatch['stage_1'][0], depth_gt['stage_1'], mask_s1)
    mae_s1_prop = safe_mae(depth_patchmatch['stage_1'][-1], depth_gt['stage_1'], mask_s1)
    mae_s2 = safe_mae(depth_patchmatch['stage_2'][-1], depth_gt['stage_2'], mask_s2)
    mae_s3 = safe_mae(depth_patchmatch['stage_3'][-1], depth_gt['stage_3'], mask_s3)

    # 平面细分指标 (GNN 平面增益验证)
    mae_s1_planar_prop = safe_mae(depth_patchmatch['stage_1'][-1], depth_gt['stage_1'], eval_mask_s1_planar)
    mae_s1_planar_pixel = safe_mae(depth_patchmatch['stage_1'][0], depth_gt['stage_1'], eval_mask_s1_planar)
    mae_s1_curved = safe_mae(depth_s1_fused, depth_gt['stage_1'], eval_mask_s1_curved)

    s1_planar_cnt = eval_mask_s1_planar.sum().item()
    s1_valid_cnt = mask_s1.sum().item()
    s1_planar_ratio = (s1_planar_cnt / s1_valid_cnt * 100.0) if s1_valid_cnt > 0 else 0.0

    err_10cm = safe_thres_error(depth_est['stage_0'], depth_gt['stage_0'], mask_s0, 0.10)
    err_20cm = safe_thres_error(depth_est['stage_0'], depth_gt['stage_0'], mask_s0, 0.20)
    err_50cm = safe_thres_error(depth_est['stage_0'], depth_gt['stage_0'], mask_s0, 0.50)

    return {
        "s0": mae_s0,
        "s1_fused": mae_s1_fused,
        "s1_raw": mae_s1_raw,
        "s1_prop": mae_s1_prop,
        "s1_planar_prop": mae_s1_planar_prop,
        "s1_planar_pixel": mae_s1_planar_pixel,
        "s1_curved": mae_s1_curved,
        "s1_planar_ratio": s1_planar_ratio,
        "s2": mae_s2,
        "s3": mae_s3,
        "err_10cm": err_10cm,
        "err_20cm": err_20cm,
        "err_50cm": err_50cm
    }


def load_clean_checkpoint(model, ckpt_path):
    """加载原始模型权重"""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint 文件不存在: {ckpt_path}")
    print(f"[Model] 载入 Checkpoint 权重: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    cleaned_state = {}
    for k, v in state_dict.items():
        clean_k = k[7:] if k.startswith('module.') else k
        cleaned_state[clean_k] = v
    model.load_state_dict(cleaned_state, strict=False)


def run_evaluation_on_dataset(model, loader, mode_tag, device, args=None):
    """在指定数据集 Loader 上执行完整单切片评估"""
    print(f"--------------------------------------------------------------------------------")
    print(f"🚀 开始执行模式 [{mode_tag}] 评测 (总样本数: {len(loader)})")
    print(f"--------------------------------------------------------------------------------")

    skip_keys = ["vertexs", "lines", "triangles", "scan", "file_id",
                 "tri_conf_cleaned", "tri_normal_cleaned", "tri_plane_cleaned",
                 "is_gt_planar", "tri_err95", "tri_weight_normal", "tri_weight_conf"]

    records = []
    start_time = time.time()

    with torch.no_grad():
        for idx, sample in enumerate(loader):
            scan_name = sample['scan'][0]
            file_id = sample['file_id'][0]
            region_name = extract_region_name(scan_name)
            sample_tag = f"{scan_name}_{file_id}"

            sample_cuda = tocuda(sample, device=device, skip_keys=skip_keys)
            metrics = forward_single_sample(model, sample_cuda, device, args=args)

            rec = {
                "idx": idx + 1,
                "scan": scan_name,
                "region": region_name,
                "tag": sample_tag,
                **metrics
            }
            records.append(rec)

            if (idx + 1) % 6 == 0 or (idx + 1) == len(loader):
                print(f"  [{mode_tag}] 进度: {idx + 1:2d}/{len(loader)} | "
                      f"切片: {sample_tag:20s} | "
                      f"S0: {metrics['s0']:.4f}m | S1融合: {metrics['s1_fused']:.4f}m | "
                      f"S1原始: {metrics['s1_raw']:.4f}m | S2: {metrics['s2']:.4f}m | S3: {metrics['s3']:.4f}m")

    elapsed = time.time() - start_time
    print(f"✓ 模式 [{mode_tag}] 评测完成，总耗时: {elapsed:.1f}s\n")
    return records


def parse_args():
    parser = argparse.ArgumentParser(description="ENRICH-Aerial 测试集 A / B 对照与精确门控融合评估系统")
    parser.add_argument('--dataset', default='enrich_aerial', help='数据集类型')
    parser.add_argument('--datapath', default='', help='数据集路径 (留空自动探查)')
    parser.add_argument('--testlist', default='lists/enrich_aerial/test.txt', help='测试列表路径')
    parser.add_argument('--trainlist', default='lists/enrich_aerial/train.txt', help='训练列表路径 (兼容保留)')
    parser.add_argument('--pre_base_depth_thresh', type=float, default=0.20, help='Pre-GNN Cross-Check 基础深度容限 (m)')
    parser.add_argument('--pre_pixel_dist_thresh', type=float, default=1.2, help='Pre-GNN Cross-Check 2D像素距离容限 (px)')
    parser.add_argument('--post_base_depth_thresh', type=float, default=0.16, help='Post-GNN Cross-Check 基础深度容限 (m)')
    parser.add_argument('--post_pixel_dist_thresh', type=float, default=1.2, help='Post-GNN Cross-Check 2D像素距离容限 (px)')
    parser.add_argument('--loadckpt', default='', help='模型 Checkpoint 路径')
    parser.add_argument('--nviews', type=int, default=5, help='输入视角数 (默认 5)')
    parser.add_argument('--outdir', default='./outputs_enrich_aerial_diag', help='报告输出目录')
    return parser.parse_args()


def main():
    args = parse_args()
    print_args(args)

    # 1. 探查数据路径
    data_path = args.datapath
    if not data_path or not os.path.isdir(data_path):
        candidates = [
            "/home/myao/ENRICH-Aerial_Cropped_0.6x_Overlap50/",
            "/home/myao/ENRICH-Aerial_Data/",
            "/home/myao/Crop_Test_Dataset/",
            "./datasets/ENRICH-Aerial_Cropped_0.6x_Overlap50/"
        ]
        for p in candidates:
            if os.path.isdir(p):
                data_path = p
                break
    if not data_path or not os.path.isdir(data_path):
        raise FileNotFoundError(f"未找到有效的数据集目录，请通过 --datapath 显式指定！")
    print(f"[Dataset] 确认数据集根目录: {data_path}")

    # 2. 探查权重
    ckpt_path = args.loadckpt
    if not ckpt_path or not os.path.isfile(ckpt_path):
        candidates = [
            "./checkpoints/tensorboard_enrich_ours/model_000033.ckpt",
            "./checkpoints/tensorboard_enrich_ours/model_000032.ckpt",
            "./checkpoints/tensorboard_train40/model_000033.ckpt"
        ]
        for c in candidates:
            if os.path.isfile(c):
                ckpt_path = c
                break
        if not ckpt_path and os.path.isdir("./checkpoints"):
            for root, _, files in os.walk("./checkpoints"):
                for f in files:
                    if f.endswith(".ckpt"):
                        ckpt_path = os.path.join(root, f)
                        break
                if ckpt_path:
                    break
    if not ckpt_path or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"未找到有效的 Checkpoint，请通过 --loadckpt 指定！")
    print(f"[Model] 确认 Checkpoint: {ckpt_path}")

    # 3. 构造测试集 Loader (BatchSize=1, 固定顺序)
    MVSDataset = find_dataset_def(args.dataset)
    eval_dataset = MVSDataset(data_path, args.testlist, mode="test", nviews=args.nviews)
    eval_loader = DataLoader(
        eval_dataset, batch_size=1, shuffle=False,
        collate_fn=collate_keep_list, num_workers=2, pin_memory=True
    )
    print(f"[Dataset] 测试集样本数: 共 {len(eval_dataset)} 切片 (BatchSize=1, nviews={args.nviews})")

    # 4. 实例化模型底座
    base_model = PatchmatchNet(
        patchmatch_interval_scale=[0.005, 0.0125, 0.025],
        propagation_range=[6, 4, 2],
        patchmatch_iteration=[1, 2, 2],
        patchmatch_num_sample=[8, 8, 16],
        propagate_neighbors=[0, 8, 16],
        evaluate_neighbors=[9, 9, 9],
        pre_base_depth_thresh=args.pre_base_depth_thresh,
        pre_pixel_dist_thresh=args.pre_pixel_dist_thresh,
        post_base_depth_thresh=args.post_base_depth_thresh,
        post_pixel_dist_thresh=args.post_pixel_dist_thresh
    ).to(device)
    load_clean_checkpoint(base_model, ckpt_path)
    clean_state_dict = copy.deepcopy(base_model.state_dict())

    # =========================================================================
    # 评测 1: 模式 A (标准 eval - 原始全局 running 统计量)
    # =========================================================================
    model_A = copy.deepcopy(base_model)
    model_A.load_state_dict(clean_state_dict)
    model_A.eval()
    for m in model_A.modules():
        if isinstance(m, _BN_TYPES):
            m.eval()
            m.track_running_stats = True
    records_A = run_evaluation_on_dataset(model_A, eval_loader, mode_tag="Mode A (原全局统计)", device=device, args=args)
    del model_A
    torch.cuda.empty_cache()

    # =========================================================================
    # 评测 2: 模式 B (动态单图 BN 统计 - track_running_stats=False 严格防污染)
    # =========================================================================
    model_B = copy.deepcopy(base_model)
    model_B.load_state_dict(clean_state_dict)
    model_B.eval()
    for m in model_B.modules():
        if isinstance(m, _BN_TYPES):
            m.train()
            m.track_running_stats = False
    records_B = run_evaluation_on_dataset(model_B, eval_loader, mode_tag="Mode B (单图动态统计)", device=device, args=args)
    del model_B
    torch.cuda.empty_cache()

    # =========================================================================
    # 指标聚合统计
    # =========================================================================
    def compute_summary(records):
        return {
            "s0": float(np.nanmean([r['s0'] for r in records])),
            "s1_fused": float(np.nanmean([r['s1_fused'] for r in records])),
            "s1_raw": float(np.nanmean([r['s1_raw'] for r in records])),
            "s1_prop": float(np.nanmean([r['s1_prop'] for r in records])),
            "s1_planar_prop": float(np.nanmean([r['s1_planar_prop'] for r in records])),
            "s1_planar_pixel": float(np.nanmean([r['s1_planar_pixel'] for r in records])),
            "s1_curved": float(np.nanmean([r['s1_curved'] for r in records])),
            "s1_planar_ratio": float(np.nanmean([r['s1_planar_ratio'] for r in records])),
            "s2": float(np.nanmean([r['s2'] for r in records])),
            "s3": float(np.nanmean([r['s3'] for r in records])),
            "err_10cm": float(np.nanmean([r['err_10cm'] for r in records])) * 100,
            "err_20cm": float(np.nanmean([r['err_20cm'] for r in records])) * 100,
            "err_50cm": float(np.nanmean([r['err_50cm'] for r in records])) * 100,
        }

    sum_A = compute_summary(records_A)
    sum_B = compute_summary(records_B)

    # 区域分组统计
    regions = sorted(list(set(r['region'] for r in records_A)))
    reg_summary = {}
    for reg in regions:
        recs_reg_A = [r for r in records_A if r['region'] == reg]
        recs_reg_B = [r for r in records_B if r['region'] == reg]
        reg_summary[reg] = {
            "count": len(recs_reg_A),
            "s0_A": float(np.nanmean([r['s0'] for r in recs_reg_A])),
            "s0_B": float(np.nanmean([r['s0'] for r in recs_reg_B])),
            "s1_fused_B": float(np.nanmean([r['s1_fused'] for r in recs_reg_B])),
            "s1_raw_B": float(np.nanmean([r['s1_raw'] for r in recs_reg_B])),
            "s1_planar_prop_B": float(np.nanmean([r['s1_planar_prop'] for r in recs_reg_B])),
            "s1_planar_pixel_B": float(np.nanmean([r['s1_planar_pixel'] for r in recs_reg_B])),
            "s1_planar_ratio_B": float(np.nanmean([r['s1_planar_ratio'] for r in recs_reg_B])),
        }

    # GNN 平面提升幅度
    gnn_gain_B = ((sum_B['s1_planar_pixel'] - sum_B['s1_planar_prop']) / sum_B['s1_planar_pixel'] * 100.0) \
        if sum_B['s1_planar_pixel'] > 0 else 0.0

    # =========================================================================
    # 生成 Markdown 报表
    # =========================================================================
    report_md = f"""# ENRICH-Aerial 测试集 A / B 对照与精确门控融合评估报告

- **评估样本集**: `TEST` (列表: `{args.testlist}`)
- **样本总切片数**: {len(eval_dataset)} 张
- **Checkpoint**: `{ckpt_path}`
- **Batch Size**: 1 (单卡单切片，严格对齐训练物理流)
- **输入视角数**: {args.nviews} 视角

---

## 🏆 1. 全局多尺度深度精度核心对比表 (Table 1: Global Accuracy)

| 对照组 (Mode) | 运行环境与 BN 机制 | S0 Refined MAE | S1 门控融合 MAE | S1 原始像素 MAE | S2 MAE | S3 MAE (1/8) | >10cm 误差率 | >20cm 误差率 | >50cm 误差率 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **模式 A** | 标准 `model.eval()` (原全局 running 统计) | **{sum_A['s0']:.4f} m** | {sum_A['s1_fused']:.4f} m | {sum_A['s1_raw']:.4f} m | {sum_A['s2']:.4f} m | {sum_A['s3']:.4f} m | {sum_A['err_10cm']:.1f}% | {sum_A['err_20cm']:.1f}% | {sum_A['err_50cm']:.1f}% |
| **模式 B** | `eval()` + BN 动态单图统计 (即时批统计) | **{sum_B['s0']:.4f} m** | **{sum_B['s1_fused']:.4f} m** | {sum_B['s1_raw']:.4f} m | {sum_B['s2']:.4f} m | {sum_B['s3']:.4f} m | **{sum_B['err_10cm']:.1f}%** | **{sum_B['err_20cm']:.1f}%** | **{sum_B['err_50cm']:.1f}%** |

---

## 🎯 2. 诊断结论与关键指标对比

1. **S0 最终精度增益**: 模式 B 将 S0 MAE 从 **{sum_A['s0']:.4f}m** 骤降至 **{sum_B['s0']:.4f}m**（相对提升 **{((sum_A['s0']-sum_B['s0'])/sum_A['s0'])*100:.1f}%**）。
2. **Stage 1 门控融合物理检验**:
   - S1 原始纯像素 MAE (模式 B): **{sum_B['s1_raw']:.4f}m**
   - S1 门控融合 MAE (模式 B): **{sum_B['s1_fused']:.4f}m**
   - 平面区域 GNN 传播 MAE: **{sum_B['s1_planar_prop']:.4f}m** vs 纯像素 **{sum_B['s1_planar_pixel']:.4f}m** (GNN 平面提升 **{gnn_gain_B:+.2f}%**，平面占比 **{sum_B['s1_planar_ratio']:.1f}%**)

---

## 📌 3. 测试集各拍摄组 / 区域表现对比表 (Table 2: Regional Breakdown)

| 拍摄组 (Region) | 切片数 | 模式 A (S0 MAE) | 模式 B (S0 MAE) | B 相比 A 提升 | S1 融合(B) | S1 原始(B) | GNN 平面提升(B) | 平面占比 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for reg, d in reg_summary.items():
        diff_BA = ((d['s0_A'] - d['s0_B']) / d['s0_A']) * 100
        gain_r = ((d['s1_planar_pixel_B'] - d['s1_planar_prop_B']) / d['s1_planar_pixel_B'] * 100.0) if d['s1_planar_pixel_B'] > 0 else 0.0
        report_md += (f"| `{reg}` | {d['count']} | {d['s0_A']:.4f} m | **{d['s0_B']:.4f} m** | "
                      f"`+{diff_BA:.1f}%` | **{d['s1_fused_B']:.4f} m** | {d['s1_raw_B']:.4f} m | "
                      f"`{gain_r:+.2f}%` | {d['s1_planar_ratio_B']:.1f}% |\n")

    report_md += f"""
---

## 📋 4. 逐切片详细指标明细表 (Sample Records Preview)

| 序号 | 切片标签 | 模式 A (S0 MAE) | 模式 B (S0 MAE) | S1 融合(B) | S1 原始(B) | 模式 B 增益 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
"""
    for i in range(len(records_A)):
        tag = records_A[i]['tag']
        mA = records_A[i]['s0']
        mB = records_B[i]['s0']
        s1_f = records_B[i]['s1_fused']
        s1_r = records_B[i]['s1_raw']
        gain = ((mA - mB) / mA) * 100
        report_md += f"| {i+1} | `{tag}` | {mA:.4f} m | **{mB:.4f} m** | {s1_f:.4f} m | {s1_r:.4f} m | `{gain:+.1f}%` |\n"

    print("\n" + report_md)

    os.makedirs(args.outdir, exist_ok=True)
    report_file = os.path.join(args.outdir, "test_ab_diagnostic_report.md")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"✓ 修正版 Markdown 对比报告已保存至: {report_file}")

    # 保存结构化 JSON 便于分析
    json_file = os.path.join(args.outdir, "test_ab_metrics.json")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {"A": sum_A, "B": sum_B},
            "regions": reg_summary
        }, f, indent=2)
    print(f"✓ 结构化 JSON 指标已保存至: {json_file}")
    print("=" * 85)


if __name__ == '__main__':
    main()
