import argparse
import os
import sys
import time
import json
import math
import re
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


def extract_region_name(scan_name):
    """
    从切片名称中提取原始拍摄组/测区名称。
    例如:
      'scan_031_p00' -> 'scan_031'
      'scan_001_p05' -> 'scan_001'
      'scan_035'     -> 'scan_035'
    """
    if "_p" in scan_name:
        parts = scan_name.split("_p")
        if len(parts) > 1 and parts[-1].isdigit():
            return "_p".join(parts[:-1])
    return scan_name


def safe_mae(depth_est, depth_gt, mask):
    """安全计算有效掩码区域内的绝对深度误差 MAE (米)"""
    if mask is None or not mask.any():
        return float('nan')
    est_valid = depth_est[mask]
    gt_valid = depth_gt[mask]
    return torch.mean((est_valid - gt_valid).abs()).item()


def safe_thres_error(depth_est, depth_gt, mask, thres):
    """安全计算误差大于特定阈值 (m) 的离群点比例 [0.0 ~ 1.0]"""
    if mask is None or not mask.any():
        return float('nan')
    est_valid = depth_est[mask]
    gt_valid = depth_gt[mask]
    err = (est_valid - gt_valid).abs()
    return (err > thres).float().mean().item()


def safe_thres_acc(depth_est, depth_gt, mask, thres):
    """安全计算误差小于等于特定阈值 (m) 的精度比例 [0.0 ~ 1.0]"""
    if mask is None or not mask.any():
        return float('nan')
    est_valid = depth_est[mask]
    gt_valid = depth_gt[mask]
    err = (est_valid - gt_valid).abs()
    return (err <= thres).float().mean().item()


@make_nograd_func
def evaluate_single_sample(model, sample, device, args=None):
    """执行单批次前向推断，与当前训练中的 test_sample 严格对齐"""
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

    skip = ["vertexs", "lines", "triangles", "scan", "file_id",
            "tri_conf_cleaned", "tri_normal_cleaned", "tri_plane_cleaned",
            "is_gt_planar", "tri_err95", "tri_weight_normal", "tri_weight_conf"]
    sample_cuda = tocuda(sample, device=device, skip_keys=skip)

    depth_gt = sample_cuda["depth"]
    mask = sample_cuda["mask"]

    kwargs = {}
    if args is not None:
        kwargs['pre_base_depth_thresh'] = getattr(args, 'pre_base_depth_thresh', 0.20)
        kwargs['pre_pixel_dist_thresh'] = getattr(args, 'pre_pixel_dist_thresh', 1.2)
        kwargs['post_base_depth_thresh'] = getattr(args, 'post_base_depth_thresh', 0.16)
        kwargs['post_pixel_dist_thresh'] = getattr(args, 'post_pixel_dist_thresh', 1.2)

    # 推理评测模式：关闭缝合/平滑强损失约束，对齐训练期验证
    outputs = model(
        sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["intrinsics_mats"],
        sample_cuda["depth_min"], sample_cuda["depth_max"],
        vertexs_batch, lines_batch, triangles_batch, depth_gt['stage_1'],
        lambda_c=0.0, lambda_s=0.0, progress=1.0, compute_edge_pixels=False,
        **kwargs
    )

    return outputs, depth_gt, mask


_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def set_bn_batch_stats(model):
    """
    BN 诊断模式 'batch'：整体 eval()，但 BN 层切回 train() 使用当前批次统计量，
    同时 momentum=0 保证 running_mean / running_var 不被改写（与训练期前向完全一致）。
    """
    model.eval()
    n = 0
    for m in model.modules():
        if isinstance(m, _BN_TYPES):
            m.train()
            m.momentum = 0.0
            n += 1
    print(f"[BN] 模式=batch：{n} 个 BN 层使用当前批次统计量 (running stats 冻结不更新)")


@torch.no_grad()
def recalibrate_bn(model, dataset_name, data_path, list_file, args):
    """
    BN 诊断模式 'recalib'：清空 running stats，用训练集前向 N 个 batch 做累计平均重估，
    然后切回标准 eval()。若重估后 eval 精度接近训练曲线，说明原 running stats 失配。
    """
    n = 0
    for m in model.modules():
        if isinstance(m, _BN_TYPES):
            m.reset_running_stats()
            m.momentum = None  # 累计平均
            n += 1

    MVSDataset = find_dataset_def(dataset_name)
    calib_dataset = MVSDataset(data_path, list_file, mode="train", nviews=args.n_views)
    calib_loader = DataLoader(
        calib_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_keep_list, num_workers=args.num_workers,
        pin_memory=True, drop_last=True
    )

    model.eval()
    for m in model.modules():
        if isinstance(m, _BN_TYPES):
            m.train()

    total = min(args.recalib_batches, len(calib_loader))
    print(f"[BN] 模式=recalib：重置 {n} 个 BN 层，使用训练集 {total} 个 batch 重估 running stats ...")
    for i, sample in enumerate(calib_loader):
        if i >= total:
            break
        evaluate_single_sample(model, sample, device, args=args)
        if (i + 1) % 20 == 0 or (i + 1) == total:
            print(f"[BN] recalib {i + 1}/{total}")

    model.eval()
    print("[BN] running stats 重估完成，已切回标准 eval()\n")


def load_checkpoint(model, ckpt_path):
    """自适应加载模型权重并兼容多卡/单卡前缀"""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"🚨 Checkpoint 文件不存在: {ckpt_path}")

    print(f"[Model] 载入 Checkpoint 权重: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))

    cleaned_state = {}
    for k, v in state_dict.items():
        clean_k = k[7:] if k.startswith('module.') else k
        cleaned_state[clean_k] = v

    missing, unexpected = model.load_state_dict(cleaned_state, strict=False)
    if len(missing) > 0:
        print(f"[Model Warning] 缺失参数 (前5个): {missing[:5]}")
    if len(unexpected) > 0:
        print(f"[Model Warning] 冗余参数 (前5个): {unexpected[:5]}")
    print("✓ 模型权重载入完成！\n")


def evaluate_dataset_split(model, dataset_name, data_path, list_file, split_name, args):
    """
    针对指定的划分列表（如 train.txt 或 test.txt），运行统一的 eval() 评测流程，
    并提取切片级和区域级统计指标。
    """
    print("=" * 85)
    print(f"🚀 开始执行 [{split_name.upper()}] 划分评测: 列表文件: {list_file}")
    print("=" * 85)

    MVSDataset = find_dataset_def(dataset_name)
    eval_dataset = MVSDataset(data_path, list_file, mode="test", nviews=args.n_views)
    eval_loader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_keep_list, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=(args.num_workers > 0), drop_last=False
    )
    print(f"[Dataset] 样本总数: {len(eval_dataset)} 切片 (共 {len(eval_loader)} 个 Batch，BatchSize={args.batch_size})\n")

    sample_records = []
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, sample in enumerate(eval_loader):
            step_start = time.time()
            B = sample['imgs']['stage_0'].shape[0]

            outputs, depth_gt, mask = evaluate_single_sample(model, sample, device, args=args)

            depth_est_s0 = outputs["refined_depth"]['stage_0']  # [B, 1, H0, W0]
            depth_pm = outputs["depth_patchmatch"]              # 各阶段深度
            output_plane = outputs.get("output_plane", {})

            # 掩码构建
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
                region = extract_region_name(scan)

                s0_mae = safe_mae(depth_est_s0[b:b+1], depth_gt['stage_0'][b:b+1], mask_gt_s0[b:b+1])
                s1_fused_mae = safe_mae(depth_s1_fused[b:b+1], depth_gt['stage_1'][b:b+1], mask_gt_s1[b:b+1])
                s2_mae = safe_mae(depth_pm['stage_2'][-1][b:b+1], depth_gt['stage_2'][b:b+1], mask['stage_2'][b:b+1] > 0.5)
                s3_mae = safe_mae(depth_pm['stage_3'][-1][b:b+1], depth_gt['stage_3'][b:b+1], mask['stage_3'][b:b+1] > 0.5)

                s0_planar_mae = safe_mae(depth_est_s0[b:b+1], depth_gt['stage_0'][b:b+1], eval_mask_s0_planar[b:b+1])
                s0_curved_mae = safe_mae(depth_est_s0[b:b+1], depth_gt['stage_0'][b:b+1], eval_mask_s0_curved[b:b+1])

                t10_err = safe_thres_error(depth_est_s0[b:b+1], depth_gt['stage_0'][b:b+1], mask_gt_s0[b:b+1], 0.10)
                t20_err = safe_thres_error(depth_est_s0[b:b+1], depth_gt['stage_0'][b:b+1], mask_gt_s0[b:b+1], 0.20)
                t50_err = safe_thres_error(depth_est_s0[b:b+1], depth_gt['stage_0'][b:b+1], mask_gt_s0[b:b+1], 0.50)

                t20_acc = safe_thres_acc(depth_est_s0[b:b+1], depth_gt['stage_0'][b:b+1], mask_gt_s0[b:b+1], 0.20)
                t50_acc = safe_thres_acc(depth_est_s0[b:b+1], depth_gt['stage_0'][b:b+1], mask_gt_s0[b:b+1], 0.50)

                record = {
                    "split": split_name,
                    "region": region,
                    "scan": scan,
                    "file_id": file_id,
                    "stage0_mae": s0_mae,
                    "stage1_fused_mae": s1_fused_mae,
                    "stage2_mae": s2_mae,
                    "stage3_mae": s3_mae,
                    "stage0_planar_mae": s0_planar_mae,
                    "stage0_curved_mae": s0_curved_mae,
                    "stage0_thres10cm_err": t10_err,
                    "stage0_thres20cm_err": t20_err,
                    "stage0_thres50cm_err": t50_err,
                    "stage0_thres20cm_acc": t20_acc,
                    "stage0_thres50cm_acc": t50_acc,
                }
                sample_records.append(record)

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(eval_loader):
                cur_s0 = record['stage0_mae']
                print(f"[{split_name.upper()}] Batch {batch_idx+1:03d}/{len(eval_loader):03d} | "
                      f"Last Scan: {scan} (Region: {region}) | S0 MAE: {cur_s0:.4f}m | "
                      f"Step Time: {time.time() - step_start:.2f}s")

    duration = time.time() - start_time
    print(f"\n✓ [{split_name.upper()}] 评测完成！共耗时 {duration:.2f} 秒。")

    # --- 聚合统计：计算全局均值 ---
    def mean_val(records, key):
        v = [r[key] for r in records if key in r and not math.isnan(r[key])]
        return float(np.mean(v)) if len(v) > 0 else float('nan')

    global_summary = {
        "split": split_name,
        "count": len(sample_records),
        "stage0_mae": mean_val(sample_records, "stage0_mae"),
        "stage1_fused_mae": mean_val(sample_records, "stage1_fused_mae"),
        "stage2_mae": mean_val(sample_records, "stage2_mae"),
        "stage3_mae": mean_val(sample_records, "stage3_mae"),
        "stage0_planar_mae": mean_val(sample_records, "stage0_planar_mae"),
        "stage0_curved_mae": mean_val(sample_records, "stage0_curved_mae"),
        "stage0_thres10cm_err": mean_val(sample_records, "stage0_thres10cm_err"),
        "stage0_thres20cm_err": mean_val(sample_records, "stage0_thres20cm_err"),
        "stage0_thres50cm_err": mean_val(sample_records, "stage0_thres50cm_err"),
        "stage0_thres20cm_acc": mean_val(sample_records, "stage0_thres20cm_acc"),
        "stage0_thres50cm_acc": mean_val(sample_records, "stage0_thres50cm_acc"),
    }

    # --- 聚合统计：按原始拍摄组 / 区域 (Region) 分组统计 ---
    region_dict = {}
    for r in sample_records:
        reg = r['region']
        if reg not in region_dict:
            region_dict[reg] = []
        region_dict[reg].append(r)

    region_summary_list = []
    global_s0_mae = global_summary['stage0_mae']

    for reg, recs in region_dict.items():
        s0_m = mean_val(recs, "stage0_mae")
        # 偏离度：(区域MAE - 全局MAE) / 全局MAE
        dev_pct = ((s0_m - global_s0_mae) / global_s0_mae * 100.0) if global_s0_mae > 0 else 0.0

        if dev_pct > 25.0:
            status = "🔴 严重偏高 (显著拉高均值)"
        elif dev_pct < -25.0:
            status = "🟢 优异拟合 (显著优于均值)"
        else:
            status = "⚪ 正常波动区间"

        region_summary_list.append({
            "region": reg,
            "count": len(recs),
            "stage0_mae": s0_m,
            "dev_pct": dev_pct,
            "stage1_fused_mae": mean_val(recs, "stage1_fused_mae"),
            "stage0_planar_mae": mean_val(recs, "stage0_planar_mae"),
            "stage0_thres10cm_err": mean_val(recs, "stage0_thres10cm_err"),
            "stage0_thres20cm_err": mean_val(recs, "stage0_thres20cm_err"),
            "stage0_thres50cm_err": mean_val(recs, "stage0_thres50cm_err"),
            "stage0_thres20cm_acc": mean_val(recs, "stage0_thres20cm_acc"),
            "stage0_thres50cm_acc": mean_val(recs, "stage0_thres50cm_acc"),
            "status": status
        })

    # 核心：按 S0 MAE 从大到小降序排列！找出害群之马
    region_summary_list = sorted(region_summary_list, key=lambda x: x['stage0_mae'], reverse=True)

    return global_summary, region_summary_list, sample_records


def generate_markdown_report(split_results, out_dir):
    """
    生成结构清晰、带有对比与诊断结论的终极 Markdown 报告
    """
    lines = []
    lines.append("# ENRICH-Aerial 航空遥感大场景深度精度诊断报告")
    lines.append(f"- **生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- **评估设置**: BN 模式 = `{os.path.basename(os.path.normpath(out_dir))}` | 5 视角 | 包含多尺度深度与平面特征细化\n")

    # --- 1. 如果同时包含 train 和 test，输出跨数据集严格同构对比表 ---
    if "test" in split_results and "train" in split_results:
        t_glob = split_results["train"]["global"]
        e_glob = split_results["test"]["global"]

        s0_gap = e_glob['stage0_mae'] - t_glob['stage0_mae']
        s1_gap = e_glob['stage1_fused_mae'] - t_glob['stage1_fused_mae']
        planar_gap = e_glob['stage0_planar_mae'] - t_glob['stage0_planar_mae']
        t20_gap = (e_glob['stage0_thres20cm_err'] - t_glob['stage0_thres20cm_err']) * 100.0

        lines.append("## 🏆 1. 训练集 vs 测试集 真实差距对比 (Train vs Test Gap)")
        lines.append("> 本表在完全相同的 `eval()`、相同分辨率与相同指标计算下生成，消除训练期统计波动影响。\n")
        lines.append("| 划分数据集 (Split) | 样本切片数 | S0 MAE (m) | S1 融合 MAE (m) | S2 MAE (m) | S3 MAE (m) | S0 平面 MAE (m) | >10cm 误差率 (%) | >20cm 误差率 (%) | >50cm 误差率 (%) |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        lines.append(f"| **训练集 (Train)** | {t_glob['count']} | **{t_glob['stage0_mae']:.4f}m** | {t_glob['stage1_fused_mae']:.4f}m | {t_glob['stage2_mae']:.4f}m | {t_glob['stage3_mae']:.4f}m | **{t_glob['stage0_planar_mae']:.4f}m** | {t_glob['stage0_thres10cm_err']*100:.2f}% | {t_glob['stage0_thres20cm_err']*100:.2f}% | {t_glob['stage0_thres50cm_err']*100:.2f}% |")
        lines.append(f"| **测试集 (Test)**  | {e_glob['count']} | **{e_glob['stage0_mae']:.4f}m** | {e_glob['stage1_fused_mae']:.4f}m | {e_glob['stage2_mae']:.4f}m | {e_glob['stage3_mae']:.4f}m | **{e_glob['stage0_planar_mae']:.4f}m** | {e_glob['stage0_thres10cm_err']*100:.2f}% | {e_glob['stage0_thres20cm_err']*100:.2f}% | {e_glob['stage0_thres50cm_err']*100:.2f}% |")
        lines.append(f"| **真实泛化差距 (Gap)** | - | **{'+' if s0_gap>=0 else ''}{s0_gap*100:.2f}cm** ({s0_gap:+.4f}m) | **{'+' if s1_gap>=0 else ''}{s1_gap*100:.2f}cm** | - | - | **{'+' if planar_gap>=0 else ''}{planar_gap*100:.2f}cm** | - | **{t20_gap:+.2f}%** | - |\n")

    # --- 2. 各划分按区域 (Region) 排序诊断表 ---
    for sp_name, sp_data in split_results.items():
        g = sp_data["global"]
        regs = sp_data["regions"]

        lines.append(f"## 📌 2. [{sp_name.upper()} 划分] 各拍摄组 / 区域 MAE 排序诊断表")
        lines.append(f"- **总切片数**: {g['count']} | **全局 S0 MAE**: **{g['stage0_mae']:.4f}m** | **全局平面 MAE**: **{g['stage0_planar_mae']:.4f}m**")
        lines.append("- **排序规则**: 按 `S0 MAE (m)` 降序排列（最差区域排在最上方）\n")

        lines.append("| 排名 | 拍摄组/区域 (Region) | 切片数 | S0 MAE (m) | 偏离全局均值 | S1 融合 MAE (m) | 平面 MAE (m) | >20cm 误差率 (%) | >50cm 误差率 (%) | 状态诊断 |")
        lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")

        for rank, r in enumerate(regs, start=1):
            sign_dev = f"{'+' if r['dev_pct']>=0 else ''}{r['dev_pct']:.1f}%"
            lines.append(
                f"| {rank} | `{r['region']}` | {r['count']} | **{r['stage0_mae']:.4f}** | "
                f"`{sign_dev}` | {r['stage1_fused_mae']:.4f} | {r['stage0_planar_mae']:.4f} | "
                f"{r['stage0_thres20cm_err']*100:.1f}% | {r['stage0_thres50cm_err']*100:.1f}% | {r['status']} |"
            )

        lines.append(
            f"| - | **全体均值 (Overall)** | **{g['count']}** | **{g['stage0_mae']:.4f}** | "
            f"`0.0%` | **{g['stage1_fused_mae']:.4f}** | **{g['stage0_planar_mae']:.4f}** | "
            f"**{g['stage0_thres20cm_err']*100:.1f}%** | **{g['stage0_thres50cm_err']*100:.1f}%** | - |\n"
        )

        # 深入诊断结论计算
        s0_maes = [r['stage0_mae'] for r in regs]
        max_mae = max(s0_maes)
        min_mae = min(s0_maes)
        cv = (np.std(s0_maes) / np.mean(s0_maes) * 100.0) if np.mean(s0_maes) > 0 else 0.0

        worst_regs = [r['region'] for r in regs if r['dev_pct'] > 25.0]

        lines.append("### 🔍 区域误差特征诊断结论：")
        lines.append(f"- **极值分布**: 最差区域 `{regs[0]['region']}` ({max_mae:.4f}m) vs 最佳区域 `{regs[-1]['region']}` ({min_mae:.4f}m)，极值差为 **{(max_mae - min_mae):.4f}m**。")
        lines.append(f"- **区域离散度 (变异系数 CV)**: **{cv:.1f}%**")

        if len(worst_regs) > 0 and cv >= 25.0:
            lines.append(f"- **诊断结论**: 🚨 **存在少数异常拉高区域**！区域 `{', '.join(worst_regs)}` 的误差显著偏高，极大地抬高了测试集的整体平均误差；其余多数区域表现明显好于平均线。")
        elif cv < 18.0:
            lines.append("- **诊断结论**: ⚖️ **误差呈现全域普遍性分布**！各拍摄区域之间的误差起伏很小（CV < 18%），说明该误差水平属于全局模型在当前航高、分辨率和视角基线下的统一表现，而非个别恶劣切片所致。")
        else:
            lines.append("- **诊断结论**: 📊 **呈现中度离散分布**。多数区域表现稳定，部分复杂建筑群区域误差相对偏高。")
        lines.append("\n" + "-" * 70 + "\n")

    report_text = "\n".join(lines)
    md_path = os.path.join(out_dir, "test_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text, md_path


def parse_args():
    parser = argparse.ArgumentParser(description="ENRICH-Aerial 航空遥感大场景深度精度诊断与区域 MAE 排序系统")

    parser.add_argument('--dataset', default='enrich_aerial', type=str, help='数据集定义名称')
    parser.add_argument('--testpath', default="/home/myao/ENRICH-Aerial_Cropped_0.6x_Overlap50/", type=str, help='ENRICH 数据集根目录')
    parser.add_argument('--trainlist', default="lists/enrich_aerial/train.txt", type=str, help='训练集划分列表文件')
    parser.add_argument('--testlist', default="lists/enrich_aerial/test.txt", type=str, help='测试集划分列表文件')
    parser.add_argument('--loadckpt', default="./checkpoints/tensorboard_train_enrich_aerial/model_000033.ckpt", type=str, help='待评测权重路径')
    parser.add_argument('--outdir', default="./outputs_enrich_aerial_crop", type=str, help='输出目录')

    parser.add_argument('--split', default='both', choices=['test', 'train', 'both'], help='评估划分: test, train, 或 both (对比两方)')
    parser.add_argument('--batch_size', type=int, default=2, help='测试批次大小 (严格对齐训练 batch_size=2)')
    parser.add_argument('--n_views', type=int, default=5, help='测试视角数 (1 参 4 源，对齐训练 5 视角)')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader 线程数')
    parser.add_argument('--seed', type=int, default=1, help='随机种子')

    parser.add_argument('--bn_mode', default='eval', choices=['eval', 'batch', 'recalib'],
                        help='BN 诊断模式: eval(标准 running stats) | batch(BN 用当前批次统计量，等价训练期前向) | recalib(用训练集重估 running stats 后再 eval)')
    parser.add_argument('--recalib_batches', type=int, default=200, help='recalib 模式下用于重估 BN 的训练 batch 数')

    # Cross-Check 阈值参数透传
    parser.add_argument('--pre_base_depth_thresh', type=float, default=0.20, help='Pre-GNN Cross-Check 基础深度容限 (m)')
    parser.add_argument('--pre_pixel_dist_thresh', type=float, default=1.2, help='Pre-GNN Cross-Check 2D像素距离容限 (px)')
    parser.add_argument('--post_base_depth_thresh', type=float, default=0.16, help='Post-GNN Cross-Check 基础深度容限 (m)')
    parser.add_argument('--post_pixel_dist_thresh', type=float, default=1.2, help='Post-GNN Cross-Check 2D像素距离容限 (px)')

    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print("=" * 85)
    print("ENRICH-Aerial_Data 航空遥感大场景深度精度诊断评测系统")
    print("  - 评估模式:  ", args.split.upper())
    print("  - 数据集路径:", args.testpath)
    print("  - 权重文件:  ", args.loadckpt)
    print("  - BatchSize: ", args.batch_size)
    print("  - 视角数量:  ", args.n_views)
    print("=" * 85)

    os.makedirs(args.outdir, exist_ok=True)

    # 1. 自动寻找 fallback checkpoint
    ckpt_path = args.loadckpt
    if not os.path.exists(ckpt_path):
        candidates = [
            "./checkpoints/tensorboard_train_enrich_aerial/model_000033.ckpt",
            "./checkpoints/tensorboard_enrich_ours_1/model_000033.ckpt",
            "./checkpoints/model_000033.ckpt",
            "./checkpoints/finalmodel.ckpt"
        ]
        found = False
        for c in candidates:
            if os.path.exists(c):
                ckpt_path = c
                found = True
                break
        if not found:
            # 搜索任意 .ckpt
            if os.path.exists("./checkpoints"):
                for root, _, files in os.walk("./checkpoints"):
                    for f in files:
                        if f.endswith(".ckpt"):
                            ckpt_path = os.path.join(root, f)
                            found = True
                            break
                    if found:
                        break

    # 2. 实例化模型
    model = PatchmatchNet(
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

    load_checkpoint(model, ckpt_path)

    # BN 诊断模式配置（三种模式的结果分别写入 outdir/<bn_mode>/）
    args.outdir = os.path.join(args.outdir, args.bn_mode)
    os.makedirs(args.outdir, exist_ok=True)
    print(f"[BN] 当前 BN 诊断模式: {args.bn_mode} | 输出目录: {args.outdir}")
    if args.bn_mode == 'eval':
        model.eval()
    elif args.bn_mode == 'batch':
        set_bn_batch_stats(model)
    elif args.bn_mode == 'recalib':
        recalibrate_bn(model, args.dataset, args.testpath, args.trainlist, args)

    # 3. 按需求执行评测
    split_results = {}

    splits_to_run = []
    if args.split in ['test', 'both']:
        splits_to_run.append(('test', args.testlist))
    if args.split in ['train', 'both']:
        splits_to_run.append(('train', args.trainlist))

    for sp_name, sp_list in splits_to_run:
        glob, regs, recs = evaluate_dataset_split(model, args.dataset, args.testpath, sp_list, sp_name, args)
        split_results[sp_name] = {
            "global": glob,
            "regions": regs,
            "records": recs
        }

    # 4. 生成 Markdown 与控制台全景报告
    report_text, md_path = generate_markdown_report(split_results, args.outdir)
    print("\n" + report_text)
    print(f"✓ 完整 Markdown 评测报告已保存至: {md_path}")

    # 保存 JSON
    json_path = os.path.join(args.outdir, "metrics_diagnostic.json")
    json_out = {}
    for sp_name, sp_data in split_results.items():
        json_out[sp_name] = {
            "global": sp_data["global"],
            "regions": sp_data["regions"]
        }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2)
    print(f"✓ 结构化诊断指标已保存至: {json_path}")
    print("=" * 85)


if __name__ == '__main__':
    main()
