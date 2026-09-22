import argparse
import os
import sys
import time
import json
import math
import cv2
import numpy as np

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


def parse_args():
    parser = argparse.ArgumentParser(description="ENRICH-Aerial_Data 航空遥感数据集独立评测系统")

    parser.add_argument('--dataset', default='enrich_aerial', type=str, help='数据集定义名称')
    parser.add_argument('--testpath', default=r"E:\Code\PythonCode\Pytorch\ENRICH-Aerial_Data", type=str, help='ENRICH-Aerial_Data 数据集根目录')
    parser.add_argument('--testlist', default=r"E:\Code\PythonCode\Pytorch\ENRICH-Aerial_Data\scan_list.txt", type=str, help='测试场景列表文件')
    parser.add_argument('--loadckpt', default="./checkpoints/params.ckpt", type=str, help='待评测模型 Checkpoint 权重路径')
    parser.add_argument('--outdir', default="./outputs_enrich_aerial", type=str, help='评测指标与预测深度图保存目录')

    parser.add_argument('--batch_size', type=int, default=1, help='测试批次大小 (默认 1)')
    parser.add_argument('--n_views', type=int, default=3, help='测试视角数 (1 参 2 源，严格 3 视角)')
    parser.add_argument('--num_workers', type=int, default=2, help='DataLoader 线程数')
    parser.add_argument('--seed', type=int, default=123, help='随机种子')

    parser.add_argument('--eval_mode', type=str, default='all', choices=['all', 'global'],
                        help='评测模式: all(全图多尺度 + 细分平面/GNN增益), global(仅全图多尺度)')

    parser.add_argument('--save_depth', action='store_true', default=True, help='保存预测深度图 (PFM 与 PNG)')
    parser.add_argument('--no_depth', dest='save_depth', action='store_false', help='不保存预测深度图')

    parser.add_argument('--save_masks', action='store_true', default=True, help='保存同源平面掩码 (供基线模型对齐评测)')
    parser.add_argument('--no_masks', dest='save_masks', action='store_false', help='不保存同源平面掩码')

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
        propagation_range=[1, 2, 3],
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

                # 保存同源平面掩码 (供 CasMVSNet / DiffMVS 直接读取对比)
                if args.save_masks:
                    mask_save_dir = os.path.join(args.outdir, scan, "masks")
                    os.makedirs(mask_save_dir, exist_ok=True)
                    s0_mask_np = (eval_mask_s0_planar[b, 0].cpu().numpy().astype(np.uint8)) * 255
                    cv2.imwrite(os.path.join(mask_save_dir, f"stage0_planar_mask_{file_id}.png"), s0_mask_np)
                    s1_mask_np = (eval_mask_s1_planar[b, 0].cpu().numpy().astype(np.uint8)) * 255
                    cv2.imwrite(os.path.join(mask_save_dir, f"stage1_planar_mask_{file_id}.png"), s1_mask_np)

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
