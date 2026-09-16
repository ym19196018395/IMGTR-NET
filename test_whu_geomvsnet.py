import argparse
import os
import sys
import time
import json
import math
import cv2
import numpy as np

# 控制 GPU ID：优先使用外部环境变量 GPU_ID，其次 CUDA_VISIBLE_DEVICES，默认 '0'
_gpu_choice = os.environ.get('GPU_ID', os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
os.environ['CUDA_VISIBLE_DEVICES'] = str(_gpu_choice)

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from datasets import find_dataset_def
from datasets.dtu_whu import collate_keep_list
from datasets.data_io import save_pfm
from utils import print_args, tocuda

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(
        description='WHU-MVS GeoMVSNet 独立基准评测适配脚本 (遵循 whu-mvs-benchmark-adapter 规范)'
    )
    # 数据集与路径配置
    parser.add_argument('--dataset', default='dtu_whu', help='select dataset (default: dtu_whu)')
    parser.add_argument('--testpath', default='/home/ym/Experiment/Datas/WHU_MVS_dataset',
                        help='WHU dataset root path')
    parser.add_argument('--testlist', default='lists/whu/minitest.txt',
                        help='testing scan list file (e.g. lists/whu/minitest.txt or newtest.txt)')
    parser.add_argument('--loadckpt', required=True, help='path to GeoMVSNet checkpoint (.ckpt)')
    parser.add_argument('--geomvsnet_code_dir', default='',
                        help='path to GeoMVSNet source code directory')
    parser.add_argument('--mask_dir', default='./outputs_minitest',
                        help='directory containing 同源 stage0_planar_mask_{file_id}.png (for Table 2)')
    parser.add_argument('--outdir', default='./outputs_geomvsnet',
                        help='output directory to save reports and depth predictions')

    # 批次与硬件设置
    parser.add_argument('--batch_size', type=int, default=1, help='testing batch size (recommend 1 to prevent OOM)')
    parser.add_argument('--n_views', type=int, default=5, help='number of views per sample')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers for DataLoader')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--save_depth', action='store_true', default=False,
                        help='save predicted depth maps in .pfm and 16-bit .png format')

    # GeoMVSNet 网络超参数 (默认对应其官方 4 级级联配置)
    parser.add_argument('--levels', type=int, default=4, help='levels of cascade stages')
    parser.add_argument('--hypo_plane_num_stages', nargs='+', type=int, default=[48, 32, 16, 8],
                        help='number of depth hypothesis planes for stages 1 to 4')
    parser.add_argument('--depth_interal_ratio_stages', nargs='+', type=float, default=[2.0, 1.0, 0.5, 0.25],
                        help='depth interval ratio for stages 1 to 4')
    parser.add_argument('--feat_base_channel', type=int, default=8, help='base channels of FPN')
    parser.add_argument('--reg_base_channel', type=int, default=8, help='base channels of 2D RegNet')
    parser.add_argument('--group_cor_dim_stages', nargs='+', type=int, default=[8, 8, 8, 4],
                        help='group correlation dimensions for stages 1 to 4')

    return parser.parse_args()


class GeoMVSNetWrapper(nn.Module):
    """
    WHU-MVS 基准测试适配包装器：将 WHU DataLoader 输出无缝转换为 GeoMVSNet 输入
    遵循 whu-mvs-benchmark-adapter 专家规范 (Mode A: 包装器模式)
    """
    def __init__(self, geomvsnet_class, levels=4, hypo_plane_num_stages=[48, 32, 16, 8],
                 depth_interal_ratio_stages=[2.0, 1.0, 0.5, 0.25],
                 feat_base_channel=8, reg_base_channel=8,
                 group_cor_dim_stages=[8, 8, 8, 4]):
        super().__init__()
        self.net = geomvsnet_class(
            levels=levels,
            hypo_plane_num_stages=hypo_plane_num_stages,
            depth_interal_ratio_stages=depth_interal_ratio_stages,
            feat_base_channel=feat_base_channel,
            reg_base_channel=reg_base_channel,
            group_cor_dim_stages=group_cor_dim_stages
        )

    def forward(self, sample_cuda, n_views=5):
        B = sample_cuda["depth_min"].shape[0]
        device = sample_cuda["depth_min"].device

        # 1. 转换图像列表: List[B, 3, H, W]
        # WHU DataLoader stage_0 是全分辨率原图 [B, N, 3, H, W]
        imgs = [sample_cuda["imgs"]["stage_0"][:, i] for i in range(n_views)]

        # 2. 构造绝对米制深度范围: [B, 2]
        depth_values = torch.stack([
            sample_cuda["depth_min"].view(B).float(),
            sample_cuda["depth_max"].view(B).float()
        ], dim=-1)

        # 3. 构造 GeoMVSNet 多阶段投影矩阵与内参字典
        # 层级对应关系：GeoMVSNet stage1~stage4 <-> WHU stage_3~stage_0 (从粗到细)
        stage_mapping = [
            ("stage1", "stage_3"),  # 1/8 粗阶段
            ("stage2", "stage_2"),  # 1/4
            ("stage3", "stage_1"),  # 1/2
            ("stage4", "stage_0"),  # 1/1 原图
        ]

        proj_matrices_dict = {}
        intrinsics_dict = {}

        for geo_st, whu_st in stage_mapping:
            P = sample_cuda["proj_matrices"][whu_st]       # [B, N, 4, 4]
            K = sample_cuda["intrinsics_mats"][whu_st]     # [B, N, 3, 3]

            # 精确代数求解外参: [R|t] = K^{-1} @ P[:3, :4]
            K_inv = torch.inverse(K)
            extrinsic = torch.eye(4, device=device, dtype=torch.float32).repeat(B, n_views, 1, 1)
            extrinsic[:, :, :3, :4] = torch.matmul(K_inv, P[:, :, :3, :4])

            # 封装为 GeoMVSNet 期望的 [B, N, 2, 4, 4] 打包格式
            proj_mat_geo = torch.zeros(B, n_views, 2, 4, 4, device=device, dtype=torch.float32)
            proj_mat_geo[:, :, 0, :4, :4] = extrinsic
            proj_mat_geo[:, :, 1, :3, :3] = K

            proj_matrices_dict[geo_st] = proj_mat_geo
            intrinsics_dict[geo_st] = K[:, 0]  # 参考视角内参 [B, 3, 3]

        # 4. GeoMVSNet 前向推理
        outputs = self.net(
            imgs=imgs,
            proj_matrices=proj_matrices_dict,
            intrinsics_matrices=intrinsics_dict,
            depth_values=depth_values
        )

        # 5. 提取最高分辨率 Stage 4 预测深度 (以米为单位)
        # GeoMVSNet outputs 结构: outputs["stage4"]["depth"] 形状为 [B, H, W]
        depth_pred = outputs["stage4"]["depth"].unsqueeze(1)  # [B, 1, H, W]
        confidence = outputs["stage4"]["photometric_confidence"].unsqueeze(1) # [B, 1, H, W]
        return depth_pred, confidence


def safe_mae(depth_est, depth_gt, mask):
    """安全计算有效区域内的绝对深度误差 MAE (m)"""
    if mask is None or not mask.any():
        return float('nan')
    est_valid = depth_est[mask]
    gt_valid = depth_gt[mask]
    return torch.mean((est_valid - gt_valid).abs()).item()


def safe_thres_error(depth_est, depth_gt, mask, thres):
    """安全计算误差大于特定物理阈值（如 1m, 2m, 4m, 8m）的离群点比例"""
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

    print("=" * 85)
    print("WHU-MVS 基准适配评估系统: GeoMVSNet (CVPR 2023)")
    print("=" * 85)
    print_args(args)

    os.makedirs(args.outdir, exist_ok=True)

    # 1. 动态导入外部 GeoMVSNet 源码
    geomvsnet_root = os.path.abspath(args.geomvsnet_code_dir)
    if geomvsnet_root and os.path.isdir(geomvsnet_root):
        if geomvsnet_root not in sys.path:
            sys.path.insert(0, geomvsnet_root)
        print(f"[Import] 已成功挂载 GeoMVSNet 代码路径: {geomvsnet_root}")
    else:
        print(f"[Warning] 未指定或未找到 --geomvsnet_code_dir: {geomvsnet_root}，尝试从本地环境直接导入...")

    try:
        from models.geomvsnet import GeoMVSNet
    except ImportError as e:
        raise ImportError(
            f"🚨 无法导入 GeoMVSNet 模型！请确认 --geomvsnet_code_dir 参数指向正确的 GeoMVSNet 源码目录。\n"
            f"原始错误: {e}"
        )

    # 2. 构建测试数据集与 DataLoader
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

    # 3. 实例化适配包装模型并载入权重
    model = GeoMVSNetWrapper(
        geomvsnet_class=GeoMVSNet,
        levels=args.levels,
        hypo_plane_num_stages=args.hypo_plane_num_stages,
        depth_interal_ratio_stages=args.depth_interal_ratio_stages,
        feat_base_channel=args.feat_base_channel,
        reg_base_channel=args.reg_base_channel,
        group_cor_dim_stages=args.group_cor_dim_stages
    ).to(device)

    print(f"[Model] 载入 Checkpoint 权重: {args.loadckpt}")
    checkpoint = torch.load(args.loadckpt, map_location=device)
    state_dict_model = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    cleaned_state = {}
    for k, v in state_dict_model.items():
        clean_k = k[7:] if k.startswith('module.') else k
        cleaned_state[clean_k] = v

    # 载入底层 GeoMVSNet 权重
    missing, unexpected = model.net.load_state_dict(cleaned_state, strict=False)
    if len(missing) > 0:
        print(f"[Model Warning] 缺失参数 (前10个): {missing[:10]}")
    if len(unexpected) > 0:
        print(f"[Model Warning] 冗余参数 (前10个): {unexpected[:10]}")
    model.eval()

    # 4. 逐样本前向评测
    per_scan_records = {}
    all_sample_records = []
    start_eval_time = time.time()

    print("\n" + "=" * 85)
    print("开始执行逐样本推断与 Table 1 / Table 2 指标计算...")
    print("=" * 85)

    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            step_start = time.time()
            B = sample['imgs']['stage_0'].shape[0]

            skip = ["vertexs", "lines", "triangles", "tri_conf_cleaned", "tri_normal_cleaned", "is_gt_planar"]
            sample_cuda = tocuda(sample, device=device, skip_keys=skip)

            # 前向推理
            depth_est, confidence = model(sample_cuda, n_views=args.n_views) # [B, 1, H, W]

            depth_gt = sample_cuda["depth"]["stage_0"]  # [B, 1, H, W]
            mask_gt = (sample_cuda["mask"]["stage_0"] > 0.5)

            for b in range(B):
                meta_idx = batch_idx * args.batch_size + b
                scan, file_id = test_dataset.metas[meta_idx]

                if scan not in per_scan_records:
                    per_scan_records[scan] = []

                m_b = mask_gt[b:b+1]
                d_est_b = depth_est[b:b+1]
                d_gt_b = depth_gt[b:b+1]

                # 读取同源平面切片掩码 (由本项目提前导出的同源 stage0_planar_mask_{file_id}.png)
                planar_mask_file = os.path.join(args.mask_dir, scan, "masks", f"stage0_planar_mask_{file_id}.png")
                has_planar_mask = False
                if os.path.exists(planar_mask_file):
                    m_planar_np = cv2.imread(planar_mask_file, cv2.IMREAD_GRAYSCALE)
                    if m_planar_np is not None:
                        m_planar_t = (torch.from_numpy(m_planar_np).to(device) > 128).unsqueeze(0).unsqueeze(0)
                        m_planar_b = m_b & m_planar_t
                        m_curved_b = m_b & (~m_planar_t)
                        has_planar_mask = True

                if not has_planar_mask:
                    m_planar_b = torch.zeros_like(m_b)
                    m_curved_b = m_b

                # 计算绝对深度误差与阈值比例
                s0_mae = safe_mae(d_est_b, d_gt_b, m_b)
                s0_planar_mae = safe_mae(d_est_b, d_gt_b, m_planar_b) if has_planar_mask else float('nan')
                s0_curved_mae = safe_mae(d_est_b, d_gt_b, m_curved_b) if has_planar_mask else float('nan')

                t1_err = safe_thres_error(d_est_b, d_gt_b, m_b, 1.0)
                t2_err = safe_thres_error(d_est_b, d_gt_b, m_b, 2.0)
                t4_err = safe_thres_error(d_est_b, d_gt_b, m_b, 4.0)
                t8_err = safe_thres_error(d_est_b, d_gt_b, m_b, 8.0)

                valid_cnt = m_b.sum().item()
                planar_cnt = m_planar_b.sum().item()
                planar_ratio = (planar_cnt / valid_cnt * 100.0) if valid_cnt > 0 else 0.0

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
                    "s0_valid_pixels": valid_cnt,
                    "s0_planar_pixels": planar_cnt,
                    "s0_planar_ratio_pct": planar_ratio,
                }
                per_scan_records[scan].append(record)
                all_sample_records.append(record)

                # (可选) 保存预测深度图
                if args.save_depth:
                    depth_save_dir = os.path.join(args.outdir, scan, "depths")
                    os.makedirs(depth_save_dir, exist_ok=True)
                    d_s0_np = d_est_b[0, 0].cpu().numpy()
                    save_pfm(os.path.join(depth_save_dir, f"geomvsnet_depth_{file_id}.pfm"), d_s0_np)
                    depth_png_s0 = np.clip(d_s0_np * 64.0, 0, 65535).astype(np.uint16)
                    cv2.imwrite(os.path.join(depth_save_dir, f"geomvsnet_depth_{file_id}.png"), depth_png_s0)

            cur_idx = batch_idx + 1
            step_time = time.time() - step_start
            planar_str = f"{s0_planar_mae:.4f}m" if not math.isnan(s0_planar_mae) else "N/A"
            print(f"[{cur_idx:03d}/{len(test_loader):03d}] Scan: {scan} | Img: {file_id} | "
                  f"GeoMVSNet MAE: {s0_mae:.4f}m | Planar MAE: {planar_str} | Time: {step_time:.2f}s")

    eval_duration = time.time() - start_eval_time
    print("\n" + "=" * 85)
    print(f"评测完成！共评估 {len(all_sample_records)} 张图像，总耗时: {eval_duration:.2f} 秒。")
    print("=" * 85)

    # 5. 汇总生成 Markdown 双表 (Table 1 & Table 2)
    def compute_mean_ignore_nan(records, key):
        vals = [r[key] for r in records if key in r and not math.isnan(r[key])]
        return float(np.mean(vals)) if len(vals) > 0 else float('nan')

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
            "s0_planar_ratio_pct": compute_mean_ignore_nan(recs, "s0_planar_ratio_pct"),
        }

    overall_summary = {
        "count": len(all_sample_records),
        "stage0_mae": compute_mean_ignore_nan(all_sample_records, "stage0_mae"),
        "stage0_planar_mae": compute_mean_ignore_nan(all_sample_records, "stage0_planar_mae"),
        "stage0_curved_mae": compute_mean_ignore_nan(all_sample_records, "stage0_curved_mae"),
        "stage0_thres1mm_err": compute_mean_ignore_nan(all_sample_records, "stage0_thres1mm_err"),
        "stage0_thres2mm_err": compute_mean_ignore_nan(all_sample_records, "stage0_thres2mm_err"),
        "stage0_thres4mm_err": compute_mean_ignore_nan(all_sample_records, "stage0_thres4mm_err"),
        "stage0_thres8mm_err": compute_mean_ignore_nan(all_sample_records, "stage0_thres8mm_err"),
        "s0_planar_ratio_pct": compute_mean_ignore_nan(all_sample_records, "s0_planar_ratio_pct"),
    }

    report_lines = []
    report_lines.append("# WHU-MVS 基准对比评测报告: GeoMVSNet (CVPR 2023)\n")
    report_lines.append(f"- **模型权重**: `{args.loadckpt}`")
    report_lines.append(f"- **测试列表**: `{args.testlist}`")
    report_lines.append(f"- **测试图像总量**: {overall_summary['count']} 张")
    report_lines.append(f"- **同源掩码目录**: `{args.mask_dir}`\n")

    report_lines.append("## Table 1: 全图全局深度精度 (Global Depth Accuracy)")
    report_lines.append("| 模型 / Scan | 样本数 | Stage 0 MAE (m) | >1mm 误差率 (%) | >2mm 误差率 (%) | >4mm 误差率 (%) | >8mm 误差率 (%) |")
    report_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for scan, s in summary_by_scan.items():
        report_lines.append(
            f"| `{scan}` | {s['count']} | **{s['stage0_mae']:.4f}** | "
            f"{s['stage0_thres1mm_err']*100:.2f}% | {s['stage0_thres2mm_err']*100:.2f}% | "
            f"{s['stage0_thres4mm_err']*100:.2f}% | {s['stage0_thres8mm_err']*100:.2f}% |"
        )
    report_lines.append(
        f"| **GeoMVSNet (平均)** | **{overall_summary['count']}** | **{overall_summary['stage0_mae']:.4f}** | "
        f"**{overall_summary['stage0_thres1mm_err']*100:.2f}%** | **{overall_summary['stage0_thres2mm_err']*100:.2f}%** | "
        f"**{overall_summary['stage0_thres4mm_err']*100:.2f}%** | **{overall_summary['stage0_thres8mm_err']*100:.2f}%** |"
    )

    report_lines.append("\n## Table 2: 细分区域精度与平面特性分析 (Regional & Planar Analysis)")
    report_lines.append("| 模型 / Scan | S0 全局 MAE (m) | S0 平面区 MAE (m) | S0 曲面/非平面 MAE (m) | 平面像素占比 (%) |")
    report_lines.append("| :--- | :---: | :---: | :---: | :---: |")
    for scan, s in summary_by_scan.items():
        p_str = f"{s['stage0_planar_mae']:.4f}" if not math.isnan(s['stage0_planar_mae']) else "N/A"
        c_str = f"{s['stage0_curved_mae']:.4f}" if not math.isnan(s['stage0_curved_mae']) else "N/A"
        report_lines.append(f"| `{scan}` | {s['stage0_mae']:.4f} | {p_str} | {c_str} | {s['s0_planar_ratio_pct']:.1f}% |")
    
    op_str = f"{overall_summary['stage0_planar_mae']:.4f}" if not math.isnan(overall_summary['stage0_planar_mae']) else "N/A"
    oc_str = f"{overall_summary['stage0_curved_mae']:.4f}" if not math.isnan(overall_summary['stage0_curved_mae']) else "N/A"
    report_lines.append(
        f"| **GeoMVSNet (总计)** | **{overall_summary['stage0_mae']:.4f}** | **{op_str}** | **{oc_str}** | **{overall_summary['s0_planar_ratio_pct']:.1f}%** |"
    )

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    # 保存报告与指标 JSON
    report_file = os.path.join(args.outdir, "geomvsnet_evaluation_report.md")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n[Output] 完整评估报表已保存至: {report_file}")

    json_file = os.path.join(args.outdir, "geomvsnet_records.json")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump({"overall": overall_summary, "by_scan": summary_by_scan, "records": all_sample_records}, f, indent=2)
    print(f"[Output] 详细样本指标记录已保存至: {json_file}")


if __name__ == '__main__':
    main()
