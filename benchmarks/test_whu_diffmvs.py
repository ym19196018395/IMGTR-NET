import argparse
import os
import sys
import time
import json
import math
import cv2
import numpy as np

# 确保项目根目录在 sys.path 中，以便无缝导入 datasets, utils 等模块
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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
        description='WHU-MVS DiffMVS / CasDiffMVS (TPAMI 2025) 独立基准评测适配脚本 (遵循 whu-mvs-benchmark-adapter 规范)'
    )
    # 数据集与路径配置
    parser.add_argument('--dataset', default='dtu_whu', help='select dataset (default: dtu_whu)')
    parser.add_argument('--testpath', default='/home/ym/Experiment/Datas/WHU_MVS_dataset',
                        help='WHU dataset root path')
    parser.add_argument('--testlist', default='lists/whu/newtest.txt',
                        help='testing scan list file (e.g. lists/whu/minitest.txt or newtest.txt)')
    parser.add_argument('--loadckpt', required=True, help='path to DiffMVS / CasDiffMVS checkpoint (.ckpt)')
    parser.add_argument('--diffmvs_code_dir', default='',
                        help='path to DiffMVS source code directory (e.g. /home/myao/diffmvs-main)')
    parser.add_argument('--mask_dir', default='./outputs_minitest',
                        help='directory containing 同源 stage0_planar_mask_{file_id}.png (for Table 2)')
    parser.add_argument('--outdir', default='./outputs_diffmvs',
                        help='output directory to save reports and depth predictions')

    # 批次与硬件设置
    parser.add_argument('--batch_size', type=int, default=1, help='testing batch size (recommend 1 to prevent OOM)')
    parser.add_argument('--n_views', type=int, default=5, help='number of views per sample')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers for DataLoader')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--save_depth', action='store_true', default=False,
                        help='save predicted depth maps in .pfm and 16-bit .png format')

    # DiffMVS / CasDiffMVS 核心超参数 (与官方 test_dtu_casdiffmvs.sh 完全对齐)
    parser.add_argument('--numdepth_initial', type=int, default=48,
                        help='number of depth samples in depth initialization')
    parser.add_argument('--numdepth', type=int, default=384,
                        help='1.0/numdepth is the sampling interval in inverse depth space')
    parser.add_argument('--ddim_eta', nargs="+", type=float, default=[0.0, 1.0, 1.0],
                        help='eta for ddim')
    parser.add_argument('--scale', nargs="+", type=float, default=[0.0, 0.5, 0.1],
                        help='scale of noise in diffusion')
    parser.add_argument('--timesteps', nargs="+", type=int, default=[1000, 1000, 1000],
                        help='total diffusion timesteps')
    parser.add_argument('--sampling_timesteps', nargs="+", type=int, default=[0, 1, 1],
                        help='DDIM sampling timesteps')
    parser.add_argument('--hidden_dim', nargs="+", type=int, default=[0, 32, 20],
                        help='feature dimension of hidden states for each stage')
    parser.add_argument('--context_dim', nargs="+", type=int, default=[32, 32, 16],
                        help='context dimension for each stage')
    parser.add_argument('--stage_iters', nargs="+", type=int, default=[1, 3, 3],
                        help='diffusion update iterations for each stage')
    parser.add_argument('--cost_dim_stage', nargs="+", type=int, default=[4, 4, 4],
                        help='feature dimension of group-wise correlation for each stage')
    parser.add_argument('--CostNum', nargs="+", type=int, default=[0, 4, 4],
                        help='number of new samples in each diffusion timestep')
    parser.add_argument('--unet_dim', nargs="+", type=int, default=[0, 16, 8],
                        help='base feature dimension of unet for each stage')
    parser.add_argument('--min_radius', type=float, default=0.125,
                        help='min scale factor for sampling radius')
    parser.add_argument('--max_radius', type=float, default=8.0,
                        help='max scale factor for sampling radius')
    parser.add_argument('--conf_weight', type=float, default=0.05,
                        help='weight for confidence learning')
    parser.add_argument('--depth_interals_ratio', nargs="+", type=float, default=[4.0, 2.0, 1.0],
                        help='sampling interval ratio of inverse depth across stages')

    return parser.parse_args()


class DiffMVSWrapper(nn.Module):
    """
    WHU-MVS 基准测试适配包装器：将 WHU DataLoader 输出无缝转换为 CasDiffMVS 输入
    遵循 whu-mvs-benchmark-adapter 专家规范 (Mode A: 包装器模式)
    """
    def __init__(self, casdiffmvs_class, args):
        super().__init__()
        self.numdepth = args.numdepth
        depth_interals_ratio = [int(r) if r.is_integer() else r for r in args.depth_interals_ratio]
        self.net = casdiffmvs_class(
            args=args,
            depth_interals_ratio=depth_interals_ratio,
            test=True
        )

    def forward(self, sample_cuda, n_views=5):
        B = sample_cuda["depth_min"].shape[0]
        dev = sample_cuda["depth_min"].device

        # 1. 转换图像列表: List[B, 3, H, W] (WHU stage_0 为全分辨率原图 [0, 1] 归一化)
        imgs = [sample_cuda["imgs"]["stage_0"][:, i] for i in range(n_views)]

        # 2. 构造视差空间线性采样网格 (Disparity Linear Sampling)
        disp_min = 1.0 / sample_cuda["depth_max"].view(B, 1).float()
        disp_max = 1.0 / sample_cuda["depth_min"].view(B, 1).float()
        t = torch.linspace(0.0, 1.0, steps=self.numdepth, device=dev, dtype=torch.float32).view(1, -1)
        depth_values = disp_min + t * (disp_max - disp_min)  # [B, numdepth]

        # 3. 构造 DiffMVS 多阶段投影矩阵字典 [B, N, 2, 4, 4]
        stage_mapping = [
            ("stage1", "stage_3"),  # 1/8 粗阶段
            ("stage2", "stage_2"),  # 1/4
            ("stage3", "stage_1"),  # 1/2
            ("stage4", "stage_0"),  # 1/1 原图
        ]

        proj_matrices_dict = {}
        for diff_st, whu_st in stage_mapping:
            P = sample_cuda["proj_matrices"][whu_st]       # [B, N, 4, 4]
            K = sample_cuda["intrinsics_mats"][whu_st]     # [B, N, 3, 3]

            K_inv = torch.inverse(K)
            extrinsic = torch.eye(4, device=dev, dtype=torch.float32).repeat(B, n_views, 1, 1)
            extrinsic[:, :, :3, :4] = torch.matmul(K_inv, P[:, :, :3, :4])

            proj_mat = torch.zeros(B, n_views, 2, 4, 4, device=dev, dtype=torch.float32)
            proj_mat[:, :, 0, :4, :4] = extrinsic
            proj_mat[:, :, 1, :3, :3] = K

            proj_matrices_dict[diff_st] = proj_mat

        # 4. DiffMVS 前向推理 (test=True 模式)
        outputs = self.net(
            imgs=imgs,
            proj_matrices=proj_matrices_dict,
            depth_values=depth_values
        )

        # 5. 提取最高分辨率 (Stage 0, 1/1) 与半分辨率 (Stage 1, 1/2) 预测深度图 (以绝对物理米 meters 为单位)
        depth_pred_s0 = outputs["depth"][-1]
        if depth_pred_s0.dim() == 3:
            depth_pred_s0 = depth_pred_s0.unsqueeze(1)  # [B, 1, H, W]

        # 动态定位 Stage 1 (1/2 分辨率) 预测深度图
        H_s0, W_s0 = depth_pred_s0.shape[-2], depth_pred_s0.shape[-1]
        target_h1, target_w1 = H_s0 // 2, W_s0 // 2
        depth_pred_s1 = None
        for d in reversed(outputs["depth"]):
            d_h, d_w = d.shape[-2], d.shape[-1]
            if d_h == target_h1 and d_w == target_w1:
                depth_pred_s1 = d.unsqueeze(1) if d.dim() == 3 else d
                break

        # 提取全分辨率置信度 (若可用)
        conf_list = outputs.get("photometric_confidence", [])
        confidence = conf_list[-1].unsqueeze(1) if len(conf_list) > 0 else None

        return depth_pred_s0, depth_pred_s1, confidence


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
    os.chdir(PROJECT_ROOT)
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print("=" * 85)
    print("WHU-MVS 基准适配评估系统: DiffMVS / CasDiffMVS (IEEE TPAMI 2025)")
    print("=" * 85)
    print_args(args)

    os.makedirs(args.outdir, exist_ok=True)

    # 1. 动态导入外部 DiffMVS 源码
    diffmvs_root = os.path.abspath(args.diffmvs_code_dir) if args.diffmvs_code_dir else ""
    if not (diffmvs_root and os.path.isdir(diffmvs_root)):
        candidates = [
            "/home/myao/diffmvs-main",
            "/home/ym/Experiment/diffmvs-main",
            os.path.abspath(os.path.join(PROJECT_ROOT, "..", "diffmvs-main")),
        ]
        for c in candidates:
            if os.path.isdir(c):
                diffmvs_root = c
                print(f"[Auto-Detect] 自动探测并挂载 DiffMVS 源码路径: {diffmvs_root}")
                break

    if diffmvs_root and os.path.isdir(diffmvs_root):
        if diffmvs_root not in sys.path:
            sys.path.insert(0, diffmvs_root)
        print(f"[Import] 已成功挂载 DiffMVS 代码路径: {diffmvs_root}")
    else:
        print(f"[Warning] 未指定或未找到 --diffmvs_code_dir: {diffmvs_root}，尝试从系统环境导入...")

    try:
        from models.diffusion import CasDiffMVS
    except ImportError as e:
        raise ImportError(
            f"🚨 无法导入 CasDiffMVS 模型！请确认 --diffmvs_code_dir 参数指向正确的 DiffMVS 源码目录。\n"
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
    model = DiffMVSWrapper(
        casdiffmvs_class=CasDiffMVS,
        args=args
    ).to(device)

    print(f"[Model] 载入 Checkpoint 权重: {args.loadckpt}")
    checkpoint = torch.load(args.loadckpt, map_location=device)
    state_dict_model = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    cleaned_state = {}
    for k, v in state_dict_model.items():
        clean_k = k[7:] if k.startswith('module.') else k
        cleaned_state[clean_k] = v

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
            depth_est_s0, depth_est_s1, confidence = model(sample_cuda, n_views=args.n_views)

            depth_gt_s0 = sample_cuda["depth"]["stage_0"]  # [B, 1, H, W]
            mask_gt_s0 = (sample_cuda["mask"]["stage_0"] > 0.5)

            depth_gt_s1 = sample_cuda["depth"]["stage_1"]  # [B, 1, H//2, W//2]
            mask_gt_s1 = (sample_cuda["mask"]["stage_1"] > 0.5)

            for b in range(B):
                meta_idx = batch_idx * args.batch_size + b
                scan, file_id = test_dataset.metas[meta_idx]

                if scan not in per_scan_records:
                    per_scan_records[scan] = []

                m_b_s0 = mask_gt_s0[b:b+1]
                d_est_b_s0 = depth_est_s0[b:b+1]
                d_gt_b_s0 = depth_gt_s0[b:b+1]

                # 读取同源平面切片掩码 (由本项目提前导出的同源 stage0_planar_mask_{file_id}.png 与 stage1_planar_mask_{file_id}.png)
                planar_mask_s0_file = os.path.join(args.mask_dir, scan, "masks", f"stage0_planar_mask_{file_id}.png")
                has_planar_mask_s0 = False
                if os.path.exists(planar_mask_s0_file):
                    m_planar_np_s0 = cv2.imread(planar_mask_s0_file, cv2.IMREAD_GRAYSCALE)
                    if m_planar_np_s0 is not None:
                        m_planar_t_s0 = (torch.from_numpy(m_planar_np_s0).to(device) > 128).unsqueeze(0).unsqueeze(0)
                        m_planar_b_s0 = m_b_s0 & m_planar_t_s0
                        m_curved_b_s0 = m_b_s0 & (~m_planar_t_s0)
                        has_planar_mask_s0 = True

                if not has_planar_mask_s0:
                    m_planar_b_s0 = torch.zeros_like(m_b_s0)
                    m_curved_b_s0 = m_b_s0

                # 计算 Stage 0 绝对深度误差与阈值比例
                s0_mae = safe_mae(d_est_b_s0, d_gt_b_s0, m_b_s0)
                s0_planar_mae = safe_mae(d_est_b_s0, d_gt_b_s0, m_planar_b_s0) if has_planar_mask_s0 else float('nan')
                s0_curved_mae = safe_mae(d_est_b_s0, d_gt_b_s0, m_curved_b_s0) if has_planar_mask_s0 else float('nan')

                t1_err = safe_thres_error(d_est_b_s0, d_gt_b_s0, m_b_s0, 1.0)
                t2_err = safe_thres_error(d_est_b_s0, d_gt_b_s0, m_b_s0, 2.0)
                t4_err = safe_thres_error(d_est_b_s0, d_gt_b_s0, m_b_s0, 4.0)
                t8_err = safe_thres_error(d_est_b_s0, d_gt_b_s0, m_b_s0, 8.0)

                valid_cnt_s0 = m_b_s0.sum().item()
                planar_cnt_s0 = m_planar_b_s0.sum().item()
                planar_ratio_s0 = (planar_cnt_s0 / valid_cnt_s0 * 100.0) if valid_cnt_s0 > 0 else 0.0

                # --- 计算 Stage 1 平面与全局指标 ---
                s1_mae = float('nan')
                s1_planar_mae = float('nan')
                planar_ratio_s1 = 0.0
                if depth_est_s1 is not None:
                    m_b_s1 = mask_gt_s1[b:b+1]
                    d_est_b_s1 = depth_est_s1[b:b+1]
                    d_gt_b_s1 = depth_gt_s1[b:b+1]

                    planar_mask_s1_file = os.path.join(args.mask_dir, scan, "masks", f"stage1_planar_mask_{file_id}.png")
                    has_planar_mask_s1 = False
                    if os.path.exists(planar_mask_s1_file):
                        m_planar_np_s1 = cv2.imread(planar_mask_s1_file, cv2.IMREAD_GRAYSCALE)
                        if m_planar_np_s1 is not None:
                            m_planar_t_s1 = (torch.from_numpy(m_planar_np_s1).to(device) > 128).unsqueeze(0).unsqueeze(0)
                            m_planar_b_s1 = m_b_s1 & m_planar_t_s1
                            has_planar_mask_s1 = True

                    s1_mae = safe_mae(d_est_b_s1, d_gt_b_s1, m_b_s1)
                    if has_planar_mask_s1:
                        s1_planar_mae = safe_mae(d_est_b_s1, d_gt_b_s1, m_planar_b_s1)
                        valid_cnt_s1 = m_b_s1.sum().item()
                        planar_cnt_s1 = m_planar_b_s1.sum().item()
                        planar_ratio_s1 = (planar_cnt_s1 / valid_cnt_s1 * 100.0) if valid_cnt_s1 > 0 else 0.0

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
                    "s0_valid_pixels": valid_cnt_s0,
                    "s0_planar_pixels": planar_cnt_s0,
                    "s0_planar_ratio_pct": planar_ratio_s0,
                    "stage1_mae": s1_mae,
                    "stage1_planar_mae": s1_planar_mae,
                    "s1_planar_ratio_pct": planar_ratio_s1,
                }
                per_scan_records[scan].append(record)
                all_sample_records.append(record)

                # (可选) 保存预测深度图
                if args.save_depth:
                    depth_save_dir = os.path.join(args.outdir, scan, "depths")
                    os.makedirs(depth_save_dir, exist_ok=True)
                    d_s0_np = d_est_b_s0[0, 0].cpu().numpy()
                    save_pfm(os.path.join(depth_save_dir, f"diffmvs_depth_{file_id}.pfm"), d_s0_np)
                    depth_png_s0 = np.clip(d_s0_np * 64.0, 0, 65535).astype(np.uint16)
                    cv2.imwrite(os.path.join(depth_save_dir, f"diffmvs_depth_{file_id}.png"), depth_png_s0)

            cur_idx = batch_idx + 1
            step_time = time.time() - step_start
            s0_planar_str = f"{s0_planar_mae:.4f}m" if not math.isnan(s0_planar_mae) else "N/A"
            s1_planar_str = f"{s1_planar_mae:.4f}m" if not math.isnan(s1_planar_mae) else "N/A"
            print(f"[{cur_idx:03d}/{len(test_loader):03d}] Scan: {scan} | Img: {file_id} | "
                  f"DiffMVS S0 MAE: {s0_mae:.4f}m | S0 Planar: {s0_planar_str} | S1 Planar: {s1_planar_str} | Time: {step_time:.2f}s")

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
            "stage1_mae": compute_mean_ignore_nan(recs, "stage1_mae"),
            "stage1_planar_mae": compute_mean_ignore_nan(recs, "stage1_planar_mae"),
            "s1_planar_ratio_pct": compute_mean_ignore_nan(recs, "s1_planar_ratio_pct"),
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
        "stage1_mae": compute_mean_ignore_nan(all_sample_records, "stage1_mae"),
        "stage1_planar_mae": compute_mean_ignore_nan(all_sample_records, "stage1_planar_mae"),
        "s1_planar_ratio_pct": compute_mean_ignore_nan(all_sample_records, "s1_planar_ratio_pct"),
    }

    report_lines = []
    report_lines.append("# WHU-MVS 基准对比评测报告: DiffMVS / CasDiffMVS (IEEE TPAMI 2025)\n")
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
        f"| **DiffMVS (平均)** | **{overall_summary['count']}** | **{overall_summary['stage0_mae']:.4f}** | "
        f"**{overall_summary['stage0_thres1mm_err']*100:.2f}%** | **{overall_summary['stage0_thres2mm_err']*100:.2f}%** | "
        f"**{overall_summary['stage0_thres4mm_err']*100:.2f}%** | **{overall_summary['stage0_thres8mm_err']*100:.2f}%** |"
    )

    report_lines.append("\n## Table 2: 细分区域精度与平面特性分析 (Regional & Planar Analysis)")
    report_lines.append("| 模型 / Scan | S0 全局 MAE (m) | S0 平面区 MAE (m) | S1 全局 MAE (m) | S1 平面区 MAE (m) | S0 曲面/非平面 MAE (m) | S0 平面占比 (%) | S1 平面占比 (%) |")
    report_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for scan, s in summary_by_scan.items():
        p0_str = f"{s['stage0_planar_mae']:.4f}" if not math.isnan(s['stage0_planar_mae']) else "N/A"
        p1_str = f"{s['stage1_planar_mae']:.4f}" if not math.isnan(s['stage1_planar_mae']) else "N/A"
        s1_m_str = f"{s['stage1_mae']:.4f}" if not math.isnan(s['stage1_mae']) else "N/A"
        c_str = f"{s['stage0_curved_mae']:.4f}" if not math.isnan(s['stage0_curved_mae']) else "N/A"
        report_lines.append(f"| `{scan}` | {s['stage0_mae']:.4f} | {p0_str} | {s1_m_str} | {p1_str} | {c_str} | {s['s0_planar_ratio_pct']:.1f}% | {s['s1_planar_ratio_pct']:.1f}% |")

    op0_str = f"{overall_summary['stage0_planar_mae']:.4f}" if not math.isnan(overall_summary['stage0_planar_mae']) else "N/A"
    op1_str = f"{overall_summary['stage1_planar_mae']:.4f}" if not math.isnan(overall_summary['stage1_planar_mae']) else "N/A"
    os1_m_str = f"{overall_summary['stage1_mae']:.4f}" if not math.isnan(overall_summary['stage1_mae']) else "N/A"
    oc_str = f"{overall_summary['stage0_curved_mae']:.4f}" if not math.isnan(overall_summary['stage0_curved_mae']) else "N/A"
    report_lines.append(
        f"| **DiffMVS (总计)** | **{overall_summary['stage0_mae']:.4f}** | **{op0_str}** | **{os1_m_str}** | **{op1_str}** | **{oc_str}** | **{overall_summary['s0_planar_ratio_pct']:.1f}%** | **{overall_summary['s1_planar_ratio_pct']:.1f}%** |"
    )

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    # 保存报告与指标 JSON
    report_file = os.path.join(args.outdir, "diffmvs_evaluation_report.md")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n[Output] 完整评估报表已保存至: {report_file}")

    json_file = os.path.join(args.outdir, "diffmvs_records.json")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump({"overall": overall_summary, "by_scan": summary_by_scan, "records": all_sample_records}, f, indent=2)
    print(f"[Output] 详细样本指标记录已保存至: {json_file}")


if __name__ == '__main__':
    main()
