import argparse
import os
import sys
import copy
import time
import json
import math
import cv2
import numpy as np

# 确保项目根目录强制位于 sys.path[0]，杜绝任何外部 baseline 仓库产生命名空间遮蔽
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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
from utils import print_args, tocuda

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="ENRICH-Aerial 航空遥感大场景 DiffMVS / CasDiffMVS (IEEE TPAMI 2025) A/B 模式独立基准评测系统"
    )
    # 数据集与路径配置
    parser.add_argument('--dataset', default='enrich_aerial', type=str, help='数据集定义名称')
    parser.add_argument('--testpath', default="", type=str, help='ENRICH 数据集根目录 (留空自动探查)')
    parser.add_argument('--testlist', default="lists/enrich_aerial/test.txt", type=str, help='测试场景列表文件')
    parser.add_argument('--loadckpt', required=True, help='DiffMVS / CasDiffMVS Checkpoint 权重路径 (.ckpt)')
    parser.add_argument('--diffmvs_code_dir', default='',
                        help='DiffMVS 源码根目录 (如 /home/myao/diffmvs-main)')
    parser.add_argument('--mask_dir', default='./outputs_enrich_aerial',
                        help='同源平面掩码目录 (由 test_enrich_aerial.py 生成的 stage0/1_planar_mask_*.png)')
    parser.add_argument('--outdir', default='./outputs_diffmvs_enrich',
                        help='评测报告与结果保存目录')

    # A/B 诊断模式配置 (与 CasMVSNet / PatchmatchNet 对齐)
    parser.add_argument('--bn_mode', default='both', choices=['both', 'eval', 'batch'],
                        help='BN诊断模式: both(A/B双路对比) | eval(模式A 固化全局统计) | batch(模式B 动态即时统计)')

    # 批次与硬件设置
    parser.add_argument('--batch_size', type=int, default=1, help='测试批次大小 (航测大图建议 1)')
    parser.add_argument('--n_views', type=int, default=3, help='测试视角数 (1 参 2 源，严格 3 视角)')
    parser.add_argument('--num_workers', type=int, default=2, help='DataLoader 线程数')
    parser.add_argument('--seed', type=int, default=123, help='随机种子')
    parser.add_argument('--save_depth', action='store_true', default=False,
                        help='保存预测深度图 (PFM 与 PNG)')

    # DiffMVS / CasDiffMVS 核心超参数 (与官方脚本完全对齐)
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
    ENRICH-Aerial 航测数据基准评测适配包装器：将 DataLoader 输出动态转换为 CasDiffMVS 输入
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

    def forward(self, sample_cuda, n_views=3):
        B = sample_cuda["depth_min"].shape[0]
        dev = sample_cuda["depth_min"].device
        actual_views = sample_cuda["imgs"]["stage_0"].shape[1]
        n_v = min(n_views, actual_views)

        # 1. 转换图像列表: List[B, 3, H, W] (ENRICH stage_0 为全分辨率原图 [0, 1] 归一化)
        imgs = [sample_cuda["imgs"]["stage_0"][:, i] for i in range(n_v)]

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
        for diff_st, enrich_st in stage_mapping:
            P = sample_cuda["proj_matrices"][enrich_st][:, :n_v]       # [B, N, 4, 4]
            K = sample_cuda["intrinsics_mats"][enrich_st][:, :n_v]     # [B, N, 3, 3]

            K_inv = torch.inverse(K)
            extrinsic = torch.eye(4, device=dev, dtype=torch.float32).repeat(B, n_v, 1, 1)
            extrinsic[:, :, :3, :4] = torch.matmul(K_inv, P[:, :, :3, :4])

            proj_mat = torch.zeros(B, n_v, 2, 4, 4, device=dev, dtype=torch.float32)
            proj_mat[:, :, 0, :4, :4] = extrinsic
            proj_mat[:, :, 1, :3, :3] = K

            proj_matrices_dict[diff_st] = proj_mat

        # 4. DiffMVS 前向推理 (test=True 模式)
        outputs = self.net(
            imgs=imgs,
            proj_matrices=proj_matrices_dict,
            depth_values=depth_values
        )

        # 5. 提取各尺度预测深度图 (以绝对物理米 meters 为单位)
        depth_pred_s0 = outputs["depth"][-1]
        if depth_pred_s0.dim() == 3:
            depth_pred_s0 = depth_pred_s0.unsqueeze(1)  # [B, 1, H, W]

        # 动态定位 Stage 1 (1/2 分辨率) 与 Stage 2 (1/4 分辨率) 预测深度图
        H_s0, W_s0 = depth_pred_s0.shape[-2], depth_pred_s0.shape[-1]
        target_h1, target_w1 = H_s0 // 2, W_s0 // 2
        target_h2, target_w2 = H_s0 // 4, W_s0 // 4
        depth_pred_s1 = None
        depth_pred_s2 = None
        for d in reversed(outputs["depth"]):
            d_h, d_w = d.shape[-2], d.shape[-1]
            if depth_pred_s1 is None and d_h == target_h1 and d_w == target_w1:
                depth_pred_s1 = d.unsqueeze(1) if d.dim() == 3 else d
            elif depth_pred_s2 is None and d_h == target_h2 and d_w == target_w2:
                depth_pred_s2 = d.unsqueeze(1) if d.dim() == 3 else d

        # 提取全分辨率置信度 (若可用)
        conf_list = outputs.get("photometric_confidence", [])
        confidence = conf_list[-1].unsqueeze(1) if len(conf_list) > 0 else None

        return depth_pred_s0, depth_pred_s1, depth_pred_s2, confidence


def safe_mae(depth_est, depth_gt, mask):
    """安全计算有效区域内的绝对深度误差 MAE (m)"""
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


def evaluate_single_pass(model, test_loader, mode_tag, device, args):
    """
    针对指定的 BN 模式执行全量测试集前向评测
    """
    print(f"\n>>> 启动评测: [{mode_tag}] (共 {len(test_loader)} 个批次) ...")
    per_scan_records = {}
    all_sample_records = []
    start_eval_time = time.time()

    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            step_start = time.time()
            B = sample['imgs']['stage_0'].shape[0]

            skip = ["vertexs", "lines", "triangles", "scan", "file_id"]
            sample_cuda = tocuda(sample, device=device, skip_keys=skip)

            # 前向推理
            depth_est_s0, depth_est_s1, depth_est_s2, confidence = model(sample_cuda, n_views=args.n_views)

            depth_gt_s0 = sample_cuda["depth"]["stage_0"]
            mask_gt_s0 = (sample_cuda["mask"]["stage_0"] > 0.5)

            depth_gt_s1 = sample_cuda["depth"]["stage_1"]
            mask_gt_s1 = (sample_cuda["mask"]["stage_1"] > 0.5)

            depth_gt_s2 = sample_cuda["depth"]["stage_2"]
            mask_gt_s2 = (sample_cuda["mask"]["stage_2"] > 0.5)

            for b in range(B):
                scan = sample['scan'][b]
                file_id = sample['file_id'][b]
                region = extract_region_name(scan)
                tag = f"{scan}_{file_id}"

                if scan not in per_scan_records:
                    per_scan_records[scan] = []

                m_b_s0 = mask_gt_s0[b:b+1]
                d_est_b_s0 = depth_est_s0[b:b+1]
                d_gt_b_s0 = depth_gt_s0[b:b+1]

                # 读取同源平面切片掩码 (由 test_enrich_aerial.py 保存)
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

                # --- Table 1: 全图多尺度全局深度精度与误差分布 ---
                s0_mae = safe_mae(d_est_b_s0, d_gt_b_s0, m_b_s0)
                s1_mae = safe_mae(depth_est_s1[b:b+1], depth_gt_s1[b:b+1], mask_gt_s1[b:b+1]) if depth_est_s1 is not None else float('nan')
                s2_mae = safe_mae(depth_est_s2[b:b+1], depth_gt_s2[b:b+1], mask_gt_s2[b:b+1]) if depth_est_s2 is not None else float('nan')

                t05_err = safe_thres_error(d_est_b_s0, d_gt_b_s0, m_b_s0, 0.05)
                t10_err = safe_thres_error(d_est_b_s0, d_gt_b_s0, m_b_s0, 0.10)
                t20_err = safe_thres_error(d_est_b_s0, d_gt_b_s0, m_b_s0, 0.20)
                t50_err = safe_thres_error(d_est_b_s0, d_gt_b_s0, m_b_s0, 0.50)

                # --- Table 2: 细分平面区与曲面区精度评测 ---
                s0_planar_mae = safe_mae(d_est_b_s0, d_gt_b_s0, m_planar_b_s0) if has_planar_mask_s0 else float('nan')
                s0_curved_mae = safe_mae(d_est_b_s0, d_gt_b_s0, m_curved_b_s0) if has_planar_mask_s0 else float('nan')

                valid_cnt_s0 = m_b_s0.sum().item()
                planar_cnt_s0 = m_planar_b_s0.sum().item()
                planar_ratio_s0 = (planar_cnt_s0 / valid_cnt_s0 * 100.0) if valid_cnt_s0 > 0 else 0.0

                # Stage 1 平面特性
                s1_planar_mae = float('nan')
                s1_curved_mae = float('nan')
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
                            m_curved_b_s1 = m_b_s1 & (~m_planar_t_s1)
                            has_planar_mask_s1 = True

                    if has_planar_mask_s1:
                        s1_planar_mae = safe_mae(d_est_b_s1, d_gt_b_s1, m_planar_b_s1)
                        s1_curved_mae = safe_mae(d_est_b_s1, d_gt_b_s1, m_curved_b_s1)
                        valid_cnt_s1 = m_b_s1.sum().item()
                        planar_cnt_s1 = m_planar_b_s1.sum().item()
                        planar_ratio_s1 = (planar_cnt_s1 / valid_cnt_s1 * 100.0) if valid_cnt_s1 > 0 else 0.0

                record = {
                    "scan": scan,
                    "region": region,
                    "file_id": file_id,
                    "tag": tag,
                    "s0": s0_mae,
                    "s1": s1_mae,
                    "s2": s2_mae,
                    "err_05cm": t05_err,
                    "err_10cm": t10_err,
                    "err_20cm": t20_err,
                    "err_50cm": t50_err,
                    "s0_planar": s0_planar_mae,
                    "s0_curved": s0_curved_mae,
                    "planar_ratio": planar_ratio_s0,
                    "s1_planar": s1_planar_mae,
                    "s1_curved": s1_curved_mae,
                    "planar_ratio_s1": planar_ratio_s1,
                }

                per_scan_records[scan].append(record)
                all_sample_records.append(record)

                # 保存预测深度图 (以模式目录隔离)
                if args.save_depth:
                    mode_sub = mode_tag.replace(" ", "_").replace("(", "").replace(")", "")
                    depth_save_dir = os.path.join(args.outdir, mode_sub, scan, "depths")
                    os.makedirs(depth_save_dir, exist_ok=True)
                    d_s0_np = d_est_b_s0[0, 0].cpu().numpy()
                    save_pfm(os.path.join(depth_save_dir, f"diffmvs_depth_{file_id}.pfm"), d_s0_np)
                    depth_png_s0 = np.clip(d_s0_np * 64.0, 0, 65535).astype(np.uint16)
                    cv2.imwrite(os.path.join(depth_save_dir, f"diffmvs_depth_{file_id}.png"), depth_png_s0)

            if (batch_idx + 1) % 6 == 0 or (batch_idx + 1) == len(test_loader):
                cur_rec = all_sample_records[-1]
                print(f"  [{mode_tag}] 进度: {batch_idx + 1:2d}/{len(test_loader)} | 切片: {cur_rec['tag']:22s} | "
                      f"S0: {cur_rec['s0']:.4f}m | S1: {cur_rec['s1']:.4f}m | S2: {cur_rec['s2']:.4f}m")

    eval_duration = time.time() - start_eval_time
    print(f"✓ [{mode_tag}] 评测完成，总耗时: {eval_duration:.1f}s\n")
    return all_sample_records, per_scan_records


def compute_summary(records):
    """计算全样本平均指标"""
    return {
        "s0": float(np.nanmean([r['s0'] for r in records])),
        "s1": float(np.nanmean([r['s1'] for r in records])),
        "s2": float(np.nanmean([r['s2'] for r in records])),
        "err_05cm": float(np.nanmean([r['err_05cm'] for r in records])) * 100.0,
        "err_10cm": float(np.nanmean([r['err_10cm'] for r in records])) * 100.0,
        "err_20cm": float(np.nanmean([r['err_20cm'] for r in records])) * 100.0,
        "err_50cm": float(np.nanmean([r['err_50cm'] for r in records])) * 100.0,
        "s0_planar": float(np.nanmean([r['s0_planar'] for r in records])),
        "s0_curved": float(np.nanmean([r['s0_curved'] for r in records])),
        "planar_ratio": float(np.nanmean([r['planar_ratio'] for r in records])),
        "s1_planar": float(np.nanmean([r['s1_planar'] for r in records])),
        "s1_curved": float(np.nanmean([r['s1_curved'] for r in records])),
        "planar_ratio_s1": float(np.nanmean([r['planar_ratio_s1'] for r in records])),
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # 1. 自动探查数据路径
    if not args.testpath:
        candidates = [
            "/home/myao/ENRICH-Aerial_Cropped_0.6x_Overlap50/",
            "/home/myao/ENRICH-Aerial_Data/",
            "/home/myao/Crop_Test_Dataset/",
            "./datasets/ENRICH-Aerial_Cropped_0.6x_Overlap50/",
        ]
        for c in candidates:
            if os.path.isdir(c):
                args.testpath = c
                break

    if not args.testpath or not os.path.isdir(args.testpath):
        raise FileNotFoundError(f"未找到 ENRICH 数据集根目录，请通过 --testpath 明确指定")

    os.makedirs(args.outdir, exist_ok=True)

    # 2. 动态挂载外部 DiffMVS 源码
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
        while diffmvs_root in sys.path:
            sys.path.remove(diffmvs_root)
        sys.path.insert(0, diffmvs_root)
        print(f"[Import] 已挂载 DiffMVS 代码路径: {diffmvs_root}")
    else:
        print(f"[Warning] 未指定或未找到 --diffmvs_code_dir: {diffmvs_root}")

    # 清理已有的 models 缓存，确保导入的是 DiffMVS 的 models
    sys.modules.pop('models', None)
    for k in list(sys.modules.keys()):
        if k.startswith('models.'):
            sys.modules.pop(k, None)

    try:
        from models.diffusion import CasDiffMVS
    except ImportError as e:
        raise ImportError(f"🚨 无法导入 CasDiffMVS 核心模块！请检查 --diffmvs_code_dir。\n错误: {e}")

    # 3. 构建 Dataset 与 DataLoader
    MVSDataset = find_dataset_def(args.dataset)
    test_dataset = MVSDataset(args.testpath, args.testlist, mode="test", nviews=args.n_views)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_keep_list, num_workers=args.num_workers,
        pin_memory=True, drop_last=False
    )

    print("=" * 85)
    print("ENRICH-Aerial DiffMVS / CasDiffMVS (IEEE TPAMI 2025) A / B 对照诊断评测系统")
    print("=" * 85)
    print(f"  数据集根目录:   {args.testpath}")
    print(f"  测试列表路径:   {args.testlist}")
    print(f"  模型 Checkpoint: {args.loadckpt}")
    print(f"  输入视角数:     {args.n_views} 视角 (1 参 2 源)")
    print(f"  BN 诊断模式:    {args.bn_mode.upper()}")
    print(f"  测试样本总数:   {len(test_dataset)} 张切片")
    print("=" * 85)

    # 4. 实例化基础模型并载入权重
    base_model = DiffMVSWrapper(
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

    missing, unexpected = base_model.net.load_state_dict(cleaned_state, strict=False)
    if len(missing) > 0:
        print(f"[Model Warning] 缺失参数 (前5个): {missing[:5]}")
    if len(unexpected) > 0:
        print(f"[Model Warning] 冗余参数 (前5个): {unexpected[:5]}")
    print("✓ 模型权重载入完成！\n")

    clean_state_dict = copy.deepcopy(base_model.state_dict())
    records_A = None
    records_B = None

    # =========================================================================
    # 评测 1: 模式 A (标准 eval - 原始全局 running 统计量)
    # =========================================================================
    if args.bn_mode in ['both', 'eval']:
        model_A = copy.deepcopy(base_model)
        model_A.load_state_dict(clean_state_dict)
        model_A.eval()
        for m in model_A.modules():
            if isinstance(m, _BN_TYPES):
                m.eval()
                if hasattr(m, 'track_running_stats'):
                    m.track_running_stats = True
        records_A, per_scan_A = evaluate_single_pass(
            model_A, test_loader, mode_tag="Mode A (全局固化统计)", device=device, args=args
        )
        del model_A
        torch.cuda.empty_cache()

    # =========================================================================
    # 评测 2: 模式 B (动态单图 BN 统计 - track_running_stats=False 严格防污染)
    # =========================================================================
    if args.bn_mode in ['both', 'batch']:
        model_B = copy.deepcopy(base_model)
        model_B.load_state_dict(clean_state_dict)
        model_B.eval()
        for m in model_B.modules():
            if isinstance(m, _BN_TYPES):
                m.train()
                if hasattr(m, 'track_running_stats'):
                    m.track_running_stats = False
                if hasattr(m, 'momentum'):
                    m.momentum = 0.0
        records_B, per_scan_B = evaluate_single_pass(
            model_B, test_loader, mode_tag="Mode B (单图动态统计)", device=device, args=args
        )
        del model_B
        torch.cuda.empty_cache()

    # =========================================================================
    # 5. 汇总与生成 Markdown 诊断报表
    # =========================================================================
    report_lines = []
    report_lines.append("# ENRICH-Aerial 航空遥感场景基准评测报告: DiffMVS (IEEE TPAMI 2025)\n")
    report_lines.append(f"- **模型权重**: `{args.loadckpt}`")
    report_lines.append(f"- **数据集路径**: `{args.testpath}`")
    report_lines.append(f"- **同源掩码目录**: `{args.mask_dir}`")
    report_lines.append(f"- **视角配置**: 1 参 2 源 (严格 {args.n_views} 视角)")
    report_lines.append(f"- **BN 诊断模式**: `{args.bn_mode.upper()}`\n")

    # 双路 A/B 对比表
    if args.bn_mode == 'both' and records_A is not None and records_B is not None:
        sum_A = compute_summary(records_A)
        sum_B = compute_summary(records_B)

        regions = sorted(list(set(r['region'] for r in records_A)))
        report_lines.append("## 🏆 A/B 模式测区对比表 (Region-Level Summary)")
        report_lines.append("| 测区 (Region) | 切片数 | 模式 A (固化BN) S0 MAE | 模式 B (动态BN) S0 MAE | 增益 / 差距 (m) | 提升幅度 (%) | 模式 B S0 平面 MAE | 模式 B S0 曲面 MAE |")
        report_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

        for reg in regions:
            recs_reg_A = [r for r in records_A if r['region'] == reg]
            recs_reg_B = [r for r in records_B if r['region'] == reg]
            s0_a = float(np.nanmean([r['s0'] for r in recs_reg_A]))
            s0_b = float(np.nanmean([r['s0'] for r in recs_reg_B]))
            p_b = float(np.nanmean([r['s0_planar'] for r in recs_reg_B]))
            c_b = float(np.nanmean([r['s0_curved'] for r in recs_reg_B]))
            diff = s0_a - s0_b
            pct = (diff / s0_a * 100.0) if s0_a > 0 else 0.0

            p_b_str = f"{p_b:.4f}" if not math.isnan(p_b) else "N/A"
            c_b_str = f"{c_b:.4f}" if not math.isnan(c_b) else "N/A"
            report_lines.append(
                f"| `{reg}` | {len(recs_reg_A)} | {s0_a:.4f}m | **{s0_b:.4f}m** | "
                f"{'+' if diff >= 0 else ''}{diff:.4f}m | {'+' if pct >= 0 else ''}{pct:.2f}% | "
                f"{p_b_str}m | {c_b_str}m |"
            )

        diff_total = sum_A['s0'] - sum_B['s0']
        pct_total = (diff_total / sum_A['s0'] * 100.0) if sum_A['s0'] > 0 else 0.0
        p_total_str = f"{sum_B['s0_planar']:.4f}" if not math.isnan(sum_B['s0_planar']) else "N/A"
        c_total_str = f"{sum_B['s0_curved']:.4f}" if not math.isnan(sum_B['s0_curved']) else "N/A"
        report_lines.append(
            f"| **全测区平均 (Overall)** | **{len(records_A)}** | **{sum_A['s0']:.4f}m** | **{sum_B['s0']:.4f}m** | "
            f"**{'+' if diff_total >= 0 else ''}{diff_total:.4f}m** | **{'+' if pct_total >= 0 else ''}{pct_total:.2f}%** | "
            f"**{p_total_str}m** | **{c_total_str}m** |\n"
        )

    # 针对选定主模式生成 Table 1 与 Table 2
    active_records = records_B if records_B is not None else records_A
    active_mode_str = "模式 B (动态单图BN)" if records_B is not None else "模式 A (全局固化BN)"

    if active_records is not None:
        summary_active = compute_summary(active_records)
        scans = sorted(list(set(r['scan'] for r in active_records)))

        report_lines.append(f"## Table 1: 全图多尺度全局深度精度与误差分布 - {active_mode_str}")
        report_lines.append("| Scan | 样本数 | S0 MAE (m) | S1 MAE (m) | S2 MAE (m) | >5cm 误差率 (%) | >10cm 误差率 (%) | >20cm 误差率 (%) | >50cm 误差率 (%) |")
        report_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

        for scan in scans:
            recs_s = [r for r in active_records if r['scan'] == scan]
            s_sum = compute_summary(recs_s)
            s1_str = f"{s_sum['s1']:.4f}" if not math.isnan(s_sum['s1']) else "N/A"
            s2_str = f"{s_sum['s2']:.4f}" if not math.isnan(s_sum['s2']) else "N/A"
            report_lines.append(
                f"| `{scan}` | {len(recs_s)} | **{s_sum['s0']:.4f}** | {s1_str} | {s2_str} | "
                f"{s_sum['err_05cm']:.2f}% | {s_sum['err_10cm']:.2f}% | {s_sum['err_20cm']:.2f}% | {s_sum['err_50cm']:.2f}% |"
            )

        os1_str = f"{summary_active['s1']:.4f}" if not math.isnan(summary_active['s1']) else "N/A"
        os2_str = f"{summary_active['s2']:.4f}" if not math.isnan(summary_active['s2']) else "N/A"
        report_lines.append(
            f"| **DiffMVS (平均)** | **{len(active_records)}** | **{summary_active['s0']:.4f}** | "
            f"**{os1_str}** | **{os2_str}** | "
            f"**{summary_active['err_05cm']:.2f}%** | **{summary_active['err_10cm']:.2f}%** | "
            f"**{summary_active['err_20cm']:.2f}%** | **{summary_active['err_50cm']:.2f}%** |\n"
        )

        report_lines.append(f"## Table 2: 细分平面区与曲面区精度评测 - {active_mode_str}")
        report_lines.append("| Scan | S0 平面 MAE (m) | S0 曲面 MAE (m) | S1 平面 MAE (m) | S1 曲面 MAE (m) | S0 平面占比 (%) | S1 平面占比 (%) |")
        report_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

        for scan in scans:
            recs_s = [r for r in active_records if r['scan'] == scan]
            s_sum = compute_summary(recs_s)
            p0_str = f"{s_sum['s0_planar']:.4f}" if not math.isnan(s_sum['s0_planar']) else "N/A"
            c0_str = f"{s_sum['s0_curved']:.4f}" if not math.isnan(s_sum['s0_curved']) else "N/A"
            p1_str = f"{s_sum['s1_planar']:.4f}" if not math.isnan(s_sum['s1_planar']) else "N/A"
            c1_str = f"{s_sum['s1_curved']:.4f}" if not math.isnan(s_sum['s1_curved']) else "N/A"
            report_lines.append(
                f"| `{scan}` | **{p0_str}** | {c0_str} | **{p1_str}** | {c1_str} | "
                f"{s_sum['planar_ratio']:.2f}% | {s_sum['planar_ratio_s1']:.2f}% |"
            )

        op0_str = f"{summary_active['s0_planar']:.4f}" if not math.isnan(summary_active['s0_planar']) else "N/A"
        oc0_str = f"{summary_active['s0_curved']:.4f}" if not math.isnan(summary_active['s0_curved']) else "N/A"
        op1_str = f"{summary_active['s1_planar']:.4f}" if not math.isnan(summary_active['s1_planar']) else "N/A"
        oc1_str = f"{summary_active['s1_curved']:.4f}" if not math.isnan(summary_active['s1_curved']) else "N/A"
        report_lines.append(
            f"| **DiffMVS (平均)** | **{op0_str}** | **{oc0_str}** | **{op1_str}** | **{oc1_str}** | "
            f"**{summary_active['planar_ratio']:.2f}%** | **{summary_active['planar_ratio_s1']:.2f}%** |"
        )

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    # 保存报告与 JSON 数据
    summary_file = os.path.join(args.outdir, "diffmvs_enrich_summary.md")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n[Output] 完整评估报表已保存至: {summary_file}")

    json_file = os.path.join(args.outdir, "diffmvs_enrich_records.json")
    json_data = {
        "mode_A": [r for r in records_A] if records_A is not None else None,
        "mode_B": [r for r in records_B] if records_B is not None else None,
    }
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    print(f"[Output] 详细指标记录已保存至: {json_file}")


if __name__ == '__main__':
    main()
