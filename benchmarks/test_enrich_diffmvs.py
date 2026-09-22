import argparse
import os
import sys
import time
import json
import math
import cv2
import numpy as np

# 确保项目根目录在 sys.path[0]，以便无缝导入 datasets, utils 等模块
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="ENRICH-Aerial_Data 航空遥感大场景 DiffMVS / CasDiffMVS (IEEE TPAMI 2025) 独立基准评测系统"
    )
    # 数据集与路径配置
    parser.add_argument('--dataset', default='enrich_aerial', type=str, help='数据集定义名称')
    parser.add_argument('--testpath', default="/home/myao/ENRICH-Aerial_Data", type=str, help='ENRICH 数据集根目录')
    parser.add_argument('--testlist', default="/home/myao/ENRICH-Aerial_Data/scan_list.txt", type=str, help='测试场景列表文件')
    parser.add_argument('--loadckpt', required=True, help='DiffMVS / CasDiffMVS Checkpoint 权重路径 (.ckpt)')
    parser.add_argument('--diffmvs_code_dir', default='',
                        help='DiffMVS 源码根目录 (如 /home/myao/diffmvs-main)')
    parser.add_argument('--mask_dir', default='./outputs_enrich_aerial',
                        help='同源平面掩码目录 (由 test_enrich_aerial.py 生成的 stage0/1_planar_mask_*.png)')
    parser.add_argument('--outdir', default='./outputs_diffmvs_enrich',
                        help='评测报告与结果保存目录')

    # 批次与硬件设置
    parser.add_argument('--batch_size', type=int, default=1, help='测试批次大小 (航测大图建议 1)')
    parser.add_argument('--n_views', type=int, default=3, help='测试视角数 (1 参 2 源，严格 3 视角)')
    parser.add_argument('--num_workers', type=int, default=2, help='DataLoader 线程数')
    parser.add_argument('--seed', type=int, default=123, help='随机种子')
    parser.add_argument('--save_depth', action='store_true', default=False,
                        help='保存预测深度图 (PFM 与 PNG)')

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
    ENRICH-Aerial_Data 航测数据基准评测适配包装器：将 DataLoader 输出动态转换为 CasDiffMVS 输入
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

        # 1. 转换图像列表: List[B, 3, H, W] (ENRICH stage_0 为全分辨率原图 [0, 1] 归一化)
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
            P = sample_cuda["proj_matrices"][whu_st][:, :n_views]       # [B, N, 4, 4]
            K = sample_cuda["intrinsics_mats"][whu_st][:, :n_views]     # [B, N, 3, 3]

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


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print("=" * 85)
    print("ENRICH-Aerial_Data 航空遥感大场景 DiffMVS / CasDiffMVS 独立基准评测系统 (严格 3 视角纯推理)")
    print("=" * 85)
    print_args(args)

    os.makedirs(args.outdir, exist_ok=True)

    # 1. 动态挂载外部 DiffMVS 源码 (安全隔离避免 models 命名空间遮蔽)
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

    if not (diffmvs_root and os.path.isdir(diffmvs_root)):
        raise RuntimeError(
            f"🚨 未找到外部 DiffMVS 源码目录！请通过 --diffmvs_code_dir 指定。"
        )

    print(f"[External] 动态挂载 DiffMVS 源码目录: {diffmvs_root}")

    # 动态安全挂载外部 DiffMVS 的 models.diffusion，杜绝与主仓 models 命名空间冲突
    old_models = sys.modules.pop('models', None)
    old_models_sub = {k: sys.modules.pop(k) for k in list(sys.modules.keys()) if k.startswith('models.')}
    if diffmvs_root and diffmvs_root not in sys.path:
        sys.path.insert(0, diffmvs_root)

    try:
        from models.diffusion import CasDiffMVS
    except ImportError as e:
        raise ImportError(
            f"🚨 无法导入 CasDiffMVS 模型！请确认 --diffmvs_code_dir 参数指向正确的 DiffMVS 源码目录。\n"
            f"原始错误: {e}"
        )
    finally:
        if diffmvs_root and diffmvs_root in sys.path:
            sys.path.remove(diffmvs_root)
        if old_models is not None:
            sys.modules['models'] = old_models
        sys.modules.update(old_models_sub)

    # 2. 构建 Dataset 与 DataLoader
    MVSDataset = find_dataset_def(args.dataset)
    print(f"\n[Dataset] 加载 ENRICH-Aerial 数据集: path={args.testpath}, listfile={args.testlist}")
    test_dataset = MVSDataset(args.testpath, args.testlist, mode="test", nviews=args.n_views)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_keep_list, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=(args.num_workers > 0), drop_last=False
    )
    print(f"[Dataset] 测试集样本总量: {len(test_dataset)} (共 {len(test_loader)} 个批次)\n")

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

    per_scan_records = {}
    all_sample_records = []
    start_eval_time = time.time()

    print("\n" + "=" * 85)
    print("开始执行 ENRICH-Aerial_Data DiffMVS 逐场景推断与 Table 1 / Table 2 精度评测...")
    print("=" * 85)

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

                # 航测场景误差阈值 (0.05m, 0.10m, 0.20m, 0.50m)
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
                    "file_id": file_id,
                    "stage0_mae": s0_mae,
                    "stage1_mae": s1_mae,
                    "stage2_mae": s2_mae,
                    "stage0_thres05cm_err": t05_err,
                    "stage0_thres10cm_err": t10_err,
                    "stage0_thres20cm_err": t20_err,
                    "stage0_thres50cm_err": t50_err,
                    "stage0_planar_mae": s0_planar_mae,
                    "stage0_curved_mae": s0_curved_mae,
                    "s0_valid_pixels": valid_cnt_s0,
                    "s0_planar_pixels": planar_cnt_s0,
                    "s0_planar_ratio_pct": planar_ratio_s0,
                    "stage1_planar_mae": s1_planar_mae,
                    "stage1_curved_mae": s1_curved_mae,
                    "s1_planar_ratio_pct": planar_ratio_s1,
                }

                per_scan_records[scan].append(record)
                all_sample_records.append(record)

                # 保存预测深度图
                if args.save_depth:
                    depth_save_dir = os.path.join(args.outdir, scan, "depths")
                    os.makedirs(depth_save_dir, exist_ok=True)
                    d_s0_np = d_est_b_s0[0, 0].cpu().numpy()
                    save_pfm(os.path.join(depth_save_dir, f"diffmvs_depth_{file_id}.pfm"), d_s0_np)
                    depth_png_s0 = np.clip(d_s0_np * 64.0, 0, 65535).astype(np.uint16)
                    cv2.imwrite(os.path.join(depth_save_dir, f"diffmvs_depth_{file_id}.png"), depth_png_s0)

            cur_idx = batch_idx + 1
            step_time = time.time() - step_start
            p0_str = f"{record['stage0_planar_mae']:.4f}m" if not math.isnan(record['stage0_planar_mae']) else "N/A"
            p1_str = f"{record['stage1_planar_mae']:.4f}m" if not math.isnan(record['stage1_planar_mae']) else "N/A"
            s0_mae_str = f"{record['stage0_mae']:.4f}m" if not math.isnan(record['stage0_mae']) else "N/A"
            print(f"[{cur_idx:03d}/{len(test_loader):03d}] Scan: {scan} | Img: {file_id} | "
                  f"DiffMVS S0 MAE: {s0_mae_str} | S0 Planar: {p0_str} | S1 Planar: {p1_str} | Time: {step_time:.2f}s")

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
        summary_by_scan[scan] = {
            "count": len(recs),
            "stage0_mae": compute_mean_ignore_nan(recs, "stage0_mae"),
            "stage1_mae": compute_mean_ignore_nan(recs, "stage1_mae"),
            "stage2_mae": compute_mean_ignore_nan(recs, "stage2_mae"),
            "stage0_thres05cm_err": compute_mean_ignore_nan(recs, "stage0_thres05cm_err"),
            "stage0_thres10cm_err": compute_mean_ignore_nan(recs, "stage0_thres10cm_err"),
            "stage0_thres20cm_err": compute_mean_ignore_nan(recs, "stage0_thres20cm_err"),
            "stage0_thres50cm_err": compute_mean_ignore_nan(recs, "stage0_thres50cm_err"),
            "stage0_planar_mae": compute_mean_ignore_nan(recs, "stage0_planar_mae"),
            "stage0_curved_mae": compute_mean_ignore_nan(recs, "stage0_curved_mae"),
            "s0_planar_ratio_pct": compute_mean_ignore_nan(recs, "s0_planar_ratio_pct"),
            "stage1_planar_mae": compute_mean_ignore_nan(recs, "stage1_planar_mae"),
            "stage1_curved_mae": compute_mean_ignore_nan(recs, "stage1_curved_mae"),
            "s1_planar_ratio_pct": compute_mean_ignore_nan(recs, "s1_planar_ratio_pct"),
        }

    overall_summary = {
        "count": len(all_sample_records),
        "stage0_mae": compute_mean_ignore_nan(all_sample_records, "stage0_mae"),
        "stage1_mae": compute_mean_ignore_nan(all_sample_records, "stage1_mae"),
        "stage2_mae": compute_mean_ignore_nan(all_sample_records, "stage2_mae"),
        "stage0_thres05cm_err": compute_mean_ignore_nan(all_sample_records, "stage0_thres05cm_err"),
        "stage0_thres10cm_err": compute_mean_ignore_nan(all_sample_records, "stage0_thres10cm_err"),
        "stage0_thres20cm_err": compute_mean_ignore_nan(all_sample_records, "stage0_thres20cm_err"),
        "stage0_thres50cm_err": compute_mean_ignore_nan(all_sample_records, "stage0_thres50cm_err"),
        "stage0_planar_mae": compute_mean_ignore_nan(all_sample_records, "stage0_planar_mae"),
        "stage0_curved_mae": compute_mean_ignore_nan(all_sample_records, "stage0_curved_mae"),
        "s0_planar_ratio_pct": compute_mean_ignore_nan(all_sample_records, "s0_planar_ratio_pct"),
        "stage1_planar_mae": compute_mean_ignore_nan(all_sample_records, "stage1_planar_mae"),
        "stage1_curved_mae": compute_mean_ignore_nan(all_sample_records, "stage1_curved_mae"),
        "s1_planar_ratio_pct": compute_mean_ignore_nan(all_sample_records, "s1_planar_ratio_pct"),
    }

    report_lines = []
    report_lines.append("# ENRICH-Aerial_Data 航空遥感场景基准评测报告: DiffMVS / CasDiffMVS (IEEE TPAMI 2025)\n")
    report_lines.append(f"- **模型权重**: `{args.loadckpt}`")
    report_lines.append(f"- **数据集路径**: `{args.testpath}`")
    report_lines.append(f"- **测试场景数**: {len(summary_by_scan)} 个 (共 {overall_summary['count']} 张样本)")
    report_lines.append(f"- **同源掩码目录**: `{args.mask_dir}`")
    report_lines.append(f"- **视角配置**: 1 参 2 源 (严格 3 视角)\n")

    # Table 1: 全图多尺度全局深度精度
    report_lines.append("## Table 1: 全图多尺度全局深度精度与误差分布 (Global Depth Accuracy)")
    report_lines.append("| Scan | 样本数 | S0 MAE (m) | S1 MAE (m) | S2 MAE (m) | >5cm 误差率 (%) | >10cm 误差率 (%) | >20cm 误差率 (%) | >50cm 误差率 (%) |")
    report_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for scan, s in summary_by_scan.items():
        s1_str = f"{s['stage1_mae']:.4f}" if not math.isnan(s['stage1_mae']) else "N/A"
        s2_str = f"{s['stage2_mae']:.4f}" if not math.isnan(s['stage2_mae']) else "N/A"
        report_lines.append(
            f"| `{scan}` | {s['count']} | **{s['stage0_mae']:.4f}** | {s1_str} | {s2_str} | "
            f"{s['stage0_thres05cm_err']*100:.2f}% | {s['stage0_thres10cm_err']*100:.2f}% | "
            f"{s['stage0_thres20cm_err']*100:.2f}% | {s['stage0_thres50cm_err']*100:.2f}% |"
        )
    os1_str = f"{overall_summary['stage1_mae']:.4f}" if not math.isnan(overall_summary['stage1_mae']) else "N/A"
    os2_str = f"{overall_summary['stage2_mae']:.4f}" if not math.isnan(overall_summary['stage2_mae']) else "N/A"
    report_lines.append(
        f"| **DiffMVS (总计)** | **{overall_summary['count']}** | **{overall_summary['stage0_mae']:.4f}** | "
        f"**{os1_str}** | **{os2_str}** | "
        f"**{overall_summary['stage0_thres05cm_err']*100:.2f}%** | **{overall_summary['stage0_thres10cm_err']*100:.2f}%** | "
        f"**{overall_summary['stage0_thres20cm_err']*100:.2f}%** | **{overall_summary['stage0_thres50cm_err']*100:.2f}%** |"
    )

    # Table 2: 细分平面与曲面特性评测
    report_lines.append("\n## Table 2: 细分平面与曲面特性评测 (Planar vs Non-Planar Accuracy)")
    report_lines.append("| Scan | S0 平面 MAE (m) | S0 曲面 MAE (m) | S1 平面 MAE (m) | S1 曲面 MAE (m) | S0 平面占比 (%) | S1 平面占比 (%) |")
    report_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for scan, s in summary_by_scan.items():
        p0_str = f"{s['stage0_planar_mae']:.4f}" if not math.isnan(s['stage0_planar_mae']) else "N/A"
        c0_str = f"{s['stage0_curved_mae']:.4f}" if not math.isnan(s['stage0_curved_mae']) else "N/A"
        p1_str = f"{s['stage1_planar_mae']:.4f}" if not math.isnan(s['stage1_planar_mae']) else "N/A"
        c1_str = f"{s['stage1_curved_mae']:.4f}" if not math.isnan(s['stage1_curved_mae']) else "N/A"
        report_lines.append(f"| `{scan}` | **{p0_str}** | {c0_str} | **{p1_str}** | {c1_str} | {s['s0_planar_ratio_pct']:.2f}% | {s['s1_planar_ratio_pct']:.2f}% |")

    op0_str = f"{overall_summary['stage0_planar_mae']:.4f}" if not math.isnan(overall_summary['stage0_planar_mae']) else "N/A"
    oc0_str = f"{overall_summary['stage0_curved_mae']:.4f}" if not math.isnan(overall_summary['stage0_curved_mae']) else "N/A"
    op1_str = f"{overall_summary['stage1_planar_mae']:.4f}" if not math.isnan(overall_summary['stage1_planar_mae']) else "N/A"
    oc1_str = f"{overall_summary['stage1_curved_mae']:.4f}" if not math.isnan(overall_summary['stage1_curved_mae']) else "N/A"
    report_lines.append(
        f"| **DiffMVS (总计)** | **{op0_str}** | **{oc0_str}** | **{op1_str}** | **{oc1_str}** | "
        f"**{overall_summary['s0_planar_ratio_pct']:.2f}%** | **{overall_summary['s1_planar_ratio_pct']:.2f}%** |"
    )

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    # 保存报告与指标 JSON
    report_file = os.path.join(args.outdir, "diffmvs_enrich_summary.md")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n[Output] 完整评估报表已保存至: {report_file}")

    json_file = os.path.join(args.outdir, "diffmvs_enrich_records.json")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump({"overall": overall_summary, "by_scan": summary_by_scan, "records": all_sample_records}, f, indent=2)
    print(f"[Output] 详细样本指标记录已保存至: {json_file}")


if __name__ == '__main__':
    main()
