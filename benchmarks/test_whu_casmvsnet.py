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
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from datasets import find_dataset_def
from datasets.dtu_whu import collate_keep_list
from datasets.data_io import save_pfm
from utils import print_args, tocuda

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True


# ==============================================================================
# 0. 优雅环境兼容层: InPlaceABN 降级保护
# 若远程服务器未编译安装 inplace_abn C++/CUDA 扩展，自动无缝降级为原生 PyTorch 实现
# ==============================================================================
try:
    import inplace_abn
except ImportError:
    import types
    abn_module = types.ModuleType('inplace_abn')

    class InPlaceABN(nn.Module):
        """
        兼容层：当未安装 inplace_abn 扩展时的等价 PyTorch 原生实现。
        同时支持 4D (B, C, H, W) 与 5D (B, C, D, H, W) 特征体，
        且权重变量名与官方 InPlaceABN 严格一致（weight, bias, running_mean, running_var）。
        """
        def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                     activation="leaky_relu", activation_param=0.01, **kwargs):
            super().__init__()
            self.num_features = num_features
            self.eps = eps
            self.momentum = momentum
            self.affine = affine
            self.activation = activation
            self.activation_param = activation_param

            if self.affine:
                self.weight = nn.Parameter(torch.ones(num_features))
                self.bias = nn.Parameter(torch.zeros(num_features))
            else:
                self.register_parameter('weight', None)
                self.register_parameter('bias', None)

            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

        def forward(self, x):
            x = F.batch_norm(
                x, self.running_mean, self.running_var,
                self.weight, self.bias,
                self.training, self.momentum, self.eps
            )
            if self.activation == "leaky_relu":
                return F.leaky_relu(x, negative_slope=self.activation_param, inplace=True)
            elif self.activation == "relu":
                return F.relu(x, inplace=True)
            elif self.activation == "elu":
                return F.elu(x, alpha=self.activation_param, inplace=True)
            return x

    class ABN(InPlaceABN):
        pass

    abn_module.InPlaceABN = InPlaceABN
    abn_module.ABN = ABN
    sys.modules['inplace_abn'] = abn_module
    print("[Compatibility] 未检测到系统级 inplace_abn 扩展，已自动挂载 PyTorch 原生 InPlaceABN 兼容层！")

try:
    import kornia
    from kornia.utils import create_meshgrid
except ImportError:
    import types
    kornia_mod = types.ModuleType('kornia')
    kornia_utils_mod = types.ModuleType('kornia.utils')

    def create_meshgrid(height, width, normalized_coordinates=False, device=None, dtype=torch.float32):
        xs = torch.linspace(0, width - 1, width, device=device, dtype=dtype)
        ys = torch.linspace(0, height - 1, height, device=device, dtype=dtype)
        try:
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        except TypeError:
            grid_y, grid_x = torch.meshgrid(ys, xs)
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)
        if normalized_coordinates:
            grid[..., 0] = grid[..., 0] / ((width - 1) / 2) - 1
            grid[..., 1] = grid[..., 1] / ((height - 1) / 2) - 1
        return grid

    kornia_utils_mod.create_meshgrid = create_meshgrid
    kornia_mod.utils = kornia_utils_mod
    sys.modules['kornia'] = kornia_mod
    sys.modules['kornia.utils'] = kornia_utils_mod
    print("[Compatibility] 未检测到系统级 kornia 库，已自动挂载 PyTorch 原生 create_meshgrid 兼容层！")


def parse_args():
    parser = argparse.ArgumentParser(
        description='WHU-MVS CasMVSNet (CVPR 2020) 独立基准评测适配脚本 (遵循 whu-mvs-benchmark-adapter 规范)'
    )
    # 数据集与路径配置
    parser.add_argument('--dataset', default='dtu_whu', help='select dataset (default: dtu_whu)')
    parser.add_argument('--testpath', default='/home/ym/Experiment/Datas/WHU_MVS_dataset',
                        help='WHU dataset root path')
    parser.add_argument('--testlist', default='lists/whu/newtest.txt',
                        help='testing scan list file (e.g. lists/whu/minitest.txt or newtest.txt)')
    parser.add_argument('--loadckpt', required=True, help='path to CasMVSNet checkpoint (.ckpt or .pth)')
    parser.add_argument('--casmvsnet_code_dir', default='',
                        help='path to CasMVSNet source code directory (e.g. /home/myao/CasMVSNet_pl-master)')
    parser.add_argument('--mask_dir', default='./outputs_minitest',
                        help='directory containing 同源 stage0_planar_mask_{file_id}.png (for Table 2)')
    parser.add_argument('--outdir', default='./outputs_casmvsnet',
                        help='output directory to save reports and depth predictions')

    # 批次与硬件设置
    parser.add_argument('--batch_size', type=int, default=1, help='testing batch size (recommend 1 to prevent OOM)')
    parser.add_argument('--n_views', type=int, default=5, help='number of views per sample')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers for DataLoader')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--save_depth', action='store_true', default=False,
                        help='save predicted depth maps in .pfm and 16-bit .png format')

    # CasMVSNet 核心超参数 (与官方 CasMVSNet / CasMVSNet_pl 完全对齐)
    parser.add_argument('--n_depths', nargs='+', type=int, default=[8, 32, 48],
                        help='number of depth hypotheses in each stage [fine, medium, coarse]')
    parser.add_argument('--interval_ratios', nargs='+', type=float, default=[1.0, 2.0, 4.0],
                        help='depth interval ratio to multiply with base depth_interval in each stage')
    parser.add_argument('--num_groups', type=int, default=1, choices=[1, 2, 4, 8],
                        help='number of groups in groupwise correlation, must be a divisor of 8 (default: 1)')

    return parser.parse_args()


class CasMVSNetWrapper(nn.Module):
    """
    WHU-MVS 基准测试适配包装器：将 WHU DataLoader 输出无缝转换为 CasMVSNet 输入
    遵循 whu-mvs-benchmark-adapter 专家规范 (Mode A: 外挂包装器模式)
    """
    def __init__(self, casmvsnet_class, n_depths=[8, 32, 48], interval_ratios=[1.0, 2.0, 4.0], num_groups=1):
        super().__init__()
        self.n_depths = n_depths
        self.interval_ratios = interval_ratios
        self.num_groups = num_groups

        self.net = casmvsnet_class(
            n_depths=n_depths,
            interval_ratios=interval_ratios,
            num_groups=num_groups
        )

        # ImageNet 标准归一化均值与方差 (CasMVSNet_pl 训练标准)
        self.register_buffer('img_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
        self.register_buffer('img_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))

    def forward(self, sample_cuda, n_views=5):
        B = sample_cuda["depth_min"].shape[0]
        dev = sample_cuda["depth_min"].device

        # 1. 图像预处理与标准化 (B, V, 3, H, W)
        # WHU stage_0 图像取值范围为 [0, 1]，应用 ImageNet 归一化
        raw_imgs = sample_cuda["imgs"]["stage_0"][:, :n_views]  # [B, V, 3, H, W]
        imgs = (raw_imgs - self.img_mean) / self.img_std

        _, _, _, H, W = imgs.shape

        # 2. 分辨率对齐 (CasMVSNet 降采样特征需能被 32 整除)
        pad_h = (32 - H % 32) % 32
        pad_w = (32 - W % 32) % 32
        if pad_h > 0 or pad_w > 0:
            # 展平为 (B*V, 3, H, W) 进行 pad
            imgs = imgs.view(B * n_views, 3, H, W)
            imgs = F.pad(imgs, (0, pad_w, 0, pad_h), mode='replicate')
            imgs = imgs.view(B, n_views, 3, H + pad_h, W + pad_w)

        # 3. 构造 3 个阶段的多尺度相对投影矩阵 [B, V-1, levels, 3, 4] (顺序: fine to coarse)
        # level 0: stage_0 (1/1)
        # level 1: stage_1 (1/2)
        # level 2: stage_2 (1/4)
        P_0 = sample_cuda["proj_matrices"]["stage_0"][:, :n_views]  # [B, V, 4, 4]
        P_1 = sample_cuda["proj_matrices"]["stage_1"][:, :n_views]  # [B, V, 4, 4]
        P_2 = sample_cuda["proj_matrices"]["stage_2"][:, :n_views]  # [B, V, 4, 4]

        # 堆叠为 [B, V, 3, 4, 4]
        P_levels = torch.stack([P_0, P_1, P_2], dim=2)

        # 参考视角的 4x4 投影矩阵逆矩阵: [B, 3, 4, 4]
        ref_proj = P_levels[:, 0]
        ref_proj_inv = torch.inverse(ref_proj)

        # 相对投影矩阵 P_rel = P_src @ ref_proj_inv 取前 3x4
        proj_mats = []
        for i in range(1, n_views):
            src_proj = P_levels[:, i]  # [B, 3, 4, 4]
            rel_proj = torch.matmul(src_proj, ref_proj_inv)  # [B, 3, 4, 4]
            proj_mats.append(rel_proj[:, :, :3, :4])  # [B, 3, 3, 4]
        proj_mats = torch.stack(proj_mats, dim=1)  # [B, V-1, 3, 3, 4]

        # 4. 深度范围与粗阶段假设步长计算 (CasMVSNet 内部要求形状必须为 (B, 1) 以支持广播及 einops 'b 1 -> b 1 1 1')
        init_depth_min = sample_cuda["depth_min"].view(B, 1).float()
        depth_max = sample_cuda["depth_max"].view(B, 1).float()

        # 最粗层 (level 2) 覆盖整个场景范围: D_coarse * interval_ratio_coarse
        coarse_coverage = float(self.n_depths[-1] * self.interval_ratios[-1])
        depth_interval = ((depth_max - init_depth_min) / coarse_coverage).view(B, 1).float()

        # 5. CasMVSNet 前向推理
        outputs = self.net(
            imgs=imgs,
            proj_mats=proj_mats,
            init_depth_min=init_depth_min,
            depth_interval=depth_interval
        )

        # 6. 提取最高分辨率 (level 0) 预测深度图 (以绝对物理米 meters 为单位)
        depth_pred = outputs["depth_0"]  # [B, H_pad, W_pad]
        if pad_h > 0 or pad_w > 0:
            depth_pred = depth_pred[:, :H, :W]
        depth_pred = depth_pred.unsqueeze(1)  # [B, 1, H, W]

        # 提取全分辨率置信度
        confidence = None
        if "confidence_0" in outputs:
            conf = outputs["confidence_0"]
            if pad_h > 0 or pad_w > 0:
                conf = conf[:, :H, :W]
            confidence = conf.unsqueeze(1)

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


def load_casmvsnet_ckpt(model, ckpt_path):
    """统一兼容加载 PyTorch-Lightning 与原生 PyTorch 权重文件"""
    print(f"[Model] 载入 Checkpoint 权重: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    if 'state_dict' in checkpoint:
        raw_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        raw_dict = checkpoint['model']
    else:
        raw_dict = checkpoint

    cleaned_state = {}
    for k, v in raw_dict.items():
        # 剥离 Lightning 'model.' 或 DDP 'module.' 前缀
        clean_k = k
        if clean_k.startswith('model.'):
            clean_k = clean_k[6:]
        if clean_k.startswith('module.'):
            clean_k = clean_k[7:]
        cleaned_state[clean_k] = v

    missing, unexpected = model.net.load_state_dict(cleaned_state, strict=False)
    if len(missing) > 0:
        print(f"[Model Warning] 缺失参数 (前10个): {missing[:10]}")
    if len(unexpected) > 0:
        print(f"[Model Warning] 冗余参数 (前10个): {unexpected[:10]}")
    if len(missing) == 0 and len(unexpected) == 0:
        print("[Model] 权重参数 100% 完美匹配加载！")


def main():
    os.chdir(PROJECT_ROOT)
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print("=" * 85)
    print("WHU-MVS 基准适配评估系统: CasMVSNet (CVPR 2020)")
    print("=" * 85)
    print_args(args)

    os.makedirs(args.outdir, exist_ok=True)

    # 1. 动态挂载外部 CasMVSNet 源码
    casmvsnet_root = os.path.abspath(args.casmvsnet_code_dir) if args.casmvsnet_code_dir else ""
    if not (casmvsnet_root and os.path.isdir(casmvsnet_root)):
        candidates = [
            "/home/myao/CasMVSNet_pl-master",
            "/home/ym/Experiment/CasMVSNet_pl-master",
            "/home/myao/CasMVSNet",
            "/home/ym/Experiment/CasMVSNet",
            os.path.abspath(os.path.join(PROJECT_ROOT, "..", "CasMVSNet_pl-master")),
            os.path.abspath(os.path.join(PROJECT_ROOT, "..", "CasMVSNet")),
        ]
        for c in candidates:
            if os.path.isdir(c):
                casmvsnet_root = c
                print(f"[Auto-Detect] 自动探测并挂载 CasMVSNet 源码路径: {casmvsnet_root}")
                break

    if casmvsnet_root and os.path.isdir(casmvsnet_root):
        if casmvsnet_root not in sys.path:
            sys.path.insert(0, casmvsnet_root)
        print(f"[Import] 已成功挂载 CasMVSNet 代码路径: {casmvsnet_root}")
    else:
        print(f"[Warning] 未指定或未找到 --casmvsnet_code_dir: {casmvsnet_root}，尝试从系统环境导入...")

    try:
        from models.mvsnet import CascadeMVSNet
    except ImportError as e:
        raise ImportError(
            f"🚨 无法导入 CascadeMVSNet 模型！请确认 --casmvsnet_code_dir 参数指向正确的 CasMVSNet 源码目录。\n"
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
    model = CasMVSNetWrapper(
        casmvsnet_class=CascadeMVSNet,
        n_depths=args.n_depths,
        interval_ratios=args.interval_ratios,
        num_groups=args.num_groups
    ).to(device)

    load_casmvsnet_ckpt(model, args.loadckpt)
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
            depth_est, confidence = model(sample_cuda, n_views=args.n_views)  # [B, 1, H, W]

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

                # 计算绝对深度误差与阈值比例 (米)
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
                    save_pfm(os.path.join(depth_save_dir, f"casmvsnet_depth_{file_id}.pfm"), d_s0_np)
                    depth_png_s0 = np.clip(d_s0_np * 64.0, 0, 65535).astype(np.uint16)
                    cv2.imwrite(os.path.join(depth_save_dir, f"casmvsnet_depth_{file_id}.png"), depth_png_s0)

            cur_idx = batch_idx + 1
            step_time = time.time() - step_start
            planar_str = f"{s0_planar_mae:.4f}m" if not math.isnan(s0_planar_mae) else "N/A"
            print(f"[{cur_idx:03d}/{len(test_loader):03d}] Scan: {scan} | Img: {file_id} | "
                  f"CasMVSNet MAE: {s0_mae:.4f}m | Planar MAE: {planar_str} | Time: {step_time:.2f}s")

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
    report_lines.append("# WHU-MVS 基准对比评测报告: CasMVSNet (CVPR 2020)\n")
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
        f"| **CasMVSNet (平均)** | **{overall_summary['count']}** | **{overall_summary['stage0_mae']:.4f}** | "
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
        f"| **CasMVSNet (总计)** | **{overall_summary['stage0_mae']:.4f}** | **{op_str}** | **{oc_str}** | **{overall_summary['s0_planar_ratio_pct']:.1f}%** |"
    )

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    # 保存报告与指标 JSON
    report_file = os.path.join(args.outdir, "casmvsnet_evaluation_report.md")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n[Output] 完整评估报表已保存至: {report_file}")

    json_file = os.path.join(args.outdir, "casmvsnet_records.json")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump({"overall": overall_summary, "by_scan": summary_by_scan, "records": all_sample_records}, f, indent=2)
    print(f"[Output] 详细样本指标记录已保存至: {json_file}")


if __name__ == '__main__':
    main()
