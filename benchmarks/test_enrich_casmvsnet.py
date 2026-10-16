import argparse
import os
import sys
import copy
import time
import json
import math
import cv2
import numpy as np

# 确保项目根目录强制位于 sys.path[0]，杜绝任何外部 baseline 仓库的同名包产生遮蔽
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
while PROJECT_ROOT in sys.path:
    sys.path.remove(PROJECT_ROOT)
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


# ==============================================================================
# 0. 环境兼容层: InPlaceABN 与 kornia 降级保护
# ==============================================================================
try:
    import inplace_abn
except ImportError:
    import types
    abn_module = types.ModuleType('inplace_abn')

    class InPlaceABN(nn.Module):
        def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                     activation="leaky_relu", activation_param=0.01, **kwargs):
            super().__init__()
            self.num_features = num_features
            self.eps = eps
            self.momentum = momentum
            self.affine = affine
            self.activation = activation
            self.activation_param = activation_param
            self.track_running_stats = True

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
            running_mean = self.running_mean if getattr(self, 'track_running_stats', True) else None
            running_var = self.running_var if getattr(self, 'track_running_stats', True) else None
            x = F.batch_norm(
                x, running_mean, running_var,
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
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
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
        description="ENRICH-Aerial 航空遥感场景 CasMVSNet (CVPR 2020) A/B 模式独立基准评测系统"
    )
    # 数据集与路径
    parser.add_argument('--dataset', default='enrich_aerial', type=str, help='数据集定义名称')
    parser.add_argument('--testpath', default="", type=str, help='ENRICH 数据集根目录 (留空自动探查)')
    parser.add_argument('--testlist', default="lists/enrich_aerial/test.txt", type=str, help='测试场景列表文件')
    parser.add_argument('--loadckpt', required=True, type=str, help='CasMVSNet Checkpoint 权重路径 (.ckpt 或 .pth)')
    parser.add_argument('--casmvsnet_code_dir', default="", type=str,
                        help='外部 CasMVSNet 源码目录 (如 /home/myao/CasMVSNet_pl-master)')
    parser.add_argument('--mask_dir', default="./outputs_enrich_aerial", type=str,
                        help='同源平面掩码目录 (由 test_enrich_aerial.py 生成的 stage0/1_planar_mask_*.png)')
    parser.add_argument('--outdir', default="./outputs_casmvsnet_enrich", type=str, help='评测报告与结果保存目录')

    # A/B 诊断模式配置
    parser.add_argument('--bn_mode', default='both', choices=['both', 'eval', 'batch'],
                        help='BN诊断模式: both(A/B双路对比) | eval(模式A 固化全局统计) | batch(模式B 动态即时统计)')

    # 批次与硬件
    parser.add_argument('--batch_size', type=int, default=1, help='测试批次大小 (航测大图建议 1)')
    parser.add_argument('--n_views', type=int, default=3, help='测试视角数 (1 参 2 源，严格 3 视角)')
    parser.add_argument('--num_workers', type=int, default=2, help='DataLoader 线程数')
    parser.add_argument('--seed', type=int, default=123, help='随机种子')
    parser.add_argument('--save_depth', action='store_true', default=False, help='保存预测深度图 (PFM 与 PNG)')

    # CasMVSNet 网络超参数
    parser.add_argument('--n_depths', nargs='+', type=int, default=[8, 32, 48],
                        help='各阶段深度假设数 [fine, medium, coarse]')
    parser.add_argument('--interval_ratios', nargs='+', type=float, default=[1.0, 2.0, 4.0],
                        help='各阶段深度采样步长倍率')
    parser.add_argument('--num_groups', type=int, default=1, choices=[1, 2, 4, 8],
                        help='分组相关性分组数 (默认: 1)')

    return parser.parse_args()


class CasMVSNetWrapper(nn.Module):
    """
    ENRICH-Aerial 航测数据基准评测适配包装器：将 DataLoader 输出动态转换为 CasMVSNet 输入
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

        # ImageNet 标准归一化均值与方差 (CasMVSNet 评测标准)
        self.register_buffer('img_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
        self.register_buffer('img_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))

    def forward(self, sample_cuda, n_views=3):
        B = sample_cuda["depth_min"].shape[0]

        # 1. 图像标准化: [B, V, 3, H, W]
        raw_imgs = sample_cuda["imgs"]["stage_0"][:, :n_views]
        imgs = (raw_imgs - self.img_mean) / self.img_std
        _, _, _, H, W = imgs.shape

        # 2. 分辨率对齐 (CasMVSNet 降采样特征需能被 32 整除)
        pad_h = (32 - H % 32) % 32
        pad_w = (32 - W % 32) % 32
        if pad_h > 0 or pad_w > 0:
            imgs = imgs.view(B * n_views, 3, H, W)
            imgs = F.pad(imgs, (0, pad_w, 0, pad_h), mode='replicate')
            imgs = imgs.view(B, n_views, 3, H + pad_h, W + pad_w)

        # 3. 构造 3 阶段多尺度相对投影矩阵 [B, V-1, 3, 3, 4] (顺序: fine to coarse)
        P_0 = sample_cuda["proj_matrices"]["stage_0"][:, :n_views]
        P_1 = sample_cuda["proj_matrices"]["stage_1"][:, :n_views]
        P_2 = sample_cuda["proj_matrices"]["stage_2"][:, :n_views]
        P_levels = torch.stack([P_0, P_1, P_2], dim=2)

        ref_proj = P_levels[:, 0]
        ref_proj_inv = torch.inverse(ref_proj)

        proj_mats = []
        for i in range(1, n_views):
            src_proj = P_levels[:, i]
            rel_proj = torch.matmul(src_proj, ref_proj_inv)
            proj_mats.append(rel_proj[:, :, :3, :4])
        proj_mats = torch.stack(proj_mats, dim=1)

        # 4. 深度范围与粗阶段假设步长计算
        init_depth_min = sample_cuda["depth_min"].view(B, 1).float()
        depth_max = sample_cuda["depth_max"].view(B, 1).float()
        coarse_coverage = float(self.n_depths[-1] * self.interval_ratios[-1])
        depth_interval = ((depth_max - init_depth_min) / coarse_coverage).view(B, 1).float()

        # 5. 前向推理
        outputs = self.net(
            imgs=imgs,
            proj_mats=proj_mats,
            init_depth_min=init_depth_min,
            depth_interval=depth_interval
        )

        # 6. 提取各尺度预测深度
        depth_pred_s0 = outputs["depth_0"]
        depth_pred_s1 = outputs["depth_1"]
        depth_pred_s2 = outputs["depth_2"]
        confidence = outputs["confidence_0"]

        # 7. 若做了 Padding，还原为原始图像有效几何区域
        if pad_h > 0 or pad_w > 0:
            depth_pred_s0 = depth_pred_s0[:, :H, :W]
            depth_pred_s1 = depth_pred_s1[:, :H//2, :W//2]
            depth_pred_s2 = depth_pred_s2[:, :H//4, :W//4]
            confidence = confidence[:, :H, :W]

        # 统一输出尺度: [B, 1, H, W]
        return depth_pred_s0.unsqueeze(1), depth_pred_s1.unsqueeze(1), depth_pred_s2.unsqueeze(1), confidence


def safe_mae(depth_est, depth_gt, mask):
    """安全计算 MAE (米)"""
    if mask is None or not mask.any():
        return float('nan')
    est_valid = depth_est[mask]
    gt_valid = depth_gt[mask]
    return torch.mean((est_valid - gt_valid).abs()).item()


def safe_thres_error(depth_est, depth_gt, mask, thres):
    """安全计算误差大于阈值的比率 [0.0 ~ 1.0]"""
    if mask is None or not mask.any():
        return float('nan')
    est_valid = depth_est[mask]
    gt_valid = depth_gt[mask]
    err = (est_valid - gt_valid).abs()
    return (err > thres).float().mean().item()


def load_casmvsnet_ckpt(model, ckpt_path):
    """稳健加载 CasMVSNet 权重"""
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"🚨 Checkpoint 文件不存在: {ckpt_path}")

    print(f"[Model] 正在载入 Checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))

    clean_state = {}
    for k, v in state_dict.items():
        clean_k = k
        if clean_k.startswith("model."):
            clean_k = clean_k[6:]
        if clean_k.startswith("module."):
            clean_k = clean_k[7:]
        clean_state[clean_k] = v

    missing, unexpected = model.net.load_state_dict(clean_state, strict=False)
    print(f"[Model] 权重加载成功！(缺失键数: {len(missing)}, 冗余键数: {len(unexpected)})")
    if len(missing) > 0:
        print(f"  -> 缺失前 5 项: {missing[:5]}")


def evaluate_single_pass(model, test_loader, mode_tag, device, args):
    """执行单个模式（Mode A 或 Mode B）的全量评测"""
    print("--------------------------------------------------------------------------------")
    print(f"🚀 开始执行 [{mode_tag}] 评测 (总样本批次: {len(test_loader)})")
    print("--------------------------------------------------------------------------------")

    all_records = []
    per_scan_records = {}
    start_time = time.time()

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

                # 读取同源平面切片掩码
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

                s0_mae = safe_mae(d_est_b_s0, d_gt_b_s0, m_b_s0)
                s1_mae = safe_mae(depth_est_s1[b:b+1], depth_gt_s1[b:b+1], mask_gt_s1[b:b+1]) if depth_est_s1 is not None else float('nan')
                s2_mae = safe_mae(depth_est_s2[b:b+1], depth_gt_s2[b:b+1], mask_gt_s2[b:b+1]) if depth_est_s2 is not None else float('nan')

                t10_err = safe_thres_error(d_est_b_s0, d_gt_b_s0, m_b_s0, 0.10)
                t20_err = safe_thres_error(d_est_b_s0, d_gt_b_s0, m_b_s0, 0.20)
                t50_err = safe_thres_error(d_est_b_s0, d_gt_b_s0, m_b_s0, 0.50)

                s0_planar_mae = safe_mae(d_est_b_s0, d_gt_b_s0, m_planar_b_s0) if has_planar_mask_s0 else float('nan')
                s0_curved_mae = safe_mae(d_est_b_s0, d_gt_b_s0, m_curved_b_s0) if has_planar_mask_s0 else float('nan')

                valid_cnt_s0 = m_b_s0.sum().item()
                planar_cnt_s0 = m_planar_b_s0.sum().item()
                planar_ratio_s0 = (planar_cnt_s0 / valid_cnt_s0 * 100.0) if valid_cnt_s0 > 0 else 0.0

                record = {
                    "idx": len(all_records) + 1,
                    "scan": scan,
                    "region": region,
                    "file_id": file_id,
                    "tag": tag,
                    "s0": s0_mae,
                    "s1": s1_mae,
                    "s2": s2_mae,
                    "err_10cm": t10_err,
                    "err_20cm": t20_err,
                    "err_50cm": t50_err,
                    "s0_planar": s0_planar_mae,
                    "s0_curved": s0_curved_mae,
                    "planar_ratio": planar_ratio_s0,
                }

                per_scan_records[scan].append(record)
                all_records.append(record)

            if (batch_idx + 1) % 6 == 0 or (batch_idx + 1) == len(test_loader):
                cur_rec = all_records[-1]
                print(f"  [{mode_tag}] 进度: {batch_idx + 1:2d}/{len(test_loader)} | 切片: {cur_rec['tag']:22s} | "
                      f"S0: {cur_rec['s0']:.4f}m | S1: {cur_rec['s1']:.4f}m | S2: {cur_rec['s2']:.4f}m")

    elapsed = time.time() - start_time
    print(f"✓ [{mode_tag}] 评测完成，总耗时: {elapsed:.1f}s\n")
    return all_records, per_scan_records


def compute_summary(records):
    """计算全样本平均指标"""
    return {
        "s0": float(np.nanmean([r['s0'] for r in records])),
        "s1": float(np.nanmean([r['s1'] for r in records])),
        "s2": float(np.nanmean([r['s2'] for r in records])),
        "err_10cm": float(np.nanmean([r['err_10cm'] for r in records])) * 100.0,
        "err_20cm": float(np.nanmean([r['err_20cm'] for r in records])) * 100.0,
        "err_50cm": float(np.nanmean([r['err_50cm'] for r in records])) * 100.0,
        "s0_planar": float(np.nanmean([r['s0_planar'] for r in records])),
        "s0_curved": float(np.nanmean([r['s0_curved'] for r in records])),
        "planar_ratio": float(np.nanmean([r['planar_ratio'] for r in records])),
    }


def main():
    args = parse_args()

    # 1. 自动探查数据路径
    if not args.testpath:
        candidates = [
            "/home/myao/ENRICH-Aerial_Cropped_0.6x_Overlap50/",
            "/home/myao/ENRICH-Aerial_Data/",
            "/home/myao/Crop_Test_Dataset/"
        ]
        for c in candidates:
            if os.path.isdir(c):
                args.testpath = c
                break

    if not args.testpath or not os.path.isdir(args.testpath):
        raise FileNotFoundError(f"未找到 ENRICH 数据集根目录，请通过 --testpath 明确指定")

    # 2. 挂载外部 CasMVSNet 源码
    casmvsnet_root = os.path.abspath(args.casmvsnet_code_dir) if args.casmvsnet_code_dir else ""
    if not (casmvsnet_root and os.path.isdir(casmvsnet_root)):
        candidates = [
            "/home/myao/CasMVSNet_pl-master",
            "/home/ym/Experiment/CasMVSNet_pl-master",
            "/home/myao/CasMVSNet",
            "/home/ym/Experiment/CasMVSNet",
            os.path.abspath(os.path.join(PROJECT_ROOT, "..", "CasMVSNet_pl-master")),
            os.path.abspath(os.path.join(PROJECT_ROOT, "..", "CasMVSNet_pl-master", "CasMVSNet_pl-master")),
            os.path.abspath(os.path.join(PROJECT_ROOT, "..", "CasMVSNet")),
        ]
        for c in candidates:
            if os.path.isdir(c):
                casmvsnet_root = c
                print(f"[Auto-Detect] 自动探测并挂载 CasMVSNet 源码路径: {casmvsnet_root}")
                break

    old_models = sys.modules.pop('models', None)
    old_models_sub = {k: sys.modules.pop(k) for k in list(sys.modules.keys()) if k.startswith('models.')}
    if casmvsnet_root and casmvsnet_root not in sys.path:
        sys.path.insert(0, casmvsnet_root)

    try:
        from models.mvsnet import CascadeMVSNet
        from models.modules import homo_warp, depth_regression
    except ImportError as e:
        raise ImportError(f"🚨 无法导入 CascadeMVSNet 核心模块！请检查 --casmvsnet_code_dir。\n错误: {e}")
    finally:
        if casmvsnet_root and casmvsnet_root in sys.path:
            sys.path.remove(casmvsnet_root)
        if old_models is not None:
            sys.modules['models'] = old_models
        sys.modules.update(old_models_sub)

    # 动态修补 CasMVSNet 在 eval 模式下因 in-place 广播操作导致崩溃的 Bug
    def patch_casmvsnet_predict_depth():
        from einops import rearrange, repeat, reduce

        def safe_predict_depth(self, feats, proj_mats, depth_values, cost_reg):
            B, V, C, H, W = feats.shape
            D = depth_values.shape[1]

            ref_feats, src_feats = feats[:, 0], feats[:, 1:]
            src_feats = rearrange(src_feats, 'b vm1 c h w -> vm1 b c h w')
            proj_mats = rearrange(proj_mats, 'b vm1 x y -> vm1 b x y')

            ref_volume = rearrange(ref_feats, 'b c h w -> b c 1 h w')
            ref_volume = repeat(ref_volume, 'b c 1 h w -> b c d h w', d=D)
            if self.G == 1:
                volume_sum = ref_volume.clone()
                volume_sq_sum = ref_volume.pow(2)
            else:
                volume_sum = repeat(ref_feats, 'b (g c) h w -> b g c 1 h w', g=self.G)
                volume_sum = repeat(volume_sum, 'b g c 1 h w -> b g c d h w', d=D).clone()

            for src_feat, proj_mat in zip(src_feats, proj_mats):
                warped_volume = homo_warp(src_feat, proj_mat, depth_values)
                if self.G == 1:
                    volume_sum += warped_volume
                    volume_sq_sum += warped_volume.pow(2)
                else:
                    warped_volume = rearrange(warped_volume, 'b (g c) d h w -> b g c d h w', g=self.G)
                    volume_sum += warped_volume
                del warped_volume
            del src_feats, proj_mats

            if self.G == 1:
                volume_variance = volume_sq_sum.div_(V).sub_(volume_sum.div_(V).pow_(2))
                del volume_sq_sum, volume_sum
            else:
                volume_variance = reduce(volume_sum * ref_volume,
                                         'b g c d h w -> b g d h w', 'mean').div_(V - 1)
                del volume_sum, ref_volume

            cost_reg = rearrange(cost_reg(volume_variance), 'b 1 d h w -> b d h w')
            prob_volume = F.softmax(cost_reg, 1)
            del cost_reg
            depth = depth_regression(prob_volume, depth_values)

            with torch.no_grad():
                prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1),
                                                          pad=(0, 0, 0, 0, 1, 2)),
                                                    (4, 1, 1), stride=1).squeeze(1)
                depth_index = depth_regression(prob_volume,
                                               torch.arange(D,
                                                            device=prob_volume.device,
                                                            dtype=prob_volume.dtype)
                                              ).long()
                depth_index = torch.clamp(depth_index, 0, D - 1)
                confidence = torch.gather(prob_volume_sum4, 1,
                                          depth_index.unsqueeze(1)).squeeze(1)

            return depth, confidence

        CascadeMVSNet.predict_depth = safe_predict_depth
        print("[Compatibility] 已成功挂载 CasMVSNet eval 模式内存安全补丁！")

    patch_casmvsnet_predict_depth()

    # 3. 构建 DataLoader
    MVSDataset = find_dataset_def(args.dataset)
    test_dataset = MVSDataset(args.testpath, args.testlist, mode="test", nviews=args.n_views)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_keep_list, num_workers=args.num_workers,
        pin_memory=True, drop_last=False
    )

    print("=" * 85)
    print("ENRICH-Aerial CasMVSNet (CVPR 2020) A / B 对照诊断评测系统")
    print("=" * 85)
    print(f"  数据集根目录:   {args.testpath}")
    print(f"  测试列表路径:   {args.testlist}")
    print(f"  模型 Checkpoint: {args.loadckpt}")
    print(f"  输入视角数:     {args.n_views}")
    print(f"  BN 诊断模式:    {args.bn_mode.upper()}")
    print(f"  测试样本总数:   {len(test_dataset)} 张")
    print("=" * 85)

    # 4. 构建基准模型并载入权重
    base_model = CasMVSNetWrapper(
        casmvsnet_class=CascadeMVSNet,
        n_depths=args.n_depths,
        interval_ratios=args.interval_ratios,
        num_groups=args.num_groups
    ).to(device)
    load_casmvsnet_ckpt(base_model, args.loadckpt)
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
            if isinstance(m, _BN_TYPES) or hasattr(m, 'track_running_stats') or m.__class__.__name__.endswith('ABN'):
                m.eval()
                if hasattr(m, 'track_running_stats'):
                    m.track_running_stats = True
        records_A, _ = evaluate_single_pass(model_A, test_loader, mode_tag="Mode A (原全局统计)", device=device, args=args)
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
            if isinstance(m, _BN_TYPES) or hasattr(m, 'track_running_stats') or m.__class__.__name__.endswith('ABN'):
                m.train()
                if hasattr(m, 'track_running_stats'):
                    m.track_running_stats = False
                if hasattr(m, 'momentum'):
                    m.momentum = 0.0
        records_B, _ = evaluate_single_pass(model_B, test_loader, mode_tag="Mode B (单图动态统计)", device=device, args=args)
        del model_B
        torch.cuda.empty_cache()

    # =========================================================================
    # 报表生成逻辑
    # =========================================================================
    os.makedirs(args.outdir, exist_ok=True)

    if args.bn_mode == 'both' and records_A is not None and records_B is not None:
        sum_A = compute_summary(records_A)
        sum_B = compute_summary(records_B)

        regions = sorted(list(set(r['region'] for r in records_A)))
        reg_summary = {}
        for reg in regions:
            recs_reg_A = [r for r in records_A if r['region'] == reg]
            recs_reg_B = [r for r in records_B if r['region'] == reg]
            reg_summary[reg] = {
                "count": len(recs_reg_A),
                "s0_A": float(np.nanmean([r['s0'] for r in recs_reg_A])),
                "s0_B": float(np.nanmean([r['s0'] for r in recs_reg_B])),
                "s1_B": float(np.nanmean([r['s1'] for r in recs_reg_B])),
                "s2_B": float(np.nanmean([r['s2'] for r in recs_reg_B])),
                "s0_planar_B": float(np.nanmean([r['s0_planar'] for r in recs_reg_B])),
                "s0_curved_B": float(np.nanmean([r['s0_curved'] for r in recs_reg_B])),
                "planar_ratio_B": float(np.nanmean([r['planar_ratio'] for r in recs_reg_B])),
            }

        s0_gain = ((sum_A['s0'] - sum_B['s0']) / sum_A['s0']) * 100.0 if sum_A['s0'] > 0 else 0.0

        report_md = f"""# ENRICH-Aerial 测试集 A / B 对照与精度评估报告 (CasMVSNet)

- **评估样本集**: `TEST` (列表: `{args.testlist}`)
- **样本总切片数**: {len(test_dataset)} 张
- **Checkpoint**: `{args.loadckpt}`
- **Batch Size**: {args.batch_size} (严格单卡单切片，与训练同构)
- **输入视角数**: {args.n_views} 视角

---

## 🏆 1. 全局多尺度深度精度核心对比表 (Table 1: Global Accuracy)

| 对照组 (Mode) | 运行环境与 BN 机制 | S0 最终 MAE | S1 中层 MAE | S2 粗层 MAE | >10cm 误差率 | >20cm 误差率 | >50cm 误差率 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **模式 A** | 标准 `model.eval()` (原全局 running 统计) | **{sum_A['s0']:.4f} m** | {sum_A['s1']:.4f} m | {sum_A['s2']:.4f} m | {sum_A['err_10cm']:.1f}% | {sum_A['err_20cm']:.1f}% | {sum_A['err_50cm']:.1f}% |
| **模式 B** | `eval()` + BN 动态单图统计 (即时批统计) | **{sum_B['s0']:.4f} m** | **{sum_B['s1']:.4f} m** | **{sum_B['s2']:.4f} m** | **{sum_B['err_10cm']:.1f}%** | **{sum_B['err_20cm']:.1f}%** | **{sum_B['err_50cm']:.1f}%** |

---

## 🎯 2. 诊断结论与关键指标对比

1. **S0 最终精度增益**: 模式 B 将 S0 MAE 从 **{sum_A['s0']:.4f}m** 调整至 **{sum_B['s0']:.4f}m**（相对变动 **{s0_gain:+.1f}%**）。
2. **多尺度收敛对比**:
   - S1 中层 MAE: 模式 A **{sum_A['s1']:.4f}m** vs 模式 B **{sum_B['s1']:.4f}m**
   - S2 粗层 MAE: 模式 A **{sum_A['s2']:.4f}m** vs 模式 B **{sum_B['s2']:.4f}m**
3. **离群误差抑制**:
   - >10cm 误差率: 模式 A **{sum_A['err_10cm']:.1f}%** vs 模式 B **{sum_B['err_10cm']:.1f}%**
   - >20cm 误差率: 模式 A **{sum_A['err_20cm']:.1f}%** vs 模式 B **{sum_B['err_20cm']:.1f}%**
   - >50cm 误差率: 模式 A **{sum_A['err_50cm']:.1f}%** vs 模式 B **{sum_B['err_50cm']:.1f}%**

---

## 📌 3. 测试集各拍摄组 / 区域表现对比表 (Table 2: Regional Breakdown)

| 拍摄组 (Region) | 切片数 | 模式 A (S0 MAE) | 模式 B (S0 MAE) | 模式 B 增益 | S1 MAE (B) | S2 MAE (B) | S0 平面 MAE (B) | 平面占比 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
        for reg, d in reg_summary.items():
            diff_BA = ((d['s0_A'] - d['s0_B']) / d['s0_A']) * 100.0 if d['s0_A'] > 0 else 0.0
            report_md += (f"| `{reg}` | {d['count']} | {d['s0_A']:.4f} m | **{d['s0_B']:.4f} m** | "
                          f"`{diff_BA:+.1f}%` | {d['s1_B']:.4f} m | {d['s2_B']:.4f} m | "
                          f"{d['s0_planar_B']:.4f} m | {d['planar_ratio_B']:.1f}% |\n")

        report_md += f"""
---

## 📋 4. 逐切片详细指标明细表 (Sample Records Preview)

| 序号 | 切片标签 | 模式 A (S0 MAE) | 模式 B (S0 MAE) | S1 (B) | S2 (B) | 模式 B 增益 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
"""
        for i in range(len(records_A)):
            tag = records_A[i]['tag']
            mA = records_A[i]['s0']
            mB = records_B[i]['s0']
            s1_b = records_B[i]['s1']
            s2_b = records_B[i]['s2']
            gain = ((mA - mB) / mA) * 100.0 if mA > 0 else 0.0
            report_md += f"| {i+1} | `{tag}` | {mA:.4f} m | **{mB:.4f} m** | {s1_b:.4f} m | {s2_b:.4f} m | `{gain:+.1f}%` |\n"

        print("\n" + report_md)

        report_file = os.path.join(args.outdir, "casmvsnet_ab_diagnostic_report.md")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_md)
        with open(os.path.join(args.outdir, "casmvsnet_enrich_summary.md"), "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"✓ 对照 Markdown 报告已保存至: {report_file}")

        json_file = os.path.join(args.outdir, "casmvsnet_ab_metrics.json")
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump({
                "summary": {"A": sum_A, "B": sum_B},
                "regions": reg_summary,
                "records_A": records_A,
                "records_B": records_B
            }, f, indent=2)
        print(f"✓ 结构化 JSON 数据已保存至: {json_file}")

    else:
        # 单模式运行 (Mode A 或 Mode B)
        recs = records_A if records_A is not None else records_B
        tag_name = "Mode A" if records_A is not None else "Mode B"
        summ = compute_summary(recs)
        print(f"\n[Summary {tag_name}] S0 MAE: {summ['s0']:.4f}m | S1: {summ['s1']:.4f}m | S2: {summ['s2']:.4f}m | "
              f">10cm: {summ['err_10cm']:.1f}% | >20cm: {summ['err_20cm']:.1f}% | >50cm: {summ['err_50cm']:.1f}%")

    print("=" * 85)


if __name__ == '__main__':
    main()
