import argparse
import os
import sys
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
        description="ENRICH-Aerial_Data 航空遥感场景 CasMVSNet (CVPR 2020) 独立基准评测系统"
    )
    # 数据集与路径
    parser.add_argument('--dataset', default='enrich_aerial', type=str, help='数据集定义名称')
    parser.add_argument('--testpath', default="/home/myao/ENRICH-Aerial_Data", type=str, help='ENRICH 数据集根目录')
    parser.add_argument('--testlist', default="/home/myao/ENRICH-Aerial_Data/scan_list.txt", type=str, help='测试场景列表文件')
    parser.add_argument('--loadckpt', required=True, type=str, help='CasMVSNet Checkpoint 权重路径 (.ckpt 或 .pth)')
    parser.add_argument('--casmvsnet_code_dir', default="", type=str,
                        help='外部 CasMVSNet 源码目录 (如 /home/myao/CasMVSNet_pl-master)')
    parser.add_argument('--mask_dir', default="./outputs_enrich_aerial", type=str,
                        help='同源平面掩码目录 (由 test_enrich_aerial.py 生成的 stage0/1_planar_mask_*.png)')
    parser.add_argument('--outdir', default="./outputs_casmvsnet_enrich", type=str, help='评测报告与结果保存目录')

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
    ENRICH-Aerial_Data 航测数据基准评测适配包装器：将 DataLoader 输出动态转换为 CasMVSNet 输入
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
        # level 0: stage_0 (1/1)
        # level 1: stage_1 (1/2)
        # level 2: stage_2 (1/4)
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

        # 4. 深度范围与粗阶段假设步长计算 (CasMVSNet 要求形状必须为 [B, 1])
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
        # Stage 0: 全分辨率 [B, 1, H, W]
        depth_pred_s0 = outputs["depth_0"]
        if pad_h > 0 or pad_w > 0:
            depth_pred_s0 = depth_pred_s0[:, :H, :W]
        depth_pred_s0 = depth_pred_s0.unsqueeze(1)

        # Stage 1: 半分辨率 [B, 1, H//2, W//2]
        depth_pred_s1 = None
        if "depth_1" in outputs:
            d1 = outputs["depth_1"]
            if pad_h > 0 or pad_w > 0:
                d1 = d1[:, :H // 2, :W // 2]
            depth_pred_s1 = d1.unsqueeze(1)

        # Stage 2: 1/4 分辨率 [B, 1, H//4, W//4]
        depth_pred_s2 = None
        if "depth_2" in outputs:
            d2 = outputs["depth_2"]
            if pad_h > 0 or pad_w > 0:
                d2 = d2[:, :H // 4, :W // 4]
            depth_pred_s2 = d2.unsqueeze(1)

        # 全分辨率置信度
        confidence = None
        if "confidence_0" in outputs:
            conf = outputs["confidence_0"]
            if pad_h > 0 or pad_w > 0:
                conf = conf[:, :H, :W]
            confidence = conf.unsqueeze(1)

        return depth_pred_s0, depth_pred_s1, depth_pred_s2, confidence


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


def load_casmvsnet_ckpt(model, ckpt_path):
    """兼容加载 PyTorch-Lightning 与原生 PyTorch 权重文件"""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"🚨 Checkpoint 文件不存在: {ckpt_path}")

    print(f"[Model] 载入 CasMVSNet 权重: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    if 'state_dict' in checkpoint:
        raw_state = checkpoint['state_dict']
    elif 'model' in checkpoint:
        raw_state = checkpoint['model']
    else:
        raw_state = checkpoint

    cleaned_state = {}
    for k, v in raw_state.items():
        clean_k = k
        if clean_k.startswith('module.'):
            clean_k = clean_k[7:]
        if clean_k.startswith('model.'):
            clean_k = clean_k[6:]
        if not clean_k.startswith('net.'):
            clean_k = 'net.' + clean_k
        cleaned_state[clean_k] = v

    missing, unexpected = model.load_state_dict(cleaned_state, strict=False)
    if len(missing) > 0:
        print(f"[Model Warning] 缺失参数 (前5个): {missing[:5]}")
    if len(unexpected) > 0:
        print(f"[Model Warning] 冗余参数 (前5个): {unexpected[:5]}")
    print("✓ CasMVSNet 模型权重载入成功！")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print("=" * 85)
    print("ENRICH-Aerial_Data 航测大场景 CasMVSNet 独立评测系统 (严格 3 视角纯推理)")
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
            os.path.join(PROJECT_ROOT, "..", "CasMVSNet_pl-master"),
            os.path.join(PROJECT_ROOT, "..", "CasMVSNet"),
        ]
        for c in candidates:
            if os.path.isdir(c):
                casmvsnet_root = os.path.abspath(c)
                break

    if not (casmvsnet_root and os.path.isdir(casmvsnet_root)):
        raise RuntimeError(
            f"🚨 未找到外部 CasMVSNet 源码目录！请通过 --casmvsnet_code_dir 指定。"
        )

    print(f"[External] 动态挂载 CasMVSNet 源码目录: {casmvsnet_root}")

    # 动态安全挂载外部 CasMVSNet 的 models.mvsnet，杜绝与主仓 models 命名空间冲突
    old_models = sys.modules.pop('models', None)
    old_models_sub = {k: sys.modules.pop(k) for k in list(sys.modules.keys()) if k.startswith('models.')}
    if casmvsnet_root and casmvsnet_root not in sys.path:
        sys.path.insert(0, casmvsnet_root)

    try:
        from models.mvsnet import CascadeMVSNet
        from models.modules import homo_warp, depth_regression
    except ImportError as e:
        raise ImportError(
            f"🚨 无法导入 CascadeMVSNet 模型！请确认 --casmvsnet_code_dir 参数指向正确的 CasMVSNet 源码目录。\n"
            f"原始错误: {e}"
        )
    finally:
        if casmvsnet_root and casmvsnet_root in sys.path:
            sys.path.remove(casmvsnet_root)
        if old_models is not None:
            sys.modules['models'] = old_models
        sys.modules.update(old_models_sub)

    # 动态修复原版 CasMVSNet 在 eval 模式下因 in-place 操作广播张量导致的崩溃问题
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
                volume_sq_sum = ref_volume ** 2
            else:
                ref_volume = ref_volume.view(B, self.G, C // self.G, *ref_volume.shape[-3:])
                volume_sum = 0
            del ref_feats

            for src_feat, proj_mat in zip(src_feats, proj_mats):
                warped_volume = homo_warp(src_feat, proj_mat, depth_values)
                warped_volume = warped_volume.to(ref_volume.dtype)
                if self.G == 1:
                    volume_sum = volume_sum + warped_volume
                    volume_sq_sum = volume_sq_sum + warped_volume ** 2
                else:
                    warped_volume = warped_volume.view_as(ref_volume)
                    if self.training:
                        volume_sum = volume_sum + warped_volume
                    else:
                        volume_sum += warped_volume
                del warped_volume, src_feat, proj_mat
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

    # 3. 实例化模型并载入权重
    model = CasMVSNetWrapper(
        casmvsnet_class=CascadeMVSNet,
        n_depths=args.n_depths,
        interval_ratios=args.interval_ratios,
        num_groups=args.num_groups
    ).to(device)

    load_casmvsnet_ckpt(model, args.loadckpt)
    model.eval()

    per_scan_records = {}
    all_sample_records = []
    start_eval_time = time.time()

    print("\n" + "=" * 85)
    print("开始执行 ENRICH-Aerial_Data CasMVSNet 逐场景推断与 Table 1 / Table 2 精度评测...")
    print("=" * 85)

    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            step_start = time.time()
            B = sample['imgs']['stage_0'].shape[0]

            skip = ["vertexs", "lines", "triangles", "scan", "file_id"]
            sample_cuda = tocuda(sample, device=device, skip_keys=skip)

            # 前向推理: 输出 Stage 0, Stage 1, Stage 2 深度预测
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
                    save_pfm(os.path.join(depth_save_dir, f"casmvsnet_depth_{file_id}.pfm"), d_s0_np)
                    depth_png_s0 = np.clip(d_s0_np * 64.0, 0, 65535).astype(np.uint16)
                    cv2.imwrite(os.path.join(depth_save_dir, f"casmvsnet_depth_{file_id}.png"), depth_png_s0)

            cur_idx = batch_idx + 1
            step_time = time.time() - step_start
            p0_str = f"{record['stage0_planar_mae']:.4f}m" if not math.isnan(record['stage0_planar_mae']) else "N/A"
            p1_str = f"{record['stage1_planar_mae']:.4f}m" if not math.isnan(record['stage1_planar_mae']) else "N/A"
            s0_mae_str = f"{record['stage0_mae']:.4f}m" if not math.isnan(record['stage0_mae']) else "N/A"
            print(f"[{cur_idx:03d}/{len(test_loader):03d}] Scan: {scan} | Img: {file_id} | "
                  f"CasMVSNet S0 MAE: {s0_mae_str} | S0 Planar: {p0_str} | S1 Planar: {p1_str} | Time: {step_time:.2f}s")

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
    report_lines.append("# ENRICH-Aerial_Data 航空遥感场景基准评测报告: CasMVSNet (CVPR 2020)\n")
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
        f"| **CasMVSNet (总计)** | **{overall_summary['count']}** | **{overall_summary['stage0_mae']:.4f}** | "
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
        f"| **CasMVSNet (总计)** | **{op0_str}** | **{oc0_str}** | **{op1_str}** | **{oc1_str}** | "
        f"**{overall_summary['s0_planar_ratio_pct']:.2f}%** | **{overall_summary['s1_planar_ratio_pct']:.2f}%** |"
    )

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    # 保存报告与指标 JSON
    report_file = os.path.join(args.outdir, "casmvsnet_enrich_summary.md")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n[Output] 完整评估报表已保存至: {report_file}")

    json_file = os.path.join(args.outdir, "casmvsnet_enrich_records.json")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump({"overall": overall_summary, "by_scan": summary_by_scan, "records": all_sample_records}, f, indent=2)
    print(f"[Output] 详细样本指标记录已保存至: {json_file}")


if __name__ == '__main__':
    main()
