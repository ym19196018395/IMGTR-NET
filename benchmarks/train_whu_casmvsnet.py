import argparse
import os
import sys
import time
import json
import math
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
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import find_dataset_def
from datasets.dtu_whu import collate_keep_list
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
        description='WHU-MVS CasMVSNet (CVPR 2020) 从零完整训练适配系统 (遵循 whu-mvs-benchmark-adapter 规范)'
    )
    # 路径与数据
    parser.add_argument('--dataset', default='dtu_whu', help='select dataset (default: dtu_whu)')
    parser.add_argument('--trainpath', default='/home/ym/Experiment/Datas/WHU_MVS_dataset', help='train data path')
    parser.add_argument('--valpath', default=None, help='validation data path (default: same as trainpath)')
    parser.add_argument('--trainlist', default='lists/whu/newtrain.txt', help='training list file')
    parser.add_argument('--vallist', default='lists/whu/minitest.txt',
                        help='validation list file (必须绑定极小集 minitest.txt，防频繁评估拖垮算力)')
    parser.add_argument('--casmvsnet_code_dir', default='',
                        help='path to CasMVSNet source code directory (e.g. /home/myao/CasMVSNet_pl-master)')
    parser.add_argument('--logdir', default='./checkpoints/casmvsnet_whu_train',
                        help='directory to save checkpoints and tensorboard logs')
    parser.add_argument('--loadckpt', default=None, help='resume training from specific checkpoint')
    parser.add_argument('--resume', action='store_true', help='auto-resume training from last checkpoint')

    # 训练超参数
    parser.add_argument('--epochs', type=int, default=16, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--lr_sche', type=str, default='onecycle', choices=['onecycle', 'mslr'],
                        help='learning rate schedule (onecycle or mslr)')
    parser.add_argument('--lrepochs', type=str, default="10,12,14:2",
                        help='for mslr: epoch ids to downscale lr and downscale rate')
    parser.add_argument('--wd', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='train batch size (建议 1 或 2，防 OOM)')
    parser.add_argument('--n_views', type=int, default=5, help='number of views per sample')
    parser.add_argument('--num_workers', type=int, default=4, help='workers for DataLoader')
    parser.add_argument('--seed', type=int, default=123, help='random seed')

    # 训练节奏控制 (严格遵循规范五：轻量过程监控)
    parser.add_argument('--summary_freq', type=int, default=20, help='print log frequency')
    parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency (epochs)')
    parser.add_argument('--eval_freq', type=int, default=2,
                        help='validation frequency (epochs，建议 2~4，严禁每轮全量验证)')

    # CasMVSNet 网络超参数 (与官方脚本完全对齐)
    parser.add_argument('--n_depths', nargs='+', type=int, default=[8, 32, 48],
                        help='number of depth hypotheses in each stage [fine, medium, coarse]')
    parser.add_argument('--interval_ratios', nargs='+', type=float, default=[1.0, 2.0, 4.0],
                        help='depth interval ratio to multiply with base depth_interval in each stage')
    parser.add_argument('--num_groups', type=int, default=1, choices=[1, 2, 4, 8],
                        help='number of groups in groupwise correlation, must be a divisor of 8 (default: 1)')

    return parser.parse_args()


class CasMVSNetTrainWrapper(nn.Module):
    """
    WHU-MVS 训练适配包装器：将 WHU DataLoader 输出动态桥接为 CasMVSNet 输入
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

        # 1. 图像标准化: [B, V, 3, H, W]
        raw_imgs = sample_cuda["imgs"]["stage_0"][:, :n_views]
        imgs = (raw_imgs - self.img_mean) / self.img_std
        _, _, _, H, W = imgs.shape

        # 2. 分辨率对齐 (CasMVSNet 需能被 32 整除)
        pad_h = (32 - H % 32) % 32
        pad_w = (32 - W % 32) % 32
        if pad_h > 0 or pad_w > 0:
            imgs = imgs.view(B * n_views, 3, H, W)
            imgs = F.pad(imgs, (0, pad_w, 0, pad_h), mode='replicate')
            imgs = imgs.view(B, n_views, 3, H + pad_h, W + pad_w)

        # 3. 构造 3 阶段多尺度相对投影矩阵 [B, V-1, 3, 3, 4] (fine to coarse)
        P_0 = sample_cuda["proj_matrices"]["stage_0"][:, :n_views]  # 1/1
        P_1 = sample_cuda["proj_matrices"]["stage_1"][:, :n_views]  # 1/2
        P_2 = sample_cuda["proj_matrices"]["stage_2"][:, :n_views]  # 1/4
        P_levels = torch.stack([P_0, P_1, P_2], dim=2)

        ref_proj = P_levels[:, 0]
        ref_proj_inv = torch.inverse(ref_proj)

        proj_mats = []
        for i in range(1, n_views):
            src_proj = P_levels[:, i]
            rel_proj = torch.matmul(src_proj, ref_proj_inv)
            proj_mats.append(rel_proj[:, :, :3, :4])
        proj_mats = torch.stack(proj_mats, dim=1)

        # 4. 深度假设步长 (CasMVSNet 内部要求形状必须为 (B, 1) 以支持 (B, 1) * (1, D) -> (B, D) 广播及 einops 'b 1 -> b 1 1 1')
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

        return outputs, (H, W, pad_h, pad_w)


def safe_smooth_l1(pred, gt, mask):
    """安全 Smooth L1 损失计算：无有效像素时安全返回 0.0 防止 NaN 污染梯"""
    if mask is None or not mask.any():
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return F.smooth_l1_loss(pred[mask], gt[mask], reduction='mean')


def compute_casmvsnet_loss(outputs, pad_info, sample_cuda):
    """
    计算 CasMVSNet 多尺度 Smooth L1 深度损失
    按官方标准权重: level 0 (细): 2.0, level 1 (中): 1.0, level 2 (粗): 0.5
    """
    H, W, pad_h, pad_w = pad_info

    # 多尺度 GT 与 Mask 映射 (level 0: stage_0, level 1: stage_1, level 2: stage_2)
    targets = {
        0: sample_cuda["depth"]["stage_0"].squeeze(1),
        1: sample_cuda["depth"]["stage_1"].squeeze(1),
        2: sample_cuda["depth"]["stage_2"].squeeze(1),
    }
    masks = {
        0: (sample_cuda["mask"]["stage_0"].squeeze(1) > 0.5),
        1: (sample_cuda["mask"]["stage_1"].squeeze(1) > 0.5),
        2: (sample_cuda["mask"]["stage_2"].squeeze(1) > 0.5),
    }

    # 官方权重: 2**(1-l) -> [2.0, 1.0, 0.5]
    loss_weights = [2.0, 1.0, 0.5]
    total_loss = 0.0
    loss_dict = {}

    for l in range(3):
        pred_l = outputs[f"depth_{l}"]
        if pad_h > 0 or pad_w > 0:
            scale = 2 ** l
            h_l = H // scale
            w_l = W // scale
            pred_l = pred_l[:, :h_l, :w_l]

        gt_l = targets[l]
        mask_l = masks[l]

        stage_loss = safe_smooth_l1(pred_l, gt_l, mask_l)
        total_loss = total_loss + stage_loss * loss_weights[l]
        loss_dict[f"loss_level_{l}"] = stage_loss.item()

    loss_dict["total_loss"] = total_loss.item()
    return total_loss, loss_dict


def safe_mae(depth_est, depth_gt, mask):
    """计算绝对深度误差 MAE (m)"""
    if mask is None or not mask.any():
        return float('nan')
    est_valid = depth_est[mask]
    gt_valid = depth_gt[mask]
    return torch.mean((est_valid - gt_valid).abs()).item()


def safe_thres_error(depth_est, depth_gt, mask, thres):
    """误差大于特定阈值 (m) 的离群率"""
    if mask is None or not mask.any():
        return float('nan')
    est_valid = depth_est[mask]
    gt_valid = depth_gt[mask]
    err = (est_valid - gt_valid).abs()
    return (err > thres).float().mean().item()


def main():
    os.chdir(PROJECT_ROOT)
    args = parse_args()
    if args.valpath is None:
        args.valpath = args.trainpath

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print("=" * 85)
    print("WHU-MVS CasMVSNet (CVPR 2020) 从零训练系统")
    print("=" * 85)
    print_args(args)

    os.makedirs(args.logdir, exist_ok=True)
    tb_writer = SummaryWriter(args.logdir)

    # 1. 动态挂载 CasMVSNet 源码
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
        print(f"[Import] 已挂载 CasMVSNet 代码路径: {casmvsnet_root}")
    else:
        print(f"[Warning] 未指定或未找到 --casmvsnet_code_dir: {casmvsnet_root}")

    try:
        from models.mvsnet import CascadeMVSNet
    except ImportError as e:
        raise ImportError(f"🚨 无法导入 CascadeMVSNet 核心模块！请检查 --casmvsnet_code_dir。\n错误: {e}")

    # 2. 构建 DataLoader
    MVSDataset = find_dataset_def(args.dataset)
    print(f"\n[Dataset] 加载训练集: {args.trainlist}")
    train_dataset = MVSDataset(args.trainpath, args.trainlist, "train", nviews=args.n_views, robust_train=False)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_keep_list, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=(args.num_workers > 0), drop_last=True
    )

    print(f"[Dataset] 加载轻量验证集: {args.vallist} (严格遵守规范五：仅绑极小集监控)")
    val_dataset = MVSDataset(args.valpath, args.vallist, "test", nviews=args.n_views, robust_train=False)
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        collate_fn=collate_keep_list, num_workers=2,
        pin_memory=True, drop_last=False
    )
    print(f"[Dataset] 训练样本: {len(train_dataset)} | 验证样本: {len(val_dataset)}\n")

    # 3. 实例化适配模型与优化器
    model = CasMVSNetTrainWrapper(
        casmvsnet_class=CascadeMVSNet,
        n_depths=args.n_depths,
        interval_ratios=args.interval_ratios,
        num_groups=args.num_groups
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd,
        eps=1e-8
    )

    start_epoch = 0
    if args.resume and not args.loadckpt:
        saved_ckpts = [f for f in os.listdir(args.logdir) if f.startswith("model_") and f.endswith(".ckpt")]
        if len(saved_ckpts) > 0:
            saved_ckpts = sorted(saved_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            args.loadckpt = os.path.join(args.logdir, saved_ckpts[-1])
            print(f"[Auto-Resume] 自动发现最新权重: {args.loadckpt}")

    if args.loadckpt:
        print(f"[Model] 载入已有 Checkpoint: {args.loadckpt}")
        ckpt = torch.load(args.loadckpt, map_location=device)
        state_dict = ckpt.get('model', ckpt.get('state_dict', ckpt))
        clean_state = {}
        for k, v in state_dict.items():
            clean_k = k
            if clean_k.startswith('model.'):
                clean_k = clean_k[6:]
            if clean_k.startswith('module.'):
                clean_k = clean_k[7:]
            clean_state[clean_k] = v

        missing, unexpected = model.net.load_state_dict(clean_state, strict=False)
        if len(missing) > 0:
            print(f"[Model Warning] 缺失参数 (前5个): {missing[:5]}")
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'epoch' in ckpt:
            start_epoch = ckpt['epoch'] + 1
            print(f"[Resume] 从 Epoch {start_epoch} 继续训练...")

    # 学习率调度器
    if args.lr_sche == "mslr":
        milestones = [int(ep) for ep in args.lrepochs.split(':')[0].split(',')]
        lr_gamma = 1.0 / float(args.lrepochs.split(':')[1])
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=lr_gamma, last_epoch=start_epoch - 1
        )
    elif args.lr_sche == "onecycle":
        total_steps = len(train_loader) * args.epochs + 100
        last_step = len(train_loader) * start_epoch - 1 if start_epoch > 0 else -1
        lr_scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, total_steps=total_steps,
            pct_start=0.05, cycle_momentum=False, anneal_strategy='linear',
            last_epoch=last_step
        )

    # 4. 主训练循环
    global_step = start_epoch * len(train_loader)
    best_val_mae = float('inf')
    print("=" * 85)
    print("开始从零训练 CasMVSNet (WHU-MVS)...")
    print("=" * 85)

    for epoch_idx in range(start_epoch, args.epochs):
        model.train()
        epoch_start = time.time()
        print(f"\n>>> Epoch {epoch_idx}/{args.epochs} | 当前 LR: {optimizer.param_groups[0]['lr']:.6f}")

        for batch_idx, sample in enumerate(train_loader):
            step_start = time.time()
            optimizer.zero_grad()

            skip = ["vertexs", "lines", "triangles", "tri_conf_cleaned", "tri_normal_cleaned", "is_gt_planar"]
            sample_cuda = tocuda(sample, device=device, skip_keys=skip)

            # 前向推理
            outputs, pad_info = model(sample_cuda, n_views=args.n_views)

            # 计算多尺度 Smooth L1 损失
            loss, loss_dict = compute_casmvsnet_loss(outputs, pad_info, sample_cuda)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            if args.lr_sche == "onecycle":
                lr_scheduler.step()

            step_time = time.time() - step_start

            # 日志记录
            if batch_idx % args.summary_freq == 0:
                print(f"[Epoch {epoch_idx:02d} | Step {batch_idx:03d}/{len(train_loader):03d}] "
                      f"Total Loss: {loss_dict['total_loss']:.4f} | "
                      f"L0: {loss_dict['loss_level_0']:.4f} | "
                      f"L1: {loss_dict['loss_level_1']:.4f} | "
                      f"L2: {loss_dict['loss_level_2']:.4f} | "
                      f"Time: {step_time:.2f}s")

                tb_writer.add_scalar("train/total_loss", loss_dict["total_loss"], global_step)
                tb_writer.add_scalar("train/loss_level_0", loss_dict["loss_level_0"], global_step)
                tb_writer.add_scalar("train/loss_level_1", loss_dict["loss_level_1"], global_step)
                tb_writer.add_scalar("train/loss_level_2", loss_dict["loss_level_2"], global_step)
                tb_writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

            global_step += 1

        if args.lr_sche == "mslr":
            lr_scheduler.step()

        epoch_duration = time.time() - epoch_start
        print(f">>> Epoch {epoch_idx} 完成，耗时: {epoch_duration:.1f}s")

        # 5. 轻量验证集监控 (严格遵守规范五：防频繁评估拖垮算力)
        if (epoch_idx + 1) % args.eval_freq == 0 or epoch_idx == args.epochs - 1:
            model.eval()
            print(f"\n--- 开始轻量验证集评估 (Epoch {epoch_idx}) ---")
            val_maes = []
            val_t1s = []
            val_t2s = []

            with torch.no_grad():
                for v_idx, v_sample in enumerate(val_loader):
                    v_cuda = tocuda(v_sample, device=device, skip_keys=skip)
                    v_outputs, v_pad_info = model(v_cuda, n_views=args.n_views)

                    H_v, W_v, pad_h_v, pad_w_v = v_pad_info
                    d_pred = v_outputs["depth_0"]
                    if pad_h_v > 0 or pad_w_v > 0:
                        d_pred = d_pred[:, :H_v, :W_v]

                    d_gt = v_cuda["depth"]["stage_0"].squeeze(1)
                    m_gt = (v_cuda["mask"]["stage_0"].squeeze(1) > 0.5)

                    v_mae = safe_mae(d_pred, d_gt, m_gt)
                    v_t1 = safe_thres_error(d_pred, d_gt, m_gt, 1.0)
                    v_t2 = safe_thres_error(d_pred, d_gt, m_gt, 2.0)

                    if not math.isnan(v_mae):
                        val_maes.append(v_mae)
                    if not math.isnan(v_t1):
                        val_t1s.append(v_t1)
                    if not math.isnan(v_t2):
                        val_t2s.append(v_t2)

            mean_val_mae = float(np.mean(val_maes)) if len(val_maes) > 0 else float('nan')
            mean_val_t1 = float(np.mean(val_t1s)) if len(val_t1s) > 0 else float('nan')
            mean_val_t2 = float(np.mean(val_t2s)) if len(val_t2s) > 0 else float('nan')

            print(f"[Validation Epoch {epoch_idx}] MAE: {mean_val_mae:.4f}m | >1m: {mean_val_t1*100:.2f}% | >2m: {mean_val_t2*100:.2f}%")
            tb_writer.add_scalar("val/mae", mean_val_mae, epoch_idx)
            tb_writer.add_scalar("val/thres1m_err", mean_val_t1, epoch_idx)
            tb_writer.add_scalar("val/thres2m_err", mean_val_t2, epoch_idx)

            if mean_val_mae < best_val_mae:
                best_val_mae = mean_val_mae
                best_ckpt_path = os.path.join(args.logdir, "model_best.ckpt")
                torch.save({
                    'epoch': epoch_idx,
                    'model': model.net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'val_mae': best_val_mae
                }, best_ckpt_path)
                print(f"🌟 [New Best] 最佳模型已保存至: {best_ckpt_path} (MAE: {best_val_mae:.4f}m)")

        # 6. 保存定期 Checkpoint
        if (epoch_idx + 1) % args.save_freq == 0 or epoch_idx == args.epochs - 1:
            save_path = os.path.join(args.logdir, f"model_{epoch_idx:02d}.ckpt")
            torch.save({
                'epoch': epoch_idx,
                'model': model.net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, save_path)
            print(f"[Checkpoint] 已保存模型权重至: {save_path}")

    print("\n" + "=" * 85)
    print("CasMVSNet 训练流程全部顺利结束！")
    print(f"最优验证集 MAE: {best_val_mae:.4f}m")
    print(f"权重保存目录: {args.logdir}")
    print("=" * 85)


if __name__ == '__main__':
    main()
