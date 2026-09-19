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


def parse_args():
    parser = argparse.ArgumentParser(
        description='WHU-MVS DiffMVS / CasDiffMVS (TPAMI 2025) 从零完整训练适配系统 (遵循 whu-mvs-benchmark-adapter 规范)'
    )
    # 路径与数据
    parser.add_argument('--dataset', default='dtu_whu', help='select dataset (default: dtu_whu)')
    parser.add_argument('--trainpath', default='/home/ym/Experiment/Datas/WHU_MVS_dataset', help='train data path')
    parser.add_argument('--valpath', default=None, help='validation data path (default: same as trainpath)')
    parser.add_argument('--trainlist', default='lists/whu/newtrain.txt', help='training list file')
    parser.add_argument('--vallist', default='lists/whu/minitest.txt',
                        help='validation list file (必须绑定极小集 minitest.txt，防频繁评估拖垮算力)')
    parser.add_argument('--diffmvs_code_dir', default='',
                        help='path to DiffMVS source code directory (e.g. /home/myao/diffmvs-main)')
    parser.add_argument('--logdir', default='./checkpoints/diffmvs_whu_train',
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
    parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
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

    # DiffMVS / CasDiffMVS 网络超参数 (与官方脚本完全对齐)
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
    parser.add_argument('--loss_rate', type=float, default=0.9,
                        help='exponential weighting rate for multi-step loss')
    parser.add_argument('--depth_interals_ratio', nargs="+", type=float, default=[4.0, 2.0, 1.0],
                        help='sampling interval ratio of inverse depth across stages')

    return parser.parse_args()


class DiffMVSTrainWrapper(nn.Module):
    """
    WHU-MVS 训练适配包装器：将 WHU DataLoader 输出动态桥接为 CasDiffMVS 输入
    """
    def __init__(self, casdiffmvs_class, args):
        super().__init__()
        self.numdepth = args.numdepth
        depth_interals_ratio = [int(r) if r.is_integer() else r for r in args.depth_interals_ratio]
        self.net = casdiffmvs_class(
            args=args,
            depth_interals_ratio=depth_interals_ratio,
            test=False
        )

    def forward(self, sample_cuda, depth_gt_ms=None, n_views=5):
        B = sample_cuda["depth_min"].shape[0]
        dev = sample_cuda["depth_min"].device

        # 1. 图像列表: List[B, 3, H, W]
        imgs = [sample_cuda["imgs"]["stage_0"][:, i] for i in range(n_views)]

        # 2. 构造视差空间线性采样网格 [B, numdepth]
        disp_min = 1.0 / sample_cuda["depth_max"].view(B, 1).float()
        disp_max = 1.0 / sample_cuda["depth_min"].view(B, 1).float()
        t = torch.linspace(0.0, 1.0, steps=self.numdepth, device=dev, dtype=torch.float32).view(1, -1)
        depth_values = disp_min + t * (disp_max - disp_min)

        # 3. 构造 DiffMVS 多阶段投影矩阵与内参字典
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

        # 4. 前向计算
        outputs = self.net(
            imgs=imgs,
            proj_matrices=proj_matrices_dict,
            depth_values=depth_values,
            depth_gt_ms=depth_gt_ms
        )
        return outputs, depth_values


def build_multiscale_gt(depth_gt_full, mask_gt_full):
    """
    为 DiffMVS loss 自动构建 4 级下采样的 GT 深度和有效掩码字典
    DiffMVS stage1 (1/8) -> stage2 (1/4) -> stage3 (1/2) -> stage4 (1/1)
    形状均为 [B, H, W]
    """
    depth_gt_ms = {}
    mask_ms = {}

    d_4d = depth_gt_full if depth_gt_full.dim() == 4 else depth_gt_full.unsqueeze(1)
    m_4d = mask_gt_full.float() if mask_gt_full.dim() == 4 else mask_gt_full.float().unsqueeze(1)

    # stage4: 原图 1/1
    depth_gt_ms["stage4"] = d_4d.squeeze(1)
    mask_ms["stage4"] = (m_4d.squeeze(1) > 0.5).float()

    # stage3: 1/2
    d3 = F.interpolate(d_4d, scale_factor=0.5, mode="nearest")
    m3 = F.interpolate(m_4d, scale_factor=0.5, mode="nearest")
    depth_gt_ms["stage3"] = d3.squeeze(1)
    mask_ms["stage3"] = (m3.squeeze(1) > 0.5).float()

    # stage2: 1/4
    d2 = F.interpolate(d_4d, scale_factor=0.25, mode="nearest")
    m2 = F.interpolate(m_4d, scale_factor=0.25, mode="nearest")
    depth_gt_ms["stage2"] = d2.squeeze(1)
    mask_ms["stage2"] = (m2.squeeze(1) > 0.5).float()

    # stage1: 1/8
    d1 = F.interpolate(d_4d, scale_factor=0.125, mode="nearest")
    m1 = F.interpolate(m_4d, scale_factor=0.125, mode="nearest")
    depth_gt_ms["stage1"] = d1.squeeze(1)
    mask_ms["stage1"] = (m1.squeeze(1) > 0.5).float()

    return depth_gt_ms, mask_ms


def main():
    os.chdir(PROJECT_ROOT)
    args = parse_args()
    if args.valpath is None:
        args.valpath = args.trainpath

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print("=" * 85)
    print("WHU-MVS DiffMVS / CasDiffMVS (IEEE TPAMI 2025) 从零训练系统")
    print("=" * 85)
    print_args(args)

    os.makedirs(args.logdir, exist_ok=True)
    tb_writer = SummaryWriter(args.logdir)

    # 1. 动态挂载 DiffMVS 源码
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
        print(f"[Import] 已挂载 DiffMVS 代码路径: {diffmvs_root}")
    else:
        print(f"[Warning] 未指定或未找到 --diffmvs_code_dir: {diffmvs_root}")

    try:
        from models.diffusion import CasDiffMVS
        from models.loss import compute_inverse_loss
    except ImportError as e:
        raise ImportError(f"🚨 无法导入 DiffMVS 核心模块！请检查 --diffmvs_code_dir。\n错误: {e}")

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
    model = DiffMVSTrainWrapper(
        casdiffmvs_class=CasDiffMVS,
        args=args
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
        clean_state = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}
        missing, unexpected = model.net.load_state_dict(clean_state, strict=False)
        if len(missing) > 0:
            print(f"[Model Warning] 缺失参数: {missing[:5]}")
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
    print("=" * 85)
    print("开始从零训练 DiffMVS / CasDiffMVS (WHU-MVS)...")
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

            # 构建多尺度 GT
            depth_gt_full = sample_cuda["depth"]["stage_0"]
            mask_gt_full = (sample_cuda["mask"]["stage_0"] > 0.5)
            depth_gt_ms, mask_ms = build_multiscale_gt(depth_gt_full, mask_gt_full)

            # 前向推理
            outputs, depth_values = model(sample_cuda, depth_gt_ms=depth_gt_ms, n_views=args.n_views)

            # 计算 DiffMVS 逆深度损失 (L1 + 扩散置信度不确定性加权)
            loss, depth_loss_dict = compute_inverse_loss(
                args=args,
                inputs=outputs["depth"],
                confs=outputs["conf"],
                depth_gt_ms=depth_gt_ms,
                mask_ms=mask_ms,
                depth_values=depth_values,
                loss_rate=args.loss_rate,
                iters=args.stage_iters
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            if args.lr_sche == "onecycle":
                lr_scheduler.step()

            global_step += 1
            step_time = time.time() - step_start

            # 打印与记录 TensorBoard 标量
            if global_step % args.summary_freq == 0:
                d_est = outputs["depth"][-1]
                d_gt = depth_gt_ms["stage4"]
                m = mask_ms["stage4"] > 0.5
                mae = torch.mean((d_est[m] - d_gt[m]).abs()).item() if m.any() else 0.0

                tb_writer.add_scalar('train/loss', loss.item(), global_step)
                tb_writer.add_scalar('train/mae', mae, global_step)
                tb_writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)

                print(f"Epoch {epoch_idx:02d} [{batch_idx:04d}/{len(train_loader):04d}] | "
                      f"Loss: {loss.item():.4f} | S0 MAE: {mae:.4f}m | Time: {step_time:.2f}s")

        if args.lr_sche == "mslr":
            lr_scheduler.step()

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch_idx:02d} 训练耗时: {epoch_time:.1f}s")

        # 保存权重
        if (epoch_idx + 1) % args.save_freq == 0 or epoch_idx == args.epochs - 1:
            save_path = os.path.join(args.logdir, f"model_{epoch_idx:06d}.ckpt")
            torch.save({
                'epoch': epoch_idx,
                'model': model.net.state_dict(),
                'optimizer': optimizer.state_dict()
            }, save_path)
            print(f"[Checkpoint] 已保存模型权重至: {save_path}")

        # 低频轻量验证 (严格遵循规范五：每 eval_freq 轮运行 80 张 minitest)
        if (epoch_idx + 1) % args.eval_freq == 0 or epoch_idx == args.epochs - 1:
            print(f"\n--- 运行极速过程验证 (minitest: {len(val_loader)} 张) ---")
            model.eval()
            val_maes = []
            with torch.no_grad():
                for v_sample in val_loader:
                    v_cuda = tocuda(v_sample, device=device, skip_keys=skip)
                    v_outputs, _ = model(v_cuda, depth_gt_ms=None, n_views=args.n_views)
                    v_est = v_outputs["depth"][-1]
                    v_gt = v_cuda["depth"]["stage_0"].squeeze(1)
                    v_m = (v_cuda["mask"]["stage_0"].squeeze(1) > 0.5)
                    if v_m.any():
                        val_maes.append(torch.mean((v_est[v_m] - v_gt[v_m]).abs()).item())
            mean_val_mae = float(np.mean(val_maes)) if val_maes else float('nan')
            tb_writer.add_scalar('val/mae', mean_val_mae, epoch_idx)
            print(f">>> [Validation] Epoch {epoch_idx:02d} Minitest 平均 MAE: {mean_val_mae:.4f}m\n")

    print("\n" + "=" * 85)
    print("DiffMVS / CasDiffMVS 训练彻底完成！")
    print(f"终极权重保存在: {args.logdir}/model_{args.epochs-1:06d}.ckpt")
    print("接下来可直接使用 test_whu_diffmvs.sh 执行最终收官基准评测！")
    print("=" * 85)


if __name__ == '__main__':
    main()
