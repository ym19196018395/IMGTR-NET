import argparse
import os
import sys
import time
import json
import math
import numpy as np

# 控制 GPU ID：优先使用外部环境变量 GPU_ID，其次 CUDA_VISIBLE_DEVICES，默认 '0'
_gpu_choice = os.environ.get('GPU_ID', os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
os.environ['CUDA_VISIBLE_DEVICES'] = str(_gpu_choice)

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import find_dataset_def
from datasets.dtu_whu import collate_keep_list
from utils import print_args, tocuda, DictAverageMeter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(
        description='WHU-MVS GeoMVSNet 从零完整训练适配系统 (遵循 whu-mvs-benchmark-adapter 规范)'
    )
    # 路径与数据
    parser.add_argument('--dataset', default='dtu_whu', help='select dataset (default: dtu_whu)')
    parser.add_argument('--trainpath', default='/home/ym/Experiment/Datas/WHU_MVS_dataset', help='train data path')
    parser.add_argument('--valpath', default=None, help='validation data path (default: same as trainpath)')
    parser.add_argument('--trainlist', default='lists/whu/train.txt', help='training list file')
    parser.add_argument('--vallist', default='lists/whu/minitest.txt',
                        help='validation list file (必须绑定极小集 minitest.txt，防频繁评估拖垮算力)')
    parser.add_argument('--geomvsnet_code_dir', default='',
                        help='path to GeoMVSNet source code directory')
    parser.add_argument('--logdir', default='./checkpoints/geomvsnet_train',
                        help='directory to save checkpoints and tensorboard logs')
    parser.add_argument('--loadckpt', default=None, help='resume training from specific checkpoint')
    parser.add_argument('--resume', action='store_true', help='auto-resume training from last checkpoint')

    # 训练超参数
    parser.add_argument('--epochs', type=int, default=16, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--lrepochs', type=str, default="1,3,5,7,9,11,13,15:1.5",
                        help='epoch ids to downscale lr and the downscale rate (GeoMVSNet 官方配置)')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='train batch size (GeoMVSNet 显存较大，建议 1 或 2，防 OOM)')
    parser.add_argument('--n_views', type=int, default=5, help='number of views per sample')
    parser.add_argument('--num_workers', type=int, default=4, help='workers for DataLoader')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    # 训练节奏控制 (规范五：轻量过程监控)
    parser.add_argument('--summary_freq', type=int, default=20, help='print log frequency')
    parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency (epochs)')
    parser.add_argument('--eval_freq', type=int, default=2,
                        help='validation frequency (epochs，建议 2~4，严禁每轮验证)')

    # GeoMVSNet 网络超参数 (与官方 opts.py 完全对齐)
    parser.add_argument('--levels', type=int, default=4, help='levels of cascade stages')
    parser.add_argument('--hypo_plane_num_stages', nargs='+', type=int, default=[8, 8, 4, 4],
                        help='number of depth hypothesis planes for stages 1 to 4 (default: 8 8 4 4)')
    parser.add_argument('--depth_interal_ratio_stages', nargs='+', type=float, default=[0.5, 0.5, 0.5, 1.0],
                        help='depth interval ratio for stages 1 to 4 (default: 0.5 0.5 0.5 1.0)')
    parser.add_argument('--feat_base_channel', type=int, default=8, help='base channels of FPN')
    parser.add_argument('--reg_base_channel', type=int, default=8, help='base channels of 2D RegNet')
    parser.add_argument('--group_cor_dim_stages', nargs='+', type=int, default=[8, 8, 4, 4],
                        help='group correlation dimensions for stages 1 to 4 (default: 8 8 4 4)')
    parser.add_argument('--stage_lw', type=str, default="1.0,1.0,1.0,1.0",
                        help='loss weights for stages 1 to 4')

    return parser.parse_args()


class GeoMVSNetTrainWrapper(nn.Module):
    """
    WHU-MVS 训练适配包装器：动态桥接 WHU DataLoader 与 GeoMVSNet 网络层级
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

        # 1. 转换图像: List[B, 3, H, W] (原图 stage_0)
        imgs = [sample_cuda["imgs"]["stage_0"][:, i] for i in range(n_views)]

        # 2. 构造绝对米制深度范围: [B, 2]
        depth_values = torch.stack([
            sample_cuda["depth_min"].view(B).float(),
            sample_cuda["depth_max"].view(B).float()
        ], dim=-1)

        # 3. 构造 GeoMVSNet 多阶段投影矩阵与内参字典
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

            K_inv = torch.inverse(K)
            extrinsic = torch.eye(4, device=device, dtype=torch.float32).repeat(B, n_views, 1, 1)
            extrinsic[:, :, :3, :4] = torch.matmul(K_inv, P[:, :, :3, :4])

            proj_mat_geo = torch.zeros(B, n_views, 2, 4, 4, device=device, dtype=torch.float32)
            proj_mat_geo[:, :, 0, :4, :4] = extrinsic
            proj_mat_geo[:, :, 1, :3, :3] = K

            proj_matrices_dict[geo_st] = proj_mat_geo
            intrinsics_dict[geo_st] = K[:, 0]  # 参考视角内参 [B, 3, 3]

        # 4. GeoMVSNet 前向多阶段推理
        outputs = self.net(
            imgs=imgs,
            proj_matrices=proj_matrices_dict,
            intrinsics_matrices=intrinsics_dict,
            depth_values=depth_values
        )
        return outputs, depth_values


def build_multiscale_gt(depth_gt_full, mask_gt_full):
    """
    为 GeoMVSNet loss 自动构建 4 级下采样的 GT 深度和有效掩码字典
    GeoMVSNet stage1 (1/8) -> stage2 (1/4) -> stage3 (1/2) -> stage4 (1/1)
    """
    depth_gt_ms = {}
    mask_ms = {}

    # stage4: 原图
    depth_gt_ms["stage4"] = depth_gt_full.squeeze(1) if depth_gt_full.dim() == 4 else depth_gt_full
    mask_ms["stage4"] = mask_gt_full.squeeze(1) if mask_gt_full.dim() == 4 else mask_gt_full

    # stage3: 1/2
    d3 = F.interpolate(depth_gt_full, scale_factor=0.5, mode="nearest")
    m3 = F.interpolate(mask_gt_full.float(), scale_factor=0.5, mode="nearest")
    depth_gt_ms["stage3"] = d3.squeeze(1)
    mask_ms["stage3"] = (m3.squeeze(1) > 0.5).float()

    # stage2: 1/4
    d2 = F.interpolate(depth_gt_full, scale_factor=0.25, mode="nearest")
    m2 = F.interpolate(mask_gt_full.float(), scale_factor=0.25, mode="nearest")
    depth_gt_ms["stage2"] = d2.squeeze(1)
    mask_ms["stage2"] = (m2.squeeze(1) > 0.5).float()

    # stage1: 1/8
    d1 = F.interpolate(depth_gt_full, scale_factor=0.125, mode="nearest")
    m1 = F.interpolate(mask_gt_full.float(), scale_factor=0.125, mode="nearest")
    depth_gt_ms["stage1"] = d1.squeeze(1)
    mask_ms["stage1"] = (m1.squeeze(1) > 0.5).float()

    return depth_gt_ms, mask_ms


def main():
    args = parse_args()
    if args.valpath is None:
        args.valpath = args.trainpath

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print("=" * 85)
    print("WHU-MVS GeoMVSNet 从零完整训练系统")
    print("=" * 85)
    print_args(args)

    os.makedirs(args.logdir, exist_ok=True)
    tb_writer = SummaryWriter(args.logdir)

    # 1. 动态挂载 GeoMVSNet 源码
    geomvsnet_root = os.path.abspath(args.geomvsnet_code_dir)
    if geomvsnet_root and os.path.isdir(geomvsnet_root):
        if geomvsnet_root not in sys.path:
            sys.path.insert(0, geomvsnet_root)
        print(f"[Import] 已挂载 GeoMVSNet 代码路径: {geomvsnet_root}")
    else:
        print(f"[Warning] 未指定或未找到 --geomvsnet_code_dir: {geomvsnet_root}")

    try:
        from models.geomvsnet import GeoMVSNet
        from models.loss import geomvsnet_loss
    except ImportError as e:
        raise ImportError(f"🚨 无法导入 GeoMVSNet 核心模块！请检查 --geomvsnet_code_dir。\n错误: {e}")

    # 2. 构建 DataLoader
    MVSDataset = find_dataset_def(args.dataset)
    print(f"\n[Dataset] 加载训练集: {args.trainlist}")
    train_dataset = MVSDataset(args.trainpath, args.trainlist, "train", nviews=args.n_views, robust_train=False)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_keep_list, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=(args.num_workers > 0), drop_last=True
    )

    print(f"[Dataset] 加载轻量验证集: {args.vallist} (严格遵守规范五：仅绑极小集)")
    val_dataset = MVSDataset(args.valpath, args.vallist, "test", nviews=args.n_views, robust_train=False)
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        collate_fn=collate_keep_list, num_workers=2,
        pin_memory=True, drop_last=False
    )
    print(f"[Dataset] 训练样本: {len(train_dataset)} | 验证样本: {len(val_dataset)}\n")

    # 3. 实例化模型与损失权重
    model = GeoMVSNetTrainWrapper(
        geomvsnet_class=GeoMVSNet,
        levels=args.levels,
        hypo_plane_num_stages=args.hypo_plane_num_stages,
        depth_interal_ratio_stages=args.depth_interal_ratio_stages,
        feat_base_channel=args.feat_base_channel,
        reg_base_channel=args.reg_base_channel,
        group_cor_dim_stages=args.group_cor_dim_stages
    ).to(device)

    stage_lw = [float(e) for e in args.stage_lw.split(",") if e]

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)

    # 学习率多步衰减
    milestones = [int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
    lr_gamma = 1.0 / float(args.lrepochs.split(':')[1])
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=lr_gamma)

    start_epoch = 0
    if args.loadckpt:
        print(f"[Model] 载入已有 Checkpoint: {args.loadckpt}")
        ckpt = torch.load(args.loadckpt, map_location=device)
        state_dict = ckpt.get('model', ckpt.get('state_dict', ckpt))
        clean_state = { (k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items() }
        model.net.load_state_dict(clean_state, strict=False)
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'epoch' in ckpt:
            start_epoch = ckpt['epoch'] + 1
            print(f"[Resume] 从 Epoch {start_epoch} 继续训练...")

    # 4. 主训练循环
    global_step = start_epoch * len(train_loader)
    print("=" * 85)
    print("开始从零训练 GeoMVSNet (WHU-MVS)...")
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
            outputs, depth_values = model(sample_cuda, n_views=args.n_views)

            # 构建多尺度 GT
            depth_gt_full = sample_cuda["depth"]["stage_0"]
            mask_gt_full = (sample_cuda["mask"]["stage_0"] > 0.5)
            depth_gt_ms, mask_ms = build_multiscale_gt(depth_gt_full, mask_gt_full)

            # 计算官方损失
            loss, epe, pw_losses, dds_losses = geomvsnet_loss(
                outputs, depth_gt_ms, mask_ms,
                stage_lw=stage_lw, depth_values=depth_values
            )

            loss.backward()
            optimizer.step()

            global_step += 1
            step_time = time.time() - step_start

            # 记录标量
            if global_step % args.summary_freq == 0:
                d_est = outputs["stage4"]["depth"]
                d_gt = depth_gt_ms["stage4"]
                m = mask_ms["stage4"] > 0.5
                mae = torch.mean((d_est[m] - d_gt[m]).abs()).item() if m.any() else 0.0

                tb_writer.add_scalar('train/loss', loss.item(), global_step)
                tb_writer.add_scalar('train/mae', mae, global_step)
                tb_writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)

                print(f"Epoch {epoch_idx:02d} [{batch_idx:04d}/{len(train_loader):04d}] | "
                      f"Loss: {loss.item():.4f} | S4 MAE: {mae:.4f}m | Time: {step_time:.2f}s")

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

        # 低频轻量验证 (规范五：每 eval_freq 轮运行 80 张 minitest)
        if (epoch_idx + 1) % args.eval_freq == 0 or epoch_idx == args.epochs - 1:
            print(f"\n--- 运行极速过程验证 (minitest: {len(val_loader)} 张) ---")
            model.eval()
            val_maes = []
            with torch.no_grad():
                for v_sample in val_loader:
                    v_cuda = tocuda(v_sample, device=device, skip_keys=skip)
                    v_outputs, _ = model(v_cuda, n_views=args.n_views)
                    v_est = v_outputs["stage4"]["depth"]
                    v_gt = v_cuda["depth"]["stage_0"].squeeze(1)
                    v_m = (v_cuda["mask"]["stage_0"].squeeze(1) > 0.5)
                    if v_m.any():
                        val_maes.append(torch.mean((v_est[v_m] - v_gt[v_m]).abs()).item())
            mean_val_mae = float(np.mean(val_maes)) if val_maes else float('nan')
            tb_writer.add_scalar('val/mae', mean_val_mae, epoch_idx)
            print(f">>> [Validation] Epoch {epoch_idx:02d} Minitest 平均 MAE: {mean_val_mae:.4f}m\n")

    print("\n" + "=" * 85)
    print("GeoMVSNet 训练彻底完成！")
    print(f"终极权重保存在: {args.logdir}/model_{args.epochs-1:06d}.ckpt")
    print("接下来可直接使用 test_whu_geomvsnet.sh 执行最终收官基准评测！")
    print("=" * 85)


if __name__ == '__main__':
    main()
