import argparse
import os

import math

from models.edge_head import EdgeLabelGenerator
from models.PlanePatchMatch import *
from models.net import compute_normal_cosine_loss

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
from datasets import find_dataset_def
from models import *
from utils import *
import gc
import sys
import datetime
from datasets.dtu_whu import collate_keep_list

# ym_add 这对应的就是实际的cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PatchmatchNet for high-resolution multi-view stereo')
parser.add_argument('--mode', default='train', help='train or val', choices=['train', 'val'])
parser.add_argument('--model', default='PatchmatchNet', help='select model')

parser.add_argument('--dataset', default='dtu_blended', help='select dataset')
parser.add_argument('--trainpath', help='train datapath')
parser.add_argument('--valpath', help='validation datapath')
parser.add_argument('--trainlist', help='train list')
parser.add_argument('--vallist', help='validation list')

parser.add_argument('--epochs', type=int, default=16, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lrepochs', type=str, default="10,12,14:2",
                    help='epoch ids to downscale lr and the downscale rate')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')

parser.add_argument('--batch_size', type=int, default=12, help='train batch size')
parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--logdir', default='./checkpoints/debug', help='the directory to save checkpoints/logs')
parser.add_argument('--resume', default=False, action='store_true', help='continue to train the model')

parser.add_argument('--summary_freq', type=int, default=2, help='print and summary frequency')
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')

parser.add_argument('--patchmatch_iteration', nargs='+', type=int, default=[1, 2, 2],
                    help='num of iteration of patchmatch on stages 1,2,3')
parser.add_argument('--patchmatch_num_sample', nargs='+', type=int, default=[8, 8, 16],
                    help='num of generated samples in local perturbation on stages 1,2,3')
parser.add_argument('--patchmatch_interval_scale', nargs='+', type=float, default=[0.005, 0.0125, 0.025],
                    help='normalized interval in inverse depth range to generate samples in local perturbation')
parser.add_argument('--patchmatch_range', nargs='+', type=int, default=[6, 4, 2],
                    help='fixed offset of sampling points for propogation of patchmatch on stages 1,2,3')
parser.add_argument('--propagate_neighbors', nargs='+', type=int, default=[0, 8, 16],
                    help='num of neighbors for adaptive propagation on stages 1,2,3')
parser.add_argument('--evaluate_neighbors', nargs='+', type=int, default=[9, 9, 9],
                    help='num of neighbors for adaptive matching cost aggregation of adaptive evaluation on stages 1,2,3')

# parse arguments and check
args = parser.parse_args()
if args.resume:  # store_true means set the variable as "True"
    assert args.mode == "train"
    assert args.loadckpt is None
if args.valpath is None:
    args.valpath = args.trainpath

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

if args.mode == "train":
    if not os.path.isdir(args.logdir):
        os.mkdir(args.logdir)

    current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    print("current time", current_time_str)

    print("creating new summary file")
    logger = SummaryWriter(args.logdir)

print("argv:", sys.argv[1:])
print_args(args)

# dataset, dataloader
MVSDataset = find_dataset_def(args.dataset)
if args.dataset == 'dtu_whu':
    train_dataset = MVSDataset(args.trainpath, args.trainlist, "train", 5, robust_train=True)
    # ym-modify 12.1 因为测试集里面没有这个数据
    test_dataset = MVSDataset(args.trainpath, args.vallist, "test", 5, robust_train=False)

# 进行了一个修改，对于有些数据不进行默认collate
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=False, collate_fn=collate_keep_list, num_workers=8,
                            drop_last=True)
TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=True, collate_fn=collate_keep_list, num_workers=4,
                           drop_last=False)

# ym-modified 为了探测问题 num_workers设置为0
# TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True,collate_fn=collate_keep_list, num_workers=0, drop_last=True)
# TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False,collate_fn=collate_keep_list, num_workers=0, drop_last=False)

# model, optimizer
model = PatchmatchNet(patchmatch_interval_scale=args.patchmatch_interval_scale,
                      propagation_range=args.patchmatch_range, patchmatch_iteration=args.patchmatch_iteration,
                      patchmatch_num_sample=args.patchmatch_num_sample,
                      propagate_neighbors=args.propagate_neighbors, evaluate_neighbors=args.evaluate_neighbors)

# if args.mode in ["train", "val"]:
#     model = nn.DataParallel(model)
# model.cuda()
model.to(device)
model_loss = patchmatchnet_loss
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)


# load 模型 parameters
start_epoch = 0
if (args.mode == "train" and args.resume) or (args.mode == "test" and not args.loadckpt):
    saved_models = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(args.logdir, saved_models[-1])
    print("resuming", loadckpt)
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])
print("start at epoch {}".format(start_epoch))
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


# main function
def  train():
    # 学习率调度器初始化
    milestones = [int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(args.lrepochs.split(':')[1])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=lr_gamma,
                                                        last_epoch=start_epoch - 1)

    total_steps=len(TrainImgLoader)*args.epochs
    for epoch_idx in range(start_epoch, args.epochs):
        print('Epoch {}:'.format(epoch_idx))
        lr_scheduler.step()
        global_step = len(TrainImgLoader) * epoch_idx

        # training 这个是一共多少批次，每批次的大小是batch-size
        # ym-issue 为什么一开始会调用很多次trainimgloader呢
        for batch_idx, sample in enumerate(TrainImgLoader):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            # 不是每一张都保存，是过一段时间才保存
            do_summary = global_step % args.summary_freq == 0
            do_summary_image = global_step % (30 * args.summary_freq) == 0
            # 处理单个样本，计算损失并反向传播
            total_loss, scalar_outputs, image_outputs = train_sample(sample, do_summary_image=do_summary_image,
                                                                     total_steps=total_steps,global_step=global_step)
            loss_depth = scalar_outputs['loss_depth']
            loss_alpha_sup=scalar_outputs['loss_alpha_sup']
            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
            if do_summary_image:
                save_images(logger, 'train', image_outputs, global_step)

            print(
                '\nEpoch {}/{}, Iter {}/{},loss_depth:{:.3f},loss_alpha_sup:{:.3f},'
                'continuity_loss:{:.3f},smoothness_loss:{:.3f},normal_loss:{:.3f}'
                'total loss:{:.3f},global_step:{:d} time = {:.3f}'.format(
                    epoch_idx, args.epochs, batch_idx,
                    len(TrainImgLoader),
                    loss_depth,loss_alpha_sup,
                    scalar_outputs['continuity_loss'],scalar_outputs['smoothness_loss'],scalar_outputs['normal_loss'],
                    total_loss,global_step,time.time() - start_time))

            del scalar_outputs, image_outputs

        # checkpoint
        if (epoch_idx + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch_idx,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()},
                "{}/model_{:0>6}.ckpt".format(args.logdir, epoch_idx))

        avg_test_scalars = DictAverageMeter()
        for batch_idx, sample in enumerate(TestImgLoader):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            # do_summary_test = global_step % (10*args.summary_freq) == 0
            do_summary_image = global_step % (10 * args.summary_freq) == 0
            loss, scalar_outputs, image_outputs = test_sample(sample, detailed_summary=do_summary_image)
            loss_depth = scalar_outputs['loss_depth']
            loss_alpha_sup=scalar_outputs['loss_alpha_sup']
            if do_summary:
                save_scalars(logger, 'test', scalar_outputs, global_step)
            if do_summary_image:
                save_images(logger, 'test', image_outputs, global_step)
            avg_test_scalars.update(scalar_outputs)
            del scalar_outputs, image_outputs
            print(
                'Epoch {}/{}, Iter {}/{},loss_depth:{:.3f},loss_alpha_sup:{:.3f},total loss:{:.3f}, time = {:.3f}'.format(
                    epoch_idx, args.epochs, batch_idx,
                    len(TrainImgLoader), loss_depth, loss_alpha_sup, loss,
                    time.time() - start_time))

        save_scalars(logger, 'fulltest', avg_test_scalars.mean(), global_step)
        print("avg_test_scalars:", avg_test_scalars.mean())
        print("当前时间（time模块）：", time.ctime())
        gc.collect()


def test():
    avg_test_scalars = DictAverageMeter()
    for batch_idx, sample in enumerate(TestImgLoader):
        start_time = time.time()
        loss, scalar_outputs, image_outputs = test_sample(sample, detailed_summary=True)
        avg_test_scalars.update(scalar_outputs)
        del scalar_outputs, image_outputs
        print('Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(batch_idx, len(TestImgLoader), loss,
                                                                    time.time() - start_time))
        if batch_idx % 100 == 0:
            print("Iter {}/{}, test results = {}".format(batch_idx, len(TestImgLoader), avg_test_scalars.mean()))
    print("final", avg_test_scalars)

def compute_edge_supervision_loss(
        edge_label_generator,
        pred_alphas_list,
        gt_depth_map,
        tri_infos,
        tri_id_map
):
    """
    封装边预测头 (EdgeHead) 的监督 Loss 计算全流程。
    包含: 伪标签生成 -> Mask 过滤 -> 加权 BCE -> 稀疏性惩罚统计。
    """
    # 1. 生成 Stage 1 伪真值标签和有效掩码
    gt_stage1_list, valid_mask_list = edge_label_generator(
        pred_alphas_list=pred_alphas_list,
        gt_depth_map=gt_depth_map,
        tri_infos=tri_infos,
        tri_id_map=tri_id_map
    )

    B = len(pred_alphas_list)
    device = gt_depth_map.device  # 提前获取 device，避免后续报错

    loss_alpha_sup = 0.0
    total_sparsity_loss = 0.0

    valid_bce_count = 0  # 记录有多少张图成功计算了 BCE
    valid_sparsity_count = 0  # 记录有多少张图成功计算了稀疏性

    for b in range(B):
        pred_alphas = pred_alphas_list[b]

        # 如果当前图没有任何边，直接跳过
        if pred_alphas.numel() == 0:
            continue

        valid_mask = valid_mask_list[b]
        gt_targets = gt_stage1_list[b]

        # --- 统计稀疏性 (只针对有边的图) ---
        total_sparsity_loss += pred_alphas.mean()
        valid_sparsity_count += 1

        # 如果当前图没有任何有效的 GT 边，跳过 BCE 计算
        if valid_mask.sum() == 0:
            continue

        # --- 计算加权 BCE ---
        valid_pred = pred_alphas[valid_mask]
        valid_target = gt_targets[valid_mask]

        loss_alpha_sup += compute_weighted_bce_core(valid_pred, valid_target)
        valid_bce_count += 1

    # ==========================================
    # 聚合求平均 (防止除 0 并保持计算图连通)
    # ==========================================

    # 1. BCE Loss 平均
    if valid_bce_count > 0:
        loss_alpha_sup = loss_alpha_sup / valid_bce_count
    else:
        # 🔥 关键修复：加上 requires_grad=True
        # 万一某个极端的 Batch 全被跳过了，直接返回 0.0 会导致 total_loss.backward() 找不到梯度图崩溃
        loss_alpha_sup = torch.tensor(0.0, device=device, requires_grad=True)

    # 2. 稀疏性 Loss 平均 (修复了你原来的 Bug)
    if valid_sparsity_count > 0:
        loss_sparsity = total_sparsity_loss / valid_sparsity_count
    else:
        loss_sparsity = torch.tensor(0.0, device=device, requires_grad=True)

    return loss_alpha_sup, loss_sparsity,gt_stage1_list


def compute_edge_supervision_loss_new(
        alphas_list,     # [B] 方案A中唯一一次预测的边断裂概率 (0~1)
        gt_stage0_list,  # [B] 绝对真值 (纯净的 0 或 1)
        valid_mask_list  # [B] 有效掩码
):
    """
    方案 A 专属：单次精准深层监督 Loss 计算。
    只接收 Iter 1 在干净平面上预测出的高质量边缘概率。
    """
    B = len(alphas_list)
    device = alphas_list[0].device if B > 0 else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    total_loss_alpha = 0.0
    total_sparsity = 0.0

    valid_count = 0

    for b in range(B):
        alphas = alphas_list[b]
        gt_target = gt_stage0_list[b]
        valid_mask = valid_mask_list[b]

        # 异常跳过
        if alphas.numel() == 0 or valid_mask.sum() == 0:
            continue

        # 1. 稀疏性惩罚 (直接对概率求平均)
        total_sparsity += alphas.mean()

        # 2. 提取有效区域的概率和真值
        valid_preds = alphas[valid_mask]
        valid_target = gt_target[valid_mask]

        # 3. 直接调用你的防爆 BCE 核心 (绝不再做 Sigmoid！)
        total_loss_alpha += compute_weighted_bce_core(valid_preds, valid_target)

        valid_count += 1

    # ==========================================
    # 4. 聚合计算
    # ==========================================
    if valid_count > 0:
        total_loss_alpha = total_loss_alpha / valid_count
        total_sparsity = total_sparsity / valid_count
    else:
        # 防崩溃保底机制
        total_loss_alpha = torch.tensor(0.0, device=device, requires_grad=True)
        total_sparsity = torch.tensor(0.0, device=device, requires_grad=True)

    return total_loss_alpha, total_sparsity

def get_dynamic_loss_weights(global_step, total_steps):
    """
    动态计算 Loss 权重
    """
    # 进度比例: 0.0 -> 1.0
    progress = global_step / total_steps

    # ==========================================
    # 1. 边缘监督权重 (alpha_sup) - 逐渐衰减
    # ==========================================
    # 起始权重为 10.0，随着训练进行，指数衰减到 1.0 甚至 0.1
    # 我们这里设计一个衰减曲线：前期保持较高，中后期迅速下降
    alpha_start = 10.0
    alpha_end = 1.0

    # 使用余弦退火曲线，过渡极其平滑
    # 当 progress=0 时为 1，progress=1 时为 0
    decay_factor = 0.5 * (1.0 + math.cos(math.pi * progress))

    weight_alpha = alpha_end + (alpha_start - alpha_end) * decay_factor

    # ==========================================
    # 2. 连续性约束权重 (continuity) - 逐渐增强 或 保持稳定
    # ==========================================
    # 刚开始拟合的平面很乱，强行做连续性约束会导致网络崩溃
    # 所以让它从 0.01 慢慢涨到 0.3，接管后期的边缘学习
    cont_start = 0.01
    cont_end = 0.3

    # 线性增长
    weight_cont = cont_start + (cont_end - cont_start) * progress

    # ==========================================
    # 3. 稀疏性惩罚 (sparsity) - 必须始终保持一定强度防止作弊
    # ==========================================
    # 随着 continuity 权重上升，作弊的收益变大，所以惩罚也要跟上
    weight_spars = 1.0 + 1.0 * progress  # 从 1.0 涨到 2.0

    return weight_alpha, weight_cont, weight_spars


def compute_weighted_bce_core(preds, targets):
    """
    纯粹的加权 BCE 计算核心 (防爆版)
    输入必须是经过 valid_mask 筛选后的 1D tensor
    """
    preds = torch.clamp(preds, 1e-6, 1.0 - 1e-6)
    num_pos = (targets > 0.5).sum().float()
    num_neg = (targets <= 0.5).sum().float()

    if num_pos < 1.0:
        return F.binary_cross_entropy(preds, targets)

    pos_weight = torch.clamp(num_neg / num_pos, min=1.0, max=40.0)

    loss_pos = - pos_weight * targets * torch.log(preds)
    loss_neg = - (1.0 - targets) * torch.log(1.0 - preds)
    return (loss_pos + loss_neg).mean()


def compute_pixel_cost_margin_loss(no_prop_depth, gt_depth, pixel_costs, tri_id_map,
                                   good_rel_thresh=0.01, bad_rel_thresh=0.05):
    """
    [完全体] 像素级 Cost Margin Loss：尺度不变 & 样本均衡 & 边界截断 & 网格掩码保护

    Args:
        no_prop_depth:   [B, 1, H, W] SVD 初值深度 (务必 detach!)
        gt_depth:        [B, 1, H, W] GT 深度
        pixel_costs:     [B, H, W, 1] 模型输出的代价 (L2归一化点积取负)
        tri_id_map:      [B, H, W] Stage 1 的三角形 ID 图，用于过滤无三角形的无效背景 (-1)
        good_rel_thresh: 好平面相对误差阈值 (默认 1%)
        bad_rel_thresh:  坏平面相对误差阈值 (默认 5%)
    """
    if pixel_costs.dim() == 4 and pixel_costs.shape[-1] == 1:
        pixel_costs = pixel_costs.permute(0, 3, 1, 2)  # [B, 1, H, W]

    # ==========================================
    # 1. 终极有效掩码 (剔除天空、无效 GT、无效网格、SVD 爆炸区)
    # ==========================================
    # 深度有效性
    valid_depth_mask = (gt_depth > 1e-3) & (no_prop_depth > 1e-3)
    # 网格有效性 (排除 tri_id_map < 0 的背景或天空)
    valid_tri_mask = (tri_id_map >= 0).unsqueeze(1)  # [B, 1, H, W]

    valid_mask = valid_depth_mask & valid_tri_mask

    # ==========================================
    # 2. 计算相对深度误差 (Scale-Invariant)
    # ==========================================
    rel_error = torch.abs(no_prop_depth - gt_depth) / (gt_depth + 1e-6)

    # ==========================================
    # 3. 划分好坏阵营
    # ==========================================
    mask_good = valid_mask & (rel_error <= good_rel_thresh)
    mask_bad = valid_mask & (rel_error >= bad_rel_thresh)

    # ==========================================
    # 4. 引入 Margin 思想 (单侧惩罚)
    # ==========================================
    loss_good = 0.0
    loss_bad = 0.0

    if mask_good.sum() > 0:
        # 只惩罚那些 "明明是好几何，但 Cost 却大于 -0.8" 的像素
        costs_good = pixel_costs[mask_good]
        # F.relu 惩罚超出 -0.8 的部分
        loss_good = F.relu(costs_good - (-0.8)).mean()

    if mask_bad.sum() > 0:
        # 只惩罚那些 "明明是坏几何，但 Cost 却居然小于 -0.2 (作弊匹配上了)" 的像素
        costs_bad = pixel_costs[mask_bad]
        # 惩罚低于 -0.2 的部分
        loss_bad = F.relu((-0.2) - costs_bad).mean()

    # ==========================================
    # 5. 正负样本解耦相加 (Hard Negative Mining)
    # ==========================================
    # 给坏平面(通常是建筑边缘)更大的权重 2.0，逼迫 FeatureNet 去抠边缘！
    total_loss = loss_good + 2.0 * loss_bad

    # 打印监控日志 (用于观察特征网的训练状态)
    with torch.no_grad():
        good_ratio = mask_good.float().mean().item()
        bad_ratio = mask_bad.float().mean().item()
        # 避免训练初期大量刷屏，你可以加上 if 限制输出频率
        print(f"[Cost Loss] Good: {good_ratio:.3f} | Bad: {bad_ratio:.3f} | "
              f"L_good: {loss_good.item() if isinstance(loss_good, torch.Tensor) else 0:.4f} | "
              f"L_bad: {loss_bad.item() if isinstance(loss_bad, torch.Tensor) else 0:.4f}")

    return total_loss

def generate_geometric_edge_gt(final_planes, current_edges, midpoints_norm, intrinsics, H, W, threshold=10.0):
    """
    动态生成几何真值 (免受斜面干扰的绝对真值)
    Args:
        final_planes: [E, 4] 边的相关平面
        current_edges: [E, 2] 边的邻接三角形索引
        midpoints_norm: [E, 2] 边的中点
    Returns:
        geom_targets: [E] 几何真值 0 or 1
    """
    with torch.no_grad():
        E = midpoints_norm.shape[0]
        device = midpoints_norm.device

        # 1. 坐标反投影
        midpoints_uv = torch.zeros_like(midpoints_norm)
        midpoints_uv[:, 0] = (midpoints_norm[:, 0] + 1.0) / 2.0 * (W - 1)
        midpoints_uv[:, 1] = (midpoints_norm[:, 1] + 1.0) / 2.0 * (H - 1)

        ones = torch.ones(E, 1, device=device)
        uv_homo = torch.cat([midpoints_uv, ones], dim=-1)
        K_inv = torch.inverse(intrinsics)  # [3, 3]
        rays = torch.matmul(K_inv, uv_homo.unsqueeze(-1)).squeeze(-1)  # [E, 3]

        # 2. 提取 T1 和 T2 平面并算深度
        idx1, idx2 = current_edges[:, 0].long(), current_edges[:, 1].long()
        planes_t1, planes_t2 = final_planes[idx1], final_planes[idx2]

        n1, d1 = planes_t1[:, :3], planes_t1[:, 3:]
        n2, d2 = planes_t2[:, :3], planes_t2[:, 3:]

        denom1 = torch.sum(n1 * rays, dim=-1, keepdim=True).clamp_min(1e-6)
        denom2 = torch.sum(n2 * rays, dim=-1, keepdim=True).clamp_min(1e-6)
        depth1, depth2 = -d1 / denom1, -d2 / denom2

        # 3. 产生几何标签
        diff = torch.abs(depth1 - depth2).squeeze(-1)
        geom_targets = (diff > threshold).float()

        # 边界边强制为断裂
        geom_targets[idx1 == idx2] = 1.0

    return geom_targets


def get_smooth_weight_with_decay(progress, start_prog, end_prog, decay_prog, max_weight, end_ratio=0.1):
    """
    终极版：带退坡松绑的平滑过渡权重 (Cosine Warmup + Plateau + Cosine Decay)

    Args:
        progress:   当前训练进度 [0.0 ~ 1.0]
        start_prog: 开始介入的节点 (例如 0.3)
        end_prog:   达到满载的节点 (例如 0.6)
        decay_prog: 开始松绑退坡的节点 (例如 0.8)
        max_weight: 满载时的最高权重
        end_ratio:  训练结束时保留的权重比例 (默认 0.1，即保留 10% 的保底约束)
    """
    # 1. 潜伏期 (尚未启动)
    if progress <= start_prog:
        return 0.0

    # 2. 爬坡期 (Cosine Warmup)
    elif progress < end_prog:
        linear_ratio = (progress - start_prog) / (end_prog - start_prog)
        # 从 0.0 平滑上升到 1.0
        smooth_ratio = (1.0 - math.cos(linear_ratio * math.pi)) / 2.0
        return max_weight * smooth_ratio

    # 3. 满载期 (Plateau - 强力塑形)
    elif progress <= decay_prog:
        return max_weight

    # 4. 🚀 退坡松绑期 (Cosine Decay - 极限抠细节)
    else:
        # 进度从 decay_prog 到 1.0 映射为 0.0 到 1.0
        linear_ratio = (progress - decay_prog) / (1.0 - decay_prog)

        # smooth_ratio_down 会从 1.0 平滑下降到 0.0
        smooth_ratio_down = (1.0 + math.cos(linear_ratio * math.pi)) / 2.0

        # 计算衰减的下限 (保底权重)
        min_weight = max_weight * end_ratio

        # 在 min_weight 和 max_weight 之间进行余弦插值
        return min_weight + (max_weight - min_weight) * smooth_ratio_down

def train_sample(sample, do_summary_image=False,global_step=0, total_steps=0):

    model.train()
    optimizer.zero_grad()

    # 将cdt_data进行一个单独处理处理,单独将这些数据放入GPU中
    # ym-issue 是不是可以不用这么早进行一个处理
    # vertexs/list-of-arrays -> 转 tensor 并 to(device)

    # vertexs_batch = []
    # lines_batch = []
    # triangles_batch = []

    vertexs_batch = [torch.from_numpy(v).to(device) for v in sample['vertexs']]
    lines_batch = [torch.from_numpy(v).to(device) for v in sample['lines']]
    triangles_batch = []
    for tri_list in sample['triangles']:  # tri_list 是一个 sample 的 triangles
        tri_processed = []
        for t in tri_list:
            v_ids = torch.from_numpy(t['vertex_ids']).to(device)
            l_ids = torch.from_numpy(t['line_ids']).to(device)
            pts = torch.from_numpy(t['valid_points']).to(device)  # variable len
            tri_processed.append((v_ids, l_ids, pts))
        triangles_batch.append(tri_processed)

    # ym-modify 重写了一下对于cdt—data数据进行了一个跳过
    skip = ["vertexs", "lines", "triangles"]
    sample_cuda = tocuda(sample, device=device, skip_keys=skip)

    depth_gt = sample_cuda["depth"]
    mask = sample_cuda["mask"]

    # 启用梯度异常检测
    # torch.autograd.set_detect_anomaly(True)

    # 一开始不启动连续性约束和光滑性约束，后面再打开，目前是直接打开
    progress = global_step / total_steps
    # 连续性约束和光滑性约束权重
    max_lambda_c = 50.0
    max_lambda_s = 1.0
    # max_lambda_c = 0.0
    # max_lambda_s = 0.0
    max_lambda_n = 0.0 # 法向 Loss 的量级通常较大，0.1 到 0.5 之间调节
    max_lambda_cost=0.2
    weight_alpha = 1.0

    # 1. 连通性约束 (早启动，早满载，晚退坡)：
    # 0.2 启动，0.4 满载，0.85 开始松绑，最后保留 10% 的防撕裂底线
    lambda_c = get_smooth_weight_with_decay(progress, 0.1, 0.4, 0.6, max_lambda_c, end_ratio=0.01)

    # 2. 光滑性约束 (中启动，中满载，早退坡)：
    # 0.3 启动，0.6 满载，0.8 开始松绑，因为平滑最容易影响高频细节，所以早点松绑
    lambda_s = get_smooth_weight_with_decay(progress, 0.2, 0.4, 0.6, max_lambda_s, end_ratio=0.03)

    # 3. 法向约束 (晚启动，晚满载，早退坡)：
    # 0.5 启动，0.7 满载，0.8 开始松绑，防止后期拟合 SVD 噪声
    lambda_n = get_smooth_weight_with_decay(progress, 0.3, 0.5, 0.55, max_lambda_n, end_ratio=0.05)

    # lambda_c,lambda_s,lambda_n=0.0,0.0,0.0

    # 自动构建计算图（动态计算图），记录每个张量的操作历史（如卷积、激活、矩阵乘法等），从而在反向传播时能通过链式法则计算梯度
    outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"],sample_cuda["intrinsics_mats"],
                    sample_cuda["depth_min"], sample_cuda["depth_max"],
                    vertexs_batch, lines_batch, triangles_batch,depth_gt['stage_1'],
                    lambda_c,lambda_s)

    depth_est = outputs["refined_depth"]

    depth_patchmatch = outputs["depth_patchmatch"]

    # 通过计算最终的损失
    # ====================================================
    # 1. 深度主任务 Loss
    # ====================================================
    # 0-1掩码 用来损失函数的
    valid_mask_s1 = (outputs["output_plane"]['tri_id_map'] >= 0).float().unsqueeze(dim=1)
    mask['stage_1']=valid_mask_s1
    loss_depth = model_loss(depth_patchmatch, depth_est, depth_gt, mask)  # 深度损失

    # ====================================================
    # 2. 边预测头的混合监督 Loss
    # ====================================================
    # EdgeConsistencyLoss（自监督 BCE）
    intrinsics_s1 = torch.unbind(sample_cuda["intrinsics_mats"]['stage_1'].float(), 1)
    ref_intrinsics = intrinsics_s1[0]  # 参考视角的内参，形状必为 [B, 3, 3]

    intrinsics_s0 = torch.unbind(sample_cuda["intrinsics_mats"]['stage_0'].float(), 1)
    ref_intrinsics_s0 = intrinsics_s0[0]  # [B, 3, 3]

    depth_threshold=0.2
    edge_label_generator = EdgeLabelGenerator(pt2plane_threshold=depth_threshold)

    # === 生成基于像素的法向量图 (使用上面定义的函数) ================================
    # 获取 Stage 1 的 GT 深度和 Mask
    gt_depth_s1 = depth_gt['stage_1']  # 假设形状 [B,1, H, W]
    gt_mask_s1 = mask['stage_1']  # 假设形状 [B, 1,H, W]

    # 获取 Stage 1 的 预测 深度 (planepatchmatch最终预测结果)
    pred_depth_s1 = depth_patchmatch['stage_1'][-1]  # 假设形状 [B,H, W]

    # 1. 生成 GT 法向量 (传入 mask 去除无效区域)
    gt_normals_math, gt_normals_vis = compute_normal_map_torch(depth_gt['stage_0'], ref_intrinsics_s0,mask=None, smooth=True)

    # 2. 生成 预测 法向量 (同样传入 mask，或者你可以传入 threshold 后的 mask)
    # normal_pred_s1 = compute_normal_map_torch(pred_depth_s1, mask=gt_mask_s1, smooth=False)

    # ----------------------------------------------------
    # 阶段 A：外部预生成 Stage 0 分辨率的基础伪标签
    # ym-add 用原分辨率的图像来计算真值，使得真值更加的准确
    # ----------------------------------------------------
    edge_alphas_gt, valid_mask_list = edge_label_generator(
        pred_alphas_list=outputs["edge_alphas"],
        gt_depth_map=depth_gt['stage_0'],  # Stage 0 的高清深度图
        tri_infos=outputs["tri_infos"],
        intrinsics=ref_intrinsics_s0,
        gt_normal_map=gt_normals_math,
    )

    # ----------------------------------------------------
    # 阶段 B：送入 Loss 函数进行动态混合和计算
    # ----------------------------------------------------

    # loss_alpha_raw, loss_sparsity_raw,edge_alphas_gt = compute_edge_supervision_loss(
    #     edge_label_generator=edge_label_generator,
    #     pred_alphas_list=outputs["edge_alphas"],
    #     gt_depth_map=depth_gt[f'stage_1'],
    #     tri_infos=outputs["tri_infos"],
    #     tri_id_map=outputs["output_plane"]['tri_id_map'],
    # )

    # 边预测头 Loss 计算
    loss_alpha_raw, loss_sparsity_raw = compute_edge_supervision_loss_new(
        alphas_list=outputs["edge_alphas"],
        gt_stage0_list=edge_alphas_gt,
        valid_mask_list=valid_mask_list
    )

    # ====================================================
    # 3. 损失函数的混合
    # ====================================================

    # 计算法向量损失
    normal_loss = compute_normal_cosine_loss(final_planes=outputs["output_plane"]["final_plane"],
                                             tri_id_map=outputs["output_plane"]['tri_id_map'],
                                             gt_normals_math_s0=gt_normals_math,
                                             depth_stage_1=depth_gt['stage_1'],
                                             )

    # 计算 Cost Margin Loss
    cost_margin_loss = compute_pixel_cost_margin_loss(
        no_prop_depth=outputs["output_plane"]['depth_no_pro'].detach(),  # SVD 的初始深度
        gt_depth=depth_gt['stage_1'],  # GT 深度
        pixel_costs=outputs["output_plane"]['pixel_costs'],  # 畅通回传给特征网的代价
        tri_id_map=outputs["output_plane"]['tri_id_map']
    )

    cost_margin_loss = cost_margin_loss * max_lambda_cost

    # 边缘监督 Loss
    # 乘上一个权重再，加上边断裂损失，防止预测头损失过小

    loss_alpha_sup=loss_alpha_raw * weight_alpha + loss_sparsity_raw

    # 3.连续性正则化 Loss (Geometric Smoothness)
    # 因为它是正则化项，绝不能喧宾夺主。建议权重设为 0.1 ~ 0.5
    # 4.光滑性约束

    continuity_loss = outputs["continuity_loss"] * lambda_c
    smoothness_loss = outputs["smoothness_loss"] * lambda_s
    normal_loss = normal_loss * lambda_n

    # 总损失：深度损失+边断裂损失+连续性损失+光滑性约束
    loss = loss_depth + loss_alpha_sup + continuity_loss + smoothness_loss + normal_loss + cost_margin_loss

    # 边断裂损失
    loss.backward()

    # todo：将梯度限制maxmax_norm以内
    # ← 必须在这里，backward 之后才有梯度可以裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

    # 优化器根据计算的梯度更新模型参数（梯度下降的具体实现）
    optimizer.step()

    scalar_outputs = {"loss": loss,
                      "loss_depth": loss_depth,
                      "loss_alpha_sup": loss_alpha_sup,
                      "continuity_loss":continuity_loss,
                      "smoothness_loss":smoothness_loss,
                      "normal_loss":normal_loss,
                      "cost_margin_loss":cost_margin_loss}

    image_outputs = []

    # ====================================================
    # 4.可视化
    # ====================================================
    if do_summary_image:
        # ================ 生成断裂图 ===============================================
        image_outputs_pre = generate_edge_alpha_overlays(
            ref_imgs=sample["imgs"]['stage_0'][:, 0],  # 注意取 ref 图
            edge_alphas_list=outputs["edge_alphas"],
            edges_pixels_list=outputs["tri_infos"][0]['edges_pixels'],
            device=device,
            overlay_alpha=0.6,  # 线条显示的透明度
            line_thickness=1  # 线条粗细
        )

        ref_img_edge_alpha_pre = image_outputs_pre["ref_img_edge_alpha"]

        image_outputs_gt = generate_edge_alpha_overlays(
            ref_imgs=sample["imgs"]['stage_0'][:, 0],  # 注意取 ref 图
            edge_alphas_list=edge_alphas_gt,
            edges_pixels_list=outputs["tri_infos"][0]['edges_pixels'],
            device=device,
            overlay_alpha=0.6,  # 线条显示的透明度
            line_thickness=1  # 线条粗细
        )

        ref_img_edge_alpha_gt = image_outputs_gt["ref_img_edge_alpha"]


        # normal_gt_s1 = compute_normal_map_perspective(
        #     gt_depth_s1,
        #     intrinsics_s1[0],
        #     mask=gt_mask_s1,
        #     smooth=False
        # )
        #
        # normal_pred_s1 = compute_normal_map_perspective(
        #     pred_depth_s1,
        #     intrinsics_s1[0],
        #     mask=gt_mask_s1,
        #     smooth=False
        # )

        # vis_depth = normalize_depth_for_display(depth_pred_phys, valid_mask=None)
        # 2. 生成用于 TensorBoard 展示的数据 ym-need-delete 1.13
        # 注意：这里我们手动归一化了，所以 save_images 里对 depth 的 normalize=True 其实就
        # 变成了在 0~1 之间归一化，不会破坏我们做好的拉伸。
        # vis_depth_gt_plane = normalize_depth_for_display(depth_gt_plane_stage_1)  # 传入 valid mask
        #
        # vis_depth_pre_plane = normalize_depth_for_display(outputs["output_plane"]['depth_pred'])  # 传入 valid mask

        # ===== tensorboard显示图片和曲线 ======================================

        image_outputs = {  # 暂时注释一些图片，输出的图片太多了
            "最终预测结果": depth_est['stage_0'] * mask['stage_0'],
            "stage1深度真值": depth_gt['stage_1'] ,
            "patchmatch预测的stage2上采样经过恢复的深度值": outputs["output_plane"]['depth_stage1_pixels'],
            "patchmatch预测的stage2深度值": depth_patchmatch['stage_2'][-1] * mask['stage_2'],
            # "depth_patchmatch_stage_3": depth_patchmatch['stage_3'][-1] * mask['stage_3'],
            "ref_img": sample["imgs"]['stage_0'][:, 0],
            # 新增：基于像素点的法向量图
            "根据深度真值生成的法向量": gt_normals_vis,
            # "经过平面传播预测的stage1深度值生成的像素法向量": normal_pred_s1,
            # 新增：基于平面的深度图和法向量图传播完
            "经过平面传播生成的平面法向量": outputs["output_plane"]['normal_final'],
            "经过平面传播预测的stage1深度值": depth_patchmatch['stage_1'][-1],
            # 新增：基于平面的深度图和法向量图，刚拟合
            "没有经过平面传播，刚拟合完初始平面深度值": outputs["output_plane"]['depth_no_pro'],
            "没有经过平面传播，刚拟合完初始平面法向量": outputs["output_plane"]['normal_no_pro'] ,
            # 新增：边预测头预测值和真值
            "ref_img_edge_alpha_pre": ref_img_edge_alpha_pre,
            "ref_img_edge_alpha_gt": ref_img_edge_alpha_gt
        }

        # image_outputs["errormap_refined_stage_0"] = (depth_est['stage_0'] - depth_gt['stage_0']).abs() * mask['stage_0']
        # image_outputs["errormap_patchmatch_stage_1"] = (depth_patchmatch['stage_1'][-1] - depth_gt['stage_1']).abs() * \
        #                                                mask['stage_1']
        # image_outputs["errormap_patchmatch_stage_2"] = (depth_patchmatch['stage_2'][-1] - depth_gt['stage_2']).abs() * \
        #                                                mask['stage_2']
        # image_outputs["errormap_patchmatch_stage_3"] = (depth_patchmatch['stage_3'][-1] - depth_gt['stage_3']).abs() * \
        #                                                mask['stage_3']



    scalar_outputs["abs_depth_error_refined_stage_0"] = AbsDepthError_metrics(depth_est['stage_0'], depth_gt['stage_0'],
                                                                              mask['stage_0'] > 0.5)
    scalar_outputs["abs_depth_error_patchmatch_stage_3"] = AbsDepthError_metrics(depth_patchmatch['stage_3'][-1],
                                                                                 depth_gt['stage_3'],
                                                                                 mask['stage_3'] > 0.5)
    scalar_outputs["abs_depth_error_patchmatch_stage_2"] = AbsDepthError_metrics(depth_patchmatch['stage_2'][-1],
                                                                                 depth_gt['stage_2'],
                                                                                 mask['stage_2'] > 0.5)
    scalar_outputs["abs_depth_error_patchmatch_stage_1"] = AbsDepthError_metrics(depth_patchmatch['stage_1'][-1],
                                                                                 depth_gt['stage_1'],
                                                                                 mask['stage_1'] > 0.5)
    # threshold = 1mm
    scalar_outputs["thres1mm_error"] = Thres_metrics(depth_est['stage_0'], depth_gt['stage_0'], mask['stage_0'] > 0.5,
                                                     1)
    # threshold = 2mm
    scalar_outputs["thres2mm_error"] = Thres_metrics(depth_est['stage_0'], depth_gt['stage_0'], mask['stage_0'] > 0.5,
                                                     2)
    # threshold = 4mm
    scalar_outputs["thres4mm_error"] = Thres_metrics(depth_est['stage_0'], depth_gt['stage_0'], mask['stage_0'] > 0.5,
                                                     4)
    # threshold = 8mm
    scalar_outputs["thres8mm_error"] = Thres_metrics(depth_est['stage_0'], depth_gt['stage_0'], mask['stage_0'] > 0.5,
                                                     8)

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


@make_nograd_func
def test_sample(sample, detailed_summary=False, global_step=0, total_steps=1):
    model.eval()

    # ====================================================
    # 1. 数据准备 (与 train_sample 严格对齐)
    # ====================================================
    vertexs_batch = [torch.from_numpy(v).to(device) for v in sample['vertexs']]
    lines_batch = [torch.from_numpy(v).to(device) for v in sample['lines']]
    triangles_batch = []
    for tri_list in sample['triangles']:
        tri_processed = []
        for t in tri_list:
            v_ids = torch.from_numpy(t['vertex_ids']).to(device)
            l_ids = torch.from_numpy(t['line_ids']).to(device)
            pts = torch.from_numpy(t['valid_points']).to(device)
            tri_processed.append((v_ids, l_ids, pts))
        triangles_batch.append(tri_processed)

    # ym-modify 重写了一下对于cdt—data数据进行了一个跳过
    skip = ["vertexs", "lines", "triangles"]
    sample_cuda = tocuda(sample, device=device, skip_keys=skip)

    depth_gt = sample_cuda["depth"]
    mask = sample_cuda["mask"]

    # ====================================================
    # 2. 模型 Forward (补齐缺失的 lambda 参数)
    # ====================================================
    # 在测试阶段，我们通常希望查看模型在"完全体"约束下的表现
    # 所以直接给出完全展开的惩罚系数
    max_lambda_c = 100.0
    max_lambda_s = 3.0
    max_lambda_n = 0.3
    max_lambda_cost = 0.2


    outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["intrinsics_mats"],
                    sample_cuda["depth_min"], sample_cuda["depth_max"],
                    vertexs_batch, lines_batch, triangles_batch, depth_gt['stage_1'],
                    max_lambda_c, max_lambda_s)  # <- 这里补齐了 train 里的新参数

    depth_est = outputs["refined_depth"]
    depth_patchmatch = outputs["depth_patchmatch"]

    # ====================================================
    # 3. 损失计算 (同步使用新的 Edge Loss 和 Normal Loss)
    # ====================================================
    valid_mask_s1 = (outputs["output_plane"]['tri_id_map'] >= 0).float().unsqueeze(dim=1)
    mask['stage_1'] = valid_mask_s1

    loss_depth = model_loss(depth_patchmatch, depth_est, depth_gt, mask)

    intrinsics_s1 = torch.unbind(sample_cuda["intrinsics_mats"]['stage_1'].float(), 1)
    ref_intrinsics = intrinsics_s1[0]
    intrinsics_s0 = torch.unbind(sample_cuda["intrinsics_mats"]['stage_0'].float(), 1)
    ref_intrinsics_s0 = intrinsics_s0[0]

    depth_threshold = 0.2
    edge_label_generator = EdgeLabelGenerator(pt2plane_threshold=depth_threshold)

    # 提取 GT 法向量 (测试时也需要计算以作对照)
    gt_normals_math, gt_normals_vis = compute_normal_map_torch(depth_gt['stage_0'], ref_intrinsics_s0, mask=None,
                                                               smooth=True)

    # 阶段 A：预生成标签
    edge_alphas_gt, valid_mask_list = edge_label_generator(
        pred_alphas_list=outputs["edge_alphas"],
        gt_depth_map=depth_gt['stage_0'],
        tri_infos=outputs["tri_infos"],
        intrinsics=ref_intrinsics_s0,
        gt_normal_map=gt_normals_math,
    )

    H1, W1 = depth_gt['stage_1'].shape[-2:]

    # 阶段 B：两段式边预测头损失
    loss_alpha_raw, loss_sparsity_raw = compute_edge_supervision_loss_new(
        alphas_list=outputs["edge_alphas"],
        gt_stage0_list=edge_alphas_gt,
        valid_mask_list=valid_mask_list
    )

    normal_loss = compute_normal_cosine_loss(final_planes=outputs["output_plane"]["final_plane"],
                                             tri_id_map=outputs["output_plane"]['tri_id_map'],
                                             gt_normals_math_s0=gt_normals_math,
                                             depth_stage_1=depth_gt['stage_1'])

    # 计算 Cost Margin Loss
    cost_margin_loss = compute_pixel_cost_margin_loss(
        no_prop_depth=outputs["output_plane"]['depth_no_pro'].detach(),  # SVD 的初始深度
        gt_depth=depth_gt['stage_1'],  # GT 深度
        pixel_costs=outputs["output_plane"]['pixel_costs'],  # 畅通回传给特征网的代价
        tri_id_map=outputs["output_plane"]['tri_id_map']
    )

    cost_margin_loss = cost_margin_loss * max_lambda_cost

    weight_alpha = 1.0
    loss_alpha_sup = loss_alpha_raw * weight_alpha + loss_sparsity_raw
    continuity_loss = outputs["continuity_loss"] * max_lambda_c
    smoothness_loss = outputs["smoothness_loss"] * max_lambda_s
    normal_loss = normal_loss * max_lambda_n

    loss = loss_depth + loss_alpha_sup + continuity_loss + smoothness_loss + normal_loss +cost_margin_loss

    # ====================================================
    # 4. 指标统计与可视化记录
    # ====================================================
    scalar_outputs = {
        "loss": loss,
        "loss_depth": loss_depth,
        "loss_alpha_sup": loss_alpha_sup,
        "continuity_loss": continuity_loss,
        "smoothness_loss": smoothness_loss,
        "normal_loss": normal_loss,
        "cost_margin_loss":cost_margin_loss
    }

    image_outputs = {}
    if detailed_summary:
        image_outputs_pre = generate_edge_alpha_overlays(
            ref_imgs=sample["imgs"]['stage_0'][:, 0],
            edge_alphas_list=outputs["edge_alphas"],
            edges_pixels_list=outputs["tri_infos"][0]['edges_pixels'],
            device=device, overlay_alpha=0.6, line_thickness=1
        )
        image_outputs_gt = generate_edge_alpha_overlays(
            ref_imgs=sample["imgs"]['stage_0'][:, 0],
            edge_alphas_list=edge_alphas_gt,
            edges_pixels_list=outputs["tri_infos"][0]['edges_pixels'],
            device=device, overlay_alpha=0.6, line_thickness=1
        )

        image_outputs = {
            "最终预测结果": depth_est['stage_0'] * mask['stage_0'],
            "stage1深度真值": depth_gt['stage_1'],
            "patchmatch预测的stage2上采样经过恢复的深度值": outputs["output_plane"]['depth_stage1_pixels'],
            "ref_img": sample["imgs"]['stage_0'][:, 0],
            "根据深度真值生成的法向量": gt_normals_vis,
            "经过平面传播生成的平面法向量": outputs["output_plane"]['normal_final'],
            "经过平面传播预测的stage1深度值": depth_patchmatch['stage_1'][-1],
            "没有经过平面传播，刚拟合完初始平面深度值": outputs["output_plane"]['depth_no_pro'],
            "没有经过平面传播，刚拟合完初始平面法向量": outputs["output_plane"]['normal_no_pro'],
            "ref_img_edge_alpha_pre": image_outputs_pre["ref_img_edge_alpha"],
            "ref_img_edge_alpha_gt": image_outputs_gt["ref_img_edge_alpha"]
        }

    # if detailed_summary:
    #     image_outputs["errormap_refined_stage_0"] = (depth_est['stage_0'] - depth_gt['stage_0']).abs() * mask['stage_0']
    #     image_outputs["errormap_patchmatch_stage_1"] = (depth_patchmatch['stage_1'][-1] - depth_gt['stage_1']).abs() * \
    #                                                    mask['stage_1']
    #     image_outputs["errormap_patchmatch_stage_2"] = (depth_patchmatch['stage_2'][-1] - depth_gt['stage_2']).abs() * \
    #                                                    mask['stage_2']
    #     image_outputs["errormap_patchmatch_stage_3"] = (depth_patchmatch['stage_3'][-1] - depth_gt['stage_3']).abs() * \
    #                                                    mask['stage_3']

    scalar_outputs["abs_depth_error_refined_stage_0"] = AbsDepthError_metrics(depth_est['stage_0'], depth_gt['stage_0'],
                                                                              mask['stage_0'] > 0.5)
    scalar_outputs["abs_depth_error_patchmatch_stage_3"] = AbsDepthError_metrics(depth_patchmatch['stage_3'][-1],
                                                                                 depth_gt['stage_3'],
                                                                                 mask['stage_3'] > 0.5)
    scalar_outputs["abs_depth_error_patchmatch_stage_2"] = AbsDepthError_metrics(depth_patchmatch['stage_2'][-1],
                                                                                 depth_gt['stage_2'],
                                                                                 mask['stage_2'] > 0.5)
    scalar_outputs["abs_depth_error_patchmatch_stage_1"] = AbsDepthError_metrics(depth_patchmatch['stage_1'][-1],
                                                                                 depth_gt['stage_1'],
                                                                                 mask['stage_1'] > 0.5)
    # threshold = 1mm
    scalar_outputs["thres1mm_error"] = Thres_metrics(depth_est['stage_0'], depth_gt['stage_0'], mask['stage_0'] > 0.5,
                                                     1)
    # threshold = 2mm
    scalar_outputs["thres2mm_error"] = Thres_metrics(depth_est['stage_0'], depth_gt['stage_0'], mask['stage_0'] > 0.5,
                                                     2)
    # threshold = 4mm
    scalar_outputs["thres4mm_error"] = Thres_metrics(depth_est['stage_0'], depth_gt['stage_0'], mask['stage_0'] > 0.5,
                                                     4)
    # threshold = 8mm
    scalar_outputs["thres8mm_error"] = Thres_metrics(depth_est['stage_0'], depth_gt['stage_0'], mask['stage_0'] > 0.5,
                                                     8)

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


if __name__ == '__main__':
    if args.mode == "train":
        train()
    elif args.mode == "val":
        test()
