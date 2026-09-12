import argparse
import os

import math

from models.edge_head import EdgeLabelGenerator
from models.PlanePatchMatch import *
from models.sum_loss import *

# 控制要暴露给进程的 GPU id：优先使用外部环境变量 GPU_ID，
# 否则使用已有的 CUDA_VISIBLE_DEVICES，最后回退到 '0'
_gpu_choice = os.environ.get('GPU_ID', os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
os.environ['CUDA_VISIBLE_DEVICES'] = str(_gpu_choice)
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
import subprocess
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

# Optional: run large-scene evaluation using the final checkpoint after training
parser.add_argument('--run_big_eval', action='store_true', help='After training, run eval_whu_big on a large testset')
parser.add_argument('--big_eval_dataset', default='dtu_whu_eval_big', help='dataset name for big eval')
parser.add_argument('--big_eval_testpath', default='/home/ym/Experiment/Datas/WHU_MVS_dataset', help='test data path for big eval')
parser.add_argument('--big_eval_testlist', default='lists/whu/bigtest.txt', help='test list file for big eval')

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
        os.makedirs(args.logdir, exist_ok=True)

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
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=collate_keep_list, num_workers=8,
                            drop_last=True)
TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=collate_keep_list, num_workers=4,
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


def load_checkpoint_with_channel_adaptation(model, state_dict_model):
    """
    自适应权重迁移与平滑加载：
    针对 Stage 0 Refinement 从 3 通道 (RGB) 升级至 4 通道 (RGB + W_plane_s0) 实行前3通道继承、第4通道置零初始化，
    确保微调或测试冷启动时输出与基线 100% 相同，实现真正平滑无冲击的 Warm-Start。
    """
    model_state = model.state_dict()
    key = 'upsample_net.conv0.conv.weight'
    if key in state_dict_model and key in model_state:
        ckpt_w = state_dict_model[key]
        tgt_w = model_state[key]
        if ckpt_w.shape[1] == 4 and tgt_w.shape[1] == 3:
            print(f"[Adaptive Loading] Adapting {key} from [8, 4, 3, 3] -> [8, 3, 3, 3] (discarding 4th channel)...")
            state_dict_model[key] = ckpt_w[:, :3, :, :]
        elif ckpt_w.shape[1] == 3 and tgt_w.shape[1] == 4:
            print(f"[Adaptive Loading] Adapting {key} from [8, 3, 3, 3] -> [8, 4, 3, 3] (Channel 4 set to 0.0 for zero-shock start)...")
            adapted_w = torch.zeros_like(tgt_w)
            adapted_w[:, :3, :, :] = ckpt_w
            state_dict_model[key] = adapted_w

    try:
        model.load_state_dict(state_dict_model)
    except RuntimeError as e:
        print(f"Warning: Exact load failed ({e}), loading with strict=False...")
        model.load_state_dict(state_dict_model, strict=False)


# load 模型 parameters
start_epoch = 0
if (args.mode == "train" and args.resume) or (args.mode == "test" and not args.loadckpt):
    saved_models = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(args.logdir, saved_models[-1])
    print("resuming", loadckpt)
    state_dict = torch.load(loadckpt)
    load_checkpoint_with_channel_adaptation(model, state_dict['model'])
    try:
        optimizer.load_state_dict(state_dict['optimizer'])
    except Exception as e:
        print(f"Warning: Failed to load optimizer state ({e}), re-initializing optimizer state fresh.")
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    load_checkpoint_with_channel_adaptation(model, state_dict['model'])
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
            loss, scalar_outputs, image_outputs = test_sample(sample, detailed_summary=do_summary_image,
                                                              global_step=global_step, total_steps=total_steps)
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


def tensor_to_pseudocolor(tensor_map, mask=None, colormap=cv2.COLORMAP_JET, invert=True):
    """
    将单通道特征图转换为 RGB 伪彩色 Tensor，专用于 TensorBoard 优雅展示。

    Args:
        tensor_map: [H, W] 或 [1, H, W] 的单通道 Tensor，值域 [0, 1]
        mask: [H, W] 或 [1, H, W] 的布尔/0-1 Tensor，可选。无效区域将被涂成纯黑。
        colormap: OpenCV 伪彩色映射表
        invert: 是否反转数值。反转后 1.0(平面)映射为蓝，0.0(曲面)映射为红。

    Returns:
        [3, H, W] 的 RGB Tensor，值域 [0, 1]
    """
    # 1. 维度降维保证是 2D
    if tensor_map.dim() == 3:
        tensor_map = tensor_map.squeeze(0)

    # 2. 转为 numpy uint8
    map_np = (tensor_map.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)

    # 3. 色彩反转 (满足红危险/蓝安全的工程直觉)
    if invert:
        map_np = 255 - map_np

    # 4. 应用 OpenCV 伪彩色映射 (输出为 BGR)
    color_map_bgr = cv2.applyColorMap(map_np, colormap)

    # 5. 绝对纯黑背景截断 (解决 JET 的 0 是深蓝的问题)
    if mask is not None:
        if mask.dim() == 3:
            mask = mask.squeeze(0)
        mask_np = mask.detach().cpu().numpy()
        color_map_bgr[mask_np == 0] = [0, 0, 0]  # BGR 纯黑

    # 6. BGR 转 RGB，再转回 Tensor
    color_map_rgb = cv2.cvtColor(color_map_bgr, cv2.COLOR_BGR2RGB)
    color_tensor = torch.from_numpy(color_map_rgb).permute(2, 0, 1).float() / 255.0

    return color_tensor.unsqueeze(0)  # 输出变为 [1, 3, H, W]

def cosine_temperature_schedule(progress, t_min=0.55, t_max=1.0):
    """余弦平滑退火控温：progress∈[0,1] 时温度从 t_max 柔和降至 t_min。"""
    prog_val = max(0.0, min(float(progress), 1.0))
    cos_decay = 0.5 * (1.0 + math.cos(prog_val * math.pi))
    return t_min + (t_max - t_min) * cos_decay


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


def build_w_plane_tensorboard_views(output_plane, mask_s1_batch0=None):
    """
    将传播前/后平面置信度转为 TensorBoard 伪彩色图及差分图。
    返回 dict，键为空则对应张量不可用。
    """
    views = {}
    if 'W_plane_pixel' not in output_plane or 'W_plane_pixel_init' not in output_plane:
        return views

    w_after = output_plane['W_plane_pixel'][0, 0]
    w_before = output_plane['W_plane_pixel_init'][0, 0]
    mask = mask_s1_batch0

    views['W_plane_传播后'] = tensor_to_pseudocolor(tensor_map=w_after, mask=mask, invert=True)
    views['W_plane_传播前_物理冷启动'] = tensor_to_pseudocolor(tensor_map=w_before, mask=mask, invert=True)

    # 替换原本的 views['W_plane_更新前后差分_abs'] 为平面置信度真值 (W_GT_pixel)
    if 'W_plane_gt_pixel' in output_plane:
        w_gt = output_plane['W_plane_gt_pixel'][0, 0]
        views['W_plane_真值'] = tensor_to_pseudocolor(tensor_map=w_gt, mask=mask, invert=True)
    else:
        # 兜底：若评估/测试阶段没有算 Loss 从而没有真值，则显示更新前后差分图
        w_diff = (w_after - w_before).abs()
        views['W_plane_更新前后差分_abs'] = tensor_to_pseudocolor(tensor_map=w_diff, mask=mask, invert=False)
    return views


def compute_stage1_flat_region_mae_tensors(outputs, pixel_depth_s1, depth_gt_s1, device):
    """
    W_plane > 0.80 高置信平面区域内的 Stage 1 深度 MAE（纯 GPU 张量，无 matplotlib）。
    
    升级特性：
    1. planar_err_vis [B,1,H,W]: 差值等比例放大图像。0.5米判定摸顶，让 0.08m 级别的优秀误差清晰可见，其余抹黑。
    2. planar_color_vis [B,3,H,W]: 刚性红蓝判定图。大于0.80的平面为纯蓝，小于0.80的曲面/碎面为纯红，背景纯黑。
    """
    # 1. 建立高鲁棒性字典探针，封杀 KeyError
    output_plane_dict = outputs.get("output_plane", {})
    # 优先采用高精度离线置信度真值用于评估，若限制缺失则退回预测值
    if 'W_plane_gt_pixel' in output_plane_dict:
        w_plane_pixel = output_plane_dict['W_plane_gt_pixel']
    elif 'W_plane_pixel' in output_plane_dict:
        w_plane_pixel = output_plane_dict['W_plane_pixel']
    else:
        w_plane_pixel = torch.zeros_like(depth_gt_s1)

    if w_plane_pixel.shape[2:] != depth_gt_s1.shape[2:]:
        w_plane_pixel = F.interpolate(
            w_plane_pixel.float(), size=depth_gt_s1.shape[2:], mode='nearest'
        )

    # 2. 密集拓扑有效合规检测
    tri_id_map = output_plane_dict.get('tri_id_map', None)
    if tri_id_map is not None:
        tri_valid = (tri_id_map >= 0)
        if tri_valid.dim() == 3:
            tri_valid = tri_valid.unsqueeze(1)
    else:
        tri_valid = torch.ones_like(depth_gt_s1, dtype=torch.bool)

    # 3. 完美合成平面单纯形掩码场
    mask_valid = (pixel_depth_s1 > 0) & (depth_gt_s1 > 0) & tri_valid
    mask_planar = mask_valid & (w_plane_pixel > 0.80)  # 严格对齐 0.80 刚性判据

    # 4. 解析空间绝对误差提取
    err_map_dense = torch.abs(pixel_depth_s1 - depth_gt_s1)

    if mask_planar.any():
        planar_mae_val = err_map_dense[mask_planar].mean()
        if 'depth_stage1_pixels' in output_plane_dict:
            depth_pixels = output_plane_dict['depth_stage1_pixels']
            err_map_pixel = torch.abs(depth_pixels - depth_gt_s1)
            pixel_mae_val = err_map_pixel[mask_planar].mean()
        else:
            pixel_mae_val = torch.tensor(0.0, device=device)
    else:
        planar_mae_val = torch.tensor(0.0, device=device)
        pixel_mae_val = torch.tensor(0.0, device=device)

    # =====================================================================
    # 👑 【核心修改点一】：单通道深度差值等比例放大（解救纯黑）
    # =====================================================================
    # 设定最高截止线为 0.5 米。任何大于 0.5 米的误差被判定为摸顶(1.0 full brightness)
    # 此时 0.1 米的误差会被等比例映射放大为 0.2 的灰度亮度，在 Tensorboard 里呈现清晰的白色/灰色轮廓
    max_error_cutoff = 0.5
    error_amplified = torch.clamp(err_map_dense / max_error_cutoff, 0.0, 1.0)
    # 平面区域外（曲面、背景）施加物理黑色放逐（归 0）
    planar_err_vis = torch.where(mask_planar, error_amplified, torch.zeros_like(error_amplified))

    # =====================================================================
    # 👑 【核心修改点二】：3通道 RGB 刚性红蓝相变判定图生成
    # =====================================================================
    B, _, H, W = depth_gt_s1.shape
    planar_color_vis = torch.zeros(B, 3, H, W, device=device)
    
    # 满足 W > 0.80 的刚性大平白墙区域 ──> 赋予纯蓝色 [0, 0, 1]
    planar_color_vis[:, 2:3, :, :] = mask_planar.float()
    
    # 有效深度区域内，但是 W <= 0.80 的高频曲面/断层小碎面区域 ──> 赋予纯红色 [1, 0, 0]
    mask_soft_curved = mask_valid & (~mask_planar)
    planar_color_vis[:, 0:1, :, :] = mask_soft_curved.float()
    
    # 其余区域（无效全黑背景）自动保持默认的零值纯黑色 [0, 0, 0]

    return planar_mae_val, pixel_mae_val, planar_err_vis, planar_color_vis, mask_planar


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

    # ym-modify 重写了一下对于cdt—data数据进行了一个跳过，同时也跳过超轻量几何变长列表的直接转换
    skip = ["vertexs", "lines", "triangles", "tri_conf_cleaned", "tri_normal_cleaned", "is_gt_planar"]
    sample_cuda = tocuda(sample, device=device, skip_keys=skip)

    # 手动转换并移动到 GPU
    sample_cuda['tri_conf_cleaned'] = [torch.from_numpy(v).to(device).float() for v in sample['tri_conf_cleaned']]
    sample_cuda['tri_normal_cleaned'] = [torch.from_numpy(v).to(device).float() for v in sample['tri_normal_cleaned']]
    if 'is_gt_planar' in sample:
        sample_cuda['is_gt_planar'] = [torch.from_numpy(v).to(device).bool() for v in sample['is_gt_planar']]

    depth_gt = sample_cuda["depth"]
    mask = sample_cuda["mask"]

    # 启用梯度异常检测
    # torch.autograd.set_detect_anomaly(True)

    # 一开始不启动连续性约束和光滑性约束，后面再打开，目前是直接打开
    progress = global_step / total_steps
    # 连续性约束和光滑性约束权重
    # max_lambda_c = 40.0
    # max_lambda_s = 1.0
    max_lambda_c = 0.0
    max_lambda_s = 0.0
    max_lambda_n_tri = 1.0  # 宏观平面级法向峰值（完全恢复至创造 0.185m/0.106m 黄金记录时的 1.0，贡献约 0.05）
    max_lambda_n_pix = 0.0  # 微观像素级 D2N 法向峰值（已关闭，彻底释放 FeatureNet 泛化能力）
    max_lambda_n_0 = 0.0
    max_lambda_cost = 0.0  # 完全关闭代价裕量损失，彻底断绝对 FeatureNet 特征底座的反向梯度污染
    weight_alpha = 0.15    # 辅助边分类损失降权，防止绑架总梯度范数
    max_lambda_dnc_s1 = 0.05 * 50
    max_lambda_plane = 0.10  # 辅助平面置信度分类损失降权，作为正则项介入

    # ====================================================
    # 物理课程学习（Curriculum Learning）黄金阶梯错峰调度
    # ====================================================
    # 1. 法向约束 (一阶方向引导，率先点火，摆正平面且不破坏深度边界)：
    # 0.15 启动，0.40 满载，0.85 开始松绑，底线保留 75%
    lambda_n_tri = get_smooth_weight_with_decay(progress, 0.15, 0.40, 0.85, max_lambda_n_tri, end_ratio=0.75)
    lambda_n_pix = get_smooth_weight_with_decay(progress, 0.15, 0.40, 0.85, max_lambda_n_pix, end_ratio=0.75) if max_lambda_n_pix > 0.0 else 0.0

    # 2. 连通性约束 (强几何缝合，等 edge_head 充分预热学会断裂后再缓坡介入)：
    # 0.30 启动，0.55 满载，0.85 开始松绑，底线保留 5%
    lambda_c = get_smooth_weight_with_decay(progress, 0.30, 0.55, 0.85, max_lambda_c, end_ratio=0.05)

    # 3. 光滑性约束 (曲率平滑，晚启动，最后微调局部细节)：
    # 0.40 启动，0.65 满载，0.80 开始松绑，底线保留 40%
    lambda_s = get_smooth_weight_with_decay(progress, 0.40, 0.65, 0.80, max_lambda_s, end_ratio=0.40)

    # 4. cost约束
    lambda_cost = get_smooth_weight_with_decay(progress, 0.0, 0.1, 0.6, max_lambda_cost, end_ratio=0.5)

    lambda_dnc_s1 = get_smooth_weight_with_decay(progress, 0.0, 0.2, 1.0, max_lambda_dnc_s1, end_ratio=1.0)

    lambda_plane_s1=get_smooth_weight_with_decay(progress, 0.05, 0.15, 1.0, max_lambda_plane, end_ratio=1.0)


    # 余弦柔和退火控温 [1.0 -> 0.55]，避免线性下坠引发 Softmax 阶跃相变
    current_temp = cosine_temperature_schedule(progress)

    # 自动构建计算图（动态计算图），记录每个张量的操作历史（如卷积、激活、矩阵乘法等），从而在反向传播时能通过链式法则计算梯度
    outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"],sample_cuda["intrinsics_mats"],
                    sample_cuda["depth_min"], sample_cuda["depth_max"],
                    vertexs_batch, lines_batch, triangles_batch,depth_gt['stage_1'],
                    lambda_c,lambda_s,current_temp)

    depth_est = outputs["refined_depth"]

    depth_patchmatch = outputs["depth_patchmatch"]

    # 通过计算最终的损失
    # ====================================================
    # 1. 深度主任务 Loss (解耦：原版 PatchmatchNet 纯像素深度主损失 + 新增平面级深度加权损失)
    # ====================================================
    # 局部三角形有效掩码 (仅供平面置信度/可视化等局部几何计算使用，严禁覆盖数据集全局真实 mask)
    valid_mask_s1 = (outputs["output_plane"]['tri_id_map'] >= 0).float().unsqueeze(dim=1)
    valid_mask_s0 = (outputs["output_plane"]['tri_id_map_stage0'] >= 0).float().unsqueeze(dim=1)

    # 1.1 原版 PatchmatchNet 纯像素深度主损失 (6项标准像素级深度损失，完全等价于官方源工程)
    loss_depth = patchmatchnet_loss(
        depth_patchmatch=depth_patchmatch,
        refined_depth=depth_est,
        depth_gt=depth_gt,
        mask=mask
    )

    # 1.2 新增解耦的 Stage 1 平面级深度异方差加权损失
    loss_depth_plane = compute_plane_depth_loss(
        depth_plane=depth_patchmatch['stage_1'][-1],
        depth_gt=depth_gt['stage_1'],
        mask=mask['stage_1'],
        W_plane_pixel=outputs["output_plane"].get("W_plane_pixel", None),
        gamma=1.5
    )

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

    # 边预测头 Loss 计算
    loss_alpha_raw, loss_sparsity_raw = compute_edge_supervision_loss_new(
        alphas_list=outputs["edge_alphas"],
        gt_stage0_list=edge_alphas_gt,
        valid_mask_list=valid_mask_list
    )

    # DNC损失
    loss_dnc_s1=0.0

    # loss_dnc_s1 = compute_gated_dnc_loss(
    #     Z_pixel=depth_patchmatch['stage_1'][-1],  # 🚀 直接复用物理防爆渲染器出的 1通道 密集真实深度
    #     N_pixel=outputs["output_plane"]['normal_pro_pure'],  # 🚀 采用我们单独剥离出来的未被可视化污染的 3通道 密集真法向
    #     W_plane_pixel=outputs["output_plane"]['W_plane_pixel'].detach(),  # 高隔离度空间软路由拦截闸
    #     tri_id_map= outputs["output_plane"]['tri_id_map'],
    #     intrinsics=ref_intrinsics,
    #     valid_mask=valid_mask_s1  # 强力剔除全黑虚空背景
    # )

    # 平面置信度
    loss_plane_s1, W_GT_pixel = compute_confidence_supervision_loss(
        pixel_depth_pred=depth_patchmatch['stage_1'][-1].detach(),
        pixel_depth_gt=depth_gt['stage_1'],
        pixel_normal_pred=outputs["output_plane"]['normal_pro_pure'],
        tri_id_map=outputs["output_plane"]['tri_id_map'],
        W_pred=outputs["output_plane"]["W_plane_tri"],
        progress=progress,
        depth_range=(sample_cuda["depth_min"], sample_cuda["depth_max"]),
        valid_mask=valid_mask_s1,
        intrinsics=ref_intrinsics,
        tri_conf_gt=sample_cuda["tri_conf_cleaned"],
        is_gt_planar=sample_cuda.get("is_gt_planar", None)
    )
    outputs["output_plane"]["W_plane_gt_pixel"] = W_GT_pixel

    # --- 🚨 新增：基于预处理 npz 数据，在线极速查表映射 Stage 1 真值置信度与法线 ---
    tri_id_map = outputs["output_plane"].get('tri_id_map', None)
    if tri_id_map is not None and "tri_conf_cleaned" in sample_cuda:
        B, H_s1, W_s1 = tri_id_map.shape
        tri_device = tri_id_map.device
        planar_soft_conf_s1_list = []
        gt_normal_map_s1_list = []
        
        for b in range(B):
            # 将 bool 掩码转为 float (True变为1.0, False变为0.0)，映射到 TensorBoard
            tri_conf_b = sample_cuda['is_gt_planar'][b].float()     # [N_tri]
            tri_normal_b = sample_cuda['tri_normal_cleaned'][b] # [N_tri, 3]
            tri_id_map_b = tri_id_map[b]                        # [H_s1, W_s1]
            
            dummy_conf = torch.cat([tri_conf_b, torch.tensor([0.0], device=tri_device)], dim=0)
            dummy_normal = torch.cat([tri_normal_b, torch.tensor([[0.0, 0.0, 0.0]], device=tri_device)], dim=0)
            
            safe_idx = torch.where(tri_id_map_b >= 0, tri_id_map_b, torch.tensor(tri_conf_b.shape[0], device=tri_device).long())
            
            planar_soft_conf_s1_b = dummy_conf[safe_idx]  # [H_s1, W_s1]
            gt_normal_map_s1_b = dummy_normal[safe_idx]    # [H_s1, W_s1, 3]
            
            planar_soft_conf_s1_list.append(planar_soft_conf_s1_b)
            gt_normal_map_s1_list.append(gt_normal_map_s1_b)
            
        planar_soft_conf_s1 = torch.stack(planar_soft_conf_s1_list, dim=0).unsqueeze(1) # [B, 1, H_s1, W_s1]
        gt_normal_map_s1 = torch.stack(gt_normal_map_s1_list, dim=0) # [B, H_s1, W_s1, 3]
        
        # 将降级且清晰的 W_plane_gt 塞入输出字典，使 TensorBoard 可视化完全对齐！
        outputs["output_plane"]["W_plane_gt_pixel"] = planar_soft_conf_s1
        outputs["output_plane"]["gt_normal_map_s1"] = gt_normal_map_s1

    loss_plane_s1 = loss_plane_s1 * lambda_plane_s1
    # ====================================================
    # 3. 损失函数的混合
    # ====================================================

    # 1. 计算宏观平面级法向量损失 (每个三角面参数对比 GT SVD 法向)
    normal_loss_tri = torch.tensor(0.0, device=device)
    if "is_gt_planar" in sample_cuda and "tri_normal_cleaned" in sample_cuda:
        pixel_counts = outputs["output_plane"]["pixel_counts"] # [B, N_tri]
        
        normal_loss_tri = compute_normal_gt_loss(
            n_pred=outputs["output_plane"]["final_plane"][..., :3],
            n_gt=sample_cuda["tri_normal_cleaned"],
            is_gt_planar=sample_cuda["is_gt_planar"],
            pixel_counts=pixel_counts,
            min_pixels=3
        )

    # 2. 计算微观像素级 D2N 法向损失（带权重门控判断：若 max_lambda_n_pix <= 0 则彻底跳过计算，零计算开销）
    if max_lambda_n_pix > 0.0:
        normal_loss_pix = compute_pixel_d2n_normal_loss(
            depth_pixel_pred=depth_patchmatch['stage_1'][0],
            depth_gt=depth_gt['stage_1'],
            intrinsics=ref_intrinsics,
            mask=valid_mask_s1,
            cliff_threshold=0.8,
            W_plane_pixel=outputs["output_plane"].get("W_plane_pixel", None),
            threshold_low=0.3,
            threshold_high=0.8
        )
    else:
        normal_loss_pix = torch.tensor(0.0, device=device)
        
    normal_loss_s0 = 0.0

    # 计算 Cost Margin Loss（带权重门控判断：若 max_lambda_cost <= 0 则彻底跳过计算）
    if max_lambda_cost > 0.0:
        cost_margin_loss = compute_pixel_cost_margin_loss(
            no_prop_depth=outputs["output_plane"]['depth_no_pro'].detach(),  # SVD 的初始深度
            gt_depth=depth_gt['stage_1'],  # GT 深度
            pixel_costs=outputs["output_plane"]['pixel_costs'],  # 畅通回传给特征网的代价
            tri_id_map=outputs["output_plane"]['tri_id_map']
        )
        cost_margin_loss = cost_margin_loss * lambda_cost
    else:
        cost_margin_loss = torch.tensor(0.0, device=device)

    # 边缘监督 Loss
    # 乘上一个权重再，加上边断裂损失，防止预测头损失过小

    loss_alpha_sup=loss_alpha_raw * weight_alpha + loss_sparsity_raw

    # 3.连续性正则化 Loss (Geometric Smoothness)
    # 因为它是正则化项，绝不能喧宾夺主。建议权重设为 0.1 ~ 0.5
    # 4.光滑性约束

    continuity_loss = outputs["continuity_loss"] * lambda_c
    smoothness_loss = outputs["smoothness_loss"] * lambda_s
    normal_loss = (normal_loss_tri * lambda_n_tri) + (normal_loss_pix * lambda_n_pix) + (normal_loss_s0 * max_lambda_n_0)

    # DNC 损失
    loss_dnc_s1 = loss_dnc_s1 * lambda_dnc_s1

    # 总损失：原版主深度损失 + 独立平面深度损失 + 边断裂损失 + 连续性损失 + 光滑性约束 + 法向量损失 + cost损失 + DNC损失 + 平面置信度
    lambda_plane_depth = 1.0  # 平面深度加权损失系数（1.0 严格保持与原方案数学等价，消融时可置为 0.0）
    loss = loss_depth + (loss_depth_plane * lambda_plane_depth) + loss_alpha_sup + continuity_loss + smoothness_loss + normal_loss + cost_margin_loss + loss_dnc_s1 + loss_plane_s1

    # 边断裂损失
    loss.backward()

    # todo：将梯度限制maxmax_norm以内
    # ← 必须在这里，backward 之后才有梯度可以裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

    # 优化器根据计算的梯度更新模型参数（梯度下降的具体实现）
    optimizer.step()

    scalar_outputs = {"loss": loss,
                      "loss_depth": loss_depth,
                      "loss_depth_plane": loss_depth_plane,
                      "loss_depth_total": loss_depth + loss_depth_plane,
                      "loss_alpha_sup": loss_alpha_sup,
                      "continuity_loss": continuity_loss,
                      "smoothness_loss": smoothness_loss,
                      "normal_loss": normal_loss,
                      "cost_margin_loss": cost_margin_loss,
                      "loss_plane_s1": loss_plane_s1,
                      "loss_dnc_s1": loss_dnc_s1
                      }

        # 记录到 scalar_outputs 以便主循环写入 Tensorboard
    if isinstance(normal_loss_tri, torch.Tensor):
        scalar_outputs["loss_normal_gt_s1"] = (normal_loss_tri * lambda_n_tri).item()
    else:
        scalar_outputs["loss_normal_gt_s1"] = 0.0
    if isinstance(normal_loss_pix, torch.Tensor):
        scalar_outputs["loss_normal_pix_s1"] = (normal_loss_pix * lambda_n_pix).item()
    else:
        scalar_outputs["loss_normal_pix_s1"] = 0.0
    scalar_outputs["loss_normal_total_s1"] = (
        scalar_outputs["loss_normal_gt_s1"] + scalar_outputs["loss_normal_pix_s1"]
    )

    image_outputs = {}

    pixel_depth_s1 = depth_patchmatch['stage_1'][-1]
    planar_mae_val, pixel_mae_val, _ , planar_err_masked , _= compute_stage1_flat_region_mae_tensors(
        outputs, pixel_depth_s1, depth_gt['stage_1'], device
    )

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

        mask_s1_b0 = mask['stage_1'][0, 0] if 'stage_1' in mask else None
        w_plane_views = build_w_plane_tensorboard_views(outputs["output_plane"], mask_s1_batch0=mask_s1_b0)
        w_plane_tb_tensor = w_plane_views.get('W_plane_传播后')

        vis_nomal_final=get_visual_normal(outputs["output_plane"]['final_normal'], valid_mask_s0)

        vis_depth_final = get_visual_depth(depth_est['stage_0'], valid_mask_s0)
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
        # 👑 构建 Stage 1 物理混合融合深度 (Z_fused) 用于可视化展示
        if "W_plane_pixel" in outputs["output_plane"]:
            w_pixel_s1 = outputs["output_plane"]["W_plane_pixel"]
            mask_planar_s1 = (w_pixel_s1 >= 0.80)
            depth_s1_fused = torch.where(mask_planar_s1, depth_patchmatch['stage_1'][-1], depth_patchmatch['stage_1'][0])
        else:
            depth_s1_fused = depth_patchmatch['stage_1'][-1]

        image_outputs = {  # 暂时注释一些图片，输出的图片太多了
            "stage0最终深度预测值": vis_depth_final,
            "stage1深度真值": depth_gt['stage_1'] ,
            "patchmatch预测的stage2上采样经过恢复的深度值": outputs["output_plane"]['depth_stage1_pixels'],
            # "patchmatch预测的stage2深度值": depth_patchmatch['stage_2'][-1] * mask['stage_2'],
            # "depth_patchmatch_stage_3": depth_patchmatch['stage_3'][-1] * mask['stage_3'],
            "ref_img": sample["imgs"]['stage_0'][:, 0],
            # todo:暂时不要真值法向量可视化
            # "根据深度真值生成的法向量": gt_normals_vis,
            "物理平面置信度_W_plane_传播后": w_plane_tb_tensor,
            "最终预测的法向量":vis_nomal_final,
            # "经过平面传播预测的stage1深度值生成的像素法向量": normal_pred_s1,
            # 新增：基于平面的深度图和法向量图传播完
            "经过平面传播生成的平面法向量": outputs["output_plane"]['normal_pro'],
            "stage1融合深度预测值": depth_s1_fused,
            # 新增：基于平面的深度图和法向量图，刚拟合
            "没有经过平面传播，刚拟合完初始平面深度值": outputs["output_plane"]['depth_no_pro'],
            "没有经过平面传播，刚拟合完初始平面法向量": outputs["output_plane"]['normal_no_pro'] ,
            # 新增：边预测头预测值和真值
            "ref_img_edge_alpha_pre": ref_img_edge_alpha_pre,
            "ref_img_edge_alpha_gt": ref_img_edge_alpha_gt,
            "stage0预测截断平面区域": outputs["output_plane"].get("is_planar_s0", torch.zeros_like(depth_patchmatch['stage_1'][-1])).float()
        }

        # image_outputs["errormap_refined_stage_0"] = (depth_est['stage_0'] - depth_gt['stage_0']).abs() * mask['stage_0']
        # image_outputs["errormap_patchmatch_stage_1"] = (depth_patchmatch['stage_1'][-1] - depth_gt['stage_1']).abs() * \
        #                                                mask['stage_1']
        # image_outputs["errormap_patchmatch_stage_2"] = (depth_patchmatch['stage_2'][-1] - depth_gt['stage_2']).abs() * \
        #                                                mask['stage_2']
        # image_outputs["errormap_patchmatch_stage_3"] = (depth_patchmatch['stage_3'][-1] - depth_gt['stage_3']).abs() * \
        #                                                mask['stage_3']

        image_outputs["stage1_planar_error_map"] = planar_err_masked
        if 'cross_check_diagnosis_rgb' in outputs.get("output_plane", {}):
            image_outputs["CrossCheck_三态诊断图(绿保留_红降分)"] = outputs["output_plane"]['cross_check_diagnosis_rgb']
        image_outputs.update(w_plane_views)

    # 平面置信度更新幅度标量（有效三角网格内）
    if 'W_plane_pixel' in outputs["output_plane"] and 'W_plane_pixel_init' in outputs["output_plane"]:
        w_a = outputs["output_plane"]['W_plane_pixel']
        w_b = outputs["output_plane"]['W_plane_pixel_init']
        tri_ok = outputs["output_plane"]['tri_id_map'] >= 0
        if tri_ok.dim() == 3:
            tri_ok = tri_ok.unsqueeze(1)
        conf_valid = tri_ok & (w_a > 0) & (w_b > 0)
        if conf_valid.any():
            scalar_outputs["w_plane_mean_abs_update"] = (w_a - w_b).abs()[conf_valid].mean()
            scalar_outputs["w_plane_mean_before"] = w_b[conf_valid].mean()
            scalar_outputs["w_plane_mean_after"] = w_a[conf_valid].mean()
        else:
            scalar_outputs["w_plane_mean_abs_update"] = 0.0
            scalar_outputs["w_plane_mean_before"] = 0.0
            scalar_outputs["w_plane_mean_after"] = 0.0

    scalar_outputs["abs_depth_error_refined_stage_0"] = AbsDepthError_metrics(depth_est['stage_0'], depth_gt['stage_0'],
                                                                              mask['stage_0'] > 0.5)
    
    if "is_planar_s0" in outputs["output_plane"]:
        is_planar_s0 = outputs["output_plane"]["is_planar_s0"]
        mask_s0 = mask['stage_0'] > 0.5
        scalar_outputs["stage0_planar_mae"] = AbsDepthError_metrics(depth_est['stage_0'], depth_gt['stage_0'], mask_s0 & is_planar_s0)
        scalar_outputs["stage0_curved_mae"] = AbsDepthError_metrics(depth_est['stage_0'], depth_gt['stage_0'], mask_s0 & (~is_planar_s0))
        
        # 👑 新增诊断指标：分析 Stage 1 原始像素深度 (depth_patchmatch['stage_1'][0]) 与传播深度 ([-1]) 在两区的表现
        # 先将 1/2 分辨率的深度图双线性上采样到 Stage 0 以对齐掩码
        depth_s1_pixel_up = F.interpolate(depth_patchmatch['stage_1'][0], size=depth_gt['stage_0'].shape[2:], mode='bilinear', align_corners=False)
        depth_s1_prop_up = F.interpolate(depth_patchmatch['stage_1'][-1], size=depth_gt['stage_0'].shape[2:], mode='bilinear', align_corners=False)
        
        scalar_outputs["stage1_pixel_planar_mae"] = AbsDepthError_metrics(depth_s1_pixel_up, depth_gt['stage_0'], mask_s0 & is_planar_s0)
        scalar_outputs["stage1_pixel_curved_mae"] = AbsDepthError_metrics(depth_s1_pixel_up, depth_gt['stage_0'], mask_s0 & (~is_planar_s0))
        scalar_outputs["stage1_prop_planar_mae"] = AbsDepthError_metrics(depth_s1_prop_up, depth_gt['stage_0'], mask_s0 & is_planar_s0)
        scalar_outputs["stage1_prop_curved_mae"] = AbsDepthError_metrics(depth_s1_prop_up, depth_gt['stage_0'], mask_s0 & (~is_planar_s0))

    scalar_outputs["abs_depth_error_patchmatch_stage_3"] = AbsDepthError_metrics(depth_patchmatch['stage_3'][-1],
                                                                                 depth_gt['stage_3'],
                                                                                 mask['stage_3'] > 0.5)
    scalar_outputs["abs_depth_error_patchmatch_stage_2"] = AbsDepthError_metrics(depth_patchmatch['stage_2'][-1],
                                                                                 depth_gt['stage_2'],
                                                                                 mask['stage_2'] > 0.5)
    scalar_outputs["abs_depth_error_patchmatch_stage_1"] = AbsDepthError_metrics(depth_patchmatch['stage_1'][-1],
                                                                                 depth_gt['stage_1'],
                                                                                 mask['stage_1'] > 0.5)
    scalar_outputs["abs_depth_error_patchmatch_stage_1_pixel_raw"] = AbsDepthError_metrics(depth_patchmatch['stage_1'][0],
                                                                                            depth_gt['stage_1'],
                                                                                            mask['stage_1'] > 0.5)
    # 👑 严格对齐 eval_whu_big.py:L2865 的 Stage 1 像素/平面全图物理混合融合深度 (Z_fused)
    # 统计全图有效真值区域 (depth_gt > 0)，三角剖分外的区域天然回退为纯像素深度 depth_patchmatch['stage_1'][0]
    if "W_plane_pixel" in outputs["output_plane"]:
        w_pixel_s1 = outputs["output_plane"]["W_plane_pixel"]
        mask_planar_s1 = (w_pixel_s1 >= 0.80)
        # 平面区走 Stage 1 最终平面深度 depth_patchmatch['stage_1'][-1]，非平面区/三角网外回退至纯像素深度 depth_patchmatch['stage_1'][0]
        depth_s1_fused = torch.where(mask_planar_s1, depth_patchmatch['stage_1'][-1], depth_patchmatch['stage_1'][0])
        valid_gt_mask_s1 = mask['stage_1'] > 0.5
        scalar_outputs["abs_depth_error_patchmatch_stage_1_fused"] = AbsDepthError_metrics(
            depth_s1_fused, depth_gt['stage_1'], valid_gt_mask_s1
        )
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
    scalar_outputs["stage1_flat_region_mae"] = planar_mae_val
    scalar_outputs["stage1_flat_region_pixel_mae"] = pixel_mae_val

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


@make_nograd_func
def test_sample(sample, detailed_summary=False, global_step=0, total_steps=1):
    model.eval()
    progress = global_step / total_steps
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

    # ym-modify 重写了一下对于cdt—data数据进行了一个跳过，同时也跳过超轻量几何变长列表的直接转换
    skip = ["vertexs", "lines", "triangles", "tri_conf_cleaned", "tri_normal_cleaned"]
    sample_cuda = tocuda(sample, device=device, skip_keys=skip)

    # 手动转换并移动到 GPU
    sample_cuda['tri_conf_cleaned'] = [torch.from_numpy(v).to(device).float() for v in sample['tri_conf_cleaned']]
    sample_cuda['tri_normal_cleaned'] = [torch.from_numpy(v).to(device).float() for v in sample['tri_normal_cleaned']]

    depth_gt = sample_cuda["depth"]
    mask = sample_cuda["mask"]

    # ====================================================
    # 2. 模型 Forward (补齐缺失的 lambda 参数)
    # ====================================================
    # 在测试阶段，与当前训练设置严格对齐（暂不开启强缝合与平滑）
    max_lambda_c = 0.0
    max_lambda_s = 0.0
    max_lambda_n = 0.3
    max_lambda_cost = 0.0
    max_lambda_dnc_s1 = 0.05

    current_temp = cosine_temperature_schedule(progress)
    outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["intrinsics_mats"],
                    sample_cuda["depth_min"], sample_cuda["depth_max"],
                    vertexs_batch, lines_batch, triangles_batch, depth_gt['stage_1'],
                    max_lambda_c, max_lambda_s,current_temp)

    depth_est = outputs["refined_depth"]
    depth_patchmatch = outputs["depth_patchmatch"]

    # --- 🚨 新增：基于预处理 npz 数据，在线极速查表映射 Stage 1 真值置信度与法线 ---
    tri_id_map = outputs["output_plane"].get('tri_id_map', None)
    if tri_id_map is not None and "tri_conf_cleaned" in sample_cuda:
        B, H_s1, W_s1 = tri_id_map.shape
        tri_device = tri_id_map.device
        planar_soft_conf_s1_list = []
        gt_normal_map_s1_list = []
        
        for b in range(B):
            tri_conf_b = sample_cuda['tri_conf_cleaned'][b]     # [N_tri]
            tri_normal_b = sample_cuda['tri_normal_cleaned'][b] # [N_tri, 3]
            tri_id_map_b = tri_id_map[b]                        # [H_s1, W_s1]
            
            dummy_conf = torch.cat([tri_conf_b, torch.tensor([0.0], device=tri_device)], dim=0)
            dummy_normal = torch.cat([tri_normal_b, torch.tensor([[0.0, 0.0, 0.0]], device=tri_device)], dim=0)
            
            safe_idx = torch.where(tri_id_map_b >= 0, tri_id_map_b, torch.tensor(tri_conf_b.shape[0], device=tri_device).long())
            
            planar_soft_conf_s1_b = dummy_conf[safe_idx]  # [H_s1, W_s1]
            gt_normal_map_s1_b = dummy_normal[safe_idx]    # [H_s1, W_s1, 3]
            
            planar_soft_conf_s1_list.append(planar_soft_conf_s1_b)
            gt_normal_map_s1_list.append(gt_normal_map_s1_b)
            
        planar_soft_conf_s1 = torch.stack(planar_soft_conf_s1_list, dim=0).unsqueeze(1) # [B, 1, H_s1, W_s1]
        gt_normal_map_s1 = torch.stack(gt_normal_map_s1_list, dim=0) # [B, H_s1, W_s1, 3]
        
        outputs["output_plane"]["W_plane_gt_pixel"] = planar_soft_conf_s1
        outputs["output_plane"]["gt_normal_map_s1"] = gt_normal_map_s1

    # ====================================================
    # 3. 损失计算 (同步使用新的 Edge Loss 和 Normal Loss)
    # ====================================================
    # 局部三角形有效掩码 (仅供局部几何计算使用，严禁覆盖数据集全局真实 mask)
    valid_mask_s1 = (outputs["output_plane"]['tri_id_map'] >= 0).float().unsqueeze(dim=1)
    valid_mask_s0 = (outputs["output_plane"]['tri_id_map_stage0'] >= 0).float().unsqueeze(dim=1)

    # 1.1 原版 PatchmatchNet 纯像素深度主损失 (6项标准像素级深度损失，完全等价于官方源工程)
    loss_depth = patchmatchnet_loss(
        depth_patchmatch=depth_patchmatch,
        refined_depth=depth_est,
        depth_gt=depth_gt,
        mask=mask
    )

    # 1.2 新增解耦的 Stage 1 平面级深度异方差加权损失
    loss_depth_plane = compute_plane_depth_loss(
        depth_plane=depth_patchmatch['stage_1'][-1],
        depth_gt=depth_gt['stage_1'],
        mask=mask['stage_1'],
        W_plane_pixel=outputs["output_plane"].get("W_plane_pixel", None),
        gamma=1.5
    )

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

    # DNC损失
    # loss_dnc_s1 = compute_gated_dnc_loss(
    #     Z_pixel=depth_patchmatch['stage_1'][-1],  # 🚀 直接复用物理防爆渲染器出的 1通道 密集真实深度
    #     N_pixel=outputs["output_plane"]['normal_pro_pure'],  # 🚀 采用我们单独剥离出来的未被可视化污染的 3通道 密集真法向
    #     W_plane_pixel=outputs["output_plane"]['W_plane_pixel'],  # 高隔离度空间软路由拦截闸
    #     tri_id_map=outputs["output_plane"]['tri_id_map'],
    #     intrinsics=ref_intrinsics,
    #     valid_mask=valid_mask_s1  # 强力剔除全黑虚空背景
    # )
    loss_dnc_s1=0.0

    # normal_loss = compute_normal_cosine_loss(final_planes=outputs["output_plane"]["final_plane"],
    #                                          tri_id_map=outputs["output_plane"]['tri_id_map'],
    #                                          gt_normals_math_s0=gt_normals_math,
    #                                          depth_stage_1=depth_gt['stage_1'])
    normal_loss = 0.0

    # 计算 Cost Margin Loss（带权重门控判断：若 max_lambda_cost <= 0 则彻底跳过计算）
    if max_lambda_cost > 0.0:
        cost_margin_loss = compute_pixel_cost_margin_loss(
            no_prop_depth=outputs["output_plane"]['depth_no_pro'].detach(),  # SVD 的初始深度
            gt_depth=depth_gt['stage_1'],  # GT 深度
            pixel_costs=outputs["output_plane"]['pixel_costs'],  # 畅通回传给特征网的代价
            tri_id_map=outputs["output_plane"]['tri_id_map']
        )
        cost_margin_loss = cost_margin_loss * max_lambda_cost
    else:
        cost_margin_loss = torch.tensor(0.0, device=device)

    weight_alpha = 0.15
    loss_alpha_sup = loss_alpha_raw * weight_alpha + loss_sparsity_raw
    continuity_loss = outputs["continuity_loss"] * max_lambda_c
    smoothness_loss = outputs["smoothness_loss"] * max_lambda_s
    normal_loss = normal_loss * max_lambda_n
    # DNC 损失
    loss_dnc_s1 = loss_dnc_s1 * max_lambda_dnc_s1
    loss = loss_depth + loss_depth_plane + loss_alpha_sup + continuity_loss + smoothness_loss + normal_loss + cost_margin_loss + loss_dnc_s1

    # ====================================================
    # 4. 指标统计与可视化记录
    # ====================================================
    scalar_outputs = {
        "loss": loss,
        "loss_depth": loss_depth,
        "loss_depth_plane": loss_depth_plane,
        "loss_depth_total": loss_depth + loss_depth_plane,
        "loss_alpha_sup": loss_alpha_sup,
        "continuity_loss": continuity_loss,
        "smoothness_loss": smoothness_loss,
        # "normal_loss": normal_loss,
        "cost_margin_loss": cost_margin_loss,
        # "loss_dnc_s1": loss_dnc_s1
    }

    image_outputs = {}

    pixel_depth_s1 = depth_patchmatch['stage_1'][-1]
    planar_mae_val, pixel_mae_val, _ , planar_err_masked , _ = compute_stage1_flat_region_mae_tensors(
        outputs, pixel_depth_s1, depth_gt['stage_1'], device
    )

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

        mask_s1_b0 = mask['stage_1'][0, 0] if 'stage_1' in mask else None
        w_plane_views = build_w_plane_tensorboard_views(outputs["output_plane"], mask_s1_batch0=mask_s1_b0)
        w_plane_tb_tensor = w_plane_views.get('W_plane_传播后')

        vis_nomal_final = get_visual_normal(outputs["output_plane"]['final_normal'], valid_mask_s0)

        vis_depth_final = get_visual_depth(depth_est['stage_0'], valid_mask_s0)

        # 👑 构建 Stage 1 物理混合融合深度 (Z_fused) 用于测试集可视化展示
        if "W_plane_pixel" in outputs["output_plane"]:
            w_pixel_s1 = outputs["output_plane"]["W_plane_pixel"]
            mask_planar_s1 = (w_pixel_s1 >= 0.80)
            depth_s1_fused = torch.where(mask_planar_s1, depth_patchmatch['stage_1'][-1], depth_patchmatch['stage_1'][0])
        else:
            depth_s1_fused = depth_patchmatch['stage_1'][-1]

        image_outputs = {
            "stage0最终深度预测值": vis_depth_final,
            "stage1深度真值": depth_gt['stage_1'],
            # "patchmatch预测的stage2上采样经过恢复的深度值": outputs["output_plane"]['depth_stage1_pixels'],
            "最终预测的法向量": vis_nomal_final,
            "ref_img": sample["imgs"]['stage_0'][:, 0],
            "根据深度真值生成的法向量": gt_normals_vis,
            "经过平面传播生成的平面法向量": outputs["output_plane"]['normal_pro'],
            "stage1融合深度预测值": depth_s1_fused,
            "没有经过平面传播，刚拟合完初始平面深度值": outputs["output_plane"]['depth_no_pro'],
            "没有经过平面传播，刚拟合完初始平面法向量": outputs["output_plane"]['normal_no_pro'],
            "ref_img_edge_alpha_pre": image_outputs_pre["ref_img_edge_alpha"],
            "ref_img_edge_alpha_gt": image_outputs_gt["ref_img_edge_alpha"],
            # "物理平面置信度_W_plane_传播后": w_plane_tb_tensor,
            "stage1_planar_error_map": planar_err_masked,
            "stage0预测截断平面区域": outputs["output_plane"].get("is_planar_s0", torch.zeros_like(depth_patchmatch['stage_1'][-1])).float()
        }
        if 'cross_check_diagnosis_rgb' in outputs.get("output_plane", {}):
            image_outputs["CrossCheck_三态诊断图(绿保留_红降分)"] = outputs["output_plane"]['cross_check_diagnosis_rgb']
        image_outputs.update(w_plane_views)

    if 'W_plane_pixel' in outputs.get("output_plane", {}) and 'W_plane_pixel_init' in outputs.get("output_plane", {}):
        w_a = outputs["output_plane"]['W_plane_pixel']
        w_b = outputs["output_plane"]['W_plane_pixel_init']
        tri_ok = outputs["output_plane"]['tri_id_map'] >= 0
        if tri_ok.dim() == 3:
            tri_ok = tri_ok.unsqueeze(1)
        conf_valid = tri_ok & (w_a > 0) & (w_b > 0)
        if conf_valid.any():
            scalar_outputs["w_plane_mean_abs_update"] = (w_a - w_b).abs()[conf_valid].mean()
            scalar_outputs["w_plane_mean_before"] = w_b[conf_valid].mean()
            scalar_outputs["w_plane_mean_after"] = w_a[conf_valid].mean()
        else:
            scalar_outputs["w_plane_mean_abs_update"] = 0.0
            scalar_outputs["w_plane_mean_before"] = 0.0
            scalar_outputs["w_plane_mean_after"] = 0.0

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
    
    if "is_planar_s0" in outputs["output_plane"]:
        is_planar_s0 = outputs["output_plane"]["is_planar_s0"]
        mask_s0 = mask['stage_0'] > 0.5
        scalar_outputs["stage0_planar_mae"] = AbsDepthError_metrics(depth_est['stage_0'], depth_gt['stage_0'], mask_s0 & is_planar_s0)
        scalar_outputs["stage0_curved_mae"] = AbsDepthError_metrics(depth_est['stage_0'], depth_gt['stage_0'], mask_s0 & (~is_planar_s0))
        
        # 👑 新增诊断指标：分析 Stage 1 原始像素深度 (depth_patchmatch['stage_1'][0]) 与传播深度 ([-1]) 在两区的表现
        depth_s1_pixel_up = F.interpolate(depth_patchmatch['stage_1'][0], size=depth_gt['stage_0'].shape[2:], mode='bilinear', align_corners=False)
        depth_s1_prop_up = F.interpolate(depth_patchmatch['stage_1'][-1], size=depth_gt['stage_0'].shape[2:], mode='bilinear', align_corners=False)
        
        scalar_outputs["stage1_pixel_planar_mae"] = AbsDepthError_metrics(depth_s1_pixel_up, depth_gt['stage_0'], mask_s0 & is_planar_s0)
        scalar_outputs["stage1_pixel_curved_mae"] = AbsDepthError_metrics(depth_s1_pixel_up, depth_gt['stage_0'], mask_s0 & (~is_planar_s0))
        scalar_outputs["stage1_prop_planar_mae"] = AbsDepthError_metrics(depth_s1_prop_up, depth_gt['stage_0'], mask_s0 & is_planar_s0)
        scalar_outputs["stage1_prop_curved_mae"] = AbsDepthError_metrics(depth_s1_prop_up, depth_gt['stage_0'], mask_s0 & (~is_planar_s0))

    scalar_outputs["abs_depth_error_patchmatch_stage_3"] = AbsDepthError_metrics(depth_patchmatch['stage_3'][-1],
                                                                                 depth_gt['stage_3'],
                                                                                 mask['stage_3'] > 0.5)
    scalar_outputs["abs_depth_error_patchmatch_stage_2"] = AbsDepthError_metrics(depth_patchmatch['stage_2'][-1],
                                                                                 depth_gt['stage_2'],
                                                                                 mask['stage_2'] > 0.5)
    scalar_outputs["abs_depth_error_patchmatch_stage_1"] = AbsDepthError_metrics(depth_patchmatch['stage_1'][-1],
                                                                                 depth_gt['stage_1'],
                                                                                 mask['stage_1'] > 0.5)
    scalar_outputs["abs_depth_error_patchmatch_stage_1_pixel_raw"] = AbsDepthError_metrics(depth_patchmatch['stage_1'][0],
                                                                                            depth_gt['stage_1'],
                                                                                            mask['stage_1'] > 0.5)
    # 👑 严格对齐 eval_whu_big.py:L2865 的 Stage 1 像素/平面全图物理混合融合深度 (Z_fused)
    # 统计全图有效真值区域 (depth_gt > 0)，三角剖分外的区域天然回退为纯像素深度 depth_patchmatch['stage_1'][0]
    if "W_plane_pixel" in outputs["output_plane"]:
        w_pixel_s1 = outputs["output_plane"]["W_plane_pixel"]
        mask_planar_s1 = (w_pixel_s1 >= 0.80)
        # 平面区走 Stage 1 最终平面深度 depth_patchmatch['stage_1'][-1]，非平面区/三角网外回退至纯像素深度 depth_patchmatch['stage_1'][0]
        depth_s1_fused = torch.where(mask_planar_s1, depth_patchmatch['stage_1'][-1], depth_patchmatch['stage_1'][0])
        valid_gt_mask_s1 = mask['stage_1'] > 0.5
        scalar_outputs["abs_depth_error_patchmatch_stage_1_fused"] = AbsDepthError_metrics(
            depth_s1_fused, depth_gt['stage_1'], valid_gt_mask_s1
        )
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
    scalar_outputs["stage1_flat_region_mae"] = planar_mae_val
    scalar_outputs["stage1_flat_region_pixel_mae"] = pixel_mae_val

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


if __name__ == '__main__':
    if args.mode == "train":
        train()
        # 如果用户请求，在训练结束后启动大图评估（使用最新保存的 checkpoint）
        if args.run_big_eval:
            # 训练结束后先释放训练过程中占用的 GPU 内存，避免子进程启动时 OOM
            print("Releasing training GPU memory before big eval...")
            try:
                del model, optimizer, train_dataset, test_dataset, TrainImgLoader, TestImgLoader
            except NameError:
                pass
            gc.collect()
            torch.cuda.empty_cache()

            # 查找最新的 ckpt
            if os.path.isdir(args.logdir):
                ckpts = [fn for fn in os.listdir(args.logdir) if fn.endswith('.ckpt')]
                parsed_ckpts = []
                for fn in ckpts:
                    try:
                        idx = int(fn.split('_')[-1].split('.')[0])
                    except ValueError:
                        continue
                    parsed_ckpts.append((idx, fn))

                # 优先选择当前训练最大 epoch 之内的 checkpoint，避免旧 run 的大序号文件被误选
                valid_ckpts = [(idx, fn) for idx, fn in parsed_ckpts if idx <= args.epochs - 1]
                if valid_ckpts:
                    valid_ckpts.sort(key=lambda x: x[0])
                    last_ckpt = os.path.join(args.logdir, valid_ckpts[-1][1])
                elif parsed_ckpts:
                    parsed_ckpts.sort(key=lambda x: x[0])
                    last_ckpt = os.path.join(args.logdir, parsed_ckpts[-1][1])
                else:
                    last_ckpt = args.loadckpt
            else:
                last_ckpt = args.loadckpt

            if last_ckpt is None:
                print('No checkpoint found to run big eval. Skipping.')
            else:
                eval_script = os.path.join(os.path.dirname(__file__), 'eval_whu_big.py')
                cmd = [sys.executable, eval_script,
                       '--dataset', args.big_eval_dataset,
                       '--testpath', args.big_eval_testpath,
                       '--testlist', args.big_eval_testlist,
                       '--loadckpt', last_ckpt,
                       '--outdir', args.logdir,
                       '--n_views', str(5)]

                # 传递 patchmatch 与 propagation 相关参数，保持评估一致性
                # 将每个 list 参数作为单次 flag 传入后跟多个值（符合 argparse with nargs='+')
                if isinstance(args.patchmatch_iteration, (list, tuple)):
                    cmd += ['--patchmatch_iteration'] + [str(v) for v in args.patchmatch_iteration]
                else:
                    cmd += ['--patchmatch_iteration', str(args.patchmatch_iteration)]

                if isinstance(args.patchmatch_num_sample, (list, tuple)):
                    cmd += ['--patchmatch_num_sample'] + [str(v) for v in args.patchmatch_num_sample]
                else:
                    cmd += ['--patchmatch_num_sample', str(args.patchmatch_num_sample)]

                if isinstance(args.patchmatch_interval_scale, (list, tuple)):
                    cmd += ['--patchmatch_interval_scale'] + [str(v) for v in args.patchmatch_interval_scale]
                else:
                    cmd += ['--patchmatch_interval_scale', str(args.patchmatch_interval_scale)]

                if isinstance(args.patchmatch_range, (list, tuple)):
                    cmd += ['--patchmatch_range'] + [str(v) for v in args.patchmatch_range]
                else:
                    cmd += ['--patchmatch_range', str(args.patchmatch_range)]

                if isinstance(args.propagate_neighbors, (list, tuple)):
                    cmd += ['--propagate_neighbors'] + [str(v) for v in args.propagate_neighbors]
                else:
                    cmd += ['--propagate_neighbors', str(args.propagate_neighbors)]

                if isinstance(args.evaluate_neighbors, (list, tuple)):
                    cmd += ['--evaluate_neighbors'] + [str(v) for v in args.evaluate_neighbors]
                else:
                    cmd += ['--evaluate_neighbors', str(args.evaluate_neighbors)]

                print('Running big-eval command:', ' '.join(cmd))
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print('Big eval failed:', e)
    elif args.mode == "val":
        test()
