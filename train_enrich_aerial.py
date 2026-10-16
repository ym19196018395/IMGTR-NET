import argparse
import os
import math
import time
import copy
import gc
import sys
import datetime
import importlib
from contextlib import contextmanager

# 控制要暴露给进程的 GPU id
_gpu_choice = os.environ.get('GPU_ID', os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
os.environ['CUDA_VISIBLE_DEVICES'] = str(_gpu_choice)

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from models.edge_head import EdgeLabelGenerator
from models.PlanePatchMatch import *
from models.sum_loss import *
from datasets import find_dataset_def
from models import *
from utils import *

# 运行设备与加速设置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def auto_find_enrich_datapath():
    """自动探查服务器与本地常见 ENRICH-Aerial 数据集路径"""
    candidates = [
        "/home/myao/ENRICH-Aerial_Cropped_0.6x_Overlap50/",
        "/home/myao/ENRICH-Aerial_Data/",
        "/home/myao/Crop_Test_Dataset/",
        "./datasets/ENRICH-Aerial_Cropped_0.6x_Overlap50/",
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return candidates[0]


parser = argparse.ArgumentParser(
    description='ENRICH-Aerial 专属训练系统 (Mode B 动态无状态验证协议 & S0 主选模)'
)
parser.add_argument('--mode', default='train', choices=['train', 'val'], help='运行模式: train 或 val')
parser.add_argument('--model', default='PatchmatchNet', help='模型架构类型')
parser.add_argument('--dataset', default='enrich_aerial', help='数据集定义名称')
parser.add_argument('--trainpath', default=auto_find_enrich_datapath(), help='训练集图像根路径')
parser.add_argument('--valpath', default=auto_find_enrich_datapath(), help='验证集图像根路径')
parser.add_argument('--trainlist', default='lists/enrich_aerial/train.txt', help='训练列表文件')
parser.add_argument('--vallist', default='lists/enrich_aerial/test.txt', help='验证列表文件')
parser.add_argument('--nviews', type=int, default=5, help='输入视角数 (1 参考 + 4 源视角)')

parser.add_argument('--epochs', type=int, default=34, help='总训练轮数')
parser.add_argument('--add_epochs', type=int, default=0, help='在当前 Checkpoint 基础上额外追加训练的轮数 (如续训追加 6 轮)')
parser.add_argument('--lr', type=float, default=0.001, help='学习率 (若续训时指定并开启 --force_lr，将覆盖 Checkpoint 中的旧 LR)')
parser.add_argument('--force_lr', action='store_true', default=False, help='续训时强制使用命令行指定的 --lr 覆盖 Checkpoint 中的优化器 LR')
parser.add_argument('--reset_epoch', action='store_true', default=False, help='载入 Checkpoint 后将起始轮数重置为 0 (迁移学习/全新微调)')
parser.add_argument('--reset_optimizer', action='store_true', default=False, help='重置优化器状态 (清空历史动量缓存，以全新状态微调)')
parser.add_argument('--lrepochs', type=str, default="22,26,30:2", help='学习率阶梯衰减轮次与倍率 (如 22,26,30:2)')
parser.add_argument('--wd', type=float, default=0.0, help='权重衰减')

parser.add_argument('--batch_size', type=int, default=2, help='训练批次大小 (推荐 1 或 2)')
parser.add_argument('--num_workers', type=int, default=2, help='DataLoader 线程数')
parser.add_argument('--loadckpt', default=None, help='载入指定 Checkpoint 权重路径')
parser.add_argument('--logdir', default='./checkpoints/tensorboard_enrich_ours', help='日志与权重保存目录')
parser.add_argument('--resume', default=False, action='store_true', help='自动断点恢复训练 (支持与 --loadckpt 联用)')

parser.add_argument('--summary_freq', type=int, default=2, help='标量日志记录步频')
parser.add_argument('--save_freq', type=int, default=1, help='检查点保存轮频')
parser.add_argument('--eval_freq', type=int, default=1, help='验证集评估轮频 (默认每轮评估)')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='随机种子')

# PatchMatch 超参数
parser.add_argument('--patchmatch_iteration', nargs='+', type=int, default=[1, 2, 2],
                    help='各 Stage PatchMatch 迭代次数')
parser.add_argument('--patchmatch_num_sample', nargs='+', type=int, default=[8, 8, 16],
                    help='各 Stage 候选深度采样数量')
parser.add_argument('--patchmatch_interval_scale', nargs='+', type=float, default=[0.005, 0.0125, 0.025],
                    help='各 Stage 逆深度扰动采样间隔尺度')
parser.add_argument('--patchmatch_range', nargs='+', type=int, default=[6, 4, 2],
                    help='各 Stage 传播范围')
parser.add_argument('--propagate_neighbors', nargs='+', type=int, default=[0, 8, 16],
                    help='各 Stage 自适应传播邻居数')
parser.add_argument('--evaluate_neighbors', nargs='+', type=int, default=[9, 9, 9],
                    help='各 Stage 代价聚合邻居数')

# 👑 监督解耦与梯度裁剪
parser.add_argument('--normal_gt_mode', default='soft', choices=['hard', 'soft'],
                    help='法向量真值监督模式: hard 或 soft(推荐)')
parser.add_argument('--conf_gt_mode', default='hard', choices=['hard', 'soft'],
                    help='平面置信度真值监督模式: hard 或 soft')
parser.add_argument('--clip_mode', default='independent', choices=['global', 'independent'],
                    help='梯度裁剪模式: global 或 independent(推荐主干/平面/精修独立裁剪)')
parser.add_argument('--clip_max_norm_bb', type=float, default=2.0, help='主干特征网梯度裁剪阈值')
parser.add_argument('--clip_max_norm_plane', type=float, default=1.0, help='平面分支梯度裁剪阈值')
parser.add_argument('--clip_max_norm_refine', type=float, default=1.0, help='Refinement梯度裁剪阈值')
parser.add_argument('--clip_max_norm_total', type=float, default=2.0, help='全局梯度裁剪阈值')

# 👑 动态 Cross-Check 几何容限透传参数
parser.add_argument('--pre_base_depth_thresh', type=float, default=0.20,
                    help='Pre-GNN 物理铁闸基础深度容限 (m, ENRICH推荐 0.16~0.20)')
parser.add_argument('--pre_pixel_dist_thresh', type=float, default=1.2,
                    help='Pre-GNN 物理铁闸 2D 像素距离容限 (px)')
parser.add_argument('--post_base_depth_thresh', type=float, default=0.16,
                    help='Post-GNN Cross-Check 基础深度容限 (m, ENRICH推荐 0.14~0.18)')
parser.add_argument('--post_pixel_dist_thresh', type=float, default=1.2,
                    help='Post-GNN Cross-Check 2D 像素距离容限 (px)')

args = parser.parse_args()
if args.resume:
    assert args.mode == "train", "resume 仅支持在 train 模式下启用"
if args.valpath is None:
    args.valpath = args.trainpath

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

if args.mode == "train":
    os.makedirs(args.logdir, exist_ok=True)
    current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    print("current time", current_time_str)
    print("creating new summary file")
    logger = SummaryWriter(args.logdir)

print("argv:", sys.argv[1:])
print_args(args)

# 构建数据集与 DataLoader
dataset_module = importlib.import_module(f'datasets.{args.dataset}')
MVSDataset = find_dataset_def(args.dataset)
dataset_collate_fn = getattr(dataset_module, 'collate_keep_list', None)

train_dataset = MVSDataset(args.trainpath, args.trainlist, "train", args.nviews, robust_train=True)
test_dataset = MVSDataset(args.valpath, args.vallist, "test", args.nviews, robust_train=False)

TrainImgLoader = DataLoader(
    train_dataset, args.batch_size, shuffle=True, collate_fn=dataset_collate_fn,
    num_workers=args.num_workers, pin_memory=True,
    persistent_workers=(args.num_workers > 0), drop_last=True
)
TestImgLoader = DataLoader(
    test_dataset, args.batch_size, shuffle=False, collate_fn=dataset_collate_fn,
    num_workers=max(2, args.num_workers // 2) if args.num_workers > 0 else 0,
    pin_memory=True, persistent_workers=(args.num_workers > 0), drop_last=False
)

print(f"[Dataset] ENRICH 训练样本数: {len(train_dataset)} | 验证样本数: {len(test_dataset)}")

# 实例化主干模型
model = PatchmatchNet(
    patchmatch_interval_scale=args.patchmatch_interval_scale,
    propagation_range=args.patchmatch_range,
    patchmatch_iteration=args.patchmatch_iteration,
    patchmatch_num_sample=args.patchmatch_num_sample,
    propagate_neighbors=args.propagate_neighbors,
    evaluate_neighbors=args.evaluate_neighbors,
    pre_base_depth_thresh=args.pre_base_depth_thresh,
    pre_pixel_dist_thresh=args.pre_pixel_dist_thresh,
    post_base_depth_thresh=args.post_base_depth_thresh,
    post_pixel_dist_thresh=args.post_pixel_dist_thresh
).to(device)

model_loss = patchmatchnet_loss
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)


def load_checkpoint_with_channel_adaptation(model, state_dict_model):
    """自适应通道平滑加载权重"""
    model_state = model.state_dict()
    key = 'upsample_net.conv0.conv.weight'
    if key in state_dict_model and key in model_state:
        ckpt_w = state_dict_model[key]
        tgt_w = model_state[key]
        if ckpt_w.shape[1] == 4 and tgt_w.shape[1] == 3:
            print(f"[Adaptive Loading] Adapting {key} from [8, 4, 3, 3] -> [8, 3, 3, 3]...")
            state_dict_model[key] = ckpt_w[:, :3, :, :]
        elif ckpt_w.shape[1] == 3 and tgt_w.shape[1] == 4:
            print(f"[Adaptive Loading] Adapting {key} from [8, 3, 3, 3] -> [8, 4, 3, 3]...")
            adapted_w = torch.zeros_like(tgt_w)
            adapted_w[:, :3, :, :] = ckpt_w
            state_dict_model[key] = adapted_w

    try:
        model.load_state_dict(state_dict_model)
    except RuntimeError as e:
        print(f"Warning: Exact load failed ({e}), loading with strict=False...")
        model.load_state_dict(state_dict_model, strict=False)


start_epoch = 0
best_val_s0_mae = float('inf')

# 统一续训与权重载入处理
target_ckpt = None
is_resuming = False

if (args.mode == "train" and args.resume) or (args.mode == "test" and not args.loadckpt):
    is_resuming = (args.mode == "train")
    if args.loadckpt:
        target_ckpt = args.loadckpt
    else:
        saved_models = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt") and fn.startswith("model_")]
        if len(saved_models) > 0:
            saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            target_ckpt = os.path.join(args.logdir, saved_models[-1])
elif args.loadckpt:
    target_ckpt = args.loadckpt
    # 若在训练模式下提供了 loadckpt 且未声明 reset_epoch，默认支持继承状态续训
    is_resuming = (args.mode == "train" and not args.reset_epoch)

if target_ckpt and os.path.exists(target_ckpt):
    print(f"\n[Checkpoint] 正在载入权重文件: {target_ckpt}")
    state_dict = torch.load(target_ckpt, map_location=device)
    model_weights = state_dict.get('model', state_dict)
    load_checkpoint_with_channel_adaptation(model, model_weights)

    if is_resuming:
        if 'epoch' in state_dict and not args.reset_epoch:
            start_epoch = state_dict['epoch'] + 1
            print(f"[Resume] 从 Checkpoint 继承轮数: 上次完成 Epoch {state_dict['epoch']} -> 本次起始 Epoch {start_epoch}")
        elif args.reset_epoch:
            print("[Resume] 开启 --reset_epoch: 轮数重置为 Epoch 0 起始！")

        if 'best_val_s0_mae' in state_dict:
            best_val_s0_mae = float(state_dict['best_val_s0_mae'])
            print(f"[Resume] 继承历史最佳验证指标 best_val_s0_mae: {best_val_s0_mae:.4f} m")

        if not args.reset_optimizer and 'optimizer' in state_dict:
            try:
                optimizer.load_state_dict(state_dict['optimizer'])
                print("[Resume] 成功载入优化器动量与参数状态！")
            except Exception as e:
                print(f"[Resume Warning] 优化器状态加载失败 ({e})，重新以全新状态初始化。")
        else:
            print("[Resume] 显式重置优化器状态 (清空历史动量缓存，以全新状态微调)。")

        # 若指定了 force_lr 或显式传参变更了 lr，强制覆盖优化器中的学习率
        if args.force_lr or (args.lr != 0.001):
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
            print(f"[Resume] 强制更新优化器学习率为命令行参数: LR = {args.lr:.6f}")
    else:
        print(f"[Model] 权重加载完成 (非续训模式，从头开始)。")
else:
    if target_ckpt:
        print(f"[Warning] 目标权重文件不存在: {target_ckpt}，将从头开始训练。")

print(f"Start at epoch {start_epoch} | Number of model parameters: {sum([p.data.nelement() for p in model.parameters()])}")


@contextmanager
def mode_b_validation_context(model):
    """
    无状态 Mode B 验证环境上下文管理器:
    1. 切换整体模型为 eval() (权重冻结，Dropout 失活)；
    2. 将全部 BatchNorm 层切换为 train() (计算单批次即时均值/方差)；
    3. 严格将 track_running_stats 设为 False (防止验证集样本统计污染全局 running 统计账本)；
    4. 退出上下文时，100% 幂等恢复原始的 model.training 状态以及每个 BN 层的 track_running_stats。
    """
    was_training = model.training
    bn_stats = {}
    model.eval()
    for name, m in model.named_modules():
        if isinstance(m, _BN_TYPES):
            bn_stats[name] = (m.training, m.track_running_stats)
            m.train()
            m.track_running_stats = False
    try:
        yield
    finally:
        for name, m in model.named_modules():
            if isinstance(m, _BN_TYPES):
                orig_train, orig_track = bn_stats.get(name, (True, True))
                m.train(orig_train)
                m.track_running_stats = orig_track
        model.train(was_training)


def tensor_to_pseudocolor(tensor_map, mask=None, colormap=cv2.COLORMAP_JET, invert=True, fixed_range=None):
    """将单通道深度/误差张量渲染为 TensorBoard 伪彩色图"""
    if tensor_map.dim() == 4:
        tensor_map = tensor_map[0, 0]
    elif tensor_map.dim() == 3:
        tensor_map = tensor_map[0]

    arr = tensor_map.detach().cpu().numpy()
    if mask is not None:
        if mask.dim() == 4:
            m_arr = mask[0, 0].detach().cpu().numpy() > 0.5
        elif mask.dim() == 3:
            m_arr = mask[0].detach().cpu().numpy() > 0.5
        else:
            m_arr = mask.detach().cpu().numpy() > 0.5
    else:
        m_arr = arr > 0

    valid_vals = arr[m_arr]
    if len(valid_vals) == 0:
        return torch.zeros((1, 3, arr.shape[0], arr.shape[1]), dtype=torch.float32)

    if fixed_range is not None:
        vmin, vmax = fixed_range
    else:
        vmin = np.percentile(valid_vals, 1)
        vmax = np.percentile(valid_vals, 99)
    if vmax == vmin:
        vmax = vmin + 1e-6

    norm_arr = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
    if invert:
        norm_arr = 1.0 - norm_arr

    norm_uint8 = (norm_arr * 255).astype(np.uint8)
    color_img = cv2.applyColorMap(norm_uint8, colormap)
    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    color_img[~m_arr] = 0

    color_tensor = torch.from_numpy(color_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return color_tensor


def cosine_temperature_schedule(progress, t_min=0.55, t_max=1.0):
    """余弦退火平滑控温"""
    p = max(0.0, min(1.0, progress))
    return t_min + 0.5 * (t_max - t_min) * (1.0 + math.cos(math.pi * p))


def get_smooth_weight_with_decay(progress, start_prog, end_prog, decay_prog, max_weight, end_ratio=0.1):
    """物理课程学习光滑权重调度"""
    if progress < start_prog:
        return 0.0
    elif progress < end_prog:
        ratio = (progress - start_prog) / (end_prog - start_prog)
        return max_weight * (1.0 - math.cos(ratio * math.pi)) / 2.0
    elif progress < decay_prog:
        return max_weight
    else:
        ratio = (progress - decay_prog) / (1.0 - decay_prog + 1e-7)
        decay_factor = 1.0 - (1.0 - end_ratio) * (1.0 - math.cos(ratio * math.pi)) / 2.0
        return max_weight * decay_factor


def build_w_plane_tensorboard_views(output_plane, mask_s1_batch0=None):
    """
    构建 W_plane 置信度 TensorBoard 可视化通道 (红蓝热力图：蓝色高置信度、红色低置信度)
    """
    views = {}
    if "W_plane_pixel" in output_plane:
        w_pixel = output_plane["W_plane_pixel"]
        if mask_s1_batch0 is not None:
            m = mask_s1_batch0.unsqueeze(0).unsqueeze(0)
        else:
            m = torch.ones_like(w_pixel) > 0
        views['W_plane_传播后'] = tensor_to_pseudocolor(w_pixel, mask=m, colormap=cv2.COLORMAP_JET, invert=True, fixed_range=(0.0, 1.0))
    if "W_plane_pixel_init" in output_plane:
        w_init = output_plane["W_plane_pixel_init"]
        if mask_s1_batch0 is not None:
            m = mask_s1_batch0.unsqueeze(0).unsqueeze(0)
        else:
            m = torch.ones_like(w_init) > 0
        views['W_plane_传播前初始'] = tensor_to_pseudocolor(w_init, mask=m, colormap=cv2.COLORMAP_JET, invert=True, fixed_range=(0.0, 1.0))
    if "W_plane_gt_pixel" in output_plane:
        w_gt = output_plane["W_plane_gt_pixel"]
        if mask_s1_batch0 is not None:
            m = mask_s1_batch0.unsqueeze(0).unsqueeze(0)
        else:
            m = torch.ones_like(w_gt) > 0
        views['W_plane_真值'] = tensor_to_pseudocolor(w_gt, mask=m, colormap=cv2.COLORMAP_JET, invert=True, fixed_range=(0.0, 1.0))
    return views


def compute_stage1_flat_region_mae_tensors(outputs, pixel_depth_s1, depth_gt_s1, device):
    """计算 Stage 1 平面区域与非平面区域的深度绝对误差"""
    output_plane_dict = outputs.get("output_plane", {})
    if "W_plane_pixel" in output_plane_dict:
        w_pixel = output_plane_dict["W_plane_pixel"]
    else:
        w_pixel = torch.zeros_like(depth_gt_s1)

    mask_valid = (depth_gt_s1 > 0)
    mask_planar = mask_valid & (w_pixel >= 0.80)
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

    max_error_cutoff = 0.5
    error_amplified = torch.clamp(err_map_dense / max_error_cutoff, 0.0, 1.0)
    planar_err_vis = torch.where(mask_planar, error_amplified, torch.zeros_like(error_amplified))

    B, _, H, W = depth_gt_s1.shape
    planar_color_vis = torch.zeros(B, 3, H, W, device=device)
    planar_color_vis[:, 2:3, :, :] = mask_planar.float()
    mask_soft_curved = mask_valid & (~mask_planar)
    planar_color_vis[:, 0:1, :, :] = mask_soft_curved.float()

    return planar_mae_val, pixel_mae_val, planar_err_vis, planar_color_vis, mask_planar


def train_sample(sample, do_summary_image=False, global_step=0, total_steps=0):
    """单批次前向训练与反向梯度优化"""
    model.train()
    optimizer.zero_grad()

    vertexs_batch = sample['vertexs']
    lines_batch = sample['lines']
    triangles_batch = []
    for tri_list in sample['triangles']:
        tri_processed = []
        for t in tri_list:
            v_ids = t['vertex_ids']
            l_ids = t['line_ids']
            pts = torch.from_numpy(t['valid_points']).to(device)
            tri_processed.append((v_ids, l_ids, pts))
        triangles_batch.append(tri_processed)

    skip = ["vertexs", "lines", "triangles", "tri_conf_cleaned", "tri_normal_cleaned", "is_gt_planar", "tri_err95", "tri_weight_normal", "tri_weight_conf"]
    sample_cuda = tocuda(sample, device=device, skip_keys=skip)

    if 'tri_conf_cleaned' in sample:
        sample_cuda['tri_conf_cleaned'] = [torch.from_numpy(v).to(device).float() for v in sample['tri_conf_cleaned']]
    if 'tri_normal_cleaned' in sample:
        sample_cuda['tri_normal_cleaned'] = [torch.from_numpy(v).to(device).float() for v in sample['tri_normal_cleaned']]
    if 'is_gt_planar' in sample:
        sample_cuda['is_gt_planar'] = [torch.from_numpy(v).to(device).bool() for v in sample['is_gt_planar']]
    if 'tri_weight_normal' in sample:
        sample_cuda['tri_weight_normal'] = [torch.from_numpy(v).to(device).float() for v in sample['tri_weight_normal']]
    if 'tri_weight_conf' in sample:
        sample_cuda['tri_weight_conf'] = [torch.from_numpy(v).to(device).float() for v in sample['tri_weight_conf']]

    depth_gt = sample_cuda["depth"]
    mask = sample_cuda["mask"]

    progress = global_step / max(1, total_steps)
    max_lambda_c = 0.0
    max_lambda_s = 0.0
    max_lambda_n_tri = 1.0
    max_lambda_n_pix = 0.0
    max_lambda_cost = 0.0
    weight_alpha = 0.05
    weight_sparsity = 0.01
    max_lambda_dnc_s1 = 0.05 * 50
    max_lambda_plane = 0.10

    lambda_n_tri = get_smooth_weight_with_decay(progress, 0.15, 0.40, 0.80, max_lambda_n_tri, end_ratio=0.75)
    lambda_c = get_smooth_weight_with_decay(progress, 0.30, 0.55, 0.85, max_lambda_c, end_ratio=0.20)
    lambda_s = get_smooth_weight_with_decay(progress, 0.40, 0.65, 0.80, max_lambda_s, end_ratio=0.40)
    lambda_cost = get_smooth_weight_with_decay(progress, 0.0, 0.1, 0.6, max_lambda_cost, end_ratio=0.5)
    lambda_dnc_s1 = get_smooth_weight_with_decay(progress, 0.0, 0.2, 1.0, max_lambda_dnc_s1, end_ratio=1.0)
    lambda_plane_s1 = get_smooth_weight_with_decay(progress, 0.05, 0.15, 1.0, max_lambda_plane, end_ratio=1.0)

    outputs = model(
        sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["intrinsics_mats"],
        sample_cuda["depth_min"], sample_cuda["depth_max"],
        vertexs_batch, lines_batch, triangles_batch, depth_gt['stage_1'],
        lambda_c, lambda_s, progress=progress, compute_edge_pixels=do_summary_image,
        pre_base_depth_thresh=args.pre_base_depth_thresh,
        pre_pixel_dist_thresh=args.pre_pixel_dist_thresh,
        post_base_depth_thresh=args.post_base_depth_thresh,
        post_pixel_dist_thresh=args.post_pixel_dist_thresh
    )

    depth_est = outputs["refined_depth"]
    depth_patchmatch = outputs["depth_patchmatch"]

    tri_id_map = outputs["output_plane"].get('tri_id_map', None)
    if (tri_id_map is not None
            and "tri_conf_cleaned" in sample_cuda
            and "tri_normal_cleaned" in sample_cuda):
        B, H_s1, W_s1 = tri_id_map.shape
        tri_device = tri_id_map.device
        planar_soft_conf_s1_list = []
        gt_normal_map_s1_list = []
        for b in range(B):
            if args.conf_gt_mode == 'soft' and "tri_weight_conf" in sample_cuda:
                tri_conf_b = sample_cuda['tri_weight_conf'][b].float()
            else:
                tri_conf_b = sample_cuda['is_gt_planar'][b].float() if "is_gt_planar" in sample_cuda else sample_cuda['tri_conf_cleaned'][b]
            tri_normal_b = sample_cuda['tri_normal_cleaned'][b]
            tri_id_map_b = tri_id_map[b]
            dummy_conf = torch.cat([tri_conf_b, torch.tensor([0.0], device=tri_device)], dim=0)
            dummy_normal = torch.cat([tri_normal_b, torch.tensor([[0.0, 0.0, 0.0]], device=tri_device)], dim=0)
            safe_idx = torch.where(tri_id_map_b >= 0, tri_id_map_b, torch.tensor(tri_conf_b.shape[0], device=tri_device).long())
            planar_soft_conf_s1_list.append(dummy_conf[safe_idx])
            gt_normal_map_s1_list.append(dummy_normal[safe_idx])
        outputs["output_plane"]["W_plane_gt_pixel"] = torch.stack(planar_soft_conf_s1_list, dim=0).unsqueeze(1)
        outputs["output_plane"]["gt_normal_map_s1"] = torch.stack(gt_normal_map_s1_list, dim=0)

    # 1. 核心深度损失
    loss_depth = patchmatchnet_loss(
        depth_patchmatch=depth_patchmatch,
        refined_depth=depth_est,
        depth_gt=depth_gt,
        mask=mask
    )

    loss_depth_plane = compute_plane_depth_loss(
        depth_plane=depth_patchmatch['stage_1'][-1],
        depth_gt=depth_gt['stage_1'],
        mask=mask['stage_1'],
        W_plane_pixel=outputs["output_plane"].get("W_plane_pixel", None),
        gamma=1.5
    )

    # 2. 边缘与法向辅助特征
    intrinsics_s1 = torch.unbind(sample_cuda["intrinsics_mats"]['stage_1'].float(), 1)
    ref_intrinsics = intrinsics_s1[0]
    intrinsics_s0 = torch.unbind(sample_cuda["intrinsics_mats"]['stage_0'].float(), 1)
    ref_intrinsics_s0 = intrinsics_s0[0]

    depth_threshold = 0.2
    edge_label_generator = EdgeLabelGenerator(pt2plane_threshold=depth_threshold)
    gt_normals_math, gt_normals_vis = compute_normal_map_torch(depth_gt['stage_0'], ref_intrinsics_s0, mask=None, smooth=True)

    edge_alphas_gt, valid_mask_list = edge_label_generator(
        pred_alphas_list=outputs["edge_alphas"],
        gt_depth_map=depth_gt['stage_0'],
        tri_infos=outputs["tri_infos"],
        intrinsics=ref_intrinsics_s0,
        gt_normal_map=gt_normals_math,
    )

    loss_alpha_raw, loss_sparsity_raw = compute_edge_supervision_loss_new(
        alphas_list=outputs["edge_alphas"],
        gt_stage0_list=edge_alphas_gt,
        valid_mask_list=valid_mask_list
    )

    loss_alpha_sup = loss_alpha_raw * weight_alpha + loss_sparsity_raw * weight_sparsity
    continuity_loss = outputs["continuity_loss"] * lambda_c
    smoothness_loss = outputs["smoothness_loss"] * lambda_s

    # 3. 👑 核心补齐：平面置信度网络硬核监督损失 (loss_plane_s1)
    valid_mask_s1 = (outputs["output_plane"]['tri_id_map'] >= 0).float().unsqueeze(dim=1)
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
        tri_conf_gt=sample_cuda.get("tri_conf_cleaned", None),
        is_gt_planar=sample_cuda.get("is_gt_planar", None),
        tri_weight_conf=sample_cuda.get("tri_weight_conf", None),
        conf_gt_mode=args.conf_gt_mode
    )
    outputs["output_plane"]["W_plane_gt_pixel"] = W_GT_pixel
    loss_plane_s1 = loss_plane_s1 * lambda_plane_s1

    # 4. 👑 核心补齐：平面法向量真值监督损失 (normal_loss_tri)
    normal_loss_tri = torch.tensor(0.0, device=device)
    if ("is_gt_planar" in sample_cuda or "tri_weight_normal" in sample_cuda) and "tri_normal_cleaned" in sample_cuda:
        pixel_counts = outputs["output_plane"]["pixel_counts"]  # [B, N_tri]
        normal_loss_tri = compute_normal_gt_loss(
            n_pred=outputs["output_plane"]["final_plane"][..., :3],
            n_gt=sample_cuda["tri_normal_cleaned"],
            is_gt_planar=sample_cuda.get("is_gt_planar", None),
            pixel_counts=pixel_counts,
            min_pixels=3,
            weights_normal=sample_cuda.get("tri_weight_normal", None),
            normal_gt_mode=args.normal_gt_mode
        )
    normal_loss = normal_loss_tri * lambda_n_tri

    # 5. 总损失闭环
    loss_depth_plane_weighted = loss_depth_plane * 1.0
    loss = (loss_depth + loss_depth_plane_weighted + loss_alpha_sup +
            continuity_loss + smoothness_loss + normal_loss + loss_plane_s1)

    loss.backward()

    # 解耦梯度裁剪
    if args.clip_mode == 'independent':
        backbone_params = list(model.feature.parameters())
        for l in range(model.stages - 1):
            pm_module = getattr(model, f'patchmatch_{l+1}')
            backbone_params.extend(list(pm_module.parameters()))
        torch.nn.utils.clip_grad_norm_(backbone_params, max_norm=args.clip_max_norm_bb)

        plane_params = list(model.plane_patchmatch_agent.parameters())
        if hasattr(model, 'stage1_refine'):
            plane_params.extend(list(model.stage1_refine.parameters()))
        torch.nn.utils.clip_grad_norm_(plane_params, max_norm=args.clip_max_norm_plane)

        refine_params = list(model.upsample_net.parameters())
        if hasattr(model, 'stage0_refiner'):
            refine_params.extend(list(model.stage0_refiner.parameters()))
        torch.nn.utils.clip_grad_norm_(refine_params, max_norm=args.clip_max_norm_refine)
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_max_norm_total)

    optimizer.step()

    scalar_outputs = {
        "loss": loss,
        "loss_depth": loss_depth,
        "loss_depth_plane": loss_depth_plane_weighted,
        "loss_depth_total": loss_depth + loss_depth_plane_weighted,
        "loss_alpha_sup": loss_alpha_sup,
        "continuity_loss": continuity_loss,
        "smoothness_loss": smoothness_loss,
        "loss_plane_s1": loss_plane_s1,
    }
    if isinstance(normal_loss_tri, torch.Tensor):
        scalar_outputs["loss_normal_gt_s1"] = (normal_loss_tri * lambda_n_tri).item()

    scalar_outputs["abs_depth_error_refined_stage_0"] = AbsDepthError_metrics(
        depth_est['stage_0'], depth_gt['stage_0'], mask['stage_0'] > 0.5
    )
    scalar_outputs["abs_depth_error_patchmatch_stage_1_pixel_raw"] = AbsDepthError_metrics(
        depth_patchmatch['stage_1'][0], depth_gt['stage_1'], mask['stage_1'] > 0.5
    )
    scalar_outputs["abs_depth_error_patchmatch_stage_1"] = AbsDepthError_metrics(
        depth_patchmatch['stage_1'][-1], depth_gt['stage_1'], mask['stage_1'] > 0.5
    )

    # 👑 核心补齐：Stage 1 门控融合深度 (Z_fused) 的绝对误差标量监控
    w_pixel_s1 = outputs["output_plane"].get("W_plane_pixel", None)
    tri_id_map_s1 = outputs["output_plane"].get("tri_id_map", None)
    if w_pixel_s1 is not None and tri_id_map_s1 is not None:
        valid_tri_s1 = (tri_id_map_s1.unsqueeze(1) >= 0)
        is_planar_s1 = (w_pixel_s1 >= 0.80) & valid_tri_s1
        depth_s1_fused = torch.where(is_planar_s1, depth_patchmatch['stage_1'][-1], depth_patchmatch['stage_1'][0])
    else:
        depth_s1_fused = depth_patchmatch['stage_1'][0]

    valid_gt_mask_s1 = mask['stage_1'] > 0.5
    scalar_outputs["abs_depth_error_patchmatch_stage_1_fused"] = AbsDepthError_metrics(
        depth_s1_fused, depth_gt['stage_1'], valid_gt_mask_s1
    )

    scalar_outputs["abs_depth_error_patchmatch_stage_2"] = AbsDepthError_metrics(
        depth_patchmatch['stage_2'][-1], depth_gt['stage_2'], mask['stage_2'] > 0.5
    )
    scalar_outputs["abs_depth_error_patchmatch_stage_3"] = AbsDepthError_metrics(
        depth_patchmatch['stage_3'][-1], depth_gt['stage_3'], mask['stage_3'] > 0.5
    )

    image_outputs = {}
    if do_summary_image:
        valid_mask_s0 = (outputs["output_plane"]['tri_id_map_stage0'] >= 0).float().unsqueeze(dim=1)
        vis_depth_final = get_visual_depth(depth_est['stage_0'], valid_mask_s0)
        vis_normal_final = get_visual_normal(outputs["output_plane"]['final_normal'], valid_mask_s0)
        mask_s1_b0 = mask['stage_1'][0, 0] if 'stage_1' in mask else None
        w_plane_views = build_w_plane_tensorboard_views(outputs["output_plane"], mask_s1_batch0=mask_s1_b0)

        image_outputs = {
            "Stage0_最终深度预测": vis_depth_final,
            "Stage0_最终法向量预测": vis_normal_final,
            "Stage1_门控融合深度": depth_s1_fused,
            "Stage1_原始像素深度": depth_patchmatch['stage_1'][0],
            "Ref_原始图像": sample["imgs"]['stage_0'][:, 0],
            "Stage1_深度真值": depth_gt['stage_1'],
        }

        # 👑 核心补齐：边断裂预测概率叠加图与几何真值图
        if "edge_alphas" in outputs and "tri_infos" in outputs and len(outputs["tri_infos"]) > 0:
            if "edges_pixels" in outputs["tri_infos"][0]:
                image_outputs_pre = generate_edge_alpha_overlays(
                    ref_imgs=sample["imgs"]['stage_0'][:, 0],
                    edge_alphas_list=outputs["edge_alphas"],
                    edges_pixels_list=outputs["tri_infos"][0]['edges_pixels'],
                    device=device,
                    overlay_alpha=0.6,
                    line_thickness=1
                )
                image_outputs["边断裂_预测概率叠加图"] = image_outputs_pre["ref_img_edge_alpha"]

                if 'edge_alphas_gt' in locals() and edge_alphas_gt is not None:
                    image_outputs_gt = generate_edge_alpha_overlays(
                        ref_imgs=sample["imgs"]['stage_0'][:, 0],
                        edge_alphas_list=edge_alphas_gt,
                        edges_pixels_list=outputs["tri_infos"][0]['edges_pixels'],
                        device=device,
                        overlay_alpha=0.6,
                        line_thickness=1
                    )
                    image_outputs["边断裂_真值叠加图"] = image_outputs_gt["ref_img_edge_alpha"]

        if 'cross_check_diagnosis_rgb' in outputs.get("output_plane", {}):
            image_outputs["CrossCheck_三态诊断图(绿保留_红降分)"] = outputs["output_plane"]['cross_check_diagnosis_rgb']
        image_outputs.update(w_plane_views)

    # 平面置信度更新标量监控
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

    return loss.item(), tensor2float(scalar_outputs), image_outputs


@make_nograd_func
def test_sample(sample, detailed_summary=False, global_step=0, total_steps=1):
    """
    单批次验证评估：
    注意：外层已由 mode_b_validation_context 严格设好 eval() 与单批次动态 BN (track_running_stats=False)，
    内部严禁重新调用 model.eval()，避免冲掉 BN 的即时统计设置！
    """
    progress = global_step / max(1, total_steps)

    vertexs_batch = sample['vertexs']
    lines_batch = sample['lines']
    triangles_batch = []
    for tri_list in sample['triangles']:
        tri_processed = []
        for t in tri_list:
            v_ids = t['vertex_ids']
            l_ids = t['line_ids']
            pts = torch.from_numpy(t['valid_points']).to(device)
            tri_processed.append((v_ids, l_ids, pts))
        triangles_batch.append(tri_processed)

    skip = ["vertexs", "lines", "triangles", "tri_conf_cleaned", "tri_normal_cleaned", "is_gt_planar", "tri_err95", "tri_weight_normal", "tri_weight_conf"]
    sample_cuda = tocuda(sample, device=device, skip_keys=skip)

    if 'tri_conf_cleaned' in sample:
        sample_cuda['tri_conf_cleaned'] = [torch.from_numpy(v).to(device).float() for v in sample['tri_conf_cleaned']]
    if 'tri_normal_cleaned' in sample:
        sample_cuda['tri_normal_cleaned'] = [torch.from_numpy(v).to(device).float() for v in sample['tri_normal_cleaned']]
    if 'is_gt_planar' in sample:
        sample_cuda['is_gt_planar'] = [torch.from_numpy(v).to(device).bool() for v in sample['is_gt_planar']]
    if 'tri_weight_normal' in sample:
        sample_cuda['tri_weight_normal'] = [torch.from_numpy(v).to(device).float() for v in sample['tri_weight_normal']]
    if 'tri_weight_conf' in sample:
        sample_cuda['tri_weight_conf'] = [torch.from_numpy(v).to(device).float() for v in sample['tri_weight_conf']]

    depth_gt = sample_cuda["depth"]
    mask = sample_cuda["mask"]

    max_lambda_c = 0.0
    max_lambda_s = 0.0
    outputs = model(
        sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["intrinsics_mats"],
        sample_cuda["depth_min"], sample_cuda["depth_max"],
        vertexs_batch, lines_batch, triangles_batch, depth_gt['stage_1'],
        max_lambda_c, max_lambda_s, progress=1.0, compute_edge_pixels=detailed_summary,
        pre_base_depth_thresh=args.pre_base_depth_thresh,
        pre_pixel_dist_thresh=args.pre_pixel_dist_thresh,
        post_base_depth_thresh=args.post_base_depth_thresh,
        post_pixel_dist_thresh=args.post_pixel_dist_thresh
    )

    depth_est = outputs["refined_depth"]
    depth_patchmatch = outputs["depth_patchmatch"]

    tri_id_map = outputs["output_plane"].get('tri_id_map', None)
    if (tri_id_map is not None
            and "tri_conf_cleaned" in sample_cuda
            and "tri_normal_cleaned" in sample_cuda):
        B, H_s1, W_s1 = tri_id_map.shape
        tri_device = tri_id_map.device
        planar_soft_conf_s1_list = []
        gt_normal_map_s1_list = []
        for b in range(B):
            if args.conf_gt_mode == 'soft' and "tri_weight_conf" in sample_cuda:
                tri_conf_b = sample_cuda['tri_weight_conf'][b].float()
            else:
                tri_conf_b = sample_cuda['is_gt_planar'][b].float() if "is_gt_planar" in sample_cuda else sample_cuda['tri_conf_cleaned'][b]
            tri_normal_b = sample_cuda['tri_normal_cleaned'][b]
            tri_id_map_b = tri_id_map[b]
            dummy_conf = torch.cat([tri_conf_b, torch.tensor([0.0], device=tri_device)], dim=0)
            dummy_normal = torch.cat([tri_normal_b, torch.tensor([[0.0, 0.0, 0.0]], device=tri_device)], dim=0)
            safe_idx = torch.where(tri_id_map_b >= 0, tri_id_map_b, torch.tensor(tri_conf_b.shape[0], device=tri_device).long())
            planar_soft_conf_s1_list.append(dummy_conf[safe_idx])
            gt_normal_map_s1_list.append(dummy_normal[safe_idx])
        outputs["output_plane"]["W_plane_gt_pixel"] = torch.stack(planar_soft_conf_s1_list, dim=0).unsqueeze(1)
        outputs["output_plane"]["gt_normal_map_s1"] = torch.stack(gt_normal_map_s1_list, dim=0)

    loss_depth = patchmatchnet_loss(
        depth_patchmatch=depth_patchmatch,
        refined_depth=depth_est,
        depth_gt=depth_gt,
        mask=mask
    )
    loss = loss_depth

    # 1. 核心深度误差指标
    mask_s0 = mask['stage_0'] > 0.5
    mask_s1 = mask['stage_1'] > 0.5

    s0_mae = AbsDepthError_metrics(depth_est['stage_0'], depth_gt['stage_0'], mask_s0)
    s1_raw_pixel_mae = AbsDepthError_metrics(depth_patchmatch['stage_1'][0], depth_gt['stage_1'], mask_s1)
    s1_prop_mae = AbsDepthError_metrics(depth_patchmatch['stage_1'][-1], depth_gt['stage_1'], mask_s1)

    # 2. 👑 Stage 1 真正物理门控融合深度与平面采用率统计
    w_pixel_s1 = outputs["output_plane"].get("W_plane_pixel", None)
    tri_id_map_s1 = outputs["output_plane"].get("tri_id_map", None)

    if w_pixel_s1 is not None and tri_id_map_s1 is not None:
        valid_tri_s1 = (tri_id_map_s1.unsqueeze(1) >= 0)
        is_planar_s1 = (w_pixel_s1 >= 0.80) & valid_tri_s1
        depth_s1_fused = torch.where(is_planar_s1, depth_patchmatch['stage_1'][-1], depth_patchmatch['stage_1'][0])
        eval_mask_s1_planar = mask_s1 & is_planar_s1
        planar_cnt = eval_mask_s1_planar.sum().item()
        valid_cnt = mask_s1.sum().item()
        planar_ratio_pct = (planar_cnt / valid_cnt * 100.0) if valid_cnt > 0 else 0.0

        if planar_cnt > 0:
            planar_prop_mae = AbsDepthError_metrics(depth_patchmatch['stage_1'][-1], depth_gt['stage_1'], eval_mask_s1_planar)
            planar_pixel_mae = AbsDepthError_metrics(depth_patchmatch['stage_1'][0], depth_gt['stage_1'], eval_mask_s1_planar)
        else:
            planar_prop_mae = 0.0
            planar_pixel_mae = 0.0
    else:
        depth_s1_fused = depth_patchmatch['stage_1'][0]
        planar_ratio_pct = 0.0
        planar_prop_mae = 0.0
        planar_pixel_mae = 0.0

    s1_fused_mae = AbsDepthError_metrics(depth_s1_fused, depth_gt['stage_1'], mask_s1)

    scalar_outputs = {
        "loss": loss,
        "loss_depth": loss_depth,
        # 👑 核心 TensorBoard 监控大三元
        "abs_depth_error_refined_stage_0": s0_mae,
        "abs_depth_error_patchmatch_stage_1_pixel_raw": s1_raw_pixel_mae,
        "abs_depth_error_patchmatch_stage_1_fused": s1_fused_mae,
        "abs_depth_error_patchmatch_stage_1": s1_prop_mae,
        "stage1_planar_ratio_pct": planar_ratio_pct,
        "stage1_planar_prop_mae": planar_prop_mae,
        "stage1_planar_pixel_mae": planar_pixel_mae,
        # 各级尺度
        "abs_depth_error_patchmatch_stage_2": AbsDepthError_metrics(depth_patchmatch['stage_2'][-1], depth_gt['stage_2'], mask['stage_2'] > 0.5),
        "abs_depth_error_patchmatch_stage_3": AbsDepthError_metrics(depth_patchmatch['stage_3'][-1], depth_gt['stage_3'], mask['stage_3'] > 0.5),
        # 👑 ENRICH 航空大场景米制离群误差率
        "thres10cm_error": Thres_metrics(depth_est['stage_0'], depth_gt['stage_0'], mask_s0, 0.10),
        "thres20cm_error": Thres_metrics(depth_est['stage_0'], depth_gt['stage_0'], mask_s0, 0.20),
        "thres50cm_error": Thres_metrics(depth_est['stage_0'], depth_gt['stage_0'], mask_s0, 0.50),
    }

    image_outputs = {}
    if detailed_summary:
        valid_mask_s0 = (outputs["output_plane"]['tri_id_map_stage0'] >= 0).float().unsqueeze(dim=1)
        vis_depth_final = get_visual_depth(depth_est['stage_0'], valid_mask_s0)
        vis_normal_final = get_visual_normal(outputs["output_plane"]['final_normal'], valid_mask_s0)
        mask_s1_b0 = mask['stage_1'][0, 0] if 'stage_1' in mask else None
        w_plane_views = build_w_plane_tensorboard_views(outputs["output_plane"], mask_s1_batch0=mask_s1_b0)

        image_outputs = {
            "Stage0_最终深度预测": vis_depth_final,
            "Stage0_最终法向量预测": vis_normal_final,
            "Stage1_门控融合深度": depth_s1_fused,
            "Stage1_原始像素深度": depth_patchmatch['stage_1'][0],
            "Ref_原始图像": sample["imgs"]['stage_0'][:, 0],
            "Stage1_深度真值": depth_gt['stage_1'],
        }
        if 'cross_check_diagnosis_rgb' in outputs.get("output_plane", {}):
            image_outputs["CrossCheck_三态诊断图(绿保留_红降分)"] = outputs["output_plane"]['cross_check_diagnosis_rgb']

        # 👑 核心补齐：验证集边断裂预测概率叠加图
        if "edge_alphas" in outputs and "tri_infos" in outputs and len(outputs["tri_infos"]) > 0:
            if "edges_pixels" in outputs["tri_infos"][0]:
                image_outputs_pre = generate_edge_alpha_overlays(
                    ref_imgs=sample["imgs"]['stage_0'][:, 0],
                    edge_alphas_list=outputs["edge_alphas"],
                    edges_pixels_list=outputs["tri_infos"][0]['edges_pixels'],
                    device=device,
                    overlay_alpha=0.6,
                    line_thickness=1
                )
                image_outputs["边断裂_预测概率叠加图"] = image_outputs_pre["ref_img_edge_alpha"]

        image_outputs.update(w_plane_views)

    return loss.item(), tensor2float(scalar_outputs), image_outputs


def train():
    global best_val_s0_mae, start_epoch

    # 智能续训轮数自适应扩充
    if args.add_epochs > 0:
        args.epochs = max(args.epochs, start_epoch + args.add_epochs)
        print(f"[Resume] 启用追加训练轮数 (--add_epochs {args.add_epochs}) -> 调整总轮数为: {args.epochs}")
    elif start_epoch >= args.epochs:
        recommended_epochs = start_epoch + 6
        print(f"⚠️ [Resume Warning] 当前起始轮数 ({start_epoch}) 已达到或超过设定总轮数 ({args.epochs})！")
        print(f"   自动为您将总轮数扩充至: {recommended_epochs} 轮 (默认追加 6 轮微调)。")
        args.epochs = recommended_epochs

    for param_group in optimizer.param_groups:
        if 'initial_lr' not in param_group:
            param_group['initial_lr'] = param_group['lr']
    milestones = [int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(args.lrepochs.split(':')[1])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=lr_gamma,
                                                        last_epoch=start_epoch - 1)
    if args.force_lr:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
            param_group['initial_lr'] = args.lr
        lr_scheduler.base_lrs = [args.lr for _ in optimizer.param_groups]
        print(f"[Resume] 强制重置学习率调度器当前 LR = {args.lr:.6f}")

    total_steps = len(TrainImgLoader) * args.epochs

    print("==============================================================================")
    print("🚀 启动 ENRICH-Aerial 专用训练系统")
    print(f"  总训练轮数 (epochs): {args.epochs}")
    print(f"  批次大小 (batch_size): {args.batch_size}")
    print(f"  验证评估协议: Mode B (动态无状态单批次 BN，严格防污染)")
    print(f"  主控选模指标: Mode B 验证集 Stage 0 MAE")
    print(f"  Pre-GNN 容限: base_depth={args.pre_base_depth_thresh}m, pixel_dist={args.pre_pixel_dist_thresh}px")
    print(f"  Post-GNN 容限: base_depth={args.post_base_depth_thresh}m, pixel_dist={args.post_pixel_dist_thresh}px")
    print(f"  日志与权重目录: {args.logdir}")
    print("==============================================================================")

    for epoch_idx in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch_idx}/{args.epochs}: [LR = {optimizer.param_groups[0]['lr']:.6f}]")
        lr_scheduler.step()

        # --------------------------------------------------------------------
        # 1. 训练阶段
        # --------------------------------------------------------------------
        model.train()
        for batch_idx, sample in enumerate(TrainImgLoader):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            do_summary_image = global_step % (10 * args.summary_freq) == 0

            loss, scalar_outputs, image_outputs = train_sample(
                sample, do_summary_image=do_summary_image,
                global_step=global_step, total_steps=total_steps
            )

            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
            if do_summary_image and len(image_outputs) > 0:
                save_images(logger, 'train', image_outputs, global_step)

            if batch_idx % 10 == 0 or batch_idx == len(TrainImgLoader) - 1:
                print(f"Epoch {epoch_idx:2d}/{args.epochs:2d} | Train [{batch_idx:3d}/{len(TrainImgLoader):3d}] "
                      f"| Loss: {loss:.4f} | S0 MAE: {scalar_outputs['abs_depth_error_refined_stage_0']:.4f}m "
                      f"| S1 Raw: {scalar_outputs['abs_depth_error_patchmatch_stage_1_pixel_raw']:.4f}m "
                      f"| Time: {time.time() - start_time:.3f}s")

            del scalar_outputs, image_outputs

        # --------------------------------------------------------------------
        # 2. 定期保存常规 Epoch Checkpoint
        # --------------------------------------------------------------------
        if (epoch_idx + 1) % args.save_freq == 0:
            ckpt_path = os.path.join(args.logdir, f"model_{epoch_idx:06d}.ckpt")
            torch.save({
                'epoch': epoch_idx,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_s0_mae': best_val_s0_mae
            }, ckpt_path)
            print(f"✓ Epoch Checkpoint 已保存至: {ckpt_path}")

        # --------------------------------------------------------------------
        # 3. 验证阶段 (严格使用无状态 Mode B: 动态批次统计量，不污染训练账本)
        # --------------------------------------------------------------------
        current_epoch = epoch_idx + 1
        do_validation = (current_epoch % args.eval_freq == 0) or (current_epoch == args.epochs)

        if do_validation:
            print(f"\n--- [Validation Mode B] 开始执行第 {epoch_idx} 轮无状态动态 BN 验证评估 ---")
            avg_test_scalars = DictAverageMeter()

            with mode_b_validation_context(model):
                for batch_idx, sample in enumerate(TestImgLoader):
                    start_time = time.time()
                    global_step = len(TrainImgLoader) * epoch_idx + batch_idx
                    do_summary = global_step % args.summary_freq == 0
                    do_summary_image = global_step % (10 * args.summary_freq) == 0

                    loss, scalar_outputs, image_outputs = test_sample(
                        sample, detailed_summary=do_summary_image,
                        global_step=global_step, total_steps=total_steps
                    )

                    if do_summary:
                        save_scalars(logger, 'test_mode_b', scalar_outputs, global_step)
                    if do_summary_image and len(image_outputs) > 0:
                        save_images(logger, 'test_mode_b', image_outputs, global_step)

                    avg_test_scalars.update(tensor2float(scalar_outputs))
                    del scalar_outputs, image_outputs

            # 全量验证集平均指标沉淀
            mean_test = avg_test_scalars.mean()
            save_scalars(logger, 'fulltest_mode_b', mean_test, global_step)

            cur_s0_mae = mean_test.get("abs_depth_error_refined_stage_0", float('inf'))
            cur_s1_fused = mean_test.get("abs_depth_error_patchmatch_stage_1_fused", float('inf'))
            cur_s1_raw = mean_test.get("abs_depth_error_patchmatch_stage_1_pixel_raw", float('inf'))
            cur_planar_ratio = mean_test.get("stage1_planar_ratio_pct", 0.0)

            print("=" * 80)
            print(f"📊 [Validation Mode B 成果] Epoch {epoch_idx}/{args.epochs}:")
            print(f"   ★ Stage 0 最终 MAE   : {cur_s0_mae:.4f} m (主选模指标)")
            print(f"   ★ Stage 1 门控融合 MAE : {cur_s1_fused:.4f} m")
            print(f"   ★ Stage 1 原始像素 MAE : {cur_s1_raw:.4f} m")
            print(f"   ★ 平面采用率 (Ratio)  : {cur_planar_ratio:.1f} %")
            if "thres10cm_error" in mean_test:
                print(f"   ★ >10cm 误差率: {mean_test['thres10cm_error']*100:.2f}% | "
                      f">20cm: {mean_test['thres20cm_error']*100:.2f}% | "
                      f">50cm: {mean_test['thres50cm_error']*100:.2f}%")
            print("=" * 80)

            # 👑 主曲线选模：基于 Mode B 验证集 Stage 0 MAE 持续沉淀最佳 Checkpoint
            if cur_s0_mae < best_val_s0_mae:
                improvement = best_val_s0_mae - cur_s0_mae
                best_val_s0_mae = cur_s0_mae
                best_ckpt_path = os.path.join(args.logdir, "best_model.ckpt")
                torch.save({
                    'epoch': epoch_idx,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_val_s0_mae': best_val_s0_mae,
                    'eval_protocol': 'Mode B (动态无状态 BN)'
                }, best_ckpt_path)
                print(f"🏆 刷新历史最佳 Checkpoint! S0 MAE 下降至 {best_val_s0_mae:.4f} m (提升 {improvement*100:.2f} cm)，已保存至: {best_ckpt_path}")

        print("当前时间:", time.ctime())
        gc.collect()


if __name__ == '__main__':
    train()
