import argparse
import os

import math

from models.sum_loss import *
from models.PlanePatchMatch import *

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
from datasets.dtu_yao import collate_keep_list

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
def train():
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
                'Epoch {}/{}, Iter {}/{},loss_depth:{:.3f},loss_alpha_sup:{:.3f},'
                'continuity_loss:{:.3f},continuity_s_loss:{:.3f}'
                'total loss:{:.3f}, time = {:.3f}'.format(
                    epoch_idx, args.epochs, batch_idx,
                    len(TrainImgLoader),
                    loss_depth,loss_alpha_sup,
                    scalar_outputs['continuity_loss'],scalar_outputs['continuity_s_loss'],
                    total_loss,time.time() - start_time))

            del scalar_outputs, image_outputs

        # checkpoint
        if (epoch_idx + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch_idx,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()},
                "{}/model_{:0>6}.ckpt".format(args.logdir, epoch_idx))

        # avg_test_scalars = DictAverageMeter()
        # for batch_idx, sample in enumerate(TestImgLoader):
        #     start_time = time.time()
        #     global_step = len(TrainImgLoader) * epoch_idx + batch_idx
        #     do_summary = global_step % args.summary_freq == 0
        #     # do_summary_test = global_step % (10*args.summary_freq) == 0
        #     do_summary_image = global_step % (50 * args.summary_freq) == 0
        #     loss, scalar_outputs, image_outputs = test_sample(sample, detailed_summary=do_summary_image)
        #     loss_depth = scalar_outputs['loss_depth']
        #     loss_alpha_sup=scalar_outputs['loss_alpha_sup']
        #     if do_summary:
        #         save_scalars(logger, 'test', scalar_outputs, global_step)
        #     if do_summary_image:
        #         save_images(logger, 'test', image_outputs, global_step)
        #     avg_test_scalars.update(scalar_outputs)
        #     del scalar_outputs, image_outputs
        #     print(
        #         'Epoch {}/{}, Iter {}/{},loss_depth:{:.3f},loss_alpha_sup:{:.3f},total loss:{:.3f}, time = {:.3f}'.format(
        #             epoch_idx, args.epochs, batch_idx,
        #             len(TrainImgLoader), loss_depth, loss_alpha_sup, loss,
        #             time.time() - start_time))
        #
        # save_scalars(logger, 'fulltest', avg_test_scalars.mean(), global_step)
        # print("avg_test_scalars:", avg_test_scalars.mean())
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

    # 自动构建计算图（动态计算图），记录每个张量的操作历史（如卷积、激活、矩阵乘法等），从而在反向传播时能通过链式法则计算梯度
    outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"],sample_cuda["intrinsics_mats"],
                    sample_cuda["depth_min"], sample_cuda["depth_max"],
                    vertexs_batch, lines_batch, triangles_batch,depth_gt['stage_1'])

    depth_est = outputs["refined_depth"]

    depth_patchmatch = outputs["depth_patchmatch"]

    # 通过计算最终的损失
    # 1. 主损失
    loss_depth = model_loss(depth_patchmatch, depth_est, depth_gt, mask)  # 深度损失

    # EdgeConsistencyLoss（自监督 BCE）
    edge_consistency_loss_fn = EdgeConsistencyLoss(depth_threshold=0.2, sparsity_weight=1e-4)
    loss_alpha_sup, info,edge_alphas_gt= edge_consistency_loss_fn(
        pred_alphas_list=outputs["edge_alphas"],
        # 1/2分辨率图的深度图
        gt_depth_map=depth_gt[f'stage_1'],  # or pass GT depth if you want pseudo from GT (but keep pred_depth for continuity)
        tri_infos=outputs["tri_infos"],
        tri_id_map=outputs["output_plane"]['tri_id_map']
    )
    # =======================手动固定权重==================
    # 2.边缘监督 Loss
    # 乘上一个权重再，加上边断裂损失，防止预测头损失过小
    weight_alpha=10
    loss_alpha_sup=loss_alpha_sup*weight_alpha

    # 3.连续性正则化 Loss (Geometric Smoothness)
    # 因为它是正则化项，绝不能喧宾夺主。建议权重设为 0.1 ~ 0.5
    lambda_c = 0.1
    # 4. 稀疏性惩罚 Loss (防止作弊)
    # 这个权重必须足够大，大到能抵消作弊带来的收益！如果 lambda_c * dist_error 大概是 0.5，那 lambda_s 至少要是 1.0 甚至 2.0
    lambda_s = 1.0

    # =======================动态权重衰减==================

    # weight_alpha, lambda_c, lambda_s = get_dynamic_loss_weights(global_step, total_steps)

    continuity_loss = outputs["continuity_loss"] * lambda_c
    continuity_s_loss = outputs["continuity_s_loss"] * lambda_s

    # 总损失：深度损失+边断裂损失+连续性损失
    loss = loss_depth + loss_alpha_sup + continuity_loss + continuity_s_loss

    # 边断裂损失
    loss.backward()

    # 优化器根据计算的梯度更新模型参数（梯度下降的具体实现）
    optimizer.step()

    scalar_outputs = {"loss": loss,
                      "loss_depth": loss_depth,
                      "loss_alpha_sup": loss_alpha_sup,
                      "continuity_loss":continuity_loss,
                      "continuity_s_loss":continuity_s_loss}

    image_outputs = []
    if do_summary_image:
        # ================ 生成断裂图 ===============================================
        image_outputs_pre = generate_edge_alpha_overlays(
            ref_imgs=sample["imgs"]['stage_1'][:, 0],  # 注意取 ref 图
            edge_alphas_list=outputs["edge_alphas"],
            edges_pixels_list=outputs["tri_infos"][0]['edges_pixels'],
            device=device,
            overlay_alpha=0.6,  # 线条显示的透明度
            line_thickness=1  # 线条粗细
        )

        ref_img_edge_alpha_pre = image_outputs_pre["ref_img_edge_alpha"]

        image_outputs_gt = generate_edge_alpha_overlays(
            ref_imgs=sample["imgs"]['stage_1'][:, 0],  # 注意取 ref 图
            edge_alphas_list=edge_alphas_gt,
            edges_pixels_list=outputs["tri_infos"][0]['edges_pixels'],
            device=device,
            overlay_alpha=0.6,  # 线条显示的透明度
            line_thickness=1  # 线条粗细
        )

        ref_img_edge_alpha_gt = image_outputs_gt["ref_img_edge_alpha"]

        # === 生成基于像素的法向量图 (使用上面定义的函数) ================================
        # 获取 Stage 1 的 GT 深度和 Mask
        gt_depth_s1 = depth_gt['stage_1']  # 假设形状 [B,1, H, W]
        gt_mask_s1 = mask['stage_1']  # 假设形状 [B, 1,H, W]

        valid_mask_s1 = (outputs["output_plane"]['tri_id_map'] >= 0).float().unsqueeze(dim=1)

        # 获取 Stage 1 的 预测 深度 (planepatchmatch最终预测结果)
        pred_depth_s1 = depth_patchmatch['stage_1'][-1]  # 假设形状 [B,H, W]

        # 1. 生成 GT 法向量 (传入 mask 去除无效区域)
        normal_gt_s1 = compute_normal_map_torch(gt_depth_s1, mask=gt_mask_s1, smooth=False)

        # 2. 生成 预测 法向量 (同样传入 mask，或者你可以传入 threshold 后的 mask)
        # normal_pred_s1 = compute_normal_map_torch(pred_depth_s1, mask=gt_mask_s1, smooth=False)

        intrinsics_s1 = torch.unbind(sample_cuda["intrinsics_mats"]['stage_1'].float(), 1)


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
            "patchmatch预测的stage1深度值": outputs["output_plane"]['depth_stage1_pixels'],
            # "depth_patchmatch_stage_2": depth_patchmatch['stage_2'][-1] * mask['stage_2'],
            # "depth_patchmatch_stage_3": depth_patchmatch['stage_3'][-1] * mask['stage_3'],
            "ref_img": sample["imgs"]['stage_1'][:, 0],
            # 新增：基于像素点的法向量图
            "根据深度真值生成的法向量": normal_gt_s1,
            # "经过平面传播预测的stage1深度值生成的像素法向量": normal_pred_s1,
            # 新增：基于平面的深度图和法向量图传播完
            "经过平面传播生成的平面法向量": outputs["output_plane"]['normal_pred'],
            "经过平面传播预测的stage1深度值": depth_patchmatch['stage_1'][-1],
            # 新增：基于平面的深度图和法向量图，刚拟合
            "没有经过平面传播，刚拟合完初始平面深度值": outputs["output_plane"]['depth_gt'],
            "没有经过平面传播，刚拟合完初始平面法向量": outputs["output_plane"]['normal_gt'] ,
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
def test_sample(sample,detailed_summary=False):
    model.eval()

    # 将cdt_data进行一个单独处理处理,单独将这些数据放入GPU中
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

    # 自动构建计算图（动态计算图），记录每个张量的操作历史（如卷积、激活、矩阵乘法等），从而在反向传播时能通过链式法则计算梯度
    outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"],sample_cuda["intrinsics_mats"],
                    sample_cuda["depth_min"], sample_cuda["depth_max"],
                    vertexs_batch, lines_batch, triangles_batch,depth_gt['stage_1'])

    depth_est = outputs["refined_depth"]
    depth_patchmatch = outputs["depth_patchmatch"]

    # 通过计算最终的损失
    # 总损失：深度损失+边断裂损失+连续性损失
    loss_depth = model_loss(depth_patchmatch, depth_est, depth_gt, mask)  # 深度损失

    # EdgeConsistencyLoss（自监督 BCE）
    edge_consistency_loss_fn = EdgeConsistencyLoss(depth_threshold=0.1, sparsity_weight=1e-4)
    loss_alpha_sup, info, edge_alphas_gt = edge_consistency_loss_fn(
        pred_alphas_list=outputs["edge_alphas"],
        # 1/2分辨率图的深度图
        gt_depth_map=depth_gt[f'stage_1'],
        # or pass GT depth if you want pseudo from GT (but keep pred_depth for continuity)
        tri_infos=outputs["tri_infos"],
        tri_id_map=outputs["output_plane"]['tri_id_map']
    )

    # # 乘上一个权重再，加上边断裂损失，防止预测头损失过小
    weight_alpha = 10
    loss_alpha_sup = loss_alpha_sup * weight_alpha
    loss = loss_depth + loss_alpha_sup

    # ================ 生成断裂图 ===============================================
    image_outputs_pre = generate_edge_alpha_overlays(
        ref_imgs=sample["imgs"]['stage_1'][:, 0],  # 注意取 ref 图
        edge_alphas_list=outputs["edge_alphas"],
        edges_pixels_list=outputs["tri_infos"][0]['edges_pixels'],
        device=device,
        overlay_alpha=0.6,  # 线条显示的透明度
        line_thickness=1  # 线条粗细
    )

    ref_img_edge_alpha_pre = image_outputs_pre["ref_img_edge_alpha"]

    image_outputs_gt = generate_edge_alpha_overlays(
        ref_imgs=sample["imgs"]['stage_1'][:, 0],  # 注意取 ref 图
        edge_alphas_list=edge_alphas_gt,
        edges_pixels_list=outputs["tri_infos"][0]['edges_pixels'],
        device=device,
        overlay_alpha=0.6,  # 线条显示的透明度
        line_thickness=1  # 线条粗细
    )

    ref_img_edge_alpha_gt = image_outputs_gt["ref_img_edge_alpha"]

    # === 生成基于像素的法向量图 (使用上面定义的函数) ================================
    # 获取 Stage 1 的 GT 深度和 Mask
    gt_depth_s1 = depth_gt['stage_1']  # 假设形状 [B, H, W]
    gt_mask_s1 = mask['stage_1']  # 假设形状 [B, H, W]


    # 获取 Stage 1 的 预测 深度 (planepatchmatch预测的结果)
    pred_depth_s1 = depth_patchmatch['stage_1'][-1]  # 假设形状 [B, H, W]

    # 1. 生成 GT 法向量 (传入 mask 去除无效区域)
    normal_gt_s1 = compute_normal_map_torch(gt_depth_s1, mask=gt_mask_s1, smooth=False)

    # 2. 生成 预测 法向量 (同样传入 mask，或者你可以传入 threshold 后的 mask)
    # normal_pred_s1 = compute_normal_map_torch(pred_depth_s1, mask=gt_mask_s1, smooth=False)

    scalar_outputs = {"loss": loss,
                      "loss_depth": loss_depth,
                      "loss_alpha_sup": loss_alpha_sup}

    image_outputs = {  # 暂时注释一些图片，输出的图片太多了
        "最终预测结果": depth_est['stage_0'] * mask['stage_0'],
        "stage1深度真值": depth_gt['stage_1'],
        "patchmatch预测的stage1深度值": outputs["output_plane"]['depth_stage1_pixels'],
        # "depth_patchmatch_stage_2": depth_patchmatch['stage_2'][-1] * mask['stage_2'],
        # "depth_patchmatch_stage_3": depth_patchmatch['stage_3'][-1] * mask['stage_3'],
        "ref_img": sample["imgs"]['stage_1'][:, 0],
        # 新增：基于像素点的法向量图
        "根据深度真值生成的法向量": normal_gt_s1,
        # "经过平面传播预测的stage1深度值生成的像素法向量": normal_pred_s1,
        # 新增：基于平面的深度图和法向量图传播完
        "经过平面传播生成的平面法向量": outputs["output_plane"]['normal_pred'],
        "经过平面传播预测的stage1深度值": depth_patchmatch['stage_1'][-1],
        # 新增：基于平面的深度图和法向量图，刚拟合
        "没有经过平面传播，刚拟合完初始平面深度值": outputs["output_plane"]['depth_gt'],
        "没有经过平面传播，刚拟合完初始平面法向量": outputs["output_plane"]['normal_gt'],
        # 新增：边预测头预测值和真值
        "ref_img_edge_alpha_pre": ref_img_edge_alpha_pre,
        "ref_img_edge_alpha_gt": ref_img_edge_alpha_gt
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
