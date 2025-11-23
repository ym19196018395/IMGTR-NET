import numpy as np
import torchvision.utils as vutils
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import random

# print arguments
def print_args(args):
    print("################################  args  ################################")
    for k, v in args.__dict__.items():
        print("{0: <10}\t{1: <30}\t{2: <20}".format(k, str(v), str(type(v))))
    print("########################################################################")


# torch.no_grad warpper for functions
def make_nograd_func(func):
    def wrapper(*f_args, **f_kwargs):
        with torch.no_grad():
            ret = func(*f_args, **f_kwargs)
        return ret

    return wrapper


# convert a function into recursive style to handle nested dict/list/tuple variables
def make_recursive_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)

    return wrapper


@make_recursive_func
def tensor2float(vars):
    if isinstance(vars, float):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.item()
    else:
        raise NotImplementedError("invalid input type {} for tensor2float".format(type(vars)))


@make_recursive_func
def tensor2numpy(vars):
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.detach().cpu().numpy().copy()
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


@make_recursive_func
def tocuda(vars):
    if isinstance(vars, torch.Tensor):
        return vars.cuda()
    elif isinstance(vars, str):
        return vars
    else:
        raise NotImplementedError("invalid input type {} for tocuda".format(type(vars)))


def save_scalars(logger, mode, scalar_dict, global_step):
    scalar_dict = tensor2float(scalar_dict)
    for key, value in scalar_dict.items():
        if not isinstance(value, (list, tuple)):
            name = '{}/{}'.format(mode, key)
            logger.add_scalar(name, value, global_step)
        else:
            for idx in range(len(value)):
                name = '{}/{}_{}'.format(mode, key, idx)
                logger.add_scalar(name, value[idx], global_step)


def save_images(logger, mode, images_dict, global_step):
    images_dict = tensor2numpy(images_dict)

    def preprocess(name, img):
        if not (len(img.shape) == 3 or len(img.shape) == 4):
            raise NotImplementedError("invalid img shape {}:{} in save_images".format(name, img.shape))
        if len(img.shape) == 3:
            img = img[:, np.newaxis, :, :]
        img = torch.from_numpy(img[:1])
        return vutils.make_grid(img, padding=0, nrow=1, normalize=True, scale_each=True)

    for key, value in images_dict.items():
        if not isinstance(value, (list, tuple)):
            name = '{}/{}'.format(mode, key)
            logger.add_image(name, preprocess(name, value), global_step)
        else:
            for idx in range(len(value)):
                name = '{}/{}_{}'.format(mode, key, idx)
                logger.add_image(name, preprocess(name, value[idx]), global_step)


class DictAverageMeter(object):
    def __init__(self):
        self.data = {}
        self.count = 0

    def update(self, new_input):
        self.count += 1
        if len(self.data) == 0:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                self.data[k] = v
        else:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                self.data[k] += v

    def mean(self):
        return {k: v / self.count for k, v in self.data.items()}


# a wrapper to compute metrics for each image individually
def compute_metrics_for_each_image(metric_func):
    def wrapper(depth_est, depth_gt, mask, *args):
        batch_size = depth_gt.shape[0]
        results = []
        # compute result one by one
        for idx in range(batch_size):
            ret = metric_func(depth_est[idx], depth_gt[idx], mask[idx], *args)
            results.append(ret)
        return torch.stack(results).mean()

    return wrapper


@make_nograd_func
@compute_metrics_for_each_image
def Thres_metrics(depth_est, depth_gt, mask, thres):
    # if thres is int or float, then True
    assert isinstance(thres, (int, float))
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    errors = torch.abs(depth_est - depth_gt)
    err_mask = errors > thres
    return torch.mean(err_mask.float())


# NOTE: please do not use this to build up training loss
@make_nograd_func
@compute_metrics_for_each_image
def AbsDepthError_metrics(depth_est, depth_gt, mask):
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    return torch.mean((depth_est - depth_gt).abs())

def tocuda(sample, device, skip_keys=None, non_blocking=True):
    """
    将 sample 中的可转为 GPU 的项搬到 device。
    - sample: dict-like (通常 DataLoader 返回的 batch)
    - device: torch.device("cuda:0") 等
    - skip_keys: iterable of top-level keys to skip (e.g. ["vertexs","lines","triangles"])
    - non_blocking: 用于 tensor.to(..., non_blocking=...)
    返回修改后的 sample（in-place 修改并返回）
    """
    if skip_keys is None:
        skip_keys = set()
    else:
        skip_keys = set(skip_keys)

    def _move(x):
        # torch tensor -> 发送到 device
        if isinstance(x, torch.Tensor):
            return x.to(device, non_blocking=non_blocking)
        # numpy array -> 转 torch 然后送 device
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(device, non_blocking=non_blocking)
        # number -> 转 tensor
        if isinstance(x, (int, float)):
            return torch.tensor(x).to(device)
        # dict/list/tuple -> 递归处理
        if isinstance(x, dict):
            return {k: _move(v) for k, v in x.items()}
        if isinstance(x, list):
            # 列表通常我们希望保留（例如 triangles 是 list-of-arrays）——
            # 这里做保守处理：如果列表里全是 torch.Tensor / np.ndarray / numbers，则把每个元素搬；
            # 否则返回原始列表（保持在 CPU，由使用处决定如何处理）
            if all(isinstance(el, (torch.Tensor, np.ndarray, int, float)) for el in x):
                return [_move(el) for el in x]
            else:
                return x  # 保持原样
        if isinstance(x, tuple):
            # 转成 tuple 返回
            return tuple(_move(el) for el in x)
        # 其他对象（自定义类、namedtuple 等）——保守返回原对象（不要试图搬）
        return x

    # 对顶层键做 skip
    for k in list(sample.keys()):
        if k in skip_keys:
            continue
        sample[k] = _move(sample[k])
    return sample


def bresenham_line(p1, p2):
    """Bresenham算法计算两点间所有像素坐标（(x, y)格式，x=列，y=行）"""
    x0, y0 = p1  # p1=(x1, y1)，x对应图像列，y对应图像行
    x1, y1 = p2  # p2=(x2, y2)
    pixels = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x1 > x0 else -1  # x方向步进
    sy = 1 if y1 > y0 else -1  # y方向步进

    if dx > dy:
        # x为主方向
        err = dx / 2.0
        while x != x1:
            pixels.append((x, y))  # 记录当前(x, y)像素
            err -= dy
            if err < 0:
                y += sy  # 调整y
                err += dx
            x += sx
        pixels.append((x, y))  # 加入终点像素
    else:
        # y为主方向
        err = dy / 2.0
        while y != y1:
            pixels.append((x, y))  # 记录当前(x, y)像素
            err -= dx
            if err < 0:
                x += sx  # 调整x
                err += dy
            y += sy
        pixels.append((x, y))  # 加入终点像素
    return pixels


def scale_pixel_coords(pixels, old_W, old_H, new_W, new_H):
    """
    缩放像素坐标（从(old_W, old_H)到(new_W, new_H)）
    参数:
        pixels: list of (x, y) 原像素坐标
        old_W/old_H: 原图像尺寸
        new_W/new_H: 目标尺寸
    返回:
        scaled_pixels: list of (x', y') 缩放后的像素坐标
    """
    # 计算缩放因子
    scale_x = new_W / old_W
    scale_y = new_H / old_H

    scaled_pixels = []
    for (x, y) in pixels:
        # 比例映射并取整
        x_scaled = int(round(x * scale_x))
        y_scaled = int(round(y * scale_y))
        # 裁剪到目标尺寸范围内
        x_scaled = np.clip(x_scaled, 0, new_W - 1)
        y_scaled = np.clip(y_scaled, 0, new_H - 1)
        scaled_pixels.append((x_scaled, y_scaled))

    return scaled_pixels


def convert_to_tri_infos(vertexs, lines, triangles, H, W, device, scale_ratio=0.5):
    """
    适配数组/张量格式的lines（每个Line对应[ p1, p2, face1, face2 ]）

    参数:
        vertexs: np.ndarray, 形状 (Nv, 2)，顶点坐标 (x, y)
        lines: np.ndarray/torch.Tensor, 形状 (Ne, 4)，每行对应[ p1, p2, face1, face2 ]
               - 若为torch.Tensor，会自动转numpy处理
        triangles: list, 每个元素为dict:
            {
                'vertex_ids': np.ndarray (3,), 三角形顶点ID
                'line_ids': np.ndarray (3,), 三角形边ID
                'valid_points': np.ndarray (M, 2), 三角形内像素 (x, y)
            }
        H/W: 图像尺寸
        device: 计算设备

    返回:
        tri_infos: dict（格式同前）
    """
    tri_infos = {
        'num_tri': len(triangles),
        'tri_masks': [],
        'edges': []
    }
    new_H = int(H * scale_ratio)
    new_W = int(W * scale_ratio)
    # -------------------------- 1. 生成缩放后的三角掩码（CPU→GPU） --------------------------
    for tri in triangles:
        valid_points = tri['valid_points']  # 原始尺寸像素坐标 (x,y)

        if len(valid_points) > 0:
            # 方案1：先缩小valid_points坐标，再生成小尺寸掩码（更省内存）
            # 缩放valid_points到新尺寸
            valid_points_scaled_x = (valid_points[:, 0] * scale_ratio).round().astype(np.int64)
            valid_points_scaled_y = (valid_points[:, 1] * scale_ratio).round().astype(np.int64)
            # 裁剪到新尺寸范围内
            valid_points_scaled_x = np.clip(valid_points_scaled_x, 0, new_W - 1)
            valid_points_scaled_y = np.clip(valid_points_scaled_y, 0, new_H - 1)

            # 在CPU上创建小尺寸numpy掩码
            tri_mask_np = np.zeros((new_H, new_W), dtype=np.float32)
            tri_mask_np[valid_points_scaled_y, valid_points_scaled_x] = 1.0  # y对应行，x对应列


        else:
            tri_mask_np = np.zeros((new_H, new_W), dtype=np.float32)

        # 转成小尺寸GPU tensor（仅占用new_H*new_W内存，远小于原始尺寸）
        tri_mask_tensor = torch.from_numpy(tri_mask_np).to(device)
        tri_infos['tri_masks'].append(tri_mask_tensor)

    # 堆叠为(N, new_H, new_W)的GPU tensor（此时尺寸已缩小，显存占用低）
    # tri_infos['tri_masks'] = torch.stack(tri_infos['tri_masks'], dim=0)
    # todo:非常严重的问题，显存爆了
    # 新代码：分批次堆叠后拼接
    stack_batch_size = 100
    mask_list = tri_infos['tri_masks']  # 所有小尺寸掩码的列表
    if len(mask_list) == 0:
        tri_infos['tri_masks'] = torch.empty(0, new_H, new_W, device=device)
    else:
        # 分批次堆叠
        stacked_batches = []
        for i in range(0, len(mask_list), stack_batch_size):
            # 取当前批次的掩码（i到i+stack_batch_size）
            batch_masks = mask_list[i:i+stack_batch_size]
            # 堆叠当前批次
            batch_stacked = torch.stack(batch_masks, dim=0)
            stacked_batches.append(batch_stacked)
        # 拼接所有批次（最终形状和一次性stack一致）
        tri_infos['tri_masks'] = torch.cat(stacked_batches, dim=0)

    # -------------------------- 3. 处理边信息（按索引取值）缩放--------------------------
    # 将lines转为numpy数组（兼容tensor输入）
    if isinstance(lines, torch.Tensor):
        lines_np = lines.cpu().numpy()
    else:
        lines_np = lines

    for line_arr in lines_np:
        # 按索引取Line字段：[p1, p2, face1, face2]
        v1_id = int(line_arr[0])  # 索引0 = p1
        v2_id = int(line_arr[1])  # 索引1 = p2
        face1 = int(line_arr[2])  # 索引2 = face1
        face2 = int(line_arr[3])  # 索引3 = face2

        # 处理face为无效值的情况（如None转成的-1或0）
        tri_ids = []
        if face1 != -1 and face1 is not None:  # 假设-1表示无相邻面
            tri_ids.append(face1)
        else: # 如果一条面为空，则给一个-1
            tri_ids.append(-1)
        if face2 != -1 and face2 is not None:
            tri_ids.append(face2)
        else:
            tri_ids.append(-1)
        tri_ids = tuple(tri_ids)

        # 计算边的像素坐标
        p1 = (vertexs[v1_id][0], vertexs[v1_id][1])
        p2 = (vertexs[v2_id][0], vertexs[v2_id][1])
        edge_pixels_original = bresenham_line(p1, p2)

        # 缩放边像素到目标尺寸
        edge_pixels_scaled = scale_pixel_coords(edge_pixels_original, W, H, new_W, new_H)
        del edge_pixels_original
        # 添加边信息
        tri_infos['edges'].append({
            'tri_ids': tri_ids,
            'edge_pixels': edge_pixels_scaled,  # 存储缩放后的像素
        })

    return tri_infos

def visualize_triangles_and_edges(tri_infos, save_path_tri, save_path_edges):
    """
    可视化tri_infos中的三角形掩码和边像素

    参数:
        tri_infos: 转换得到的tri_infos字典
        save_path_tri: 三角形可视化图片保存路径（如"triangles.png"）
        save_path_edges: 边可视化图片保存路径（如"edges.png"）
    """
    # 获取图像尺寸 (H, W)
    H = tri_infos['tri_masks'].shape[1]
    W = tri_infos['tri_masks'].shape[2]
    num_tri = tri_infos['num_tri']
    num_edges = len(tri_infos['edges'])

    # -------------------------- 1. 可视化三角形掩码 --------------------------
    # 创建空白RGB图片（白色背景）
    img_tri = Image.new('RGB', (W, H), color=(255, 255, 255))
    draw_tri = ImageDraw.Draw(img_tri)

    # 为每个三角形生成随机颜色（半透明效果，避免完全覆盖）
    tri_colors = [
        (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for _ in range(num_tri)
    ]

    # 遍历每个三角形，绘制掩码区域
    for tri_id in range(num_tri):
        # 获取当前三角形的掩码（转为CPU numpy数组）
        tri_mask = tri_infos['tri_masks'][tri_id].cpu().numpy()
        # 找到掩码中值为1的像素坐标（y, x）
        y_coords, x_coords = np.where(tri_mask == 1.0)
        # 遍历像素并绘制（用随机颜色）
        color = tri_colors[tri_id]
        for x, y in zip(x_coords, y_coords):
            # 确保坐标在有效范围内（防止越界）
            if 0 <= x < W and 0 <= y < H:
                draw_tri.point((x, y), fill=color)

    # 保存三角形图片
    img_tri.save(save_path_tri)
    print(f"三角形可视化图片已保存至: {save_path_tri}")

    # -------------------------- 2. 可视化边像素 --------------------------
    # 创建空白RGB图片（白色背景）
    img_edges = Image.new('RGB', (W, H), color=(255, 255, 255))
    draw_edges = ImageDraw.Draw(img_edges)

    # 为每条边生成随机颜色
    edge_colors = [
        (0, 0, 255)
        for _ in range(num_edges)
    ]

    # 遍历每条边，绘制边像素
    for edge_id, edge in enumerate(tri_infos['edges']):
        edge_pixels = edge['edge_pixels']  # list of (x, y)
        color = edge_colors[edge_id]
        for (x, y) in edge_pixels:
            # 确保坐标在有效范围内
            if 0 <= x < W and 0 <= y < H:
                draw_edges.point((x, y), fill=color)

    # 保存边图片
    img_edges.save(save_path_edges)
    print(f"边可视化图片已保存至: {save_path_edges}")


# 简单版：把 tri_infos 的边像素画成图片并直接上传到 TensorBoard
def visualize_edges_to_tb(tri_infos_list, writer, global_step, tag_prefix='Edges'):
    """
    简单：把每个 sample 的 edges (edge_pixels) 画到白底图上并写入 TensorBoard
    - tri_infos_list: list, batch 的 tri_infos（如果是单个 dict，也可传 [tri_infos]）
      每个 tri_infos 需包含 'tri_masks'（用于推 H,W）和 'edges'（edge dict 包含 'edge_pixels' 列表）
      edge_pixels 格式按你原先：[(x,y), ...]
    - writer: torch.utils.tensorboard.SummaryWriter 实例
    - global_step: int
    - tag_prefix: tensorboard 的标签前缀
    """
    if not isinstance(tri_infos_list, list):
        tri_infos_list = [tri_infos_list]

    for b, tri_infos in enumerate(tri_infos_list):
        # 从 tri_infos 获得尺寸（按你原代码）
        H = tri_infos['tri_masks'].shape[1]
        W = tri_infos['tri_masks'].shape[2]

        # 白底 RGB 图
        img = Image.new('RGB', (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        edges = tri_infos.get('edges', [])
        # 固定颜色：蓝色 (B,G,R)
        color = (0, 0, 255)

        for edge in edges:
            # 支持 edge 为 dict 或 tuple；优先取 edge_pixels 字段
            if isinstance(edge, dict):
                pixs = edge.get('edge_pixels', None)
            else:
                pixs = None
            if not pixs:
                continue
            for (x, y) in pixs:
                # 简单边界检查，避免越界报错
                if 0 <= x < W and 0 <= y < H:
                    draw.point((x, y), fill=color)

        # 转 numpy HWC uint8，然后写到 tensorboard（dataformats='HWC'）
        np_img = np.array(img)  # shape (H, W, 3), dtype uint8
        writer.add_image(f"{tag_prefix}/edges_{b}", np_img, global_step, dataformats='HWC')

    # 不返回复杂内容，按你要求只上传到 tensorboard
    return


def batch_convert_to_tri_infos(vertexs_batch, lines_batch, triangles_batch, H, W, device,scale_ratio=0.5):
    """
    批量转换多个样本（适配顶点坐标为(x, y)格式）

    参数:
        vertexs_batch: list of tensor, 每个元素为单个样本的顶点（已to(device)，形状(Nv, 2)，(x, y)）
        lines_batch: list of tensor, 每个元素为单个样本的边（已to(device)，形状(Ne, 2)，(v1_id, v2_id)）
        triangles_batch: list of list of tuple, 每个元素为单个样本的三角形数据:
                        每个三角形是 (v_ids, l_ids, pts)，其中：
                        - v_ids: tensor (3,) 顶点ID
                        - l_ids: tensor (3,) 边ID
                        - pts: tensor (M, 2) 有效像素坐标 (x, y)
        H, W: 图像高度和宽度
        device: 计算设备

    返回:
        tri_infos_batch: list of dict, 每个元素为单个样本的tri_infos
    """
    tri_infos_batch = []
    for b in range(len(vertexs_batch)):
        # 1. 提取单个样本数据（从tensor转回numpy处理坐标）
        vertexs = vertexs_batch[b].cpu().numpy()  # (Nv, 2)，(x, y)
        lines = lines_batch[b]  # tensor (Ne,4)，每行[ p1,p2,face1,face2 ]

        # 2. 转换triangles_batch格式为convert_to_tri_infos所需的list of dict
        triangles = []
        for tri in triangles_batch[b]:
            v_ids = tri[0].cpu().numpy()  # 顶点ID：(3,)
            l_ids = tri[1].cpu().numpy()  # 边ID：(3,)
            valid_points = tri[2].cpu().numpy()  # 有效像素：(M, 2)，(x, y)
            triangles.append({
                'vertex_ids': v_ids,
                'line_ids': l_ids,
                'valid_points': valid_points
            })

        # 3. 转换为tri_infos格式,并且进行一个缩放
        tri_info = convert_to_tri_infos(vertexs, lines, triangles, H, W, device)
        tri_infos_batch.append(tri_info)

        # 4. 可视化并保存，正常
        visualize_triangles_and_edges(
            tri_info,
            save_path_tri="triangles_visual_tensor.png",
            save_path_edges="edges_visual_tensor.png"
        )

    return tri_infos_batch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_edge_alpha_overlays(ref_imgs, edge_alphas_list, tri_infos_list,
                                 out_key='ref_img_edge_alpha', image_outputs=None,
                                 overlay_alpha=0.6, cmap='jet', device=None):
    """
    生成 edge alpha 的 heatmap overlay 并写入 image_outputs[out_key]
    - ref_imgs: torch.Tensor, shape [B, C, H, W] or [B, H, W] (C can be 1 or 3)
                值可以是 0..1 或 0..255（函数会自动归一化到 0..1）
    - edge_alphas_list: list length B, each is tensor or ndarray of shape [E] (alpha in 0..1)
    - tri_infos_list: list length B, 每项 dict 至少包含 'edges'（list），
                      优先使用 edges[i]['edge_pixels']（list of (y,x)）
                      退回方案：使用 tri_infos['tri_masks']（tensor [N,H,W] 或 list）
    - out_key: 保存到 image_outputs 的键名
    - image_outputs: dict （若为 None 会创建一个新的 dict 并返回）
    - overlay_alpha: heatmap 覆盖原图时的透明度（0~1）
    - cmap: matplotlib colormap 名称
    - device: torch device 用于返回 tensor，默认取 ref_imgs.device
    返回:
      image_outputs (dict) ，并把 overlay tensor 存在 image_outputs[out_key]
      overlay shape: torch.FloatTensor [B, 3, H, W], 值范围 0..1
    """

    if image_outputs is None:
        image_outputs = {}

    if device is None:
        device = ref_imgs.device if isinstance(ref_imgs, torch.Tensor) else torch.device('cpu')

    # 规范 ref_imgs 到 numpy float 0..1，保留 batch
    if isinstance(ref_imgs, torch.Tensor):
        imgs = ref_imgs.detach().cpu()
    else:
        imgs = torch.from_numpy(np.array(ref_imgs))

    # imgs shape handling
    # 如果是 [B, H, W] -> 变为 [B,1,H,W]
    if imgs.dim() == 3:
        imgs = imgs.unsqueeze(1)
    B, C, H, W = imgs.shape

    # 归一化图片到 0..1 float
    imgs_np = imgs.clone().float()
    if imgs_np.max() > 1.1:
        imgs_np = imgs_np / 255.0
    imgs_np = imgs_np.numpy()  # numpy for blending with matplotlib cmap output

    # 结果容器
    overlays = np.zeros((B, 3, H, W), dtype=np.float32)

    cmap_func = plt.get_cmap(cmap)

    for b in range(B):
        # 1) 生成空 alpha_map (H,W)
        alpha_map = np.zeros((H, W), dtype=np.float32)

        # 得到 edge_alphas (E,)
        alphas = edge_alphas_list[b]
        if isinstance(alphas, torch.Tensor):
            alphas = alphas.detach().cpu().numpy()
        alphas = np.asarray(alphas).astype(np.float32)

        tri_infos = tri_infos_list[b]

        # 2) 优先使用 edges[].edge_pixels（每条边的像素列表）
        edges_data = tri_infos.get('edges', None)
        used_pixels = False
        if edges_data is not None and len(edges_data) > 0:
            # 判断第一个 edge entry 是否为 dict 且包含 'edge_pixels'
            if isinstance(edges_data[0], dict) and ('edge_pixels' in edges_data[0]):
                used_pixels = True
                for ei, e in enumerate(edges_data):
                    pixs = e.get('edge_pixels', None)
                    if not pixs:
                        continue
                    arr = np.array(pixs, dtype=np.int32)
                    if arr.size == 0:
                        continue
                    xs = arr[:, 0]; ys = arr[:, 1]
                    a = float(alphas[ei]) if ei < len(alphas) else 0.0
                    # 若多个边写到同一像素，取最大值，避免覆盖掉强 alpha
                    alpha_map[ys, xs] = np.maximum(alpha_map[ys, xs], a)

        # 3) 回退：若没有 edge_pixels，则使用 tri_masks 找跨三角边像素
        if not used_pixels:
            # tri_masks 可以是 tensor [N,H,W] 或 list of masks
            tri_masks = tri_infos.get('tri_masks', None)
            if tri_masks is None:
                raise ValueError("tri_infos 中既没有 'edge_pixels' 也没有 'tri_masks'，无法构建 alpha 映射")
            # 把 tri_masks 转为 numpy [N,H,W]
            if isinstance(tri_masks, list):
                tri_masks_np = np.stack([m.astype(np.int8) if isinstance(m, np.ndarray) else m.numpy().astype(np.int8)
                                         for m in tri_masks], axis=0)
            else:
                # 可能为 torch tensor
                if isinstance(tri_masks, torch.Tensor):
                    tri_masks_np = tri_masks.detach().cpu().numpy()
                else:
                    tri_masks_np = np.array(tri_masks)
            # tri_id_map via argmax (若有重叠，argmax 取第一个最大的)
            if tri_masks_np.shape[0] == 0:
                # 没有三角，保持 alpha_map 全 0
                pass
            else:
                tri_id_map = np.argmax(tri_masks_np, axis=0).astype(np.int32)  # [H,W]
                # edge list: tri_infos['edges'] 给出 tri_ids 对应的顺序，我们按 edges 列表索引获取 alpha
                # 构造一个 map from (t1,t2) -> alpha value
                pair_to_alpha = {}
                for ei, e in enumerate(edges_data):
                    if isinstance(e, dict):
                        tid = e.get('tri_ids', None)
                        if tid is None:
                            continue
                        t1, t2 = int(tid[0]), int(tid[1])
                    else:
                        t1, t2 = int(e[0]), int(e[1])
                    # 规范 (min,max) 以便查找
                    pair_to_alpha[(t1, t2)] = float(alphas[ei]) if ei < len(alphas) else 0.0
                    pair_to_alpha[(t2, t1)] = pair_to_alpha[(t1, t2)]
                # 找右邻和下邻跨三角像素，将对应 alpha 填到 alpha_map
                left = tri_id_map[:, :-1]; right = tri_id_map[:, 1:]
                diff_mask_r = (left != right)
                if diff_mask_r.any():
                    ys, xs = np.where(diff_mask_r)
                    t_left = left[ys, xs]; t_right = right[ys, xs]
                    for y, x, tl, tr in zip(ys, xs, t_left, t_right):
                        alpha_map[y, x] = max(alpha_map[y, x], pair_to_alpha.get((int(tl), int(tr)), 0.0))
                up = tri_id_map[:-1, :]; down = tri_id_map[1:, :]
                diff_mask_d = (up != down)
                if diff_mask_d.any():
                    ys, xs = np.where(diff_mask_d)
                    t_up = up[ys, xs]; t_down = down[ys, xs]
                    for y, x, tu, td in zip(ys, xs, t_up, t_down):
                        alpha_map[y, x] = max(alpha_map[y, x], pair_to_alpha.get((int(tu), int(td)), 0.0))

        # 4) alpha_map 现在是 coarse/full-res 对应的像素位置（假设与 ref_img 分辨率一致）
        #    将 alpha_map 限制到 [0,1]
        alpha_map = np.clip(alpha_map, 0.0, 1.0)

        # 5) 用 colormap 映射 alpha_map -> RGB (float 0..1)
        heatmap = cmap_func(alpha_map)[:, :, :3]  # shape [H,W,3]

        # 6) 获取背景图（ref image）并确保为 float 0..1
        bg = imgs_np[b]  # shape [C,H,W]
        # 转为 H,W,3
        if bg.shape[0] == 1:
            bg_rgb = np.stack([bg[0]]*3, axis=2)  # [H,W,3]
        else:
            bg_rgb = bg.transpose(1,2,0)  # [H,W,3]
        if bg_rgb.max() > 1.1:
            bg_rgb = bg_rgb / 255.0

        # 7) overlay
        overlay = bg_rgb * (1.0 - overlay_alpha) + heatmap * overlay_alpha
        overlay = np.clip(overlay, 0.0, 1.0)

        # 8) 存回 overlays 容器（CHW）
        overlays[b] = overlay.transpose(2,0,1)

    # 转回 torch tensor [B,3,H,W] float 0..1，在指定 device 上
    overlays_t = torch.from_numpy(overlays).float().to(device)

    # 写入 image_outputs
    image_outputs[out_key] = overlays_t

    return image_outputs



def visualize_edges_with_alpha(tri_infos_list, edge_alphas_list,
                               ref_imgs=None,    # optional: torch.Tensor [B,C,H,W] or None
                               overlay_alpha=0.6,
                               cmap_name='RdYlGn_r',
                               device=None,
                               verbose=False):
    """
    用颜色（绿->黄->红）可视化每条边的断裂概率 alpha（越红表示越“断裂”）。
    - tri_infos_list: list length B，每项为 tri_infos dict（含 'num_tri','tri_masks' 或 'edges'）
    - edge_alphas_list: list length B，每项 tensor or ndarray shape [E_b]（alpha in [0,1]）
    - ref_imgs: optional 原图 tensor [B, C, H, W]（C 1 或 3），若提供则返回 overlay tensor 也会被归一化到 0..1
    - out_key: 用于写回 image_outputs 的键名（如果你集成到 image_outputs）
    - overlay_alpha: 覆盖透明度（heatmap 覆盖到原图时）
    - device: 返回 overlay tensor 的 device（默认与 ref_imgs 相同或 cpu）
    - 返回: (pil_images, overlay_tensor or None)
        - pil_images: list 长度 B，每项 PIL.Image RGB
        - overlay_tensor: 若 ref_imgs 提供，返回 torch.FloatTensor [B,3,H,W], 值 0..1；否则 None
    """
    assert isinstance(tri_infos_list, list), "tri_infos_list must be a list (batch)"
    B = len(tri_infos_list)
    cmap = plt.get_cmap(cmap_name)

    # 处理 ref_imgs 和尺寸信息
    if ref_imgs is not None:
        if isinstance(ref_imgs, torch.Tensor):
            imgs_t = ref_imgs.detach().cpu()
        else:
            imgs_t = torch.from_numpy(np.array(ref_imgs))
        if imgs_t.dim() == 3:   # [H,W] -> [1,1,H,W]
            imgs_t = imgs_t.unsqueeze(0).unsqueeze(1)
        if imgs_t.dim() == 4 and imgs_t.shape[1] in (1,3):
            pass
        else:
            raise ValueError("ref_imgs must be [B,C,H,W] with C=1 or 3 or None")
        B_img, C, H, W = imgs_t.shape
        assert B_img == B, "ref_imgs batch size must match tri_infos_list length"
        # 规范到 0..1 float numpy
        imgs_np = imgs_t.clone().float().numpy()
        if imgs_np.max() > 1.1:
            imgs_np = imgs_np / 255.0
    else:
        # 没有 ref_imgs 时，从 tri_infos 中尝试读取 tri_masks 的尺寸作为 H,W
        # 若无法获得尺寸则报错
        sample_info = tri_infos_list[0]
        tm = sample_info.get('tri_masks', None)
        if tm is None:
            raise ValueError("没有传入 ref_imgs，且 tri_infos 中也没有 tri_masks 可推断尺寸，请传入 ref_imgs 或 tri_masks")
        if isinstance(tm, list):
            H, W = tm[0].shape
        else:
            H, W = tm.shape[1], tm.shape[2]
        imgs_np = None

    # 结果容器
    pil_images = []
    overlays_tensor = None
    if ref_imgs is not None:
        overlays = np.zeros((B, 3, H, W), dtype=np.float32)

    # 主循环：生成每张图的 pixel-level alpha_map（每像素取覆盖到该像素的最大 alpha）
    for b in range(B):
        tri_infos = tri_infos_list[b]
        alphas = edge_alphas_list[b]
        if isinstance(alphas, torch.Tensor):
            alphas = alphas.detach().cpu().numpy()
        alphas = np.asarray(alphas).astype(np.float32)
        H_local = H; W_local = W

        # 初始化 per-pixel alpha map（0..1）
        alpha_map = np.zeros((H_local, W_local), dtype=np.float32)
        # 如果你想也保留每像素对应的 edge id，可建立一个 int map（-1 表示无边）
        # edge_id_map = -1 * np.ones((H_local, W_local), dtype=np.int32)

        edges_data = tri_infos.get('edges', None)
        used_pixels = False
        if edges_data is not None and len(edges_data) > 0 and isinstance(edges_data[0], dict) and ('edge_pixels' in edges_data[0]):
            used_pixels = True
            # 逐边写入（以 numpy vector 化方式尽量加速）
            for ei, e in enumerate(edges_data):
                pixs = e.get('edge_pixels', None)
                if not pixs:
                    continue
                arr = np.array(pixs, dtype=np.int64)  # shape (K,2) 可能为 (x,y) 或 (y,x)
                if arr.ndim != 2 or arr.shape[1] < 2:
                    continue
                xs = arr[:, 0].copy()
                ys = arr[:, 1].copy()
                # 自动检测并修正 (x,y) vs (y,x)
                # 若 xs.max() > W-1 but ys.max() <= W-1 and ys.max() <= H-1 => 很可能原来是 (y,x)，交换
                if (xs.max() >= W_local or ys.max() >= H_local) and not (ys.max() >= W_local or xs.max() >= H_local):
                    # 交换
                    xs, ys = ys, xs
                # clip 防止越界
                xs = np.clip(xs, 0, W_local - 1)
                ys = np.clip(ys, 0, H_local - 1)
                if xs.size == 0:
                    continue
                a = float(alphas[ei]) if ei < len(alphas) else 0.0
                # 多个边覆盖同一像素时取最大 alpha（突出最严重）
                # 由于可能存在重复像素索引，我们先以一维索引方式处理以减少 Python 循环
                idx_1d = ys * W_local + xs
                # current values
                cur_vals = alpha_map.reshape(-1)
                # 用最大值赋值： cur[idx] = max(cur[idx], a)
                # 为减少 Python 循环，使用 numpy.maximum.at
                np.maximum.at(cur_vals, idx_1d, a)
                alpha_map = cur_vals.reshape(H_local, W_local)
        else:
            # 退回：使用 tri_masks -> tri_id_map -> 边对映射
            tri_masks = tri_infos.get('tri_masks', None)
            if tri_masks is None:
                # 无 edge_pixels 且无 tri_masks，跳过
                if verbose:
                    print(f"[visualize_edges] tri_infos[{b}] 没有 'edge_pixels' 也没有 'tri_masks'，跳过该样本")
                pil_images.append(Image.new('RGB', (W_local, H_local), color=(255,255,255)))
                if ref_imgs is not None:
                    overlays[b] = np.stack([np.zeros((H_local,W_local))]*3, axis=0)
                continue
            # 把 tri_masks 变为 numpy [N,H,W]
            if isinstance(tri_masks, list):
                tri_masks_np = np.stack([m.astype(np.uint8) if isinstance(m, np.ndarray) else m.numpy().astype(np.uint8)
                                         for m in tri_masks], axis=0)
            else:
                tri_masks_np = tri_masks.detach().cpu().numpy()
            if tri_masks_np.shape[0] == 0:
                # 没有三角
                pil_images.append(Image.new('RGB', (W_local, H_local), color=(255,255,255)))
                if ref_imgs is not None:
                    overlays[b] = np.stack([np.zeros((H_local,W_local))]*3, axis=0)
                continue
            tri_id_map = np.argmax(tri_masks_np, axis=0).astype(np.int32)  # [H,W]
            # 构建 (t1,t2)->alpha 的映射（考虑双向）
            pair_to_alpha = {}
            for ei, e in enumerate(edges_data):
                if isinstance(e, dict):
                    tid = e.get('tri_ids', None)
                    if tid is None:
                        continue
                    t1, t2 = int(tid[0]), int(tid[1])
                else:
                    t1, t2 = int(e[0]), int(e[1])
                a = float(alphas[ei]) if ei < len(alphas) else 0.0
                pair_to_alpha[(t1, t2)] = a
                pair_to_alpha[(t2, t1)] = a
            # 找横向和纵向邻接不等处
            left = tri_id_map[:, :-1]; right = tri_id_map[:, 1:]
            diff_r = (left != right)
            if diff_r.any():
                ys, xs = np.where(diff_r)
                for y, x in zip(ys, xs):
                    tL = int(left[y, x]); tR = int(right[y, x])
                    a = pair_to_alpha.get((tL, tR), 0.0)
                    alpha_map[y, x] = max(alpha_map[y, x], a)
            up = tri_id_map[:-1, :]; down = tri_id_map[1:, :]
            diff_d = (up != down)
            if diff_d.any():
                ys, xs = np.where(diff_d)
                for y, x in zip(ys, xs):
                    tU = int(up[y, x]); tD = int(down[y, x])
                    a = pair_to_alpha.get((tU, tD), 0.0)
                    alpha_map[y, x] = max(alpha_map[y, x], a)

        # clamp
        alpha_map = np.clip(alpha_map, 0.0, 1.0)

        # 用 colormap 将 alpha_map -> RGB (float 0..1)
        heat_rgb = cmap(alpha_map)[:, :, :3]  # shape [H,W,3] float 0..1

        if ref_imgs is not None:
            bg = imgs_np[b]
            if bg.shape[0] == 1:
                bg_rgb = np.stack([bg[0]]*3, axis=2)  # [H,W,3]
            else:
                bg_rgb = bg.transpose(1,2,0)  # [H,W,3]
            if bg_rgb.max() > 1.1:
                bg_rgb = bg_rgb / 255.0
            overlay = np.clip(bg_rgb * (1.0 - overlay_alpha) + heat_rgb * overlay_alpha, 0.0, 1.0)
            overlays[b] = overlay.transpose(2,0,1)  # CHW
            pil_img = Image.fromarray((overlay * 255).astype(np.uint8))
        else:
            # 仅显示 heat map
            pil_img = Image.fromarray((heat_rgb * 255).astype(np.uint8))

        pil_images.append(pil_img)

    # 输出 overlay tensor
    overlay_tensor = None
    if ref_imgs is not None:
        overlay_tensor = torch.from_numpy(overlays).float().to(device if device is not None else ref_imgs.device)

    return pil_images, overlay_tensor


def normalized_loss_fusion(loss_depth, loss_alpha_sup, alpha_weight=3.0, beta_weight=1.0, eps=1e-8):
    """
    归一化损失融合：先消除量级差异，再按 alpha:beta = 3:1 加权
    Args:
        loss_depth: 深度损失（范围大的损失）
        loss_alpha_sup: alpha监督损失（范围小的损失）
        alpha_weight: 深度损失权重（默认3）
        beta_weight: alpha损失权重（默认1）
        eps: 防止分母为0的微小值
    Returns:
        total_loss: 归一化后融合的总损失
    """
    # 1. 动态归一化：用批次内的“均值+标准差”标准化（适配损失范围动态变化）
    # 若损失是标量（单值），直接用自身做归一化；若为批量tensor，用批次统计
    if loss_depth.dim() > 0:  # 批量tensor（如[B,]）
        depth_mean = loss_depth.mean()
        depth_std = loss_depth.std() + eps
        loss_depth_norm = (loss_depth - depth_mean) / depth_std
    else:  # 标量tensor
        loss_depth_norm = loss_depth / (loss_depth.abs() + eps)  # 归一到[-1,1]附近

    if loss_alpha_sup.dim() > 0:
        alpha_mean = loss_alpha_sup.mean()
        alpha_std = loss_alpha_sup.std() + eps
        loss_alpha_norm = (loss_alpha_sup - alpha_mean) / alpha_std
    else:
        loss_alpha_norm = loss_alpha_sup / (loss_alpha_sup.abs() + eps)

    # 2. 按 3:1 权重融合（此时两者量级一致，权重直接对应占比）

    return alpha_weight * loss_depth_norm,alpha_weight * loss_depth_norm