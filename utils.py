import os

import cv2
import numpy as np
import torchvision.utils as vutils
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import random

from tensorboard.plugins.hparams.metadata import NULL_TENSOR


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
def tensor2numpy(obj):
    """
    递归地将 Tensor 转换为 Numpy，同时保留 int, float, list, dict 等结构。
    """
    if isinstance(obj, dict):
        # 递归处理字典的值
        return {k: tensor2numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # 递归处理列表/元组的元素
        return [tensor2numpy(v) for v in obj]
    elif isinstance(obj, torch.Tensor):
        # 核心：Tensor -> Numpy
        return obj.detach().cpu().numpy()
    elif isinstance(obj, np.ndarray):
        # 已经是 Numpy，直接返回
        return obj
    elif isinstance(obj, (int, float, str, bool, type(None))):
        # 基础类型，直接返回 (解决了你的 'int' 报错)
        return obj
    else:
        # 其他未知类型，原样返回，防止报错
        return obj


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

#====================ym-add==================================

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

def norm_pixel_coords(pixels, old_W, old_H):
    """
    归一化像素坐标
    """

    scaled_pixels = []
    for (x, y) in pixels:
        # 归一化具体操作
        x_norm = (x / max(old_W - 1, 1)) * 2.0 - 1.0
        y_norm = (y / max(old_H - 1, 1)) * 2.0 - 1.0
        scaled_pixels.append((x_norm, y_norm))

    return scaled_pixels


def convert_to_tri_infos_normal_new(vertexs, lines, triangles, H, W, device, scale_ratio=1.0):
    """
    适配数组/张量格式的lines，生成三角形的【原始】顶点、质心，以及缩放/归一化后的边信息。
    优化：移除了对顶点和质心的缩放操作，直接返回原始坐标。
    参数:
           vertexs: np.ndarray, 形状 (Nv, 2)，顶点坐标 (x, y)（x=列，y=行）
           lines: np.ndarray/torch.Tensor, 形状 (Ne, 4)，每行对应[ p1, p2, face1, face2 ]
                  - 若为torch.Tensor，会自动转numpy处理
           triangles: list, 每个元素为dict:
               {
                   'vertex_ids': np.ndarray (3,), 三角形顶点ID
                   'line_ids': np.ndarray (3,), 三角形边ID
                   'valid_points': np.ndarray (M, 2), 三角形内像素 (x, y)
               }
           H/W: 原始图像尺寸（高/宽）
           device: 计算设备（此处仅为兼容参数，实际未使用）
           scale_ratio: 缩放比例（默认0.5）

       返回:
           tri_infos: dict
               {
                   'num_tri': 三角形数量,
                   'centroids': list of np.ndarray, 每个元素为(2,)，三角形缩放后的质心坐标 (x, y)
                   'vertices': list of np.ndarray, 每个元素为(3,2)，三角形缩放后的三个顶点坐标
                   'edges': list of dict, 每条边的缩放后信息（含tri_ids和edge_pixels）
               }
    """
    tri_infos = {
        'num_tri': len(triangles),
        'centroids': [],
        'vertices': [],
        'edges': []
    }

    # -------------------------- 1. 获取原始顶点和质心 (不缩放) --------------------------
    # 预先将所有顶点转换为 float32 方便计算
    # vertexs 是 (Nv, 2)
    # 不进行缩放，因为后面统一归一化
    if(scale_ratio==1.0):
        for tri in triangles:
            # 1.1 获取三角形原始顶点
            vertex_ids = tri['vertex_ids']  # (3,)
            tri_vertices_original = vertexs[vertex_ids].astype(np.float32)  # (3, 2) 原始顶点坐标 (x, y)

            # 1.2 计算原始质心 todo：质心之后后续不需要了，需要改成边中点的区域
            centroid_original = np.mean(tri_vertices_original, axis=0)  # (2,)

            # 1.3 直接存入，不做 scale_ratio 处理
            # 统一归一化会在 batch_convert_to_tri_infos 中进行
            tri_infos['vertices'].append(tri_vertices_original)
            tri_infos['centroids'].append(centroid_original)
    else:
        new_H = int(H * scale_ratio)
        new_W = int(W * scale_ratio)

        scale_x = scale_ratio  # x方向缩放因子
        scale_y = scale_ratio  # y方向缩放因子

        for tri in triangles:
            # 1.1 获取三角形的三个顶点ID对应的原始坐标
            vertex_ids = tri['vertex_ids']  # (3,) 顶点ID数组
            tri_vertices_original = vertexs[vertex_ids]  # (3, 2) 原始顶点坐标 (x, y)

            # 1.2 缩放顶点坐标（与边像素缩放逻辑一致）
            tri_vertices_scaled = np.zeros_like(tri_vertices_original, dtype=np.float32)
            tri_vertices_scaled[:, 0] = tri_vertices_original[:, 0] * scale_x  # x坐标缩放
            tri_vertices_scaled[:, 1] = tri_vertices_original[:, 1] * scale_y  # y坐标缩放
            # 可选：裁剪到目标尺寸范围内（防止越界）
            # tri_vertices_scaled[:, 0] = np.clip(tri_vertices_scaled[:, 0], 0, new_W - 1)
            # tri_vertices_scaled[:, 1] = np.clip(tri_vertices_scaled[:, 1], 0, new_H - 1)

            # 1.3 计算三角形质心（质点）：三个顶点坐标的平均值
            centroid_original = np.mean(tri_vertices_original, axis=0)  # (2,) 原始质心
            # 缩放质心坐标
            centroid_scaled = np.array([
                centroid_original[0] * scale_x,
                centroid_original[1] * scale_y
            ], dtype=np.float32)
            # 可选：裁剪质心坐标
            # centroid_scaled[0] = np.clip(centroid_scaled[0], 0, new_W - 1)
            # centroid_scaled[1] = np.clip(centroid_scaled[1], 0, new_H - 1)

            # 1.4 存入tri_infos
            tri_infos['vertices'].append(tri_vertices_scaled)
            tri_infos['centroids'].append(centroid_scaled)

    # -------------------------- 2. 处理边信息 --------------------------
    # 注意：边像素的处理通常依赖于 bresenham 和 norm_pixel_coords
    # 如果 norm_pixel_coords 内部依赖 H/W 进行归一化，这里保持不变

    if isinstance(lines, torch.Tensor):
        lines_np = lines.cpu().numpy()
    else:
        lines_np = lines

    for line_arr in lines_np:
        v1_id = int(line_arr[0]) # 索引0 = p1
        v2_id = int(line_arr[1]) # 索引1 = p2
        face1 = int(line_arr[2]) # 索引2 = face1
        face2 = int(line_arr[3]) # 索引3 = face2

        #  处理face为无效值的情况（如None转成的-1或0）
        tri_ids = []
        # 简化逻辑：-1 或 None 都视为无效,假设-1表示无相邻面
        tri_ids.append(face1 if (face1 != -1 and face1 is not None) else -1)
        tri_ids.append(face2 if (face2 != -1 and face2 is not None) else -1)
        tri_ids = tuple(tri_ids)

        # 计算边的像素坐标 (使用原始顶点坐标进行 Bresenham)
        # 注意：这里得到的是原始分辨率下的像素点
        p1 = (vertexs[v1_id][0], vertexs[v1_id][1])
        p2 = (vertexs[v2_id][0], vertexs[v2_id][1])

        # 假设 bresenham_line 返回的是 list of (x,y)
        edge_pixels_original = bresenham_line(p1, p2)

        # 归一化像素坐标 (依赖外部函数 norm_pixel_coords)
        # 这里的 H, W 是原图尺寸，norm_pixel_coords 应该将其映射到特定区间(如 [-1,1] 或 [0,1])
        # 这部分保持你原有的逻辑，因为它通常用于 EdgeHead 的 grid_sample
        edge_pixels_scaled = norm_pixel_coords(edge_pixels_original, W, H)

        # 添加边信息
        tri_infos['edges'].append({
            'tri_ids': tri_ids,
            'edge_pixels': edge_pixels_scaled,
            'endpoints': [p1, p2]  # 边的端点
        })

    return tri_infos

def convert_to_tri_infos_normal(vertexs, lines, triangles, H, W, device, scale_ratio=0.5):
    """
       适配数组/张量格式的lines，生成三角形的缩放后顶点、质心，以及缩放后的边信息

       参数:
           vertexs: np.ndarray, 形状 (Nv, 2)，顶点坐标 (x, y)（x=列，y=行）
           lines: np.ndarray/torch.Tensor, 形状 (Ne, 4)，每行对应[ p1, p2, face1, face2 ]
                  - 若为torch.Tensor，会自动转numpy处理
           triangles: list, 每个元素为dict:
               {
                   'vertex_ids': np.ndarray (3,), 三角形顶点ID
                   'line_ids': np.ndarray (3,), 三角形边ID
                   'valid_points': np.ndarray (M, 2), 三角形内像素 (x, y)
               }
           H/W: 原始图像尺寸（高/宽）
           device: 计算设备（此处仅为兼容参数，实际未使用）
           scale_ratio: 缩放比例（默认0.5）

       返回:
           tri_infos: dict
               {
                   'num_tri': 三角形数量,
                   'centroids': list of np.ndarray, 每个元素为(2,)，三角形缩放后的质心坐标 (x, y)
                   'vertices': list of np.ndarray, 每个元素为(3,2)，三角形缩放后的三个顶点坐标
                   'edges': list of dict, 每条边的缩放后信息（含tri_ids和edge_pixels）
               }
       """
    tri_infos = {
        'num_tri': len(triangles),
        'centroids':[],
        'vertices':[],
        'edges': []
    }
    new_H = int(H * scale_ratio)
    new_W = int(W * scale_ratio)

    scale_x = scale_ratio       # x方向缩放因子
    scale_y = scale_ratio       # y方向缩放因子

    # -------------------------- 1. 每个三角形的顶点和和三角形的质点，并进行一个缩放 --------------------------


    for tri in triangles:
        # 1.1 获取三角形的三个顶点ID对应的原始坐标
        vertex_ids = tri['vertex_ids']  # (3,) 顶点ID数组
        tri_vertices_original = vertexs[vertex_ids]  # (3, 2) 原始顶点坐标 (x, y)

        # 1.2 缩放顶点坐标（与边像素缩放逻辑一致）
        tri_vertices_scaled = np.zeros_like(tri_vertices_original, dtype=np.float32)
        tri_vertices_scaled[:, 0] = tri_vertices_original[:, 0] * scale_x  # x坐标缩放
        tri_vertices_scaled[:, 1] = tri_vertices_original[:, 1] * scale_y  # y坐标缩放
        # 可选：裁剪到目标尺寸范围内（防止越界）
        # tri_vertices_scaled[:, 0] = np.clip(tri_vertices_scaled[:, 0], 0, new_W - 1)
        # tri_vertices_scaled[:, 1] = np.clip(tri_vertices_scaled[:, 1], 0, new_H - 1)

        # 1.3 计算三角形质心（质点）：三个顶点坐标的平均值
        centroid_original = np.mean(tri_vertices_original, axis=0)  # (2,) 原始质心
        # 缩放质心坐标
        centroid_scaled = np.array([
            centroid_original[0] * scale_x,
            centroid_original[1] * scale_y
        ], dtype=np.float32)
        # 可选：裁剪质心坐标
        # centroid_scaled[0] = np.clip(centroid_scaled[0], 0, new_W - 1)
        # centroid_scaled[1] = np.clip(centroid_scaled[1], 0, new_H - 1)

        # 1.4 存入tri_infos
        tri_infos['vertices'].append(tri_vertices_scaled)
        tri_infos['centroids'].append(centroid_scaled)


    # -------------------------- 2. 处理边信息（按索引取值）缩放--------------------------
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

        # 归一化像素坐标供后续处理
        edge_pixels_scaled = norm_pixel_coords(edge_pixels_original, W, H)

        del edge_pixels_original
        # 添加边信息
        tri_infos['edges'].append({
            'tri_ids': tri_ids,
            'edge_pixels': edge_pixels_scaled,  # 存储缩放后的像素
        })

    return tri_infos

def convert_to_tri_infos(vertexs, lines, triangles, H, W, device, scale_ratio=0.5):
    """
       适配数组/张量格式的lines，生成三角形的缩放后顶点、质心，以及缩放后的边信息

       参数:
           vertexs: np.ndarray, 形状 (Nv, 2)，顶点坐标 (x, y)（x=列，y=行）
           lines: np.ndarray/torch.Tensor, 形状 (Ne, 4)，每行对应[ p1, p2, face1, face2 ]
                  - 若为torch.Tensor，会自动转numpy处理
           triangles: list, 每个元素为dict:
               {
                   'vertex_ids': np.ndarray (3,), 三角形顶点ID
                   'line_ids': np.ndarray (3,), 三角形边ID
                   'valid_points': np.ndarray (M, 2), 三角形内像素 (x, y)
               }
           H/W: 原始图像尺寸（高/宽）
           device: 计算设备（此处仅为兼容参数，实际未使用）
           scale_ratio: 缩放比例（默认0.5）

       返回:
           tri_infos: dict
               {
                   'num_tri': 三角形数量,
                   'centroids': list of np.ndarray, 每个元素为(2,)，三角形缩放后的质心坐标 (x, y)
                   'vertices': list of np.ndarray, 每个元素为(3,2)，三角形缩放后的三个顶点坐标
                   'edges': list of dict, 每条边的缩放后信息（含tri_ids和edge_pixels）
               }
       """
    tri_infos = {
        'num_tri': len(triangles),
        'centroids':[],
        'vertices':[],
        'edges': []
    }
    new_H = int(H * scale_ratio)
    new_W = int(W * scale_ratio)

    scale_x = scale_ratio       # x方向缩放因子
    scale_y = scale_ratio       # y方向缩放因子

    # -------------------------- 1. 每个三角形的顶点和和三角形的质点，并进行一个缩放 --------------------------
    for tri in triangles:
        # 1.1 获取三角形的三个顶点ID对应的原始坐标
        vertex_ids = tri['vertex_ids']  # (3,) 顶点ID数组
        tri_vertices_original = vertexs[vertex_ids]  # (3, 2) 原始顶点坐标 (x, y)

        # 1.2 缩放顶点坐标（与边像素缩放逻辑一致）
        tri_vertices_scaled = np.zeros_like(tri_vertices_original, dtype=np.float32)
        tri_vertices_scaled[:, 0] = tri_vertices_original[:, 0] * scale_x  # x坐标缩放
        tri_vertices_scaled[:, 1] = tri_vertices_original[:, 1] * scale_y  # y坐标缩放
        # 可选：裁剪到目标尺寸范围内（防止越界）
        tri_vertices_scaled[:, 0] = np.clip(tri_vertices_scaled[:, 0], 0, new_W - 1)
        tri_vertices_scaled[:, 1] = np.clip(tri_vertices_scaled[:, 1], 0, new_H - 1)

        # 1.3 计算三角形质心（质点）：三个顶点坐标的平均值
        centroid_original = np.mean(tri_vertices_original, axis=0)  # (2,) 原始质心
        # 缩放质心坐标
        centroid_scaled = np.array([
            centroid_original[0] * scale_x,
            centroid_original[1] * scale_y
        ], dtype=np.float32)
        # 可选：裁剪质心坐标
        centroid_scaled[0] = np.clip(centroid_scaled[0], 0, new_W - 1)
        centroid_scaled[1] = np.clip(centroid_scaled[1], 0, new_H - 1)

        # 1.4 存入tri_infos
        tri_infos['vertices'].append(tri_vertices_scaled)
        tri_infos['centroids'].append(centroid_scaled)


    # -------------------------- 2. 处理边信息（按索引取值）缩放--------------------------
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

def visualize_centroids_and_edges(tri_infos, H, W, save_path):
    """
    将三角形质心和边像素绘制在同一张图上（支持手动传入H,W）

    参数:
        tri_infos: 转换得到的tri_infos字典（含centroids、edges）
        H: 图片高度（用户手动指定）
        W: 图片宽度（用户手动指定）
        save_path: 最终可视化图片保存路径（如"centroids_edges.png"）
    """
    # -------------------------- 创建画布 --------------------------
    # 创建白色背景的RGB图片（尺寸为W×H，对应图像的宽和高）
    img = Image.new('RGB', (W, H), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # -------------------------- 1. 绘制边像素（先画边，后画质心，质心更显眼） --------------------------
    edge_color = (0, 0, 255)  # 边用蓝色，可根据需要调整
    for edge in tri_infos['edges']:
        edge_pixels = edge['edge_pixels']
        for (x, y) in edge_pixels:
            # 确保坐标在用户指定的H,W范围内
            if 0 <= x < W and 0 <= y < H:
                draw.point((x, y), fill=edge_color)

    # -------------------------- 2. 绘制三角形质心（含标记和ID） --------------------------
    num_tri = tri_infos['num_tri']
    # 为每个质心生成唯一颜色（深色系，避免与蓝色边冲突）
    centroid_colors = [
        (random.randint(0, 100), random.randint(100, 200), random.randint(0, 100))  # 绿/红色系
        for _ in range(num_tri)
    ]

    for tri_id, centroid in enumerate(tri_infos['centroids']):
        cx, cy = int(centroid[0]), int(centroid[1])
        color = centroid_colors[tri_id]

        # 绘制质心标记：外圆（直径8像素）+ 实心点（增强辨识度）
        if 0 <= cx < W and 0 <= cy < H:
            draw.ellipse([cx - 1, cy - 1, cx + 1, cy + 1], outline=color, width=1)  # 外圆
            draw.point((cx, cy), fill=color)  # 中心点

    # -------------------------- 保存图片 --------------------------
    img.save(save_path)
    print(f"质心+边可视化图片已保存至: {save_path}")


def visualize_tri_id_map(tri_id_map, save_dir="outputs/debug_tri_maps", prefix="tri_id"):
    """
    将三角形 ID 索引图可视化为彩色分割图。
    相邻 ID 会被分配完全不同的随机颜色，以便于肉眼区分。

    Args:
        tri_id_map: [B, H, W] 或 [H, W] 的 Tensor (long/int).
                    值域: 0 ~ N-1 (三角形ID), -1 (无效/背景)
        save_dir: 图片保存目录
        prefix: 保存文件的前缀 (e.g., "batch0_stage1")
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 1. 确保输入是 CPU numpy 格式
    if isinstance(tri_id_map, torch.Tensor):
        tri_id_map = tri_id_map.detach().cpu().numpy()

    # 兼容 [H, W] 输入，自动扩展为 [1, H, W]
    if tri_id_map.ndim == 2:
        tri_id_map = tri_id_map[np.newaxis, ...]

    B, H, W = tri_id_map.shape

    # 2. 获取最大 ID 数，用于生成调色板
    # 注意：我们要忽略 -1
    valid_mask = (tri_id_map >= 0)
    if not valid_mask.any():
        print(f"Warning: {prefix} 中全是无效区域 (-1)，跳过可视化。")
        return

    max_id = tri_id_map.max()
    num_colors = max_id + 1

    # 3. 生成随机调色板 (N, 3)
    # 使用随机种子确保颜色在同一批次内是确定的，但不同 ID 颜色差异大
    np.random.seed(42)
    # 生成 0-255 的随机颜色
    colors = np.random.randint(0, 255, size=(num_colors, 3), dtype=np.uint8)

    # 4. 逐张处理
    for b in range(B):
        id_img = tri_id_map[b]  # [H, W]

        # 初始化一张全黑图片 [H, W, 3]
        vis_img = np.zeros((H, W, 3), dtype=np.uint8)

        # 获取当前图的有效 mask
        mask = (id_img >= 0)

        # --- 核心映射逻辑 ---
        # 利用 numpy 的高级索引，直接将 ID 映射为颜色
        # id_img[mask] 得到所有有效的 ID
        # colors[...] 得到对应的 RGB
        if mask.any():
            valid_ids = id_img[mask]
            # 这里的 valid_ids 必须是整数索引
            vis_img[mask] = colors[valid_ids]

        # 5. 保存
        save_path = os.path.join(save_dir, f"{prefix}_b{b}.png")
        cv2.imwrite(save_path, vis_img)
        print(f"✅ 已保存三角形索引可视化: {save_path}")


def visualize_downsampled_tri_id(tri_id_map, scale_factor=0.5, save_dir="outputs/debug_tri_maps",
                                 prefix="tri_id_downsampled"):
    """
    对三角形 ID 索引图进行下采样，并可视化保存。
    注意：ID图必须使用 'nearest' 插值，不能使用 bilinear。

    Args:
        tri_id_map (torch.Tensor): 原始分辨率的 ID 图 [B, H, W]
        scale_factor (float): 下采样倍率 (e.g., 0.5 表示 1/2 分辨率)
        save_dir (str): 保存路径
        prefix (str): 文件名前缀

    Returns:
        tri_id_map_down (torch.Tensor): 下采样后的 ID 图 [B, H_new, W_new] (LongTensor)
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 1. 确保输入维度正确 [B, 1, H, W] 用于 interpolate
    if tri_id_map.dim() == 2:
        tri_id_map = tri_id_map.unsqueeze(0)  # [1, H, W]

    # 转换为 Float 才能进 interpolate，但在 nearest 模式下数值不会变
    input_tensor = tri_id_map.unsqueeze(1).float()

    # 2. 执行下采样 (关键: mode='nearest')
    # warning: recompute_scale_factor=False 是为了兼容新版 PyTorch
    tri_id_map_down = F.interpolate(
        input_tensor,
        scale_factor=scale_factor,
        mode='nearest',
        recompute_scale_factor=False
    )

    # 转回 [B, H_new, W_new] 和 Long 类型
    tri_id_map_down = tri_id_map_down.squeeze(1).long()

    # ================= VISUALIZATION =================
    # 转 numpy 处理图片
    id_map_np = tri_id_map_down.detach().cpu().numpy()
    B, H_new, W_new = id_map_np.shape

    # 获取最大 ID 用于生成颜色表 (忽略 -1)
    max_id = id_map_np.max()
    # 避免全是 -1 的情况
    if max_id < 0:
        print(f"Warning: {prefix} 全是无效区域 (-1)")
        return tri_id_map_down

    # 生成随机调色板
    # 技巧：为了让颜色在多次运行中保持一致以便对比，可以固定 seed
    # 但为了让不同 ID 区分度大，使用 randint
    np.random.seed(42)
    num_colors = max_id + 1
    colors = np.random.randint(0, 255, size=(num_colors, 3), dtype=np.uint8)

    for b in range(B):
        # 取单张图
        curr_map = id_map_np[b]

        # 初始化画布 (黑色背景)
        vis_img = np.zeros((H_new, W_new, 3), dtype=np.uint8)

        # 掩码操作：只给有效区域上色
        mask = (curr_map >= 0)
        if mask.any():
            valid_ids = curr_map[mask]
            # 查表赋值颜色
            vis_img[mask] = colors[valid_ids]

        # 保存
        filename = f"{prefix}_b{b}_{int(H_new)}x{int(W_new)}.png"
        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, vis_img)
        print(f"✅ [1/{int(1 / scale_factor)} Scale] ID Map saved: {save_path}")

    return tri_id_map_down

def batch_convert_to_tri_infos(vertexs_batch, lines_batch, triangles_batch, H, W, device,scale_ratio=0.5):
    """
    批量转换多个样本（适配顶点坐标为(x, y)格式）,
    并对其进行一个格式调整归一化等等,用来传入预测头
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
    tri_infos: list length B, 每项为 dict:
                {
                    'batch_num_tri': int,三角形的数量
                    'centroids': [B,n_tri,2] 每个三角形的质点
                    'vertices': [B,n_tri,3,2] 每个三角形的顶点
                    'edges_list' :进行了一个归一化处理边像素点集合,以及其邻接面，去除掉了单邻接面的线段
                    boundary_local_idxs_per_batch: 存储着断裂边，也就是只有一个面的边
                }
    """
    tri_infos_batch_normal = []
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
        scale_ratio = 1.0
        # 将边像素转化为normal，目的是给后面预测头用
        tri_info_normal = convert_to_tri_infos_normal(vertexs, lines, triangles, H, W, device,scale_ratio)
        tri_infos_batch_normal.append(tri_info_normal)

        # # 4. 可视化并保存，正常
        # # 将边像素缩小，目的是测试用输出图片
        # tri_info=convert_to_tri_infos(vertexs, lines, triangles, H, W, device,scale_ratio)
        # visualize_centroids_and_edges(
        #     tri_info,int(H*scale_ratio),int(W*scale_ratio),
        #     save_path="outputs\edges_photo\edges_visual_tensor{}.png".format(b)
        # )
        # tri_infos_batch_normal.append(tri_info)

    # 5) 将 tri_infos 的 centroids / vertices 转为统一的 padded tensor
    #    并做归一化到 后续的 grid_sample 要求的 [-1,1]（注意 x 对应宽 W，y 对应高 H）
    #    知道断裂边的位置

    batch_num_tri = []
    centers_list = []
    vertices_list = []
    edges_list = []  # 临时存储每个样本的 edges，已经处理完毕的，去除掉了单邻接面的线段
    edges_pixels=[] # 存储着每个线段的像素已经做归一化处理
    boundary_local_idxs_per_batch = []  # 存储着断裂边，也就是只有一个面的边

    for b in range(len(vertexs_batch)):
        info = tri_infos_batch_normal[b]
        # 取出 lists
        centroids_py = info.get('centroids', [])  # list of np.ndarray (2,)
        vertices_py = info.get('vertices', [])  # list of np.ndarray (3,2)
        edges_py = info.get('edges', [])  # list of dicts with 'tri_ids'

        n_tri = len(centroids_py)
        if(n_tri==0):
            print("qweqeqeq")

        batch_num_tri.append(n_tri)

        # 将 centroids 转为 numpy array (n_tri, 2)，若为空则用 zeros
        if n_tri == 0:
            cent_np = np.zeros((0, 2), dtype=np.float32)
            vert_np = np.zeros((0, 3, 2), dtype=np.float32)
        else:
            # centroids_py 每项形如 (2,)
            cent_np = np.stack([np.asarray(c, dtype=np.float32) for c in centroids_py], axis=0)  # [n_tri,2]
            # vertices_py 每项形如 (3,2)
            vert_np = np.stack([np.asarray(v, dtype=np.float32) for v in vertices_py], axis=0)  # [n_tri,3,2]

        # 归一化: 分别对顶点和质点进行一个归一化，为了后续用双线性插值采样
        # x_norm = (x / (W-1)) * 2 - 1 ; y_norm = (y / (H-1)) * 2 - 1
        if cent_np.shape[0] > 0:
            x = cent_np[:, 0]
            y = cent_np[:, 1]
            # 归一化具体操作
            x_norm = (x / max(W - 1, 1)) * 2.0 - 1.0
            y_norm = (y / max(H - 1, 1)) * 2.0 - 1.0
            cent_norm = np.stack([x_norm, y_norm], axis=1)  # [n_tri,2]
        else:
            cent_norm = cent_np.reshape(0, 2)

        if vert_np.shape[0] > 0:
            vx = vert_np[:, :, 0]
            vy = vert_np[:, :, 1]  # [n_tri,3]
            vx_norm = (vx / max(W - 1, 1)) * 2.0 - 1.0
            vy_norm = (vy / max(H - 1, 1)) * 2.0 - 1.0
            vert_norm = np.stack([vx_norm, vy_norm], axis=2)  # [n_tri,3,2]
        else:
            vert_norm = vert_np.reshape(0, 3, 2)

        centers_list.append(torch.from_numpy(cent_norm).float())  # [n_tri,2]
        vertices_list.append(torch.from_numpy(vert_norm).float())  # [n_tri,3,2]

        # 处理 edges：把 tri_ids 提取成 (E_b,2) 的 LongTensor（局部索引）
        edge_ids = []
        # 记录本 sample 内为 boundary 的边在 edge_ids 中的局部索引
        boundary_local_idxs = []
        pixels=[]
        for ed in edges_py:
            # 获取每个线段的归一化像素集
            pixels.append(ed.get('edge_pixels', None))
            # 支持 ed 为 dict 或 tuple/list
            if isinstance(ed, dict):
                tid = ed.get('tri_ids', None)
            else:
                tid = ed
            if tid is None:
                continue
            t1, t2 = int(tid[0]), int(tid[1])

            # 三种情形：
            # 1) 两侧面都存在 (常规)：加入 [t1, t2]
            # 2) 只有一侧存在（boundary）：加入 [t_valid, t_valid] 并记录为 boundary（稍后把 alpha 设为 1）
            # 3) 两侧都不存在（非法）,则警告
            valid1 = (0 <= t1 < n_tri)
            valid2 = (0 <= t2 < n_tri)

            if valid1 and valid2:
                edge_ids.append([t1, t2])
            elif valid1 and not valid2:
                # 这里进行一个处理，如果只有一个三角面，就将唯一的三角面赋值给两个id
                edge_ids.append([t1, t1])
                boundary_local_idxs.append(len(edge_ids) - 1)
            elif valid2 and not valid1:
                edge_ids.append([t2, t2])
                boundary_local_idxs.append(len(edge_ids) - 1)
            else:
                # 两侧都非法，有问题
                raise RuntimeError("一条直线没有相邻三角形")

        if len(edge_ids) == 0:
            edges_list.append(torch.zeros((0, 2), dtype=torch.long))
        else:
            edges_list.append(torch.tensor(edge_ids, dtype=torch.long))
        edges_pixels.append(pixels)
        boundary_local_idxs_per_batch.append(boundary_local_idxs)

    new_tri_infos = []
    new_tri_infos.append({
        'batch_num_tri': batch_num_tri,
        'centers_list': centers_list,
        'vertices_list': vertices_list,
        'edges_list': edges_list,
        'edges_pixels': edges_pixels,
        'boundary_local_idxs_per_batch': boundary_local_idxs_per_batch
    })

    return new_tri_infos


def batch_convert_to_tri_infos_new(vertexs_batch, lines_batch, triangles_batch, H, W, device, scale_ratio=1.0):
    """
    批量转换多个样本。（适配顶点坐标为(x, y)格式）
    优化点：合并循环，统一在最后一步进行 [-1, 1] 归一化，移除了中间冗余的缩放操作。

    参数:
        vertexs_batch: list of tensor, 每个元素为单个样本的顶点（已to(device)，形状(Nv, 2)，(x, y)）
        lines_batch: list of tensor, 每个元素为单个样本的边（已to(device)，形状(Ne, 2)，(v1_id, v2_id)）[Line(p1, p2, face1, face2)]
        triangles_batch: list of list of tuple, 每个元素为单个样本的三角形数据:
                        每个三角形是 (v_ids, l_ids, pts)，其中：
                        - v_ids: tensor (3,) 顶点ID
                        - l_ids: tensor (3,) 边ID
                        - pts: tensor (M, 2) 有效像素坐标 (x, y)
        H, W: 图像高度和宽度
        device: 计算设备

    返回:
    tri_infos: list length B, 每项为 dict:
                {
                    'batch_num_tri': int,三角形的数量
                    'centroids': [B,n_tri,2] 每个三角形的质点
                    'vertices': [B,n_tri,3,2] 每个三角形的顶点
                    'edges_list' :边两个邻接面，如果只有一个面，则是两个相等的面id
                    ‘edges_pixels’:进行了一个归一化处理边像素点集合
                    'tri_id_map': [B, H, W] 密集三角形索引图 (值域 0~N-1, -1为无效) <--- 新增
                    boundary_local_idxs_per_batch: 存储着断裂边，也就是只有一个面的边
                    'tri_edge_ids_list':每个三角形的边ID列表
                    'edges_midpoints': 边对应的中点已经归一化 List[B] of [E, 2]
                }
    """

    # 结果容器
    batch_num_tri = []
    centers_list = []
    vertices_list = []
    edges_list = []
    edges_pixels = []
    boundary_local_idxs_per_batch = []
    edges_midpoints_list=[] #边中点

    # 新增：存储每个样本的 tri_id_map
    tri_id_maps_list = []
    # ym-add-26.3.7 每个三角形的边ID列表
    tri_edge_ids_list = []

    # 合并后的单次遍历
    for b in range(len(vertexs_batch)):
        # 1. 提取单个样本数据
        vertexs = vertexs_batch[b].cpu().numpy()  # (Nv, 2)
        lines = lines_batch[b]  # tensor (Ne, 4)

        # 新增：初始化当前样本的 tri_id_map (-1 表示无效/背景)
        current_tri_id_map = torch.full((H, W), -1, dtype=torch.long, device=device)

        # 2. 转换 triangles 格式 并 填充 Map
        current_triangles_data = []
        # 收集每个三角形的边ID
        current_tri_edge_ids = []

        # 遍历当前样本的所有三角形
        for tri_idx, tri in enumerate(triangles_batch[b]):
            # 解析数据
            v_ids = tri[0].cpu().numpy()
            l_ids = tri[1].cpu().numpy()
            pts = tri[2]  # tensor (M, 2) (x, y) on device

            # === 🔥 核心新增逻辑：生成 tri_id_map ===
            if pts.shape[0] > 0:
                # pts 是 (x, y) -> 对应 (width, height)
                xs = pts[:, 0].long()
                ys = pts[:, 1].long()
                # 边界安全检查 (clamp 防止越界崩溃)
                xs = xs.clamp(0, W - 1)
                ys = ys.clamp(0, H - 1)
                # 填入 ID (tri_idx)
                # 注意：PyTorch 的索引顺序是 [H, W] 即 [y, x]
                current_tri_id_map[ys, xs] = tri_idx

            # === 保存边ID ===
            current_tri_edge_ids.append(l_ids)  # numpy [3]

            # 收集数据给 helper 函数
            current_triangles_data.append({
                'vertex_ids': v_ids,
                'line_ids': l_ids
                # 'valid_points': pts.cpu().numpy() 不需要这个了
            })

        # 转换为Tensor [N_tri, 3]
        if len(current_tri_edge_ids) > 0:
            tri_edge_ids_tensor = torch.from_numpy(
                np.stack(current_tri_edge_ids, axis=0)
            ).long()
        else:
            tri_edge_ids_tensor = torch.zeros((0, 3), dtype=torch.long)
        # 放入总容器中，B,N_tri,3
        tri_edge_ids_list.append(tri_edge_ids_tensor)

        # 执行下采样 (关键: mode='nearest')
        # todo: 会产生0像素的三角形，这个地方以后可能更改一下逻辑
        if current_tri_id_map.dim() == 2:
            current_tri_id_map = current_tri_id_map.unsqueeze(0)  # [1, H, W]
        # 转换为 Float 才能进 interpolate，但在 nearest 模式下数值不会变
        input_tensor = current_tri_id_map.unsqueeze(1).float()

        # warning: recompute_scale_factor=False 是为了兼容新版 PyTorch
        tri_id_map_down = F.interpolate(
            input_tensor,scale_factor=0.5,mode='nearest',recompute_scale_factor=False
        )
        # 转回 [B, H_new, W_new] 和 Long 类型
        tri_id_map_down = tri_id_map_down.squeeze(1).long()

        # 将下采样好的,map 加入列表
        tri_id_maps_list.append(tri_id_map_down)

        # 3. 获取原始坐标的三角形信息 (不进行缩放)
        # 注意：这里我们传入 H, W 主要是为了 edge_pixels 的处理，vertex 不受 scale_ratio 影响
        tri_info = convert_to_tri_infos_normal_new(vertexs, lines, current_triangles_data, H, W, device, scale_ratio)

        # test 分别可视化边和三角图并保存，可视化正常=========================================
        # scale_ratio=0.5
        # tri_info=convert_to_tri_infos(vertexs, lines, current_triangles_data, H, W, device,scale_ratio)
        # visualize_centroids_and_edges(
        #     tri_info,int(H*scale_ratio),int(W*scale_ratio),
        #     save_path="/home/ym/Experiment/PatchmatchNet-new/outputs/photo_test/edges_visual_tensor{}.png".format(b)
        # )
        # # 假设 tri_id_map 是原分辨率的 (H, W)
        # # 原分辨率图
        # visualize_tri_id_map(current_tri_id_map, save_dir="/home/ym/Experiment/PatchmatchNet-new/outputs/photo_test")
        #
        # # 调用函数生成 1/2 分辨率 map 并保存图片
        # tri_id_map_half = visualize_downsampled_tri_id(
        #     current_tri_id_map,
        #     scale_factor=0.5,
        #     save_dir="/home/ym/Experiment/PatchmatchNet-new/outputs/photo_test",
        #     prefix="stage1_tri_ids"
        # )

        # 4. 处理从 convert_to_tri_infos_normal 返回的数据
        centroids_py = tri_info['centroids']  # list of np.ndarray (2,) 原始坐标
        vertices_py = tri_info['vertices']  # list of np.ndarray (3,2) 原始坐标
        edges_py = tri_info['edges']  # list of dicts

        n_tri = len(centroids_py)
        batch_num_tri.append(n_tri)

        # 5. 统一归一化处理：映射到 [-1, 1],分别对顶点和质点进行一个归一化，为了后续用双线性插值采样
        # x_norm = (x / (W-1)) * 2 - 1
        # y_norm = (y / (H-1)) * 2 - 1

        # 预计算分母，防止除零
        div_w = max(W - 1, 1)
        div_h = max(H - 1, 1)


        if n_tri > 0:
            # 堆叠为 numpy 数组进行批量计算，比 list comprehension 更快
            cent_np = np.stack(centroids_py, axis=0).astype(np.float32)  # [n_tri, 2]
            vert_np = np.stack(vertices_py, axis=0).astype(np.float32)  # [n_tri, 3, 2]

            # 归一化质心
            cent_np[:, 0] = (cent_np[:, 0] / div_w) * 2.0 - 1.0
            cent_np[:, 1] = (cent_np[:, 1] / div_h) * 2.0 - 1.0

            # 归一化顶点
            vert_np[:, :, 0] = (vert_np[:, :, 0] / div_w) * 2.0 - 1.0
            vert_np[:, :, 1] = (vert_np[:, :, 1] / div_h) * 2.0 - 1.0
        else:
            cent_np = np.zeros((0, 2), dtype=np.float32)
            vert_np = np.zeros((0, 3, 2), dtype=np.float32)

        # 转 Tensor
        centers_list.append(torch.from_numpy(cent_np).float())
        vertices_list.append(torch.from_numpy(vert_np).float())

        # 6. 处理 edges：把 tri_ids 提取成 (E_b,2) 的 LongTensor（局部索引）
        edge_ids = []
        boundary_local_idxs = []
        current_pixels = []
        current_midpoints = []  # 收集当前batch的边中点

        for i, ed in enumerate(edges_py):
            # 获取像素集合
            current_pixels.append(ed.get('edge_pixels', None))

            # 获取 tri_ids 三角面id
            tid = ed.get('tri_ids', None) if isinstance(ed, dict) else ed
            if tid is None:
                continue

            t1, t2 = int(tid[0]), int(tid[1])

            valid1 = (0 <= t1 < n_tri)
            valid2 = (0 <= t2 < n_tri)

            if valid1 and valid2:
                # 两个面都存在，正常边
                edge_ids.append([t1, t2])
            elif valid1 and not valid2:
                # 只有面1，断裂边/边界
                edge_ids.append([t1, t1])
                boundary_local_idxs.append(len(edge_ids) - 1)
            elif valid2 and not valid1:
                # 只有面2，断裂边/边界
                edge_ids.append([t2, t2])
                boundary_local_idxs.append(len(edge_ids) - 1)
            else:
                raise RuntimeError(f"Sample {b}, Edge {i}: 一条直线没有相邻三角形")

            # 🔥 提取端点，计算原始像素中点
            pts = ed.get('endpoints')
            if pts is not None:
                p1, p2 = pts
                current_midpoints.append([(p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0])
            else:
                current_midpoints.append([0.0, 0.0])
                raise RuntimeError(f"线段没有端点")


        if len(edge_ids) == 0:
            edges_list.append(torch.zeros((0, 2), dtype=torch.long))
        else:
            edges_list.append(torch.tensor(edge_ids, dtype=torch.long))

        # 对中点进行统一的 [-1, 1] 归一化
        if len(current_midpoints) > 0:
            mid_np = np.array(current_midpoints, dtype=np.float32)  # [E, 2]

            # 归一化公式: (x / W) * 2 - 1
            mid_np[:, 0] = (mid_np[:, 0] / div_w) * 2.0 - 1.0
            mid_np[:, 1] = (mid_np[:, 1] / div_h) * 2.0 - 1.0

            midpoints_tensor = torch.from_numpy(mid_np).to(device)
        else:
            midpoints_tensor = torch.zeros((0, 2), dtype=torch.float32, device=device)

        edges_midpoints_list.append(midpoints_tensor)
        edges_pixels.append(current_pixels)
        boundary_local_idxs_per_batch.append(boundary_local_idxs)

    # 7. 组装返回结果
    new_tri_infos = []
    new_tri_infos.append({
        'batch_num_tri': batch_num_tri,
        'centers_list': centers_list,
        'vertices_list': vertices_list,
        'edges_list': edges_list,
        'edges_pixels': edges_pixels,
        'tri_id_map': tri_id_maps_list,
        'boundary_local_idxs_per_batch': boundary_local_idxs_per_batch,
        'tri_edge_ids_list': tri_edge_ids_list,
        'edges_midpoints': edges_midpoints_list # List[B] of [E, 2]
    })

    return new_tri_infos


def build_neighbor_indices(tri_infos, max_tri_num, device, return_batched=True):
    """
    构建邻居索引，支持返回list或batched tensor

    Args:
        tri_infos: 三角形信息
        max_tri_num: 最大三角形数量（用于padding）
        device: 计算设备
        return_batched: True返回[B, N_max, 3]，False返回list[B]

    Returns:
        neighbor_indices: [B, N_max, 3] Tensor 或 list[B] of [N, 3]
    """
    tri_edge_ids_list = tri_infos[0]['tri_edge_ids_list']
    edges_list = tri_infos[0]['edges_list']

    B = len(tri_edge_ids_list)
    neighbor_indices_list = []

    for b in range(B):
        tri_edge_ids = tri_edge_ids_list[b].to(device)  # [N, 3]
        edges = edges_list[b].to(device)  # [E, 2]

        N_tri = tri_edge_ids.shape[0]

        if N_tri == 0:
            print("=================三角形等于0================")
            return NULL_TENSOR

        # 向量化计算
        E_max = edges.shape[0]
        tri_edge_ids = torch.clamp(tri_edge_ids, 0, E_max - 1)  # 安全裁剪

        # 获取“构成每个三角形的3条边”所连接的“两个三角形面的ID”,广播机制
        connected_faces = edges[tri_edge_ids]  # [N, 3, 2]
        # 自身三角形id，为了跟后面进行一个比对判断，来得到三角形邻居面
        self_ids = torch.arange(N_tri, device=device).view(N_tri, 1, 1)
        # 判断三角形每条边对应的三角面对中，当前三角形是不是自己本身，如果是为True，不是为false
        mask_is_t1 = (connected_faces[..., 0] == self_ids.squeeze(-1))
        # torch.where(条件, 为真时取值, 为假时取值)，去取相反的一个面
        neighbors = torch.where(mask_is_t1, connected_faces[..., 1], connected_faces[..., 0])

        neighbor_indices_list.append(neighbors)

    # === 根据参数决定返回格式 ===
    if return_batched:
        # Stack并Pad为 [B, max_tri_num, 3]
        batched_neighbors = torch.zeros((B, max_tri_num, 3), dtype=torch.long, device=device)

        for b in range(B):
            N_tri = neighbor_indices_list[b].shape[0]
            if N_tri > 0:
                batched_neighbors[b, :N_tri, :] = neighbor_indices_list[b]

        return batched_neighbors
    else:
        return neighbor_indices_list


def convert_edge_features_to_tri_format(edge_alphas_list, tri_infos, max_tri_num, device):
    """
    将边级别的特征 (alphas 和 midpoints) 转换为三角形级别的格式 [B, N_max, 3, ...]
    Args:
        edge_alphas_list: list[B], 每个元素是 [E_b] (EdgeHead输出)
        tri_infos: 包含 tri_edge_ids_list 和 edges_midpoints 的字典
        max_tri_num: int, 最大三角形数量（用于Padding）
        device: 计算设备

    Returns:
        edge_probs: Tensor [B, max_tri_num, 3]
        aligned_midpoints_norm: Tensor [B, max_tri_num, 3, 2] (归一化到 [-1, 1] 的边中点)
    """
    tri_edge_ids_list = tri_infos[0]['tri_edge_ids_list']
    edges_midpoints_list = tri_infos[0]['edges_midpoints']  # 你之前新增的归一化中点列表
    batch_num_tri = tri_infos[0]['batch_num_tri']
    B = len(edge_alphas_list)

    # 1. 初始化返回容器
    # edge_probs 初始化为 1.0（默认阻断，安全垫底）
    edge_probs = torch.ones((B, max_tri_num, 3), dtype=torch.float32, device=device)

    # aligned_midpoints_norm 初始化为 0.0 (中心点，实际上无效边不会产生 loss，所以填什么都行)
    aligned_midpoints_norm = torch.zeros((B, max_tri_num, 3, 2), dtype=torch.float32, device=device)

    total_invalid = 0

    for b in range(B):
        edge_alphas = edge_alphas_list[b]  # [E_b]
        edge_midpoints = edges_midpoints_list[b].to(device)  # [E_b, 2]
        tri_edge_ids = tri_edge_ids_list[b].to(device)  # [N_tri, 3]
        N_tri = batch_num_tri[b]

        if N_tri == 0:
            continue

        E_max = edge_alphas.shape[0]

        # === 安全检查 ===
        invalid_mask = (tri_edge_ids < 0) | (tri_edge_ids >= E_max)
        num_invalid = invalid_mask.sum().item()
        if num_invalid > 0:
            total_invalid += num_invalid
            # print(f"⚠️ Batch {b}: {num_invalid}/{tri_edge_ids.numel()} edge IDs invalid")

        # === Padding Trick (为 -1 的无效边准备垫片) ===
        # 1. 垫概率: 末尾追加 1.0 (代表断裂)
        padded_alphas = torch.cat([edge_alphas, torch.tensor([1.0], dtype=edge_alphas.dtype, device=device)])

        # 2. 垫中点: 末尾追加 [0.0, 0.0]
        padded_midpoints = torch.cat(
            [edge_midpoints, torch.tensor([[0.0, 0.0]], dtype=edge_midpoints.dtype, device=device)], dim=0)

        # === 安全映射 ===
        # 将无效的 id (-1 或 越界) 映射为最后一行的索引 (即垫片的位置)
        # 注意：这里我们使用 E_max 作为垫片的索引，因为 padded 数组的长度是 E_max + 1
        safe_edge_ids = torch.where(
            ~invalid_mask,
            tri_edge_ids,
            torch.tensor(E_max, dtype=torch.long, device=device)
        )

        # === 同步提取 (Gather) ===
        # 提取概率 -> [N_tri, 3]
        gathered_probs = padded_alphas[safe_edge_ids]
        edge_probs[b, :N_tri, :] = gathered_probs

        # 提取中点 -> [N_tri, 3, 2]
        gathered_midpoints = padded_midpoints[safe_edge_ids]
        aligned_midpoints_norm[b, :N_tri, :, :] = gathered_midpoints

    if total_invalid > 0:
        print(f"📊 Total invalid edge IDs mapped to safe padding across all batches: {total_invalid}")

    return edge_probs, aligned_midpoints_norm

# --------------------------------------------
# 通用采样函数：对三角形的 (质心 + 3个顶点) 进行采样并取平均
# sample_points: [B, N_max, 4, 2] -> reshape 为 [B, N*4, 1, 2] 供 grid_sample 使用
# --------------------------------------------
def _sample_map(map_tensor, centers, vertices):
    """
    通用采样函数：对三角形的 (质心 + 3个顶点) 进行采样并取平均。

    Args:
        map_tensor: [B, C, H, W] (可以是图像特征，也可以是深度图)
        centers:    [B, N, 1, 2] (归一化坐标 -1 到 1)
        vertices:   [B, N, 3, 2] (归一化坐标 -1 到 1)

    Returns:
        tri_feats:  [B, N, C] 采样并平均后的特征
    """
    B_local = map_tensor.shape[0]
    C_map = map_tensor.shape[1]
    N = centers.shape[1]

    # 拼接 1 + 3 = 4 个点
    sample_pts = torch.cat([centers, vertices], dim=2)  # [B, N, 4, 2]
    # reshape 为 grid_sample 要求的 grid： [B, N*4, 1, 2]
    grid = sample_pts.view(B_local, N * 4, 1, 2).contiguous()
    # grid_sample -> out [B, C_map, N*4, 1]
    sampled = F.grid_sample(map_tensor, grid, align_corners=False, mode='bilinear', padding_mode='border')
    # 采样完之后恢复原状，变为每个三角形关于这个四点特征的矩阵 -> [B, C_map, N, 4]，
    sampled = sampled.view(B_local, C_map, N, 4).contiguous()
    # 将四个点特征取一个平均 -> [B, C_map, N] -> permute -> [B, N, C_map]
    tri_feats = sampled.mean(dim=3).permute(0, 2, 1).contiguous()
    return tri_feats  # [B, N, C_map]


def generate_edge_alpha_overlays(ref_imgs, edge_alphas_list, edges_pixels_list, device='cpu', overlay_alpha=0.7,
                                 line_thickness=2):
    """
    基于真实边像素(edges_pixels)生成断裂预测热力图。(修复版)
    """

    batch_size = ref_imgs.shape[0]

    # 获取列表实际长度，用于防止越界
    len_alphas = len(edge_alphas_list)
    len_pixels = len(edges_pixels_list)

    # 结果容器
    output_tensor_list = []

    # 1. 预计算色盘 (0~255) -> BGR (OpenCV format)
    # JET: 0(Blue) -> 128(Green) -> 255(Red)
    colormap_lut = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in range(256):
        colormap_lut[i, 0] = np.array([i, i, i])
    colormap_lut = cv2.applyColorMap(colormap_lut, cv2.COLORMAP_JET).squeeze(1)

    for b in range(batch_size):
        # --- A. 准备底图 ---
        img_tensor = ref_imgs[b].detach().cpu()

        # 反归一化处理
        if img_tensor.min() < 0:
            img_tensor = img_tensor - img_tensor.min()
            img_tensor = img_tensor / (img_tensor.max() + 1e-6)

        # [C, H, W] -> [H, W, C] -> uint8 numpy
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

        # 格式统转 BGR
        if img_np.shape[2] == 1:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        else:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        H, W, _ = img_np.shape

        # --- B. 安全检查与数据获取 ---
        # 如果当前 batch 索引超出了列表长度，说明数据不对齐，直接返回原图
        if b >= len_alphas or b >= len_pixels:
            # 转回 Tensor 并添加到输出
            final_tensor = torch.from_numpy(img_np).permute(2, 0, 1).to(torch.float32) / 255.0
            output_tensor_list.append(final_tensor)
            print("当前 batch 索引超出了列表长度，说明数据不对齐")
            continue  # 跳过绘制

        # 获取当前数据
        curr_alphas_tensor = edge_alphas_list[b]
        curr_pixels_list = edges_pixels_list[b]

        # 检查 alpha 是否为空或 None
        if curr_alphas_tensor is None or curr_alphas_tensor.numel() == 0:
            # 如果没有预测值，直接返回原图
            final_tensor = torch.from_numpy(img_np).permute(2, 0, 1).to(torch.float32) / 255.0
            output_tensor_list.append(final_tensor)
            print("alpha 为空")
            continue

        # 转 numpy
        curr_alphas = curr_alphas_tensor.detach().cpu().numpy()

        # 再次检查长度对齐 (防止 alphas 和 pixels 数量不一致)
        num_pixels_groups = len(curr_pixels_list)
        num_probs = len(curr_alphas)
        safe_len = min(num_pixels_groups, num_probs)

        # --- C. 准备画板 ---
        overlay_layer = np.zeros_like(img_np)

        # --- D. 遍历画线 ---
        for i in range(safe_len):
            prob = curr_alphas[i]
            pixels = curr_pixels_list[i]  # [(x1,y1), ...]

            # 如果像素点列表为空，跳过
            if not pixels:
                print("像素点列表为空，跳过")
                continue

            # 颜色映射: 0.0(Blue) -> 1.0(Red)
            # 确保 prob 在 0-1 之间
            prob = np.clip(prob, 0.0, 1.0)
            color_idx = int(prob * 255)
            color = colormap_lut[color_idx].tolist()  # (B, G, R)

            # 1. 转为 numpy float
            pts_norm = np.array(pixels, dtype=np.float32)

            # 2. 坐标反归一化 [-1, 1] -> [0, W/H]
            # 你的 edges_pixels 如果已经是绝对坐标，请注释掉这两行
            pts_x = (pts_norm[:, 0] + 1) * (W - 1) / 2.0
            pts_y = (pts_norm[:, 1] + 1) * (H - 1) / 2.0

            # 3. 组合
            pts_real = np.stack([pts_x, pts_y], axis=1).astype(np.int32)

            # 4. Reshape
            pts_to_draw = pts_real.reshape((-1, 1, 2))

            # 绘制
            cv2.polylines(overlay_layer, [pts_to_draw], isClosed=False, color=color,
                          thickness=line_thickness, lineType=cv2.LINE_AA)

        # --- E. 图像融合 ---
        mask = np.any(overlay_layer > 0, axis=-1)
        final_img = img_np.copy()

        # 只混合有线条的区域，保持背景亮度
        weighted_overlay = cv2.addWeighted(img_np, 1.0 - overlay_alpha, overlay_layer, overlay_alpha, 0)
        final_img[mask] = weighted_overlay[mask]

        # OpenCV 操作完是 BGR，但 TensorBoard 需要 RGB
        final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)

        # --- F. 转回 Tensor ---
        # 归一化到 0-1 范围 (通常 image_outputs 期望 float 0-1)
        final_tensor = torch.from_numpy(final_img).permute(2, 0, 1).to(torch.float32) / 255.0
        output_tensor_list.append(final_tensor)

    # 堆叠
    if len(output_tensor_list) > 0:
        output_stack = torch.stack(output_tensor_list).to(device)
    else:
        output_stack = torch.zeros_like(ref_imgs).to(device)

    # 你的原代码返回的是 uint8 注释，但 tensor 转换时通常 float 更通用，这里我输出了 float [0,1]

    return {"ref_img_edge_alpha": output_stack}

# 检查tensor是否有问题，并报错
def check_tensor(name, t):
    if not isinstance(t, torch.Tensor):
        return
    if torch.isnan(t).any():
        print(f"[NaN CHECK] {name} contains NaN; shape={tuple(t.shape)}")
        # 打印少量元素供定位
        print(getattr(t, 'detach', lambda : t)().cpu().flatten()[:20])
        raise RuntimeError(f"NaN found in {name}")
    if torch.isinf(t).any():
        print(f"[NaN CHECK] {name} contains Inf; shape={tuple(t.shape)}")
        raise RuntimeError(f"Inf found in {name}")


def compute_normal_map_torch(depth_tensor, mask=None, smooth=True):
    """
    在 GPU 上从深度图生成法向量图。

    Args:
        depth_tensor (torch.Tensor): 深度图，形状可以是 [B, 1, H, W] 或 [B, H, W]。
                                     如果是 [B, 3, H, W]，会自动取第一个通道。
        mask (torch.Tensor, optional): 有效像素掩码，形状同 depth_tensor。
                                       无效区域的法向量会被置为 0 (黑色) 或特定颜色。
        smooth (bool): 是否进行简单的高斯平滑以减少噪声（推荐 True）。

    Returns:
        normal_map (torch.Tensor): 形状 [B, 3, H, W]，数值范围 [0, 1]，用于 tensorboard 可视化。
    """
    # 1. 维度处理
    if depth_tensor.dim() == 3:
        depth_tensor = depth_tensor.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]

    if depth_tensor.shape[1] == 3:
        # 如果输入是 3 通道 (比如已经是 RGB 渲染)，取均值或单通道作为深度
        depth_tensor = depth_tensor.mean(dim=1, keepdim=True)

    B, C, H, W = depth_tensor.shape
    device = depth_tensor.device

    # 2. 高斯平滑 (减少深度图噪声导致的法向量破碎)
    if smooth:
        # 简单的 3x3 高斯核
        gaussian_kernel = torch.tensor([[1., 2., 1.],
                                        [2., 4., 2.],
                                        [1., 2., 1.]], device=device) / 16.0
        gaussian_kernel = gaussian_kernel.view(1, 1, 3, 3)
        # Reflect pad 避免边缘伪影
        depth_tensor = F.pad(depth_tensor, (1, 1, 1, 1), mode='reflect')
        depth_tensor = F.conv2d(depth_tensor, gaussian_kernel)

    # 3. 定义 Sobel 算子 (计算梯度 dz/dx, dz/dy)
    sobel_x = torch.tensor([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]], device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.],
                            [0., 0., 0.],
                            [1., 2., 1.]], device=device).view(1, 1, 3, 3)

    # 4. 计算梯度
    # Padding 保证尺寸不变
    depth_pad = F.pad(depth_tensor, (1, 1, 1, 1), mode='reflect')
    dzdx = F.conv2d(depth_pad, sobel_x)
    dzdy = F.conv2d(depth_pad, sobel_y)

    # 5. 构造法向量 (-dz/dx, -dz/dy, 1)
    # 注意：这里 Z 轴设为 1，如果你想要更强的凹凸感，可以把 dzdx, dzdy 乘以一个系数 (sensitivity)
    normal_x = -dzdx
    normal_y = -dzdy
    normal_z = torch.ones_like(normal_x)

    # 堆叠通道 [B, 3, H, W]
    normals = torch.cat([normal_x, normal_y, normal_z], dim=1)

    # 6. 归一化 (Normalize)
    # norm = sqrt(x^2 + y^2 + z^2)
    norm = torch.norm(normals, dim=1, keepdim=True)
    # 避免除 0
    normals = normals / (norm + 1e-8)

    # 7. 映射到 [0, 1] 区间用于可视化
    # 原范围 [-1, 1] -> 新范围 [0, 1]
    # RGB 对应关系: R:X(左右), G:Y(上下), B:Z(指向相机)
    normal_map = (normals + 1.0) / 2.0

    # normals_vis = normals.clone()
    # normals_vis[:, 2, :, :] = -normals_vis[:, 2, :, :]  # 翻转 Z 用于显示
    # normal_map = (normals_vis + 1.0) / 2.0

    # 8. 应用 Mask (如果有)
    if mask is not None:
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        if mask.shape[1] == 3:  # 如果 mask 是 3 通道，取单通道
            mask = mask[:, :1, :, :]

        # 确保 mask 大小匹配 (防止上采样带来的细微尺寸差异)
        if mask.shape[-2:] != normal_map.shape[-2:]:
            mask = F.interpolate(mask.float(), size=normal_map.shape[-2:], mode='nearest')

        # 将无效区域设为黑色 (0,0,0) 或者灰色 (0.5, 0.5, 0.5)
        normal_map = normal_map * mask

    return normal_map

def compute_normal_map_perspective(depth_tensor, intrinsics, mask=None, smooth=True):
    """
    [高精度版] 基于透视投影 (Perspective Projection) 计算法向量。

    原理：
    1. 将深度图反投影为 3D 点云 (Vertex Map)。
    2. 使用链式法则计算 3D 空间中相对于像素 u, v 的切向量 Tu, Tv。
    3. 法向量 n = normalize(Tu x Tv)。
    此方法比直接 Sobel 深度图更精准，因为它考虑了 FOV 和视线角度。

    Args:
        depth_tensor: [B, 1, H, W] 深度图
        intrinsics:   [B, 3, 3] 相机内参
        mask:         [B, 1, H, W] 有效区域
        smooth:       bool, 是否预平滑深度图 (推荐 True, 否则微分噪声大)

    Returns:
        normal_map:   [B, 3, H, W], 数值范围 [-1, 1], 指向相机方向 (Z < 0)
    """
    B, C, H, W = depth_tensor.shape
    device = depth_tensor.device

    # 1. 预处理：高斯平滑 (减少微分对噪声的放大)
    if smooth:
        # 3x3 高斯核
        gaussian_kernel = torch.tensor([[1., 2., 1.],
                                        [2., 4., 2.],
                                        [1., 2., 1.]], device=device) / 16.0
        gaussian_kernel = gaussian_kernel.view(1, 1, 3, 3)
        depth_tensor = F.pad(depth_tensor, (1, 1, 1, 1), mode='reflect')
        depth_tensor = F.conv2d(depth_tensor, gaussian_kernel)

    # 2. 准备网格坐标 (u, v)
    y_range = torch.arange(0, H, dtype=torch.float32, device=device)
    x_range = torch.arange(0, W, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')  # [H, W]

    # 扩展到 Batch: [B, H, W]
    u = grid_x.unsqueeze(0).expand(B, -1, -1)
    v = grid_y.unsqueeze(0).expand(B, -1, -1)

    # 3. 解析内参
    # intrinsics: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    fx = intrinsics[:, 0, 0].view(B, 1, 1)
    fy = intrinsics[:, 1, 1].view(B, 1, 1)
    cx = intrinsics[:, 0, 2].view(B, 1, 1)
    cy = intrinsics[:, 1, 2].view(B, 1, 1)

    # 4. 计算深度图的梯度 (dz/du, dz/dv)
    # 使用 Sobel 算子计算像素坐标系下的梯度
    sobel_x = torch.tensor([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]], device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.],
                            [0., 0., 0.],
                            [1., 2., 1.]], device=device).view(1, 1, 3, 3)

    depth_pad = F.pad(depth_tensor, (1, 1, 1, 1), mode='reflect')
    dz_du = F.conv2d(depth_pad, sobel_x)  # 深度在 u 方向的变化率
    dz_dv = F.conv2d(depth_pad, sobel_y)  # 深度在 v 方向的变化率

    # 将 B,1,H,W 压缩为 B,H,W 方便后续计算
    Z = depth_tensor.squeeze(1)
    dz_du = dz_du.squeeze(1)
    dz_dv = dz_dv.squeeze(1)

    # =================================================================
    # 5. 核心逻辑：透视投影下的切向量计算 (Chain Rule)
    # =================================================================
    # 3D点 P = [X, Y, Z]
    # X = (u - cx) * Z / fx
    # Y = (v - cy) * Z / fy

    # 我们需要求 P 对 u 和 v 的偏导数，作为切平面上的两个向量 Tu, Tv

    # --- 计算 Tu = dP/du ---
    # dX/du = (Z + (u-cx)*dz_du) / fx
    # dY/du = ((v-cy)*dz_du) / fy
    # dZ/du = dz_du
    dX_du = (Z + (u - cx) * dz_du) / fx
    dY_du = ((v - cy) * dz_du) / fy
    dZ_du = dz_du
    Tu = torch.stack([dX_du, dY_du, dZ_du], dim=1)  # [B, 3, H, W]

    # --- 计算 Tv = dP/dv ---
    # dX/dv = ((u-cx)*dz_dv) / fx
    # dY/dv = (Z + (v-cy)*dz_dv) / fy
    # dZ/dv = dz_dv
    dX_dv = ((u - cx) * dz_dv) / fx
    dY_dv = (Z + (v - cy) * dz_dv) / fy
    dZ_dv = dz_dv
    Tv = torch.stack([dX_dv, dY_dv, dZ_dv], dim=1)  # [B, 3, H, W]

    # 6. 计算法向量：叉积 (Cross Product)
    # n = Tu x Tv
    # 注意顺序：u 是向右，v 是向下。根据右手定则，Right x Down = Forward (Z > 0)
    # 我们希望法向量指向相机 (Z < 0)，所以我们用 Tv x Tu 或者最后取反
    # 这里直接计算 cross(Tu, Tv) 然后强制 Z 为负
    normals = torch.cross(Tu, Tv, dim=1)

    # 7. 归一化
    norm = torch.norm(normals, dim=1, keepdim=True)
    normals = normals / (norm + 1e-8)

    # normals = (normals + 1.0) / 2.0

    normals_vis = normals.clone()
    normals_vis[:, 2, :, :] = -normals_vis[:, 2, :, :]  # 翻转 Z 用于显示
    normals = (normals_vis + 1.0) / 2.0

    # 8. 强制指向相机 (Z < 0)
    # 检查 Z 分量的符号。如果 Z > 0，说明指向了屏幕里面，需要翻转。
    # 大部分情况下 Tu x Tv 得到的 Z 应该已经是正的(背离相机)，所以这里统一翻转比较稳妥
    # 或者使用 dot product 检测法

    # 简单粗暴且正确的方法：因为我们知道表面是看着相机的，所以 n_z 必须小于 0
    # 获取 n_z 的符号
    # sign_z = torch.sign(normals[:, 2:3, :, :])
    # # 如果是正数 (1.0)，我们要变成负数，所以乘以 -1
    # # 如果是负数 (-1.0)，我们要保持，所以乘以 1
    # # 也就是乘以 -sign_z (除了0的情况)
    # flip_mask = -sign_z
    # flip_mask[flip_mask == 0] = -1.0  # 处理 z=0 的边界情况，默认翻转
    #
    # normals = normals * flip_mask

    # 9. 应用 Mask
    if mask is not None:
        if mask.dim() == 3: mask = mask.unsqueeze(1)
        if mask.shape[1] == 3: mask = mask[:, :1, :, :]
        if mask.shape[-2:] != normals.shape[-2:]:
            mask = F.interpolate(mask.float(), size=normals.shape[-2:], mode='nearest')
        normals = normals * mask

    return normals

# --- 辅助函数：可视化转换 ---
def visualize_normal_map(normal_tensor):
    """
    将 [-1, 1] 的物理法向量转换为 [0, 1] 的 RGB 图片用于显示。
    为了符合人类直觉（蓝色为正面），我们会翻转 Z 轴用于显示。
    """
    vis_normals = normal_tensor.clone()
    # 物理上 Z<0 是正面，但可视化习惯用 RGB=(0.5, 0.5, 1.0) 代表正面(Z>0)
    # 所以显示时翻转 Z
    vis_normals[:, 2, :, :] = -vis_normals[:, 2, :, :]
    return (vis_normals + 1.0) / 2.0

def normalize_depth_for_display(depth, valid_mask=None):
    """
    专门用于显示的深度图归一化函数。
    将有效范围 [min, max] 线性映射到 [0, 1]，背景保持 0。
    """
    # 1. 复制一份，不影响原始数据
    vis_depth = depth.clone()

    # 2. 确定有效区域
    if valid_mask is None:
        valid_mask = vis_depth > 1e-4  # 假设大于0也是有效

    # 3. 只在有效区域计算 Min 和 Max
    if valid_mask.sum() > 0:
        d_min = vis_depth[valid_mask].min()
        d_max = vis_depth[valid_mask].max()

        # 防止 max == min 导致除零
        diff = d_max - d_min
        if diff < 1e-6:
            diff = 1.0

        # 4. 核心：只拉伸有效区域
        # (val - min) / (max - min) -> 范围变回 0~1
        vis_depth[valid_mask] = (vis_depth[valid_mask] - d_min) / diff

        # 5. 可选：反转颜色 (通常近处亮，远处暗，或者反过来，看你习惯)
        # 现在的逻辑是：近处(min) -> 0(黑), 远处(max) -> 1(白)
        # 如果你想反过来，可以取消下面这行的注释：
        # vis_depth[valid_mask] = 1.0 - vis_depth[valid_mask]

    # 6. 确保背景是纯黑
    vis_depth[~valid_mask] = 0.0

    return vis_depth
