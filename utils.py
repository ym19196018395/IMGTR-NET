import cv2
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
        scale_ratio = 0.5
        # 将边像素转化为normal，目的是给后面预测头用
        tri_info_normal = convert_to_tri_infos_normal(vertexs, lines, triangles, H, W, device,scale_ratio)
        tri_infos_batch_normal.append(tri_info_normal)

        # # 4. 可视化并保存，正常
        # # 将边像素缩小，目的是测试用输出图片
        # tri_info=convert_to_tri_infos(vertexs, lines, triangles, H, W, device,scale_ratio)
        # visualize_centroids_and_edges(
        #     tri_info,int(H*scale_ratio),int(W*scale_ratio),
        #     save_path="edges_visual_tensor{}.png".format(b)
        # )

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
    基于真实边像素(edges_pixels)生成断裂预测热力图。

    Args:
        ref_imgs: [B, 3, H, W] 输入图像 Tensor (标准化过的或0-1)
        edge_alphas_list: list len=B, 元素为 Tensor [E], 预测的断裂概率
        edges_pixels_list: list len=B, 元素为 list len=E (每条边的像素集合).
                           结构: batch_list[ edge_list[ pixel_list[(x,y),...] ] ]
                           注意：假设 pixel 坐标是 (x, y) 格式，适配 OpenCV。
        device: 输出 Tensor 的设备
        overlay_alpha: 叠加透明度
        line_thickness: 线条粗细 (建议设为2，看的更清楚)

    Returns:
        dict: {"ref_img_edge_alpha": tensor [B, 3, H, W] uint8}
    """

    batch_size = ref_imgs.shape[0]

    # 结果容器
    output_tensor_list = []

    # 1. 预计算色盘 (0~255) -> BGR
    # 使用 JET Colormap: 0(蓝) -> 0.5(青/黄) -> 1(红)
    colormap_lut = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in range(256):
        colormap_lut[i, 0] = np.array([i, i, i])
    colormap_lut = cv2.applyColorMap(colormap_lut, cv2.COLORMAP_JET).squeeze(1)

    for b in range(batch_size):
        # --- A. 准备底图 ---
        img_tensor = ref_imgs[b].detach().cpu()

        # 反归一化处理 (简单 MinMax 归一化到 0-255，确保可视化正常)
        if img_tensor.min() < 0:
            img_tensor = img_tensor - img_tensor.min()
            img_tensor = img_tensor / (img_tensor.max() + 1e-6)

        # [C, H, W] -> [H, W, C] -> uint8 numpy
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

        # 格式统转 BGR (OpenCV 默认)
        if img_np.shape[2] == 1:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        else:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        H, W, _ = img_np.shape

        # --- B. 准备画板 ---
        # 创建纯黑层用于画线
        overlay_layer = np.zeros_like(img_np)

        # 获取当前 Batch 数据
        # edges_pixels: List[List[Tuple(x,y)]]
        curr_pixels_list = edges_pixels_list[b]
        # alphas: Tensor [E]
        curr_alphas = edge_alphas_list[b].detach().cpu().numpy()

        num_edges = len(curr_pixels_list)
        # 安全检查：如果边数量和预测数量不一致，取最小值防止越界
        safe_len = min(num_edges, len(curr_alphas))

        # --- C. 遍历画线 ---
        for i in range(safe_len):
            prob = curr_alphas[i]
            pixels = curr_pixels_list[i]  # [(x1,y1), (x2,y2), ...]

            # 过滤：概率太小的边(完全连通)可以选择不画，或者画得很淡
            # 这里设置 > 0.05 才画，保持画面干净,全部都需要画
            # if prob < 0.05 or len(pixels) == 0:
            #     continue

            # 颜色映射: 0.0 -> Blue, 1.0 -> Red
            color_idx = int(np.clip(prob * 255, 0, 255))
            color = colormap_lut[color_idx].tolist()  # (B, G, R)

            # 1. 转为 numpy float 数组
            pts_norm = np.array(pixels, dtype=np.float32) # shape [N, 2]

            # 2. 根据归一化类型转换
            # 情况 A: 如果坐标范围是 [-1, 1] (PyTorch grid_sample 标准)
            # x_real = (x_norm + 1) / 2 * (W - 1)
            pts_x = (pts_norm[:, 0] + 1) * (W - 1) / 2.0
            pts_y = (pts_norm[:, 1] + 1) * (H - 1) / 2.0

            # 3. 组合并取整
            pts_real = np.stack([pts_x, pts_y], axis=1).astype(np.int32)

            # 4. Reshape 为 cv2.polylines 需要的 (N, 1, 2)
            pts_to_draw = pts_real.reshape((-1, 1, 2))

            # 绘制
            cv2.polylines(overlay_layer, [pts_to_draw], isClosed=False, color=color, thickness=line_thickness,
                          lineType=cv2.LINE_AA)

        # --- D. 图像融合 ---
        # 只有画了线的地方才有 mask
        mask = np.any(overlay_layer > 0, axis=-1)

        # 融合: Original * (1-alpha) + Overlay * alpha
        final_img = img_np.copy()

        # 使用 addWeighted 会让整体变暗，我们只混合 Mask 区域
        weighted_overlay = cv2.addWeighted(img_np, 1.0 - overlay_alpha, overlay_layer, overlay_alpha, 0)

        final_img[mask] = weighted_overlay[mask]

        # --- E. 转回 Tensor ---
        # HWC -> CHW
        final_tensor = torch.from_numpy(final_img).permute(2, 0, 1).to(torch.float32)
        output_tensor_list.append(final_tensor)

    # 堆叠 Batch
    if len(output_tensor_list) > 0:
        output_stack = torch.stack(output_tensor_list).to(device)
    else:
        # 如果 batch 为空或出错，返回全黑
        output_stack = torch.zeros_like(ref_imgs).to(torch.uint8).to(device)

    return {"ref_img_edge_alpha": output_stack}


def save_edge_prob_map(save_path, ref_img, edge_alphas, edge_pixels, overlay_alpha=0.7, line_thickness=1):
    """
    保存断裂边热力图为 PNG
    Args:
        save_path: 保存路径 (e.g., '.../000000_edge.png')
        ref_img: [3, H, W] 或 [H, W, 3] 的 numpy 数组 (原始 RGB 图像)
        edge_alphas: [E] numpy 数组 (预测概率)
        edge_pixels: list of list [(x,y)...] (像素坐标，假设归一化 [-1, 1])
    """
    # 1. 处理底图 (Ref Image)
    # 如果是 CHW 格式 (3, H, W)，转为 HWC
    if ref_img.shape[0] == 3:
        ref_img = np.transpose(ref_img, (1, 2, 0))

    # 反归一化并转 uint8 (处理 ImageNet Norm 或 简单的 min-max)
    if ref_img.dtype != np.uint8:
        ref_img = ref_img - ref_img.min()
        ref_img = ref_img / (ref_img.max() + 1e-8)
        ref_img = (ref_img * 255.0).astype(np.uint8)

    # RGB -> BGR (OpenCV使用)
    ref_img_bgr = cv2.cvtColor(ref_img, cv2.COLOR_RGB2BGR)
    H, W, _ = ref_img_bgr.shape

    # 2. 准备画板
    overlay_layer = np.zeros_like(ref_img_bgr)

    # 3. 预计算色盘 (蓝->红)
    colormap_lut = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in range(256):
        colormap_lut[i, 0] = np.array([i, i, i])
    colormap_lut = cv2.applyColorMap(colormap_lut, cv2.COLORMAP_JET).squeeze(1)

    # 4. 遍历画线
    safe_len = min(len(edge_pixels), len(edge_alphas))

    for i in range(safe_len):
        prob = edge_alphas[i]
        pixels = edge_pixels[i]  # 归一化坐标点列表

        # 过滤掉概率太小的，保持画面干净 (可选)
        # if prob < 0.05 or len(pixels) == 0:
        #     continue

        # 获取颜色
        color_idx = int(np.clip(prob * 255, 0, 255))
        color = colormap_lut[color_idx].tolist()

        # 坐标反归一化 [-1, 1] -> [0, W]
        # 注意：这里假设 pixel 是 (x, y) 格式
        pts_norm = np.array(pixels, dtype=np.float32)
        pts_x = (pts_norm[:, 0] + 1) * (W - 1) / 2.0
        pts_y = (pts_norm[:, 1] + 1) * (H - 1) / 2.0

        pts_real = np.stack([pts_x, pts_y], axis=1).astype(np.int32)
        pts_to_draw = pts_real.reshape((-1, 1, 2))

        cv2.polylines(overlay_layer, [pts_to_draw], isClosed=False, color=color,
                      thickness=line_thickness, lineType=cv2.LINE_AA)

    # 5. 叠加与保存
    mask = np.any(overlay_layer > 0, axis=-1)
    final_img = ref_img_bgr.copy()
    weighted_overlay = cv2.addWeighted(ref_img_bgr, 1.0 - overlay_alpha, overlay_layer, overlay_alpha, 0)
    final_img[mask] = weighted_overlay[mask]

    # 保存
    cv2.imwrite(save_path, final_img)

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