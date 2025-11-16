import numpy as np
import torchvision.utils as vutils
import torch
import torch.nn.functional as F


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
