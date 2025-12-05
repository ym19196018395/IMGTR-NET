import os
import numpy as np
from PIL import Image
import torchvision.utils as vutils
import torch.nn.functional as F
import torch

def ensure_dir(path):

    os.makedirs(path, exist_ok=True)

def tensor_to_uint8_image(tensor):
    """tensor: CHW, float in [0,1] -> HWC uint8"""
    arr = tensor.cpu().numpy()
    if arr.ndim == 3:
        arr = np.transpose(arr, (1,2,0))
    arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(arr)

def save_feature_maps_as_grid(feat, out_path, nrow=8):
    """
    feat: Tensor [C,H,W] or [1,C,H,W] or [B,C,H,W] (we will take first in batch)
    将每个通道当作一张灰度图并拼接成网格保存。
    """
    ensure_dir(os.path.dirname(out_path) or ".")
    if feat.dim() == 4:
        feat = feat[0]          # [C,H,W]
    if feat.dim() == 2:
        feat = feat.unsqueeze(0)
    # make a batch where each channel is a single-channel image: [C,1,H,W]
    imgs = feat.unsqueeze(1)   # [C,1,H,W]
    # use torchvision make_grid (normalize 每个通道到0-1)
    grid = vutils.make_grid(imgs, nrow=nrow, normalize=True, scale_each=True)
    # grid: 3-channel if normalize True; but may be 1-channel, handle both
    if grid.shape[0] == 1:
        img = tensor_to_uint8_image(grid.squeeze(0))
    else:
        img = Image.fromarray((grid.cpu().numpy().transpose(1,2,0)*255).astype(np.uint8))
    img.save(out_path)

def save_feature_heatmap(feat, out_path, agg='mean'):
    """
    feat: [C,H,W] or [1,C,H,W] or [B,C,H,W]
    把通道进行平均或最大化得到热力图并保存为彩色图像。
    """
    ensure_dir(os.path.dirname(out_path) or ".")
    if feat.dim() == 4:
        feat = feat[0]   # [C,H,W]
    # reduce channels
    if agg == 'mean':
        heat = feat.mean(0).cpu().numpy()
    else:
        heat = feat.max(0)[0].cpu().numpy()
    # normalize 0-1
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-9)
    # apply colormap (matplotlib) without importing pyplot heavy; use PIL palette via numpy:
    import matplotlib.cm as cm
    cmap = cm.get_cmap('jet')
    colored = cmap(heat)[:, :, :3]  # H W 3
    img = Image.fromarray((colored * 255).astype(np.uint8))
    img.save(out_path)
