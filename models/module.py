import torch
import torch.nn as nn
import torch.nn.functional as F



class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1, dilation=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        

    def forward(self,x):
        return F.relu(self.bn(self.conv(x)), inplace=True)

class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1, dilation=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)

class ConvBnReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1, dilation=1):
        super(ConvBnReLU1D, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)
        

class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))



def differentiable_warping(src_fea, src_proj, ref_proj, depth_samples):
    """Differentiable homography-based warping, implemented in Pytorch.

    Args:
        src_fea: [B, C, Hin, Win] source features, for each source view in batch
        src_proj: [B, 4, 4] source camera projection matrix, for each source view in batch
        ref_proj: [B, 4, 4] reference camera projection matrix, for each ref view in batch
        depth_samples: [B, Ndepth, Hout, Wout] virtual depth layers
    Returns:
        warped_src_fea: [B, C, Ndepth, Hout, Wout] features on depths after perspective transformation
    """
    batch, channels, height, width = src_fea.shape
    num_depth = depth_samples.shape[1]
    # Point_s=Ps*逆(Pr)*(d*point_r) 从参考像素点到源像素点公式 todo：之后结合数学理解再细看，非常重要扭曲操作
    with torch.no_grad():
        # proj=Ps*逆(Pr)
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                            torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        # 组成齐次坐标 (x, y, 1)，形状 [B, 3, Hout*Wout]
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        # rot * (u,v,1)^T * d
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]

        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_samples.view(batch, 1, num_depth,
                                                                                            height * width)  # [B, 3, Ndepth, H*W]
        # 再加上平移
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        # 如果 z（proj_xyz 第三行）太小或负（即投影到源相机后点在镜头后面或接近 0）,这里将无效点的坐标设到源视图图像外
        negative_depth_mask = proj_xyz[:, 2:] <= 1e-3
        proj_xyz[:, 0:1][negative_depth_mask] = width
        proj_xyz[:, 1:2][negative_depth_mask] = height
        proj_xyz[:, 2:3][negative_depth_mask] = 1

        # 透视除法：相机投影的核心步骤
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]

        # 将x，y归一化
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1 # [B, Ndepth, H*W]
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1

        # grid 是深度坐标网格，参考视图每个像素在每个深度下，对应源视图的哪个点
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy      

    # 这个grid是通过计算出来的，而传播时的邻居网格是通过训练出来的
    # 得到的是参考图每个像素点在不同深度下源图的特征，然后再重投影在参考图
    warped_src_fea = F.grid_sample(
        src_fea,
        grid.view(batch, num_depth * height, width, 2),
        mode='bilinear',
        padding_mode='zeros',align_corners=True)
    
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea

# p: probability volume [B, D, H, W]
# depth_values: discrete depth values [B, D]
# get expected value, soft argmin
# return: depth [B, 1, H, W]
def depth_regression(p, depth_values):
    depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(p * depth_values, 1)
    depth = depth.unsqueeze(1)
    return depth

def depth_regression_1(p, depth_values):
    depth = torch.sum(p * depth_values, 1)
    depth = depth.unsqueeze(1)
    return depth
