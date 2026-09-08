from typing import List, Tuple, Dict

from .PlanePatchMatch import *
from .feature_map import *
from utils import (batch_convert_to_tri_infos_new, build_neighbor_indices, 
                   map_tri_to_pixel_single, compute_normal_map_torch,
                   generate_cross_check_diagnosis_rgb, compute_cross_check_score_gpu)
import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *
from .patchmatch import *

# 对单张图像提取三个尺度的特征（细→中→粗），类似 FPN
class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        
        self.conv0 = ConvBnReLU(3, 8, 3, 1, 1)
        # [B,8,H,W]
        self.conv1 = ConvBnReLU(8, 8, 3, 1, 1)
        # [B,16,H/2,W/2]
        self.conv2 = ConvBnReLU(8, 16, 5, 2, 2)
        self.conv3 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv4 = ConvBnReLU(16, 16, 3, 1, 1)
        # [B,32,H/4,W/4]
        self.conv5 = ConvBnReLU(16, 32, 5, 2, 2)
        self.conv6 = ConvBnReLU(32, 32, 3, 1, 1)
        self.conv7 = ConvBnReLU(32, 32, 3, 1, 1)
        # [B,64,H/8,W/8]
        self.conv8 = ConvBnReLU(32, 64, 5, 2, 2)
        self.conv9 = ConvBnReLU(64, 64, 3, 1, 1)
        self.conv10 = ConvBnReLU(64, 64, 3, 1, 1)
        
    
        self.output1 = nn.Conv2d(64, 64, 1, bias=False)
        self.inner1 = nn.Conv2d(32, 64, 1, bias=True)
        self.inner2 = nn.Conv2d(16, 64, 1, bias=True)

        self.output2 = nn.Conv2d(64, 32, 1, bias=False)
        self.output3 = nn.Conv2d(64, 16, 1, bias=False)

        # 新增：将 FPN 主干的 64 通道压缩回 8 通道，供 Stage 0 抛光使用
        self.output0 = nn.Conv2d(64, 8, 1, bias=False)
        # 新增：将 conv1 的 8 通道映射到 FPN 主干的 64 通道
        self.inner3 = nn.Conv2d(8, 64, 1, bias=True)
        
     
    def forward(self, x):
        output_feature={}
        
        conv1 = self.conv1(self.conv0(x))
        # 分别对应不同分辨率的特征图 高中低
        conv4 = self.conv4(self.conv3(self.conv2(conv1)))
        conv7 = self.conv7(self.conv6(self.conv5(conv4)))
        conv10 = self.conv10(self.conv9(self.conv8(conv7)))

        # 粗尺度特征
        output_feature['stage_3'] = self.output1(conv10)
        # 把粗特征上采样并与更高分辨率特征相加（类似 FPN 思路)
        intra_feat = F.interpolate(conv10, scale_factor=2, mode="bilinear") + self.inner1(conv7)
        del conv7, conv10
        # 中尺度特征
        output_feature['stage_2'] = self.output2(intra_feat)
        
        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="bilinear") + self.inner2(conv4)
        del conv4
        # 细尺度
        output_feature['stage_1'] = self.output3(intra_feat)
        del conv1, intra_feat
            
        return output_feature


class GeometricRefinement(nn.Module):
    def __init__(self, in_channels=16):  # 👈 精准匹配你 FeatureNet 的 stage_1 通道数
        super(GeometricRefinement, self).__init__()

        # 提取 Stage 1 的高频特征 (16 -> 16)
        # 假设 ConvBnReLU(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv0 = ConvBnReLU(in_channels, 16, 3, 1, 1)

        # 提取低分辨率深度特征 (1 -> 16)
        self.conv1 = ConvBnReLU(1, 16, 3, 1, 1)
        self.conv2 = ConvBnReLU(16, 16, 3, 1, 1)

        # 反卷积上采样深度特征
        self.deconv = nn.ConvTranspose2d(16, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(16)

        # 融合 RGB特征 与 深度特征 (16 + 16 = 32)
        self.conv3 = ConvBnReLU(32, 16, 3, 1, 1)

        # 预测残差 (16 -> 1)
        self.res = nn.Conv2d(16, 1, 3, padding=1, bias=False)

    def forward(self, ref_feature, depth_0, depth_min, depth_max):
        """
        ref_feature: Stage 1 的特征 [B, 16, H/2, W/2]
        depth_0: Stage 2 的深度图 [B, 1, H/4, W/4]
        """
        batch_size = depth_min.size(0)

        # 1. 深度归一化到 [0, 1]
        d_min = depth_min.view(batch_size, 1, 1, 1)
        d_max = depth_max.view(batch_size, 1, 1, 1)
        depth_norm = (depth_0 - d_min) / (d_max - d_min + 1e-6)

        # 2. 特征提取
        feat_out = self.conv0(ref_feature)
        depth_out = F.relu(self.bn(self.deconv(self.conv2(self.conv1(depth_norm)))), inplace=True)

        # 3. 拼接并预测残差
        cat = torch.cat((depth_out, feat_out), dim=1)
        res = self.res(self.conv3(cat))

        # 4. 🔥 致命修复：双线性插值基础面 (绝不使用 nearest)
        depth_up = F.interpolate(depth_norm, scale_factor=2, mode="bilinear", align_corners=False)

        # 5. 加残差并反归一化
        depth_refined_norm = depth_up + res
        depth_refined = depth_refined_norm * (d_max - d_min) + d_min

        return depth_refined

class Refinement(nn.Module):
    def __init__(self):
        
        super(Refinement, self).__init__()
        
        # img: [B,3,H,W]
        self.conv0 = ConvBnReLU(3, 8)
        # depth map:[B,1,H/2,W/2]
        self.conv1 = ConvBnReLU(1, 8)
        self.conv2 = ConvBnReLU(8, 8)
        # 转置卷积（反卷积）
        self.deconv = nn.ConvTranspose2d(8, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False)
        
        self.bn = nn.BatchNorm2d(8)
        self.conv3 = ConvBnReLU(16, 8)
        self.res = nn.Conv2d(8, 1, 3, padding=1, bias=False)
        
        
    def forward(self, img, depth_0, depth_min, depth_max):
        batch_size = depth_min.size()[0]
        # pre-scale the depth map into [0,1]
        depth = (depth_0-depth_min.view(batch_size,1,1,1))/(depth_max.view(batch_size,1,1,1)-depth_min.view(batch_size,1,1,1))
        
        conv0 = self.conv0(img)
        deconv = F.relu(self.bn(self.deconv(self.conv2(self.conv1(depth)))), inplace=True)
        cat = torch.cat((deconv, conv0), dim=1)
        del deconv, conv0
        # depth residual
        res = self.res(self.conv3(cat))
        del cat

        depth = F.interpolate(depth, scale_factor=2, mode="nearest") + res
        # convert the normalized depth back
        depth = depth * (depth_max.view(batch_size,1,1,1)-depth_min.view(batch_size,1,1,1)) + depth_min.view(batch_size,1,1,1)

        return depth


class Stage0RefinementNet_V2(nn.Module):
    def __init__(self, in_channels=13):
        """
        全分辨率几何抛光机 (14通道极限内聚版)
        输入通道编队 (总计 14):
        8 (Stage0语义特征) + 1 (无损安全视差) + 3 (物理级Base法向) + 1 (无泄漏网格门控) + 1 (合法晶格掩码)
        """
        super().__init__()

        # 针对全分辨率小批次 (B=1~2) 训练，全面废除 BatchNorm，采用 InstanceNorm2d 稳固量纲
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),

            # 膨胀卷积（空洞系数=2），在不增加参数的前提下，跨越网格边缘捕获大尺度曲率
            nn.Conv2d(16, 16, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),

            # 输出层：1通道深度位移残差，2通道切空间偏转微扰 (留作后用)
            nn.Conv2d(16, 3, kernel_size=3, padding=1, bias=True)
        )

        # 终极物理断流：强制零初始化。确保训练初始状态输出绝对为 0，誓死捍卫 Stage 1 刚性大面成果
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, feat_s0, final_planes, tri_id_map_stage0, W_plane_tri, intrinsics_s0, depth_range, depth_stage1_pixels):
        """
        在前向传播内部全自动流转几何计算图
        Args:
            feat_s0:           Stage 0 原图分辨率特征 [B, 8, H0, W0]
            final_planes:      Stage 1 优化的稀疏平面参数 [B, N_tri, 4]
            tri_id_map_stage0: Stage 0 的密集三角形 ID 索引图 [B, H0, W0] (无效区为-1)
            W_plane_tri:      平面置信度 [B, N_tri, 1] 或 [B, N_tri]
            intrinsics_s0:     Stage 0 的相机内参矩阵 [B, 3, 3]
            depth_range:       当前场景的深度裁剪边界 (min_d, max_d)
        """
        B, H0, W0 = tri_id_map_stage0.shape
        device = final_planes.device

        # =====================================================================
        # 1. 密集网格索引安全离散映射 (Advanced Indexing)
        # =====================================================================
        valid_mask_s0 = (tri_id_map_stage0 >= 0).unsqueeze(1)  # [B, 1, H0, W0]
        safe_id_map = torch.where(tri_id_map_stage0 >= 0, tri_id_map_stage0, torch.zeros_like(tri_id_map_stage0)).long()
        batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(-1, H0, W0)

        # 1.a 提取并广播基础平面的 N 和 d
        pixel_planes_raw = final_planes[batch_idx, safe_id_map].permute(0, 3, 1, 2)
        N_base = pixel_planes_raw[:, 0:3, :, :]
        d_base = pixel_planes_raw[:, 3:4, :, :]

        # 对无效区域赋默认法向防爆
        N_base = torch.where(valid_mask_s0, N_base, torch.tensor([0.0, 0.0, -1.0], device=device).view(1, 3, 1, 1))

        # 1.b 🚀 【物理语义反转】：置信度 (1=平) -> 残差门控 (0=平, 关死残差)
        W_plane_s0 = W_plane_tri[batch_idx, safe_id_map]  # 得到 [B, H0, W0, 1]
        W_plane_s0 = W_plane_s0.permute(0, 3, 1, 2)  # 转换为 [B, 1, H0, W0]
        M_gating_s0 = 1.0 - W_plane_s0  # 核心反转操作！

        # 无效区域不需要任何修改，门控死死关掉
        M_gating_s0 = torch.where(valid_mask_s0, M_gating_s0, torch.zeros_like(M_gating_s0)).detach()

        # =====================================================================
        # 2. 解析刚性射线求交深度场 (Z_base) 并无损双线性对齐像素级自由深度场 (Z_pixel)
        # =====================================================================
        # 2.a 计算刚性面片解析深度场
        Z_base = self._analytical_ray_intersection(N_base, d_base, intrinsics_s0, depth_range, valid_mask_s0)

        # 2.b 强力对齐原版自由像素深度：将较低分辨率的自由深度双线性插值扩展到 Stage 0 全分辨率
        Z_pixel_s0 = F.interpolate(depth_stage1_pixels, size=(H0, W0), mode='bilinear', align_corners=False)

        # 强制断开所有先验深度底图的因果链，Stage 0 只能作为抛光机，不能反向教 Stage 1 做人！
        Z_base_safe = Z_base.detach()
        Z_pixel_s0_safe = Z_pixel_s0.detach()

        Z_hybrid = Z_base_safe * (1.0 - M_gating_s0) + Z_pixel_s0_safe * M_gating_s0

        # =====================================================================
        # 3. 构建视差空间护城河 (无量纲化相对视差)
        # =====================================================================
        disparity_base = torch.where(
            valid_mask_s0,
            1.0 / (Z_hybrid.detach() + 1e-6),
            torch.zeros_like(Z_hybrid)
        )

        # =====================================================================
        # 4. 极致压缩的 13 通道异构数据总线拼装
        # =====================================================================
        cnn_input = torch.cat([
            feat_s0,  # [8 通道]
            disparity_base,  # [1 通道]
            N_base.detach(),  # [3 通道]
            M_gating_s0,  # [1 通道]
        ], dim=1)  # 严格 13 通道

        # =====================================================================
        # 5. 轰出残差！
        # =====================================================================
        PLANAR_THRESHOLD = 0.8  # 硬路由阈值
        is_planar = (W_plane_s0 > PLANAR_THRESHOLD).detach()  # [B, 1, H0, W0]

        res = self.refine(cnn_input)

        # [通道 0]: 深度残差;
        delta_z_raw = res[:, 0:1, :, :]

        # 7. 相对深度比例钳制 (Relative Depth Percentage Clamping)
        gamma_pct = 0.12
        delta_z_B = Z_pixel_s0_safe * gamma_pct * torch.tanh(delta_z_raw)
        Z_branch_B = Z_pixel_s0_safe + delta_z_B

        # 8. 硬路由应用：深度场合成
        Z_final = torch.where(is_planar, Z_base_safe, Z_branch_B)

        # 9. 切空间法向合成 (解耦物理法向)
        N_analytic = N_base.detach()
        N_screenspace, _ = compute_normal_map_torch(Z_branch_B.detach(), intrinsics_s0, mask=None, smooth=True)
        N_final = torch.where(is_planar.repeat(1, 3, 1, 1), N_analytic, N_screenspace)

        return res, Z_final, N_final, is_planar, valid_mask_s0

    def apply_safe_tangent_refinement(self,N_base: torch.Tensor, delta_uv: torch.Tensor,
                                      M_gating: torch.Tensor) -> torch.Tensor:
        """
        【工业级】基于免奇点切空间投射的法向残差安全合成算子
        采用 Duff et al. 2017 分支无感正交基构建，彻底免疫 Gimbal Lock 与 Normalize 梯度核弹。

        参数:
            N_base:   [B, 3, H, W] 基础法向量 (Stage 1 提供，假定已归一化，务必是 detach 过的)
            delta_uv: [B, 2, H, W] CNN输出的切向微扰 (推荐外部已用 tanh 限幅)
            M_gating: [B, 1, H, W] 残差释放阀门 (1=允许修正的曲面，0=绝对平滑的刚性墙面)
        """
        # 1. 物理分量剥离
        x = N_base[:, 0:1, :, :]
        y = N_base[:, 1:2, :, :]
        z = N_base[:, 2:3, :, :]

        # 2. 符号流形提取与无分支正交基构建 (Duff et al. 2017)
        # torch.where 维持 SIMT 计算并行，无分支损耗
        sign_z = torch.where(z >= 0.0, torch.ones_like(z), -torch.ones_like(z))

        # 架构师注：sign_z + z 的绝对值恒 >= 1，数学上绝对不可能为 0，摒弃多余的 eps！
        a = -1.0 / (sign_z + z)
        b = x * y * a

        # 构造绝对正交的切线向量 T1 与副法线向量 T2
        t1_x = 1.0 + sign_z * (x ** 2) * a
        t1_y = sign_z * b
        t1_z = -sign_z * x
        T1 = torch.cat([t1_x, t1_y, t1_z], dim=1)

        t2_x = b
        t2_y = sign_z + (y ** 2) * a
        t2_z = -y
        T2 = torch.cat([t2_x, t2_y, t2_z], dim=1)

        # 3. 门控钳制与步长封锁 (Gated Clamping)
        # 乘以 0.1 作为最大偏转弧度约束，再乘以门控矩阵
        # 白墙区 (M_gating=0): delta_u/v 彻底归零，T1/T2 被抛弃，强力维持 N_base
        ratio=0.1
        delta_u = delta_uv[:, 0:1, :, :] * ratio * M_gating
        delta_v = delta_uv[:, 1:2, :, :] * ratio * M_gating

        # 4. 几何解析归一化 (Analytic Normalization)
        # 由于 T1, T2 垂直于 N_base，合成向量长度平方恒为 1 + u^2 + v^2。
        # 绝不使用 F.normalize()，从根本上消灭由于向量可能为 0 带来的除零梯度爆炸！
        len_sq = 1.0 + (delta_u ** 2) + (delta_v ** 2)
        N_final = (N_base + delta_u * T1 + delta_v * T2) / torch.sqrt(len_sq)

        return N_final

    def _analytical_ray_intersection(self, N_base, d_base, intrinsics, depth_range, valid_mask):
        """
        【内部轻量级算子】纯数学解析射线求交，无任何可视化脏操作干扰
        """
        B, _, H, W = N_base.shape
        device = N_base.device

        fx = intrinsics[:, 0, 0].view(B, 1, 1, 1)
        fy = intrinsics[:, 1, 1].view(B, 1, 1, 1)
        cx = intrinsics[:, 0, 2].view(B, 1, 1, 1)
        cy = intrinsics[:, 1, 2].view(B, 1, 1, 1)

        # 构造网格
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        x_grid = x_grid.view(1, 1, H, W).expand(B, 1, -1, -1)
        y_grid = y_grid.view(1, 1, H, W).expand(B, 1, -1, -1)

        # 射线方向
        ray_x = (x_grid - cx) / fx
        ray_y = (y_grid - cy) / fy
        # ray_z = 1.0 (隐式存在)

        # 解析点积： N_x * ray_x + N_y * ray_y + N_z * 1.0
        dot_product = N_base[:, 0:1] * ray_x + N_base[:, 1:2] * ray_y + N_base[:, 2:3]

        # 安全除法保护
        valid_dot = torch.abs(dot_product) > 1e-4
        dot_safe = torch.where(valid_dot, dot_product, torch.sign(dot_product + 1e-10) * 1e-4)

        # Z = -d / (N dot Ray)
        depth = (-d_base / dot_safe).abs()

        # 数值截断 (Clamp)
        if depth_range is not None:
            min_d, max_d = depth_range
            if isinstance(min_d, torch.Tensor): min_d = min_d.view(-1, 1, 1, 1).to(device)
            if isinstance(max_d, torch.Tensor): max_d = max_d.view(-1, 1, 1, 1).to(device)
            depth = torch.clamp(depth, min=min_d, max=max_d)
            # fill_val = min_d.expand_as(depth) if isinstance(min_d, torch.Tensor) else min_d
        else:
            depth = torch.clamp(depth, max=600.0)
            # fill_val = depth.mean()

        # 未命中区域填充保底值
        fallback_depth = torch.zeros_like(depth)
        depth = torch.where(valid_mask, depth, fallback_depth)
        return depth

class PatchmatchNet(nn.Module):
    """ 主体网络，执行 coarse→fine 的可学习 PatchMatch 深度估计"""
    def __init__(self, patchmatch_interval_scale = [0.005, 0.0125, 0.025], propagation_range = [6,4,2],
                patchmatch_iteration = [1,2,2], patchmatch_num_sample = [8,8,16], propagate_neighbors = [0,8,16],
                evaluate_neighbors = [9,9,9]):
        """Initialize modules in PatchmatchNet

        Args:
            patchmatch_interval_scale: depth interval scale in patchmatch module,每个 stage 的深度采样间隔尺度（影响随机扰动与采样间隔）
            propagation_range: propagation range，每个 stage 的传播范围（邻域半径或数量相关）
            patchmatch_iteration: patchmatch iteration number，每个 stage 的迭代次数（PatchMatch 内部的迭代步数）
            patchmatch_num_sample: patchmatch number of samples，每个 stage 初始随机采样数量（候选深度数）
            propagate_neighbors: number of propagation neighbors，在传播/evaluation 阶段每像素考虑多少邻域位置
            evaluate_neighbors: number of propagation neighbors for evaluation
        """
        super(PatchmatchNet, self).__init__()

        self.stages = 4
        # 获取特征
        self.feature = FeatureNet()
        self.patchmatch_num_sample = patchmatch_num_sample
        
        num_features = [8, 16, 32, 64]

        self.propagate_neighbors = propagate_neighbors
        self.evaluate_neighbors = evaluate_neighbors
        # number of groups for group-wise correlation
        self.G = [4,8,8]

        # 为每一个阶段设置一个PatchMatch模块，从0开始，对应1,2,3阶段
        for l in range(self.stages-1):
            #如果是stage_3需要随机初始化
            if l == 2:
                patchmatch = PatchMatch(True, propagation_range[l], patchmatch_iteration[l], 
                            patchmatch_num_sample[l], patchmatch_interval_scale[l],
                            num_features[l+1], self.G[l], self.propagate_neighbors[l], l+1,
                            evaluate_neighbors[l])
            else:
                patchmatch = PatchMatch(False, propagation_range[l], patchmatch_iteration[l], 
                            patchmatch_num_sample[l], patchmatch_interval_scale[l], 
                            num_features[l+1], self.G[l], self.propagate_neighbors[l], l+1,
                            evaluate_neighbors[l])
            # 使用 setattr 函数将创建的 patchmatch 实例设置为 self 对象的一个属性
            setattr(self, f'patchmatch_{l+1}', patchmatch)
        # 最后进行上采样 输出完整的深度图
        self.upsample_net = Refinement()

        # ym—need-modify 后面可能需要加入传播里面
        # self.edge_head = EdgeHead(num_features[1])

        # ym-add 1.21 添加PlanePatchMatch
        self.num_hypotheses = 3
        self.tau = 1.0

        self.plane_patchmatch_agent = PlanePatchMatchModule(num_hypotheses=self.num_hypotheses, G=8,
                                                            feat_channels=num_features[1],
                                                            propagator_iter=3)

        self.stage1_refine = GeometricRefinement(in_channels=16)

        # 注入高度工程集成的 Stage 0 抛光引擎
        self.stage0_refiner = Stage0RefinementNet_V2(in_channels=13)

    def forward(self, imgs, proj_matrices,intrinsics_mats,depth_min, depth_max,vertexs,lines,triangles,depth_stage_1,lambda_c, lambda_s,current_temp):
        

        imgs_0 = torch.unbind(imgs['stage_0'], 1)
        imgs_1 = torch.unbind(imgs['stage_1'], 1)
        imgs_2 = torch.unbind(imgs['stage_2'], 1)
        imgs_3 = torch.unbind(imgs['stage_3'], 1)
        del imgs
        
        self.imgs_0_ref = imgs_0[0]
        self.imgs_1_ref = imgs_1[0]
        self.imgs_2_ref = imgs_2[0]
        self.imgs_3_ref = imgs_3[0]
        del imgs_1, imgs_2, imgs_3

        # 这个是已经处理好的投影矩阵
        self.proj_matrices_0 = torch.unbind(proj_matrices['stage_0'].float(), 1)
        self.proj_matrices_1 = torch.unbind(proj_matrices['stage_1'].float(), 1)
        self.proj_matrices_2 = torch.unbind(proj_matrices['stage_2'].float(), 1)
        self.proj_matrices_3 = torch.unbind(proj_matrices['stage_3'].float(), 1)
        del proj_matrices
        
        assert len(imgs_0) == len(self.proj_matrices_0), "Different number of images and projection matrices"

        # step 0 ym—problem,已经在dataloader里面进行了处理 调整图像尺寸：保证输入宽高是 8 的倍数（便于下采样），同时更新相机内参

        # step 1. Multi-scale feature extraction
        features = []
        for img in imgs_0:
            output_feature = self.feature(img)
            features.append(output_feature)
            # ym_need_add 打印出特征图的样子以及特征数据
            # features_to_tensorboard(features,img[0])
        del imgs_0
        ref_feature, src_features = features[0], features[1:]
        
        depth_min = depth_min.float()
        depth_max = depth_max.float()

        # step 2. Learning-based patchmatch
        depth = None
        view_weights = None
        depth_patchmatch = {}
        refined_depth = {}

        continuity_loss = [] # 连续性损失
        continuity_s_loss = [] # 正则化损失，防止边断裂概率都为1
        output_plane={
            'final_plane':[], # 最终结果平面 B,N,4
            'depth_stage1_pixels':[],# stage1的深度图
            'normal_pro':[], # 传播后法向量--可视化
            'normal_pro_pure': [],  # 传播后法向量像素级
            'tri_id_map':[],# 三角形stage1下的id图
            'tri_id_map_stage0': [],  # 三角形stage1下的id图
            'depth_no_pro': [],  # 刚拟合完的深度值
            'normal_no_pro': [],  # 刚拟合完的法向量可视化
            'pixel_costs':[], # 计算出来的代价
            'W_plane_pixel':[], # 平面置信度 像素级（传播后）
            'W_plane_pixel_init': [],  # 传播前物理冷启动置信度 像素级
            'W_plane_tri':[], # 平面置信度 三角级别（传播后）
            'W_plane_tri_init': [],  # 传播前物理冷启动 三角级别
            'final_normal':[], # 最终法向量
            'pixel_counts':[] #  每个三角形包含的密集像素计数 ,原分辨率的
        }
        score = []
        
        for l in reversed(range(1, self.stages)):# for（int i = stages-1; i>0 ;i--）
            # 取出当前尺度的源特征列表 src_features_l
            src_features_l = [src_fea[f'stage_{l}'] for src_fea in src_features]
            projs_l = getattr(self, f'proj_matrices_{l}')
            # 参考和源图的投影矩阵，已经在处理完毕
            ref_proj, src_projs = projs_l[0], projs_l[1:]

            # 初始化patchmatch，通过getattr方式，分别对应stage3和stage2
            if l > 1:
                # 只在第一回合获得视图权重
                depth, _, view_weights = getattr(self, f'patchmatch_{l}')(ref_feature[f'stage_{l}'], src_features_l, 
                                        ref_proj, src_projs, 
                                        depth_min, depth_max, depth=depth, img=getattr(self,f'imgs_{l}_ref'), view_weights=view_weights)
            else:
                # 原先的操作
                # depth, score, _ = getattr(self, f'patchmatch_{l}')(ref_feature[f'stage_{l}'], src_features_l,
                #                         ref_proj, src_projs,
                #                         depth_min, depth_max, depth=depth,img=getattr(self,f'imgs_{l}_ref'), view_weights=view_weights)

                #===================对stage1 进行planepatchmatch===========================

                # 1. 运行原 Stage 1 的标准像素级 PatchMatch 获取更精细的 1/2 深度图与视图权重
                depth_pm, score, view_weights = getattr(self, f'patchmatch_{l}')(
                    ref_feature[f'stage_{l}'], src_features_l,
                    ref_proj, src_projs,
                    depth_min, depth_max, depth=depth, img=getattr(self, f'imgs_{l}_ref'), view_weights=view_weights
                )

                # 根据stage2粗深度图来拟合stage1的法向量和深度图
                tri_infos = []

                # 不经过双边残差网络上采样，直接使用 Stage 1 PatchMatch 的输出深度图
                depth_stage1_init = depth_pm[-1]

                # 获得stage1的长和宽
                _, _, height, width = depth_stage1_init.size()
                device = depth_stage1_init.get_device()

                # ================================================================
                # 1. 处理数据
                # ================================================================
                    # 对数据进行一个转化，会在里面得到边的像素集合，以及三角面的顶点和质点（归一化的）
                    # 传入原分辨率的图片，里面会进行一个归一化操作 传入的是原分辨率的
                tri_infos = batch_convert_to_tri_infos_new(vertexs, lines, triangles, height * 2, width * 2, device)

                del vertexs,lines,triangles

                intrinsics_s1 = torch.unbind(intrinsics_mats['stage_1'].float(), 1)
                # 获得stage1参考图的内参矩阵
                ref_intrinsics = intrinsics_s1[0]

                # 批次里面最大三角形数量
                max_tri_num = max(item['batch_num_tri'] for item in tri_infos)
                max_tri_num = max(max_tri_num)

                # 每个三角形的邻居面，如果只有两个面，另外一个面是自己 # [B, N_max, 3]
                neighbor_indices_batched = build_neighbor_indices(tri_infos, max_tri_num,device)


                # ================================================================
                # 2. 传入PlanePatchmatch模型,根据stage2的深度信息拟合stage1的平面
                # ================================================================

                # 平面拟合
                self.dense_plane_fitter = DensePlaneFitter(height, width, device,num_hypotheses=self.num_hypotheses,perturbation_range=0.05,
                                                           depth_max=depth_max,
                                                           depth_min=depth_min)

                # 提取全部 4 个源视角纯像素级 Stage 1 原生深度 (Views 1, 2, 3, 4)，全向立体覆盖几何死角
                src_raw_depths = self.compute_source_raw_depths(features, depth_min, depth_max, src_indices=[1, 2, 3, 4])

                (depth_samples, pixel_costs, view_weights, normal_samples, output_plane['final_plane'], edge_alpha,
                 continuity_loss, smoothness_loss, W_plane_pixel, W_plane_tri, W_plane_tri_polarized,
                 pixel_counts_tensor, W_plane_pixel_init, W_plane_tri_init) = self.plane_patchmatch_agent.forward(
                                                                    self.dense_plane_fitter,
                                                                    depth_stage1_init.detach(), tri_infos, # todo：暂时不让传播阶段去影响原来pixelpatchmatch阶段
                                                                    ref_feature[f'stage_{l}'],
                                                                    src_features_l,
                ref_proj, src_projs, intrinsics_s1,depth_min, depth_max, view_weights.detach(),
                neighbor_indices_batched = neighbor_indices_batched,
                lambda_c=lambda_c,lambda_s=lambda_s,current_temp=current_temp,
                src_raw_depths=src_raw_depths
                )

                # Score (Confidence) = -min_cost
                score = -torch.min(pixel_costs, dim=3)[0].unsqueeze(1)  # [B, 1, H, W]

                # ================================================================
                # 3. 可视化结果，返回结果
                # ================================================================

                # 转化为tensor形式(B,H,W)
                # 步骤1：去掉每个Tensor中长度为1的维度（把[1, H, W]转成[H, W]）
                processed_list = [tensor.squeeze(0)for tensor in tri_infos[0]['tri_id_map']]
                processed_list_stage0 = [tensor.squeeze(0) for tensor in tri_infos[0]['tri_id_map_stage0']]
                # 步骤2：在第0维（batch维）堆叠，得到[B, H, W]
                tri_id_map_tensor = torch.stack(processed_list, dim=0)
                tri_id_map_stage0_tensor = torch.stack(processed_list_stage0, dim=0)

                pixel_planes_s1 = map_tri_to_pixel_single(output_plane['final_plane'], tri_id_map_tensor, height,
                                                               width)
                pixel_normal_s1_pure = pixel_planes_s1[:, :3, :, :]  # 获得绝对纯净的 [B, 3, H, W] 物理法向量！

                output_plane['normal_pro_pure'] = pixel_normal_s1_pure
                output_plane['tri_id_map'] = tri_id_map_tensor
                output_plane['tri_id_map_stage0'] = tri_id_map_stage0_tensor

                # stage2 产生的深度图放大后，patchmatch产生的深度图
                output_plane['depth_stage1_pixels']=depth_stage1_init
                # 经过传播得到的平面
                output_plane['normal_pro']=normal_samples[1]

                output_plane['pixel_counts'] = pixel_counts_tensor

                # 没有进行传播得到的平面，刚拟合完的初始平面
                # 将B,N,4 分别转化为,B,H,W,1 和B,H,W,3 可视化用
                output_plane['depth_no_pro'] = depth_samples[0]
                output_plane['normal_no_pro'] = normal_samples[0]

                output_plane['pixel_costs'] = pixel_costs

                output_plane['W_plane_pixel'] = W_plane_pixel
                output_plane['W_plane_pixel_init'] = W_plane_pixel_init
                output_plane['W_plane_tri'] = W_plane_tri
                output_plane['W_plane_tri_init'] = W_plane_tri_init

                # ==============================================================================
                # 👑 [实验零：Post-GNN 四向源视角 Cross-Check 几何连续软降级与门控同步]
                # ==============================================================================
                # 对 GNN 最终深度 depth_samples[-1] 执行 GPU 极速四向重投影 (传入 pixel_normal_s1_pure 做大倾角自适应补偿)
                max_ratio_deg = compute_cross_check_score_gpu(
                    depth_ref=depth_samples[-1],
                    ref_proj=self.proj_matrices_1[0],
                    src_raw_depths=src_raw_depths,
                    src_projs=self.proj_matrices_1[1:],
                    tri_id_map=tri_id_map_tensor,
                    max_tri_num=W_plane_tri.shape[1],
                    base_depth_thresh=0.12,
                    pixel_dist_thresh=1.2,
                    normal_ref=pixel_normal_s1_pure
                )
                
                # 3. 连续乘法软降级约束 (Sigmoid 门控压制)
                if W_plane_tri.dim() == 2:
                    W_plane_tri = W_plane_tri.unsqueeze(-1)
                
                W_plane_tri_pre_cc = W_plane_tri.clone().detach()

                # Sigmoid 陡峭连续软压制门控 (tau=0.40 严格对齐 Pre-GNN 门限, T=0.04, min_penalty=0.25)
                tau_deg = 0.40
                temperature = 0.04
                min_penalty = 0.25
                penalty = min_penalty + (1.0 - min_penalty) * torch.sigmoid((max_ratio_deg - tau_deg) / temperature)
                W_plane_tri = W_plane_tri * penalty
                
                # 4. 同步广播生成像素级置信度 W_plane_pixel 并更新输出字典 (彻底废除极化算子)
                W_plane_pixel = self.plane_patchmatch_agent._scatter_triangle_to_pixel(W_plane_tri, tri_id_map_tensor)
                output_plane['W_plane_tri'] = W_plane_tri
                output_plane['W_plane_pixel'] = W_plane_pixel
                output_plane['cross_check_ratio_deg'] = max_ratio_deg

                # 5. 👑 生成 Cross-Check 三态诊断图并打包进 output_plane (绿: 保持真面 | 红: 降级伪面 | 蓝黑: 背景非平面)
                W_plane_pixel_orig = self.plane_patchmatch_agent._scatter_triangle_to_pixel(W_plane_tri_pre_cc, tri_id_map_tensor)
                output_plane['cross_check_diagnosis_rgb'] = generate_cross_check_diagnosis_rgb(
                    W_plane_pixel_orig=W_plane_pixel_orig,
                    W_plane_pixel_final=W_plane_pixel,
                    tri_id_map=tri_id_map_tensor,
                    threshold=0.80
                )
                
                # 计算用于光度 Ambiguity 诊断的 pixel_cost_min 和 view_weights_mean
                # pixel_costs 形状为 [B, H, W, K], 在 K 维度取 min 得到最匹配代价并升维
                pixel_cost_min = torch.min(pixel_costs, dim=3)[0].unsqueeze(1) # [B, 1, H, W]
                # view_weights 形状为 [B, Nview-1, H, W], 在视角维度求均值并升维
                view_weights_mean = torch.mean(view_weights, dim=1, keepdim=True) # [B, 1, H, W]

                output_plane['pixel_cost_min'] = pixel_cost_min
                output_plane['view_weights_mean'] = view_weights_mean

                # 读取并塞入未加权代价与方差图
                pixel_costs_raw = getattr(self.plane_patchmatch_agent, 'pixel_costs_raw', None)
                cost_variance = getattr(self.plane_patchmatch_agent, 'cost_variance', None)
                if pixel_costs_raw is not None:
                    output_plane['pixel_cost_min_raw'] = pixel_costs_raw.permute(0, 3, 1, 2)  # [B, 1, H, W]
                if cost_variance is not None:
                    output_plane['cost_variance'] = cost_variance.permute(0, 3, 1, 2)  # [B, 1, H, W]
                output_plane['view_weights_mean'] = view_weights_mean

                # 取法向量
                # tri_normals = before_guess_planes[..., :3]  # 形状变为 [B, N_tri, 3]

                # 生成gt-stage-1的三角平面深度图和法向量图
                # plane_hypothesis_svd_gt = self.dense_plane_fitter.By_SVD_Plane(
                #     depth_stage2=depth_stage_1,
                #     tri_id_map=tri_id_map_tensor,
                #     intrinsics_s1=ref_intrinsics,
                #     max_num_triangles=max_tri_num
                # )
                #
                # depth_gt_plane_stage_1, normal_gt_plane_stage_1 = visualizer.render_from_planes(
                #     plane_hypothesis_svd_gt, tri_id_map_tensor, ref_intrinsics
                #     , depth_range=(depth_min, depth_max))
                #
                # output_plane['depth_gt'] = depth_gt_plane_stage_1
                # output_plane['normal_gt'] = normal_gt_plane_stage_1

                # planepatchmatch最终预测结果，里面也有两份一份是像素级深度 另一份是平面级深度
                depth_samples[0]=depth_stage1_init
                depth = depth_samples

            
            del src_features_l, ref_proj, src_projs, projs_l

            # 存放各个阶段的深度图，并将最新得到的深度图进行分出来进行一个上采样来适应下一个阶段
            depth_patchmatch[f'stage_{l}'] = depth
            # 这里数据分为两份，一份用来上采样，一份用来返回
            depth = depth[-1].detach()


            if l > 1:
                # upsampling the depth map and pixel-wise view weight for next stage
                depth = F.interpolate(depth,
                                    scale_factor=2, mode='nearest')
                # 视图权重进行上采样，以便于后续阶段用
                view_weights = F.interpolate(view_weights,
                                    scale_factor=2, mode='nearest')

        # step 3. Refinement: 使用原版 PatchmatchNet 经过验证的 Refinement (upsample_net) 进行全图残差细化
        # 构建 Stage 1 物理混合融合深度 (Z_fused) 作为 Stage 0 上采样的初始输入底图
        w_pixel_s1 = output_plane.get("W_plane_pixel", None)
        if w_pixel_s1 is not None:
            mask_planar_s1 = (w_pixel_s1 >= 0.80)
            depth_stage1_fused = torch.where(mask_planar_s1, depth, output_plane['depth_stage1_pixels'].detach())
        else:
            depth_stage1_fused = depth

        depth = self.upsample_net(self.imgs_0_ref, depth_stage1_fused, depth_min, depth_max)
        refined_depth['stage_0'] = depth

        # 计算 Stage 0 法向量（用于 TensorBoard 与可视化系统兼容）
        N_final, _ = compute_normal_map_torch(depth.detach(), intrinsics_mats['stage_0'][:, 0], mask=None, smooth=True)
        output_plane['final_normal'] = N_final

        # 提取 Stage 0 平面掩码，供 train_whu.py 记录 stage0_planar_mae 和 stage0_curved_mae 指标
        tri_id_map_stage0 = output_plane['tri_id_map_stage0']
        H0, W0 = tri_id_map_stage0.shape[1], tri_id_map_stage0.shape[2]
        W_tri_in = W_plane_tri.detach() if W_plane_tri.dim() == 3 else W_plane_tri.unsqueeze(-1).detach()
        W_plane_s0 = map_tri_to_pixel_single(W_tri_in, tri_id_map_stage0, H0, W0)
        output_plane['is_planar_s0'] = (W_plane_s0 >= 0.80) & (tri_id_map_stage0.unsqueeze(1) >= 0)
        
        # 👑 架构师诊断探测针：向外输送 Stage 1 的特征网络张量，用于外围可视化纯净 DoH
        output_plane['ref_feature_s1'] = ref_feature['stage_1'].detach()
        
        del depth, ref_feature, src_features

        if self.training:
            return {"refined_depth": refined_depth, 
                        "depth_patchmatch": depth_patchmatch,
                        "tri_infos": tri_infos,
                        "edge_alphas": edge_alpha,
                        "output_plane":output_plane,
                        "continuity_loss":continuity_loss, # 连续性损失
                        "smoothness_loss":smoothness_loss # 光滑性损失
                    }
        else:
            photometric_confidence = None
            return {"refined_depth": refined_depth, 
                        "depth_patchmatch": depth_patchmatch, 
                        "photometric_confidence": photometric_confidence,
                        "tri_infos": tri_infos,
                        "edge_alphas": edge_alpha,
                        "output_plane": output_plane,
                        "continuity_loss": continuity_loss,  # 连续性损失
                        "smoothness_loss": smoothness_loss
                    }

    def compute_source_raw_depths(self, features, depth_min, depth_max, src_indices=[1, 2]):
        """
        轻量化提取主源视角的纯像素级 Stage 1 原生深度 (用于 Cross-Check 几何照妖镜)
        零三角剖分、零SVD、零GNN，仅复用已有特征跑纯像素级 PatchMatch
        """
        src_raw_depths = {}
        with torch.no_grad():
            for s_idx in src_indices:
                if s_idx >= len(features):
                    continue
                ref_feat_s = features[s_idx]
                other_feats = [features[i] for i in range(len(features)) if i != s_idx]
                
                depth_s = None
                vw_s = None
                for l in reversed(range(1, self.stages)):  # Stage 3 -> 2 -> 1
                    src_feats_l = [f[f'stage_{l}'] for f in other_feats]
                    projs_l = getattr(self, f'proj_matrices_{l}')
                    ref_proj_s = projs_l[s_idx]
                    other_projs_s = [projs_l[i] for i in range(len(projs_l)) if i != s_idx]
                    
                    if l > 1:
                        depth_s_list, _, vw_s = getattr(self, f'patchmatch_{l}')(
                            ref_feat_s[f'stage_{l}'], src_feats_l,
                            ref_proj_s, other_projs_s,
                            depth_min, depth_max, depth=depth_s, img=None, view_weights=vw_s
                        )
                        # 取出当前 stage 的最终输出深度 Tensor，并上采样 2 倍供下一 stage 使用
                        depth_s = depth_s_list[-1].detach()
                        depth_s = F.interpolate(depth_s, scale_factor=2, mode='nearest')
                        vw_s = F.interpolate(vw_s, scale_factor=2, mode='nearest')
                    else:
                        depth_s_pm, _, _ = getattr(self, f'patchmatch_{l}')(
                            ref_feat_s[f'stage_{l}'], src_feats_l,
                            ref_proj_s, other_projs_s,
                            depth_min, depth_max, depth=depth_s, img=None, view_weights=vw_s
                        )
                        src_raw_depths[s_idx] = depth_s_pm[-1].detach()  # [B, 1, H1, W1]
        return src_raw_depths
        
        
def patchmatchnet_loss(depth_patchmatch, refined_depth, depth_gt, mask):
    """
    损失函数有所改变，损失函数只计算mask标记有深度值的
    """
    stage = 4

    loss = 0
    # 进行patchmatch环境 stage1,2,3
    for l in range(1, stage):
        depth_gt_l = depth_gt[f'stage_{l}']
        # 代表这个地方深度是有效的
        mask_l = mask[f'stage_{l}'] > 0.5

        depth2 = depth_gt_l[mask_l]

        depth_patchmatch_l = depth_patchmatch[f'stage_{l}']
        for i in range(len(depth_patchmatch_l)):
            depth1 = depth_patchmatch_l[i][mask_l]
            loss = loss + F.smooth_l1_loss(depth1, depth2, reduction='mean')
    # stage 0 损失
    l = 0
    depth_refined_l = refined_depth[f'stage_{l}']
    depth_gt_l = depth_gt[f'stage_{l}']
    mask_l = mask[f'stage_{l}'] > 0.5

    depth1 = depth_refined_l[mask_l]
    depth2 = depth_gt_l[mask_l]
    # 相较之前加入了一个refine图片的损失函数
    loss = loss + F.smooth_l1_loss(depth1, depth2, reduction='mean')
    # 将损失去求一个平均

    return loss




def adjust_image_dims(
        images: List[torch.Tensor], intrinsics: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor, int, int]:
    """
    :param images: 数据集
    :param intrinsics: 相机参数集
    :return:
    """
    # stretch or compress image slightly to ensure width and height are multiples of 8
    # B 3 H W 图片的格式
    _, _, ref_height, ref_width = images[0].size()
    for i in range(len(images)):
        _, _, height, width = images[i].size()
        new_height = int(round(height / 8)) * 8
        new_width = int(round(width / 8)) * 8
        # 如果不是，则进行一个相机参数，和图片的下采样 让其符合标准
        if new_width != width or new_height != height:
            intrinsics[:, i, 0] *= new_width / width
            intrinsics[:, i, 1] *= new_height / height
            images[i] = nn.functional.interpolate(
                images[i], size=[new_height, new_width], mode='bilinear', align_corners=False)

    return images, intrinsics, ref_height, ref_width

def features_to_tensorboard(features:List[Dict[int, torch.Tensor]],input_image_tensor):
    """
    ym_add 将特征打印到tensorboard
    :param features: 特征图
    :return:
    """
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir='/home/ym/Experiment/PatchmatchNet-new/checkpoints/featuremap')

    step = 0
    for i, feat_dict in enumerate(features):
        for layer_id, feat in feat_dict.items():
            f = feat.detach().cpu()
            # 把每个channel当作一张图片 -> 将 C 作为 batch 维
            if f.dim() == 3:  # C,H,W
                imgs = f.unsqueeze(1)  # C,1,H,W
            elif f.dim() == 4:
                imgs = f.squeeze(0)  # C,H,W -> C,1,H,W
                imgs = imgs.unsqueeze(1)
            else:
                continue
            # normalize=True 会把每张小图独立归一化到0-1
            grid = vutils.make_grid(imgs, nrow=8, normalize=True, scale_each=True)
            writer.add_image(f"features/sample{i}/layer{layer_id}", grid, global_step=step)
        step += 1

    # 如果想把原始输入图也写进去（CHW, float 0-1）
    writer.add_image('input/sample0', input_image_tensor, 0, dataformats='CHW')
    writer.close()
