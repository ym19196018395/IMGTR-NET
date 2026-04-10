from typing import List, Tuple, Dict

from tensorboard.plugins.hparams.metadata import NULL_TENSOR
from .PlanePatchMatch import *
from utils import batch_convert_to_tri_infos_new,build_neighbor_indices
from .feature_map import *
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
        
        del intra_feat
            
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
        # 上采样后的原始深度 + res 预测残差
        depth = F.interpolate(depth, scale_factor=2, mode="nearest") + res
        # convert the normalized depth back
        depth = depth * (depth_max.view(batch_size,1,1,1)-depth_min.view(batch_size,1,1,1)) + depth_min.view(batch_size,1,1,1)

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


    def forward(self, imgs, proj_matrices,intrinsics_mats,depth_min, depth_max,vertexs,lines,triangles,depth_stage_1,lambda_c, lambda_s):
        

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
           'depth_stage1_pixels':[],# stage2放大后产生的深度图
            'normal_final':[], # 传播后stage1最终平面
            'tri_id_map':[],# 三角形stage1下的id图
            'tri_id_map_stage0': [],  # 三角形stage1下的id图
            'depth_no_pro': [],  # 刚拟合完的深度值
            'normal_no_pro': []  # 刚拟合完的法向量
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

                # 根据stage2粗深度图来拟合stage1的法向量和深度图
                tri_infos = []
                depth_stage2_raw = depth_patchmatch['stage_2'][-1]

                # 利用 Stage 1 的高频图像特征，进行保边平滑上采样
                depth_stage1_init = self.stage1_refine(
                    ref_feature=ref_feature[f'stage_{l}'],  # [B, 16, H/2, W/2]
                    depth_0=depth_stage2_raw,  # [B, 1, H/4, W/4]
                    depth_min=depth_min,
                    depth_max=depth_max
                )

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

                num_hypotheses=4
                # 平面拟合
                self.dense_plane_fitter = DensePlaneFitter(height, width, device,num_hypotheses=1, perturbation_range=0.05,
                                                           depth_max=depth_max,
                                                           depth_min=depth_min)


                (depth_samples, score, view_weights,normal_samples,output_plane['final_plane'],edge_alphas,
                 continuity_loss,smoothness_loss) = self.plane_patchmatch_agent.forward(
                                                                    self.dense_plane_fitter,
                                                                    depth_stage1_init, tri_infos, # todo：暂时不让传播阶段去影响原来pixelpatchmatch阶段
                                                                    ref_feature[f'stage_{l}'],
                                                                    src_features_l,
                ref_proj, src_projs, intrinsics_s1,depth_min, depth_max, view_weights.detach(),
                neighbor_indices_batched = neighbor_indices_batched,
                lambda_c=lambda_c,lambda_s=lambda_s
                )

                # ================================================================
                # 3. 可视化结果，返回结果
                # ================================================================

                # 转化为tensor形式(B,H,W)
                # 步骤1：去掉每个Tensor中长度为1的维度（把[1, H, W]转成[H, W]）
                processed_list = [tensor.squeeze(0) for tensor in tri_infos[0]['tri_id_map']]
                processed_list_stage0 = [tensor.squeeze(0) for tensor in tri_infos[0]['tri_id_map_stage0']]
                # 步骤2：在第0维（batch维）堆叠，得到[B, H, W]
                tri_id_map_tensor = torch.stack(processed_list, dim=0)
                tri_id_map_stage0_tensor = torch.stack(processed_list_stage0, dim=0)

                output_plane['tri_id_map'] = tri_id_map_tensor
                output_plane['tri_id_map_stage0'] = tri_id_map_stage0_tensor

                # stage2 产生的深度图放大后，patchmatch产生的深度图
                output_plane['depth_stage1_pixels']=depth_stage1_init
                # 经过传播得到的平面
                output_plane['normal_final']=normal_samples[1]

                # 没有进行传播得到的平面，刚拟合完的初始平面
                # 将B,N,4 分别转化为,B,H,W,1 和B,H,W,3 可视化用
                output_plane['depth_no_pro'] = depth_samples[0]
                output_plane['normal_no_pro'] = normal_samples[0]

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

                # planepatchmatch最终预测结果，里面也有两份一份用来上采样一份用来输出
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

        # step 3. Refinement  
        depth = self.upsample_net(self.imgs_0_ref, depth, depth_min, depth_max)
        refined_depth['stage_0'] = depth

        del depth, ref_feature, src_features

        if self.training:
            return {"refined_depth": refined_depth, 
                        "depth_patchmatch": depth_patchmatch,
                        "tri_infos": tri_infos,
                        "edge_alphas": edge_alphas,
                        "output_plane":output_plane,
                        "continuity_loss":continuity_loss, # 连续性损失
                        "smoothness_loss":smoothness_loss # 光滑性损失
                    }
        else:
            num_depth = self.patchmatch_num_sample[0]
            score_sum4 = 4 * F.avg_pool3d(F.pad(score.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0).squeeze(1)
            # [B, 1, H, W]
            depth_index = depth_regression(score, depth_values=torch.arange(num_depth, device=score.device, dtype=torch.float)).long()
            depth_index = torch.clamp(depth_index, 0, num_depth-1)
            photometric_confidence = torch.gather(score_sum4, 1, depth_index)
            photometric_confidence = F.interpolate(photometric_confidence,
                                        scale_factor=2, mode='nearest')
            photometric_confidence = photometric_confidence.squeeze(1)

            return {"refined_depth": refined_depth, 
                        "depth_patchmatch": depth_patchmatch, 
                        "photometric_confidence": photometric_confidence,
                        "tri_infos": tri_infos,
                        "edge_alphas": edge_alphas,
                        "output_plane": output_plane,
                        "continuity_loss": continuity_loss,  # 连续性损失
                        "smoothness_loss": smoothness_loss
                    }
        
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


def compute_normal_cosine_loss(final_planes, tri_id_map, gt_normals_math_s0, depth_stage_1):
    """
    计算 Stage 1 预测平面与 Stage 0 GT 法向量之间的余弦相似度损失。

    Args:
        final_planes: [B, N, 4] Stage 1 的平面参数
        tri_id_map: [B, H, W] Stage 1 的三角形 ID 图
        gt_normals_math_s0: [B, 3, H0, W0] Stage 0 的数学真值法向量 [-1, 1]
        depth_stage_1: [B, 1, H, W] Stage 1 的预测/GT 深度，用于过滤无效背景

    Returns:
        normal_loss: 标量 Loss
    """
    B, N, _ = final_planes.shape
    _, H, W = tri_id_map.shape
    device = final_planes.device

    # =======================================================
    # 步骤 A: 对 Stage 0 的 GT 法向量进行正确的下采样
    # =======================================================
    # 必须使用 bilinear 插值，防止法向量出现最近邻的“马赛克锯齿”
    gt_normals_s1 = F.interpolate(
        gt_normals_math_s0,
        size=(H, W),  # 直接对齐目标尺寸，比 scale_factor 更安全
        mode='bilinear',
        align_corners=False
    )
    # 🌟 极其关键：插值后向量长度缩水，必须重新 L2 归一化
    gt_normals_s1 = F.normalize(gt_normals_s1, p=2, dim=1)

    # =======================================================
    # 步骤 B: 高效提取 Stage 1 预测的“纯数学”法向量
    # =======================================================
    # 1. 取出预测的法向量 n [B, N, 3]，并归一化
    pred_tri_normals = final_planes[..., :3]
    pred_tri_normals = F.normalize(pred_tri_normals, p=2, dim=-1)

    # 2. 处理无效区域 ID (将 -1 暂时替换为 0 以防查表越界)
    safe_id_map = tri_id_map.clone()
    invalid_mask = (safe_id_map < 0)
    safe_id_map[invalid_mask] = 0

    # 3. 极其高效的查表渲染 (替代复杂的 for 循环)
    batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, H, W)
    pred_pixel_normals = pred_tri_normals[batch_idx, safe_id_map]  # [B, H, W, 3]
    pred_pixel_normals = pred_pixel_normals.permute(0, 3, 1, 2)  # [B, 3, H, W]

    # =======================================================
    # 步骤 C: 计算余弦相似度损失
    # =======================================================
    # 1. 计算点积 sum(N_pred * N_gt)
    cos_sim = torch.sum(pred_pixel_normals * gt_normals_s1, dim=1, keepdim=True)  # [B, 1, H, W]

    # 2. 转换为 Loss (1.0 - cos_sim)，完全同向时 Loss 为 0
    loss_map = 1.0 - cos_sim

    # 3. 构建有效 mask (去除原来 tri_id_map < 0 的区域，以及深度为 0 的背景)
    valid_mask = (~invalid_mask.unsqueeze(1)) & (depth_stage_1 > 1e-4)

    # 4. 最终法向损失
    normal_loss = loss_map[valid_mask].mean()

    # 容错：如果整张图全部无效（极少见），返回 0 梯度
    if torch.isnan(normal_loss):
        normal_loss = torch.tensor(0.0, device=device, requires_grad=True)

    return normal_loss

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