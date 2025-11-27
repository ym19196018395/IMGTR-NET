from typing import List, Tuple

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
        

class Refinement(nn.Module):
    def __init__(self):
        
        super(Refinement, self).__init__()
        
        # img: [B,3,H,W]
        self.conv0 = ConvBnReLU(3, 8)
        # depth map:[B,1,H/2,W/2]
        self.conv1 = ConvBnReLU(1, 8)
        self.conv2 = ConvBnReLU(8, 8)
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

    def forward(self, imgs, proj_matrices, depth_min, depth_max,vertexs,lines,triangles):
        
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

        # ym-issue 这个是已经处理好的投影矩阵，有时间看一下dataloader
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
        del imgs_0
        ref_feature, src_features = features[0], features[1:]
        
        depth_min = depth_min.float()
        depth_max = depth_max.float()

        # step 2. Learning-based patchmatch
        depth = None
        view_weights = None
        depth_patchmatch = {}
        refined_depth = {}
        
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
                # ym-add 在stage1阶段进行一个边断裂预测,暂时不用
                # depth, score, _ ,edge_alphas,edge_mats= getattr(self, f'patchmatch_{l}')(ref_feature[f'stage_{l}'], src_features_l,
                #                         ref_proj, src_projs,
                #                         depth_min, depth_max, depth=depth,img=getattr(self,f'imgs_{l}_ref'), view_weights=view_weights,
                #                         vertexs=vertexs,lines=lines,triangles=triangles)

                depth, score, _ = getattr(self, f'patchmatch_{l}')(ref_feature[f'stage_{l}'], src_features_l,
                                        ref_proj, src_projs,
                                        depth_min, depth_max, depth=depth,img=getattr(self,f'imgs_{l}_ref'), view_weights=view_weights)
            
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
            
        # 因为评估数据还没有进行处理，所以评估数据进行一个跳过,ym-need-modify
        tri_infos=[]
        edge_alphas,edge_mats=[],[]
        # if self.training:
        #     _, _, height, width = depth.size()
        #     device = depth.get_device()
        #     # 对数据进行一个转化，会在里面得到边的像素集合，以及三角面的顶点和质点（归一化的）
        #     # 因为传入的是原分辨率的三角信息，在里面会进行一个1/2的缩放
        #     tri_infos = batch_convert_to_tri_infos(vertexs, lines, triangles, height*2 , width*2 , device)
        #     # 操作是在1 / 2分辨率下面进行,少了一个edge—mat
        #     ref_stage1_feature=ref_feature['stage_1'].detach()
        #     edge_alphas = self.edge_head(ref_stage1_feature, img=None, depth_map=depth, tri_infos=tri_infos)

        # step 3. Refinement  
        depth = self.upsample_net(self.imgs_0_ref, depth, depth_min, depth_max)
        refined_depth['stage_0'] = depth

        del depth, ref_feature, src_features

        if self.training:
            return {"refined_depth": refined_depth, 
                        "depth_patchmatch": depth_patchmatch,
                        "tri_infos": tri_infos,
                        "edge_alphas": edge_alphas
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