import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import _sample_map, check_tensor


class EdgeHead(nn.Module):
    """
    EdgeHead —— 终极版边级断裂预测器 (Analytical Plane-based)
    """
    def __init__(self, feat_channels, tri_feat_dim=128, edge_mlp_hidden=128):
        """
        Args:
          feat_channels: backbone 特征通道数 C
          tri_feat_dim: 三角特征维度（masked pooling 后投影输出）
          edge_mlp_hidden: edge MLP 隐藏层大小
        """
        super().__init__()
        # 减少后续 pooling 与 MLP 的输入维度，降低参数与计算，同时让后续 tri_feats 维度固定，便于 MLP 设计
        # 用 1x1 conv 把 feat 投影到 tri_feat_dim，方便后续 pooling 与 MLP 输入维度一致
        self.proj = nn.Conv2d(feat_channels, tri_feat_dim, kernel_size=1)
        # edge_stat_dim = 1（只有 depth_diff）

        # ==========================================================
        # 🔥 优化后的输入维度 (精简版混合表示)
        # ==========================================================
        # 1. feat_mid          (中点环境特征): tri_feat_dim 维
        # 2. dense_depth_diff  (CNN 深度落差): 1 维
        # 3. plane_depth_diff  (解析面深度落差): 1 维
        # ----------------------------------------------------------
        # 总维度 = tri_feat_dim + 1 + 1 = tri_feat_dim + 2
        # ==========================================================
        mlp_in = tri_feat_dim + 2

        self.edge_mlp = nn.Sequential(
            nn.Linear(mlp_in, edge_mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(edge_mlp_hidden, edge_mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(edge_mlp_hidden, 1)
        )

        # 可学习缩放，用来调节 sigmoid 输入尺度（训练稳定）
        self.register_parameter("edge_scale", nn.Parameter(torch.tensor(10.0)))


    def _compute_analytical_depth_diff(self, midpoints_norm, planes_t1, planes_t2, intrinsics, H, W):
        """
        使用平面方程精确计算边中点两侧的深度差 (无限分辨率，无视三角形大小)
        Args:
            midpoints_norm: [E, 2] 归一化坐标 [-1, 1]
            planes_t1: [E, 4] T1 的平面参数 (n, d)
            planes_t2: [E, 4] T2 的平面参数 (n, d)
            intrinsics: [3, 3] 当前视角的相机内参
            H, W: 图像高宽
        Returns:
            depth_diff: [E, 1]
        """
        E = midpoints_norm.shape[0]
        device = midpoints_norm.device

        # 1. 解除归一化: [-1, 1] -> 像素坐标 [0, W-1], [0, H-1]
        midpoints_uv = torch.zeros_like(midpoints_norm)
        midpoints_uv[:, 0] = (midpoints_norm[:, 0] + 1.0) / 2.0 * (W - 1)
        midpoints_uv[:, 1] = (midpoints_norm[:, 1] + 1.0) / 2.0 * (H - 1)

        # 2. 构造齐次坐标并反投影为相机系射线方向 (Ray)
        ones = torch.ones(E, 1, device=device)
        uv_homo = torch.cat([midpoints_uv, ones], dim=-1)  # [E, 3]

        K_inv = torch.inverse(intrinsics.float())  # [3, 3]
        rays = torch.matmul(K_inv, uv_homo.unsqueeze(-1)).squeeze(-1)  # [E, 3]

        # 3. 提取两端三角形解析平面参数 n 和 d
        n1, d1 = planes_t1[:, :3], planes_t1[:, 3:]
        n2, d2 = planes_t2[:, :3], planes_t2[:, 3:]

        # 4. 分别计算射线与法向的点积（余弦分量）
        denom1 = torch.sum(n1 * rays, dim=-1, keepdim=True)
        denom2 = torch.sum(n2 * rays, dim=-1, keepdim=True)

        # 保底自卫，消灭绝对零点
        denom1_safe = torch.where(torch.abs(denom1) < 1e-6, torch.sign(denom1 + 1e-10) * 1e-6, denom1)
        denom2_safe = torch.where(torch.abs(denom2) < 1e-6, torch.sign(denom2 + 1e-10) * 1e-6, denom2)

        # 5. 解析求解当前视线上的绝对 Z 深度
        depth1 = -d1 / denom1_safe  # [E, 1]
        depth2 = -d2 / denom2_safe  # [E, 1]

        # 🎯 【第一性原理修正点】：计算 3D 空间射线上两点之间的相对位移向量
        # 射线绝对视差 |Z1 - Z2| 点乘余弦张角，瞬间原位转化为不随面倾斜而产生特征爆炸的正交点到面投影距离！
        cos_projection = torch.max(torch.abs(denom1), torch.abs(denom2))
        
        # 物理量纲完美收拢，最大值被刚性卡死在真实场景几何落差内部
        orthogonal_metric_diff = torch.abs(depth1 - depth2) * cos_projection # [E, 1]

        return orthogonal_metric_diff

    def extract_static_features(self, feat, tri_infos, dense_depth=None, intrinsics=None):
        """
        【EdgeHead·常数流形寄存器】只在循环外执行 O(1) 一次。
        物理机制：利用固定深度图反解出 3D 空间中死死焊接的探测点云位移向量 vec_L2R，作为循环内的常数铁锚。
        """
        device = feat.device
        B, C, H, W = feat.shape
        feat_proj = self.proj(feat)

        static_feats_list = []
        pixel_step = 1.25
        step_tensor = torch.tensor([pixel_step * (2.0 / W), pixel_step * (2.0 / H)], device=device).view(1, 2)

        for b in range(B):
            current_edges = tri_infos[0]['edges_list'][b].to(device)
            midpoints_norm = tri_infos[0]['edges_midpoints'][b].to(device)
            E = current_edges.shape[0]

            if E == 0:
                static_feats_list.append((None, None, None))
                continue

            # 1. 提取中点高维图像环境特征
            curr_feat_map = feat_proj[b].unsqueeze(0)
            mid_grid = midpoints_norm.unsqueeze(0).unsqueeze(2)
            feat_mid = F.grid_sample(curr_feat_map, mid_grid, align_corners=True).view(feat_proj.shape[1], E).permute(1, 0)

            # 2. 🪐【核心重构】：计算 3D 视锥绝对坐标位移差常数张量
            if dense_depth is not None and intrinsics is not None:
                curr_dense_depth = dense_depth[b].unsqueeze(0).detach()
                endpoints = tri_infos[0]['edges_endpoints'][b].to(device)

                # 探测点正交方向偏转
                edge_vec = endpoints[:, 1, :] - endpoints[:, 0, :]
                ortho_vec = torch.stack([-edge_vec[:, 1], edge_vec[:, 0]], dim=-1)
                ortho_vec = F.normalize(ortho_vec, p=2, dim=-1)
                scaled_ortho = ortho_vec * step_tensor

                probe_left_grid = (midpoints_norm + scaled_ortho).unsqueeze(0).unsqueeze(2).clamp(-1.0, 1.0)
                probe_right_grid = (midpoints_norm - scaled_ortho).unsqueeze(0).unsqueeze(2).clamp(-1.0, 1.0)

                # 重采样固定深度场
                d_left = F.grid_sample(curr_dense_depth, probe_left_grid, align_corners=True).view(E)
                d_right = F.grid_sample(curr_dense_depth, probe_right_grid, align_corners=True).view(E)

                # 像素坐标反解反投影
                u_left = (probe_left_grid[0, :, 0, 0] + 1.0) / 2.0 * (W - 1)
                v_left = (probe_left_grid[0, :, 0, 1] + 1.0) / 2.0 * (H - 1)
                u_right = (probe_right_grid[0, :, 0, 0] + 1.0) / 2.0 * (W - 1)
                v_right = (probe_right_grid[0, :, 0, 1] + 1.0) / 2.0 * (H - 1)

                curr_K = intrinsics[b] if intrinsics.dim() == 3 else intrinsics
                fx, fy = curr_K[0, 0], curr_K[1, 1]
                cx, cy = curr_K[0, 2], curr_K[1, 2]

                # 熔炼出 3D 绝对空间点云常数
                X_L = (u_left - cx) * d_left / fx
                Y_L = (v_left - cy) * d_left / fy
                P_left = torch.stack([X_L, Y_L, d_left], dim=1)  # [E, 3]

                X_R = (u_right - cx) * d_right / fx
                Y_R = (v_right - cy) * d_right / fy
                P_right = torch.stack([X_R, Y_R, d_right], dim=1)  # [E, 3]

                # 刚性备份常数位移向量，死死寄存
                vec_L2R = P_right - P_left  # [E, 3]
            else:
                vec_L2R = torch.zeros((E, 3), device=device)

            static_feats_list.append((feat_mid, vec_L2R))

        return static_feats_list

    def dynamic_forward(self, static_feats_list, tri_infos, tri_planes, intrinsics, H, W, W_plane_tri=None):
        """
        【EdgeHead·流形双动态进化版】循环内调用。
        
        机制精谱：
        - 让 `dense_depth_diff`（固定深度图）与 `plane_depth_diff`（解析面方程）
          共同通过每一轮最新优化的 tri_planes 法向实施动态规约点投影！
        - 零渲染开销，在稀疏拓扑空间内部全速合流更新。
        """
        device = tri_planes.device
        B = len(static_feats_list)
        output_alphas_list = []

        for b in range(B):
            feat_mid, vec_L2R = static_feats_list[b]
            if feat_mid is None:
                output_alphas_list.append(torch.zeros(0, device=device))
                continue

            current_edges = tri_infos[0]['edges_list'][b].to(device)
            current_planes = tri_planes[b]  # 最新的、不断被 GNN 熨平进化的平面方程 [N, 4]
            curr_intrinsics = intrinsics[b] if intrinsics.dim() == 3 else intrinsics
            midpoints_norm = tri_infos[0]['edges_midpoints'][b].to(device)
            E = current_edges.shape[0]

            idx1, idx2 = current_edges[:, 0].long(), current_edges[:, 1].long()
            planes_t1 = current_planes[idx1]
            planes_t2 = current_planes[idx2]

            # 🎯 【动态特征一：随法向自愈而协同沉降的稠密点到面垂直距离】
            # 查表提取两端更新后的高纯度法向量
            n_left = planes_t1[:, :3]
            n_right = planes_t2[:, :3]
            
            # 无任何采样开销，原位稀疏点乘！假断层信号会随朝向变准而瞬间塌陷归零！
            dist_to_plane_L = torch.abs(torch.sum(vec_L2R * n_left, dim=1, keepdim=True))
            dist_to_plane_R = torch.abs(torch.sum((-vec_L2R) * n_right, dim=1, keepdim=True))
            dense_depth_diff = torch.max(dist_to_plane_L, dist_to_plane_R).clamp(max=500.0) # [E, 1]

            # 🎯 【动态特征二：全新解析面方程几何视差距离（保持动态）】
            plane_depth_diff = self._compute_analytical_depth_diff(
                midpoints_norm, planes_t1, planes_t2, curr_intrinsics, H, W
            ).clamp(max=500.0)

            # # --- 3. 👑【终局范式：流形共识偏差自适应阻尼锁】---
            if W_plane_tri is not None:
                active_conf_map = W_plane_tri.squeeze(-1) if W_plane_tri.dim() == 3 else W_plane_tri
                W_left = active_conf_map[b, idx1].unsqueeze(-1)  # [E, 1]
                W_right = active_conf_map[b, idx2].unsqueeze(-1) # [E, 1]
                
                # 🪐【数理因果】：仅在【存在显著置信度极化差异】时才激活几何惩罚
                # 计算两端平面与曲面的置信度差值绝对值（代表流形主权冲突）
                conf_diff = torch.abs(W_left - W_right)
                # 计算两端的置信度均值（代表是否处于高置信度流形内）
                conf_mean = (W_left + W_right) / 2.0
                
                # 👑 核心逻辑：
                # 只有当：置信度差异大（存在平面侵略曲面的可能） AND 均值较高（至少有一端是平面）
                # 才会引发深度的“伪放大”，诱导边缘检测器落锁断开。
                # 如果两端都是低分曲面（W_mean < 0.3），该乘子趋于 1.0，完全解开平滑限制，任由网络拟合！
                
                # 使用非线性函数：在置信度差异超过 0.3 且均值较高时，才激进放大视差
                # 这样既保护了曲面的自由度，又铁腕阻断了平面对曲面的蚕食
                trigger = (conf_diff > 0.3) & (conf_mean > 0.4)
                
                # 放大系数设为动态，避免固定倍数带来的数值波动
                base_multiplier = 1.0 + (conf_diff * 4.0) 
                planar_consensus_multiplier = torch.where(trigger, base_multiplier, torch.ones_like(base_multiplier))
                
                # 只有当 planar_consensus_multiplier > 1.0 时，才会放大视差诱导断裂
                plane_depth_diff = plane_depth_diff * planar_consensus_multiplier

            # --- 4. 两条动态几何总线无伤并网，递交 MLP 推理 ---
            mlp_input = torch.cat([
                feat_mid,          # 图像高频语义环境 [E, D]
                dense_depth_diff,  # 🌟 升级：随法向收敛而自愈的密集几何距离 [E, 1]
                plane_depth_diff   # 随平面演进而收敛的解析几何视差 [E, 1]
            ], dim=1)

            logits = self.edge_mlp(mlp_input).squeeze(1) * self.edge_scale
            alphas = torch.sigmoid(logits)

            # 边界刚性锁
            is_boundary = (idx1 == idx2)
            if is_boundary.any():
                alphas = alphas.clone()
                alphas[is_boundary] = 1.0

            output_alphas_list.append(alphas)

        return output_alphas_list

    def forward(self, feat, tri_infos, tri_planes, intrinsics, dense_depth=None):
        """
        兼容性接口：保留原始 forward 调用习惯，但底层由动静分离逻辑支撑。
        """
        static_feats = self.extract_static_features(feat, tri_infos, dense_depth)
        H, W = feat.shape[2], feat.shape[3]
        return self.dynamic_forward(static_feats, tri_infos, tri_planes, intrinsics, H, W)


class EdgeLabelGenerator:
    '''
    终极版：专门针对 C0 连续性（深度遮挡断裂）的伪标签生成器
    采用 Point-to-Plane 物理垂直距离判定，完美免疫倾斜墙面，
    并将 C1 连续性（法向折痕）的阻断任务完全交由传播模块的 gate_net 自适应学习。
    '''

    def __init__(self, pt2plane_threshold=0.4, pixel_step=1.25):
        # 绝对物理阈值：点到平面垂直距离（单位通常为米）
        # 0.4 米足以抓出遮挡边缘（如车顶到地面），同时放过平滑的斜面和连通的折痕
        self.pt2plane_threshold = pt2plane_threshold
        self.pixel_step = pixel_step

    def __call__(self, pred_alphas_list, gt_depth_map, tri_infos, intrinsics, gt_normal_map):
        B, _, H0, W0 = gt_depth_map.shape
        device = gt_depth_map.device

        gt_labels_list = []
        valid_mask_list = []

        step_u = self.pixel_step * (2.0 / W0)
        step_v = self.pixel_step * (2.0 / H0)
        step_tensor = torch.tensor([step_u, step_v], device=device).view(1, 2)

        for b in range(B):
            pred_alphas = pred_alphas_list[b]
            E = pred_alphas.shape[0]

            # 防御性编程：如果没有边，直接返回空 Tensor
            if E == 0:
                gt_labels_list.append(torch.empty(0, device=device))
                valid_mask_list.append(torch.empty(0, dtype=torch.bool, device=device))
                continue

            # 1. 获取端点与中点
            endpoints = tri_infos[0]['edges_endpoints'][b].to(device)
            v1 = endpoints[:, 0, :]  # [E, 2]
            v2 = endpoints[:, 1, :]  # [E, 2]
            midpoints = (v1 + v2) / 2.0  # [E, 2]

            # 2. 计算正交法向量 (Orthogonal Vector)
            edge_vec = v2 - v1  # [E, 2]
            # 顺时针/逆时针旋转90度: (x, y) -> (-y, x)
            ortho_vec = torch.stack([-edge_vec[:, 1], edge_vec[:, 0]], dim=-1)
            ortho_vec = F.normalize(ortho_vec, p=2, dim=-1)  # [E, 2]

            # 3. 生成探测点 (Probe Points) 并防越界
            scaled_ortho = ortho_vec * step_tensor  # [E, 2]
            probe_left = (midpoints + scaled_ortho).clamp(-1.0, 1.0)
            probe_right = (midpoints - scaled_ortho).clamp(-1.0, 1.0)

            # --- 提取当前 Batch 的内参 ---
            curr_K = intrinsics[b] if intrinsics.dim() == 3 else intrinsics
            fx, fy = curr_K[0, 0], curr_K[1, 1]
            cx, cy = curr_K[0, 2], curr_K[1, 2]

            # --- 采样深度和法向 ---
            depth_b = gt_depth_map[b:b + 1]  # [1, 1, H0, W0]
            normal_b = gt_normal_map[b:b + 1]  # [1, 3, H0, W0]

            # 🔥 安全维度处理：使用 view() 和 permute() 保证 E=1 时不会崩溃
            depth_left = F.grid_sample(depth_b, probe_left.view(1, E, 1, 2), align_corners=True).view(E)
            depth_right = F.grid_sample(depth_b, probe_right.view(1, E, 1, 2), align_corners=True).view(E)

            normal_left = F.grid_sample(normal_b, probe_left.view(1, E, 1, 2), align_corners=True).view(3, E).permute(1,
                                                                                                                      0)
            normal_right = F.grid_sample(normal_b, probe_right.view(1, E, 1, 2), align_corners=True).view(3, E).permute(
                1, 0)

            # =======================================================
            # 核心降维打击：计算 3D 点坐标 (X, Y, Z)
            # =======================================================
            # 把 [-1, 1] 的坐标转回像素坐标 [0, W0-1]
            u_left = (probe_left[:, 0] + 1.0) / 2.0 * (W0 - 1)
            v_left = (probe_left[:, 1] + 1.0) / 2.0 * (H0 - 1)
            u_right = (probe_right[:, 0] + 1.0) / 2.0 * (W0 - 1)
            v_right = (probe_right[:, 1] + 1.0) / 2.0 * (H0 - 1)

            # 射线反投影
            X_L = (u_left - cx) * depth_left / fx
            Y_L = (v_left - cy) * depth_left / fy
            P_left = torch.stack([X_L, Y_L, depth_left], dim=1)  # [E, 3]

            X_R = (u_right - cx) * depth_right / fx
            Y_R = (v_right - cy) * depth_right / fy
            P_right = torch.stack([X_R, Y_R, depth_right], dim=1)  # [E, 3]

            # =======================================================
            # 唯一判定: 真正的遮挡断层 (Point-to-Plane Distance)
            # =======================================================
            vec_L2R = P_right - P_left  # [E, 3]

            # 计算 P_right 到 "P_left所在平面" 的垂直距离: D = |(P_R - P_L) · N_L|
            dist_to_plane_L = torch.abs(torch.sum(vec_L2R * normal_left, dim=1))  # [E]

            # 对称地，计算 P_left 到 "P_right所在平面" 的垂直距离
            dist_to_plane_R = torch.abs(torch.sum((-vec_L2R) * normal_right, dim=1))  # [E]

            # 取两者较大的那个作为最终距离（增加鲁棒性）
            pt2plane_dist = torch.max(dist_to_plane_L, dist_to_plane_R)

            # 断裂条件：点到平面距离大于物理阈值
            is_break = pt2plane_dist > self.pt2plane_threshold
            labels = is_break.float()

            # =======================================================
            # 掩码与边界处理
            # =======================================================
            # 有效掩码：探测的两个点都必须有 GT 深度 (排除边界黑边干扰)
            valid_mask = (depth_left > 1e-4) & (depth_right > 1e-4)

            # --- 强制边界边断裂 ---
            # 如果这条边只属于一个三角形 (边界边)，必须强制为断裂
            current_edges = tri_infos[0]['edges_list'][b].to(device)
            is_boundary = (current_edges[:, 0] == current_edges[:, 1])
            labels[is_boundary] = 1.0

            # 边界边不需要探测点必须有效，强制视为有效
            valid_mask[is_boundary] = True

            gt_labels_list.append(labels)
            valid_mask_list.append(valid_mask)

        return gt_labels_list, valid_mask_list

