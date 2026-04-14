import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import _sample_map, check_tensor


class EdgeHead(nn.Module):
    """
    EdgeHead —— 终极版边级断裂预测器 (Analytical Plane-based)
    升级特性：
      - 彻底抛弃 3x3 离散像素采样，免疫狭长三角形的特征崩溃
      - 基于平面方程 (n, d) 直接求解边中点的解析深度差，无限分辨率
      - 单点采样 Backbone 图像特征，保留 CNN 原生感受野
    输入:
      - feat: [B, C, Hf, Wf]  （backbone / FPN 特征）
      - img: [B, C_img, H, W] （原图，可为 None）
      - depth_map: [B,1,H,W] 或 [B,H,W] （当前网络的深度预测）
      - tri_infos: list length B, 每项为 dict:
                {
                    'batch_num_tri': int,三角形的数量
                    'centroids': [B,n_tri,2] 每个三角形的质点，已经归一化处理
                    'vertices': [B,n_tri,3,2] 每个三角形的顶点，已进行归一化处理
                    'edges_list' :每个边的邻接面，
                     boundary_local_idxs_per_batch: 存储着断裂边，也就是只有一个面的边
                    'tri_edge_ids_list':每个三角形的边ID列表
                    'edges_midpoints': 边对应的中点已经归一化 List[B] of [E, 2]
                }
    输出:
      - per_image_edges_alpha: list len=B，每个 tensor [E]，E = edges 数（按 edges 列表顺序）
      - per_image_edge_mat: list len=B，每个 tensor [num_tri, num_tri] 对称矩阵（不邻接处为0）
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

        # ==========================================
        # 🔥 新增：闭门初始化 (Closed-gate Initialization)
        # ==========================================
        # 因为后面有 logits = mlp_out * self.edge_scale (10.0)
        # 我们把 bias 设为 0.2，那么初始 logits 就是 0.2 * 10 = 2.0
        # sigmoid(2.0) ≈ 0.88，这意味着一开始 88% 的概率被认为是边缘（阻断传播）
        # nn.init.constant_(self.edge_mlp[4].bias, 0.2)
        # nn.init.normal_(self.edge_mlp[4].weight, mean=0.0, std=0.01)


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

        # 2. 构造齐次坐标并反投影为射线方向 (Ray)
        ones = torch.ones(E, 1, device=device)
        uv_homo = torch.cat([midpoints_uv, ones], dim=-1)  # [E, 3]

        K_inv = torch.inverse(intrinsics)  # [3, 3]
        rays = torch.matmul(K_inv, uv_homo.unsqueeze(-1)).squeeze(-1)  # [E, 3]

        # 3. 提取平面参数 n 和 d
        n1, d1 = planes_t1[:, :3], planes_t1[:, 3:]
        n2, d2 = planes_t2[:, :3], planes_t2[:, 3:]

        # 4. 分别用 T1 和 T2 的平面方程计算该射线上的深度 Z
        # Z = -d / (n * ray)
        denom1 = torch.sum(n1 * rays, dim=-1, keepdim=True)
        denom1 = torch.where(torch.abs(denom1) < 1e-6, torch.tensor(1e-6, device=device), denom1)
        depth1 = -d1 / denom1  # [E, 1]

        denom2 = torch.sum(n2 * rays, dim=-1, keepdim=True)
        denom2 = torch.where(torch.abs(denom2) < 1e-6, torch.tensor(1e-6, device=device), denom2)
        depth2 = -d2 / denom2  # [E, 1]

        # 5. 获得绝对精确的深度差
        depth_diff = torch.abs(depth1 - depth2)  # [E, 1]

        return depth_diff

    def forward(self, feat, tri_infos, tri_planes, intrinsics,dense_depth=None):
        """
        EdgeHead.forward（兼容原始 tri_infos 格式）

        Args:
            - feat: [B, C, H, W]   (torch.Tensor)
            - tri_infos: list length B, 每项为 dict:
                {
                    'batch_num_tri': int,三角形的数量
                    'centers_list': [B,n_tri,2] 每个三角形的质点，已经归一化处理
                    'vertices_list': [B,n_tri,3,2] 每个三角形的顶点，已进行归一化处理
                    'edges_list' :每个边的邻接面，
                    'edges_pixels': 每个边的像素（归一化）集合，后续需要引入作为一个特征传入mlp中
                    'boundary_local_idxs_per_batch': 存储着断裂边，也就是只有一个面的边
                }
            - tri_planes: [B, N_max, 4] (当前拟合出的平面参数, n 和 d)
            - intrinsics: [B, 3, 3] (相机内参矩阵)
        Returns:
            output_alphas_list: list length B, 每项是 torch.Tensor shape [E_b]，类型 float，位于 feat.device 上
            tri_infos：已经处理完毕的数据
        """
        device = feat.device
        B, C, H, W = feat.shape

        feat_proj = self.proj(feat)
        output_alphas_list = []

        # 探测步长仅用于采样稠密深度
        pixel_step = 1.25
        step_u = pixel_step * (2.0 / W)
        step_v = pixel_step * (2.0 / H)
        step_tensor = torch.tensor([step_u, step_v], device=device).view(1, 2)

        for b in range(B):
            current_edges = tri_infos[0]['edges_list'][b].to(device)
            current_planes = tri_planes[b].to(device)
            curr_intrinsics = intrinsics[b].to(device) if intrinsics.dim() == 3 else intrinsics.to(device)
            curr_feat_map = feat_proj[b].unsqueeze(0)

            if dense_depth is not None:
                curr_dense_depth = dense_depth[b].unsqueeze(0).detach()
            else:
                curr_dense_depth = None

            E = current_edges.shape[0]
            if E == 0:
                output_alphas_list.append(torch.zeros(0, device=device))
                continue

            idx1 = current_edges[:, 0].long()
            idx2 = current_edges[:, 1].long()
            midpoints_norm = tri_infos[0]['edges_midpoints'][b].to(device)

            # --- Step A: 解析面深度差 (证人 C) ---
            planes_t1 = current_planes[idx1]
            planes_t2 = current_planes[idx2]
            plane_depth_diff = self._compute_analytical_depth_diff(
                midpoints_norm, planes_t1, planes_t2, curr_intrinsics, H, W
            )
            plane_depth_diff = torch.clamp(plane_depth_diff, max=500.0)

            # --- Step B: 提取中点图像特征 (上下文环境) ---
            mid_grid = midpoints_norm.unsqueeze(0).unsqueeze(2)
            feat_mid = F.grid_sample(curr_feat_map, mid_grid, align_corners=True).view(feat_proj.shape[1], E).permute(1,
                                                                                                                      0)

            # --- Step C: 提取稠密深度差 (证人 B) ---
            if curr_dense_depth is not None:
                # 依然需要计算左右探测点，因为深度图是标量，必须有 diff 才有意义
                endpoints = tri_infos[0]['edges_endpoints'][b].to(device)
                edge_vec = endpoints[:, 1, :] - endpoints[:, 0, :]
                ortho_vec = torch.stack([-edge_vec[:, 1], edge_vec[:, 0]], dim=-1)
                ortho_vec = F.normalize(ortho_vec, p=2, dim=-1)
                scaled_ortho = ortho_vec * step_tensor

                probe_left_grid = (midpoints_norm + scaled_ortho).unsqueeze(0).unsqueeze(2).clamp(-1.0, 1.0)
                probe_right_grid = (midpoints_norm - scaled_ortho).unsqueeze(0).unsqueeze(2).clamp(-1.0, 1.0)

                depth_left = F.grid_sample(curr_dense_depth, probe_left_grid, align_corners=True).view(1, E).permute(1,
                                                                                                                     0)
                depth_right = F.grid_sample(curr_dense_depth, probe_right_grid, align_corners=True).view(1, E).permute(
                    1, 0)
                dense_depth_diff = torch.abs(depth_left - depth_right)
                dense_depth_diff = torch.clamp(dense_depth_diff, max=500.0)
            else:
                dense_depth_diff = torch.zeros_like(plane_depth_diff)

            # --- Step D: 拼接并预测 ---
            mlp_input = torch.cat([
                feat_mid,  # [E, D]
                dense_depth_diff,  # [E, 1]
                plane_depth_diff  # [E, 1]
            ], dim=1)

            logits = self.edge_mlp(mlp_input).squeeze(1) * self.edge_scale
            alphas = torch.sigmoid(logits)

            is_boundary = (idx1 == idx2)
            if is_boundary.any():
                alphas = alphas.clone()
                alphas[is_boundary] = 1.0

            output_alphas_list.append(alphas)

        return output_alphas_list


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

