import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import _sample_map, check_tensor


class EdgeHead(nn.Module):
    """
    EdgeHead —— 升级版边级断裂预测器
    升级特性：
      - 基于几何边中点 (Edge Midpoint) 采样
      - 局部 3x3 窗口深度差分 (Patch-based Depth Diff)
      - 引入法向量特征 (Normal Features)
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

        # 定义 MLP 输入维度
        # 输入组成:
        #   - Depth Diff (1 dim)
        #   - Normals (3+3=6 dims) -> T1 normal, T2 normal
        #   - Edge Feature (tri_feat_dim) -> 中点处的图像特征
        self.edge_stat_dim = 1 + 6
        mlp_in = tri_feat_dim + self.edge_stat_dim

        self.edge_mlp = nn.Sequential(
            nn.Linear(mlp_in, edge_mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(edge_mlp_hidden, edge_mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(edge_mlp_hidden, 1)
        )

        # 可学习缩放，用来调节 sigmoid 输入尺度（训练稳定）
        self.register_parameter("edge_scale", nn.Parameter(torch.tensor(10.0)))

    def _compute_edge_midpoints(self, v1, v2):
        """
        计算两个三角形公共边的中点 (向量化实现)
        Args:
            v1: [E, 3, 2] 三角形1的三个顶点坐标 (归一化 [-1, 1])
            v2: [E, 3, 2] 三角形2的三个顶点坐标
        Returns:
            midpoints: [E, 1, 2] 边的中点坐标
        """
        # 寻找公共点：计算 v1 和 v2 顶点之间的两两距离
        # v1: [E, 3, 1, 2], v2: [E, 1, 3, 2]
        dist = torch.norm(v1.unsqueeze(2) - v2.unsqueeze(1), p=2, dim=-1)  # [E, 3, 3]

        # 判定重合点：距离小于极小值 (考虑浮点误差)
        mask = dist < 1e-4  # [E, 3, 3] bool

        # 对于标准的三角网格，两个邻接三角形应该恰好有2个公共顶点
        # 我们需要找到这2个顶点并求平均

        # 方法：利用 mask 提取 v1 中属于公共边的顶点
        # mask.any(dim=2) -> [E, 3]，表示 v1 的第 i 个点是否在 v2 中出现过
        is_shared_v1 = mask.any(dim=2).float().unsqueeze(-1)  # [E, 3, 1]

        # 计算中点：Sum(v1 * is_shared) / Sum(is_shared)
        # 正常情况下 sum(is_shared) 应该是 2
        denom = is_shared_v1.sum(dim=1).clamp(min=1.0)  # [E, 1]
        midpoints = (v1 * is_shared_v1).sum(dim=1) / denom  # [E, 2]

        return midpoints.unsqueeze(1)  # [E, 1, 2] 适配 grid_sample

    def _sample_local_patch(self, midpoints, depth_map, tri_id_map, t1_ids, t2_ids):
        """
        在边中点周围采样 3x3 Patch，并计算 T1 和 T2 区域的平均深度
        Args:
            midpoints: [E, 1, 2] 归一化坐标
            depth_map: [1, 1, H, W] 当前图的深度
            tri_id_map: [1, 1, H, W] 当前图的三角形索引 (float)
            t1_ids: [E] 三角形1的ID
            t2_ids: [E] 三角形2的ID
        Returns:
            depth_diff: [E, 1]
        """
        E = midpoints.shape[0]
        H, W = depth_map.shape[-2:]
        device = midpoints.device

        # 1. 生成 3x3 偏移网格 (Kernel)
        # 偏移量需要转为归一化坐标尺度: 2/W, 2/H
        dx = torch.linspace(-1, 1, 3, device=device) * (2.0 / W)
        dy = torch.linspace(-1, 1, 3, device=device) * (2.0 / H)
        grid_y, grid_x = torch.meshgrid(dy, dx, indexing='ij')
        offsets = torch.stack((grid_x, grid_y), dim=-1).view(1, 9, 2)  # [1, 9, 2]

        # 2. 生成采样坐标: Midpoint + Offsets
        # midpoints: [E, 1, 2] -> [E, 9, 2]
        sample_grid = midpoints + offsets
        sample_grid = sample_grid.unsqueeze(1)  # [E, 1, 9, 2] for grid_sample

        # 3. 采样深度和 ID
        # 深度用 bilinear 插值
        # tri_id_map 必须用 nearest 插值 !!
        # 注意：grid_sample 输入需要 batch 维度，这里我们将 E 视为 batch 处理
        # depth_map expand: [1, 1, H, W] -> [E, 1, H, W] (显存消耗大，不可取)
        # 优化：不expand map，而是 reshape sample_grid
        # 但 pytorch grid_sample 要求 grid 和 input batch 维度一致
        # 技巧：我们将 sample_grid 压扁为 [1, E*9, 1, 2]，只对单张图采样，然后再 reshape 回来

        flat_grid = sample_grid.view(1, E * 9, 1, 2)

        sampled_depth = F.grid_sample(depth_map, flat_grid, mode='bilinear', align_corners=True, padding_mode='border')
        sampled_id_map = F.grid_sample(tri_id_map, flat_grid, mode='nearest', align_corners=True, padding_mode='border')

        # Reshape 回 [E, 9]
        patches_depth = sampled_depth.view(E, 9)
        patches_id = sampled_id_map.view(E, 9)

        # 4. 区分 T1 和 T2 区域
        # t1_ids: [E] -> [E, 1]
        mask_t1 = (torch.abs(patches_id - t1_ids.unsqueeze(1)) < 0.1).float()
        mask_t2 = (torch.abs(patches_id - t2_ids.unsqueeze(1)) < 0.1).float()

        # 5. 计算区域平均深度
        # 防止除零 (有些patch可能只采到了T1或者只采到了T2)
        sum_t1 = mask_t1.sum(dim=1, keepdim=True).clamp(min=1e-6)
        sum_t2 = mask_t2.sum(dim=1, keepdim=True).clamp(min=1e-6)

        mean_d1 = (patches_depth * mask_t1).sum(dim=1, keepdim=True) / sum_t1
        mean_d2 = (patches_depth * mask_t2).sum(dim=1, keepdim=True) / sum_t2

        # 如果某个三角形在Patch里完全没出现(mask sum approx 0)，则深度差可能无效
        # 这种情况下用中心点深度代替或者深度差置0
        valid_edge = (mask_t1.sum(dim=1, keepdim=True) > 0.5) & (mask_t2.sum(dim=1, keepdim=True) > 0.5)

        depth_diff = torch.abs(mean_d1 - mean_d2)
        depth_diff = depth_diff * valid_edge.float()  # 无效边深度差设为0

        return depth_diff


    def forward(self, feat, pred_depth_map, tri_infos, tri_id_map, tri_normals):
        """
        EdgeHead.forward（兼容原始 tri_infos 格式）

        Args:
            - feat: [B, C, H, W]   (torch.Tensor)
            - pred_depth_map: [B, 1, H, W] (torch.Tensor)
            - 推荐传入 depth_map.detach() 如果不想让 edgehead 影响 depth predictor
            - tri_infos: list length B, 每项为 dict:
                {
                    'batch_num_tri': int,三角形的数量
                    'centers_list': [B,n_tri,2] 每个三角形的质点，已经归一化处理
                    'vertices_list': [B,n_tri,3,2] 每个三角形的顶点，已进行归一化处理
                    'edges_list' :每个边的邻接面，
                    'edges_pixels': 每个边的像素（归一化）集合，后续需要引入作为一个特征传入mlp中
                    'boundary_local_idxs_per_batch': 存储着断裂边，也就是只有一个面的边
                }
            - tri_id_map: [B, H, W] (dense index map)
            - tri_normals: [B, N_max, 3] (每个三角形的法向量)
        Returns:
            output_alphas_list: list length B, 每项是 torch.Tensor shape [E_b]，类型 float，位于 feat.device 上
            tri_infos：已经处理完毕的数据
        """
        device = feat.device
        B, C, H, W = feat.shape

        # 1. 特征投影
        feat_proj = self.proj(feat)  # [B, D, H, W]

        # 2. 准备输出容器
        output_alphas_list = []

        # 3. 逐个 Batch 处理 (因为 edges 数量不一致)
        for b in range(B):
            # 获取当前样本数据
            # tri_infos 里的 tensor 默认可能在 CPU，必须转到 GPU 才能和 feat/depth 计算
            current_edges = tri_infos[0]['edges_list'][b].to(device)  # [E, 2]
            current_vertices = tri_infos[0]['vertices_list'][b].to(device)  # [N_tri, 3, 2]
            current_normals = tri_normals[b].to(device)  # [N_max, 3]

            # tri_id_map 需要转为 [1, 1, H, W] float 格式供 grid_sample 使用
            curr_id_map = tri_id_map[b].unsqueeze(0).unsqueeze(0).float().to(device)
            curr_depth_map = pred_depth_map[b].unsqueeze(0).to(device)  # [1, 1, H, W]
            curr_feat_map = feat_proj[b].unsqueeze(0)  # [1, D, H, W]

            num_edges = current_edges.shape[0]
            if num_edges == 0:
                output_alphas_list.append(torch.zeros(0, device=device))
                continue

            # --- Step A: 定位边中点 ---
            # 获取边两侧三角形的索引
            idx1 = current_edges[:, 0].long()
            idx2 = current_edges[:, 1].long()

            # 获取对应的顶点坐标
            v1 = current_vertices[idx1]  # [E, 3, 2]
            v2 = current_vertices[idx2]  # [E, 3, 2]

            # 计算中点
            midpoints = self._compute_edge_midpoints(v1, v2)  # [E, 1, 2]

            # --- Step B: 局部深度采样 (Patch Sampling) ---
            depth_diff = self._sample_local_patch(
                midpoints, curr_depth_map, curr_id_map, idx1, idx2
            )  # [E, 1]

            # 深度差截断，防止异常值干扰
            depth_diff = torch.clamp(depth_diff, max=500.0)

            # --- Step C: 整合法向量 ---
            n1 = current_normals[idx1]  # [E, 3]
            n2 = current_normals[idx2]  # [E, 3]
            normal_feat = torch.cat([n1, n2], dim=1)  # [E, 6]

            # --- Step D: 采样边特征 ---
            # 直接在中点处采样 Backbone 特征
            # midpoints: [E, 1, 2] -> grid_sample -> [1, D, E, 1]
            # 这里为了效率，同样利用 reshape 技巧
            flat_midpoints = midpoints.unsqueeze(0)  # [1, E, 1, 2]
            edge_feats = F.grid_sample(
                curr_feat_map, flat_midpoints, align_corners=True
            ).view(feat_proj.shape[1], num_edges).permute(1, 0)  # [E, D]

            # --- Step E: MLP 预测 ---
            mlp_input = torch.cat([edge_feats, normal_feat, depth_diff], dim=1)  # [E, D+6+1]

            logits = self.edge_mlp(mlp_input).squeeze(1) * self.edge_scale
            alphas = torch.sigmoid(logits)  # [E]

            # 处理边界边 (self-loop edges, idx1 == idx2)
            # 这种情况下通常认为是边界，设为 1
            is_boundary = (idx1 == idx2)
            if is_boundary.any():
                # === 🛠️ 修复：先 clone 再修改，或者是创建一个新变量 ===
                # alphas.clone() 会创建一个新的 Tensor，但共享梯度历史。
                # 修改 clone 后的对象不会影响 SigmoidBackward 需要的原始数据。

                alphas = alphas.clone()
                alphas[is_boundary] = 1.0

            output_alphas_list.append(alphas)

        return output_alphas_list


