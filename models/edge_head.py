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



"""
continuity_loss_edge

作用（高层）：
基于三角边级的断裂概率 (edge_alpha_mat)，对像素级深度变化做加权连续性约束：
在非断裂（或弱断裂）边上期望深度连续（z_p ≈ z_q），在断裂边上允许跳变。
同时对 edge predictor 输出的 alpha 做稀疏正则（鼓励尽量少判为断裂）。

接口（函数说明）：
    continuity_loss_edge(depth_map, tri_id_map_list, edge_alpha_mat_list,
                         mask=None, lambda_cont=1.0, lambda_sparsity=1e-4)

核心思路（数学表达）：
L_cont = avg_{图片 b} [  0.5 * ( sum_{(p,q) in right} (1-α_{t_p,t_q}) * (z_p - z_q)^2 / #valid_right
                                  + sum_{(p,q) in down} (1-α_{t_p,t_q}) * (z_p - z_q)^2 / #valid_down )  ]
L_sparsity = lambda_sparsity * mean_over_images(mean_alpha_over_existing_edges)
total = lambda_cont * L_cont + L_sparsity

实现注意事项：
- 对相邻像素对 (p,q) 只计算右方向和下方向，避免重复（cover full 4-neighbour）。
- 若两个像素属于同一三角（t_p == t_q），认为不是跨三角边（alpha 应视为 0），因此不会被 edge_alpha 惩罚。
- 对每幅图按有效邻居数归一化，避免图像分辨率影响 loss 尺度。
- 稀疏正则统计 edge_mat 的上三角（去重），只统计存在邻接（非零 entry）部分。
"""

def continuity_loss_edge(depth_map, tri_id_map_list, edge_alpha_mat_list,
                         mask=None, lambda_cont=1.0, lambda_sparsity=1e-4):
    """
    计算边级连续性损失并返回诊断信息（注释详见上方模块 docstring）。
    说明：函数内部对 batch 中每张图单独处理（便于应对不同 num_tri）。
    Args：
    - depth_map: torch.Tensor，形状 [B,1,H,W] 或 [B,H,W]。模型预测的深度图（或 ground-truth 深度）。
    - tri_id_map_list: list 长度 B，每项为 torch.LongTensor [H, W]，
        表示每像素所属三角 ID（范围 0 .. num_tri-1）。
    - edge_alpha_mat_list: list 长度 B，每项为 torch.Tensor [num_tri, num_tri]，
        对称矩阵，entry(t1,t2) 为两三角间边的断裂概率 alpha ∈ [0,1]（若 t1,t2 非邻接可为 0）。
    - mask: optional，torch.Tensor [B,1,H,W] 或 [B,H,W]，有效像素掩码（1 表示该像素参与损失）。
    - lambda_cont: 连续性损失项的权重（scalar）。
    - lambda_sparsity: 对 edge alpha 的稀疏正则权重（scalar）。

    return：
    - total_loss: 标量 tensor = continuity_loss + sparsity_loss
    - diagnostics: dict，包含 'cont_loss','sparsity_loss','alpha_mean_per_image'，用于可视化与调试
    """

    # 1) 统一 depth_map 的形状到 [B,1,H,W] 还需要判断valid mask
    #    如果用户传入的是 [B,H,W]，我们在第一个维度插入 channel 维
    if depth_map.dim() == 3:
        depth_map = depth_map.unsqueeze(1)  # -> [B,1,H,W]

    B, _, H, W = depth_map.shape

    # 2) mask 默认全部有效（若外部不传掩码）
    if mask is None:
        mask = torch.ones_like(depth_map)

    # 累积量初始化（用于 batch 内平均）
    total_cont = 0.0       # 累积每张图的归一化连续性损失
    total_images = 0.0     # 有效图计数（应等于 B，保留灵活性）
    sparsity_terms = []    # 存储每张图 edge alpha 的均值（用于稀疏正则）
    alpha_means = []       # 辅助诊断（保存每张图的 alpha 均值供可视化）

    # 逐张图处理：保持实现简单且支持每图不同 num_tri 的情形
    for b in range(B):
        # 3) 取出该图的深度与相应 tri_map / edge_mat
        z = depth_map[b, 0]  # [H, W] （提取为 2D 张量，便于差分）
        tri_map = tri_id_map_list[b].long().to(z.device)  # [H, W]，每像素的 tri id
        edge_mat = edge_alpha_mat_list[b].to(z.device)   # [num_tri, num_tri]

        # ---- 右方向邻接对 (p, q = right neighbor) ----
        # z_right 为 p 与其右邻 q 的深度差 (p - q)
        # z_right shape: [H, W-1]
        z_right = z[:, :-1] - z[:, 1:]

        # valid_r 标记左右两端像素都在有效 mask 内（boolean）
        valid_r = (mask[b, 0, :, :-1] * mask[b, 0, :, 1:]) > 0.5  # [H, W-1] bool

        # 对应两端的三角 id（索引两侧 tri_map）
        t_left = tri_map[:, :-1]   # 三角 id for pixel p
        t_right = tri_map[:, 1:]    # 三角 id for pixel q

        # 通过 tri_id 去 edge_mat 查 alpha 值（使用张量索引）
        # alpha_r shape: [H, W-1]
        # 注意：如果 t_left == t_right（在同一三角内部），我们想把 alpha 视为 0（不是三角间边）
        # 由于 edge_mat 在这些对角位置通常为 0（或未定义），但为了保险，显式清零同三角对
        alpha_r = edge_mat[t_left, t_right]
        same_tri_r = (t_left == t_right)
        alpha_r = alpha_r * (~same_tri_r).float()  # 将同三角位置的 alpha 置为 0

        # 权重由 (1 - alpha) 给出：alpha 越大（断裂概率高）权重越小（对差分不惩罚）
        w_r = (1.0 - alpha_r)

        # 平方差项
        sq_r = (z_right ** 2)

        # 加权并乘以 valid mask，统计 sum
        cont_r = (w_r * sq_r * valid_r.float()).sum()  # 标量（对所有有效右邻对求和）

        # 有效右邻对数量（用于归一化，避免不同图像尺寸或有效像素数影响）
        edges_r = valid_r.float().sum()

        # ---- 下方向邻接对 (p, q = down neighbor) ----
        z_down = z[:-1, :] - z[1:, :]  # [H-1, W]
        valid_d = (mask[b, 0, :-1, :] * mask[b, 0, 1:, :]) > 0.5
        t_up = tri_map[:-1, :]
        t_down = tri_map[1:, :]
        alpha_d = edge_mat[t_up, t_down]
        same_tri_d = (t_up == t_down)
        alpha_d = alpha_d * (~same_tri_d).float()
        w_d = (1.0 - alpha_d)
        sq_d = (z_down ** 2)
        cont_d = (w_d * sq_d * valid_d.float()).sum()
        edges_d = valid_d.float().sum()

        # 合并右/下两个方向的累积和与计数
        cont_sum = (cont_r + cont_d)
        edges_sum = (edges_r + edges_d).clamp(min=1.0)  # 至少为 1 避免除零

        # 对该图做归一化（按有效邻居数），然后累积到 total_cont
        total_cont += cont_sum / edges_sum
        total_images += 1.0

        # ---- 对 edge_mat 做稀疏正则统计（只统计上三角避免重复） ----
        # 目的是鼓励 edge predictor 输出较小的 alpha（即尽量不预测太多断裂）
        num_tri = edge_mat.shape[0]
        if num_tri > 0:
            # mask_upper 取上三角（diagonal=1 表示不包含对角线）
            mask_upper = torch.triu(torch.ones_like(edge_mat), diagonal=1).to(edge_mat.device)
            values = edge_mat * mask_upper  # 只保留上三角的值
            adjacency_mask = (values > 0.0).float()  # 这里以 >0 判断是否存在邻接（非零 entry）
            # 若 adjacency_mask.sum() == 0（没有任何邻接被列出），用 clamp 保证分母非零
            adj_count = adjacency_mask.sum().clamp(min=1.0)
            # mean_alpha: 只对存在邻接的位置求平均（这是对已列出的边的平均 alpha）
            mean_alpha = (values.sum() / adj_count)
            sparsity_terms.append(mean_alpha)
            alpha_means.append(mean_alpha.detach())
        else:
            # 若没有三角，填 0 以保持长度一致
            sparsity_terms.append(torch.tensor(0.0, device=z.device))
            alpha_means.append(torch.tensor(0.0, device=z.device))

    # ---- batch 平均化与加权 ----
    # cont_loss: batch 内每图的归一化连续性损失均值，再乘以 lambda_cont
    cont_loss = (total_cont / total_images) * lambda_cont

    # sparsity: 所有图上 mean_alpha 的均值（代表总体上 edge predictor 的平均断裂率）
    if len(sparsity_terms) > 0:
        sparsity = torch.stack(sparsity_terms).mean()
    else:
        sparsity = torch.tensor(0.0, device=depth_map.device)
    sparsity_loss = lambda_sparsity * sparsity

    total_loss = cont_loss + sparsity_loss

    # diagnostics：用于记录/调试/可视化
    diagnostics = {
        'cont_loss': cont_loss.detach() if isinstance(cont_loss, torch.Tensor) else torch.tensor(cont_loss),
        'sparsity_loss': sparsity_loss.detach() if isinstance(sparsity_loss, torch.Tensor) else torch.tensor(sparsity_loss),
        # 'alpha_mean_per_image' 是个 list，包含每张图的 alpha 平均（用于可视化）
        'alpha_mean_per_image': alpha_means,
    }
    return total_loss, diagnostics


# ----- 补充：如何解读 diagnostics（供训练时观察） -----
# - diagnostics['cont_loss']：越小表示在非断裂边上深度越平滑（如果过小可能过度平滑）。
# - diagnostics['sparsity_loss']：越小说明 edge predictor 输出越稀疏（alpha 趋向 0）。
# - diagnostics['alpha_mean_per_image']：便于检查是否退化（全 0 或全 1）。一般期望 alpha_mean 在一个合理范围（例如 0.05~0.3），
#   具体依数据集和三角划分稠密度而异。若 alpha_mean ≈ 0，说明 predictor 不预测断裂；若 ≈1，说明 predictor 过度标注断裂。
#
# 调参建议：
# - 若 alpha 全 0（不预测断裂）：降低 lambda_sparsity 或增加伪标签监督（pseudo-label from depth diff）来告诉模型哪里是真断裂。
# - 若 alpha 全 1（总是预测断裂）：增大 lambda_sparsity，或在初期降低 lambda_cont（避免模型以“把所有边设为断裂”来轻易降低连续性损失）。
#
# 性能提示：
# - 本函数使用 Python for-loop 遍历 batch 内每张图以处理不同 num_tri；对于批量很大或三角数很多的场景，
#   可把 tri 数据 pad 到相同长度并尝试批量化实现以加速（需注意 Mask 与无效项处理）。
#
# 典型调用（伪代码）：
# loss_cont, diag = continuity_loss_edge(pred_depth, tri_id_map_list, edge_mats_list,
#                                       mask=valid_mask, lambda_cont=0.8, lambda_sparsity=1e-4)
# loss = loss + loss_cont
# logging: record diag['cont_loss'], diag['sparsity_loss'], diag['alpha_mean_per_image']
#


"""
propagation gate helper:
- allow_propagation_hard(tri_id_map, edge_mat, p_coord, q_coord, gate_thresh)
    若跨边 alpha > gate_thresh 则返回 False（不允许传播），否则 True
- propagation_weight_soft(tri_id_map, edge_mat, p_coord, q_coord)
    返回权重 w = 1 - alpha_edge（用于软加权）
Note: tri_id_map: tensor [H,W] (int), edge_mat: tensor [num_tri,num_tri]
"""

def allow_propagation_hard(tri_id_map, edge_mat, p_y, p_x, q_y, q_x, gate_thresh=0.7):
    t_p = int(tri_id_map[p_y, p_x].item())
    t_q = int(tri_id_map[q_y, q_x].item())
    if t_p == t_q:
        return True
    # 如果 edge_mat 索引越界或为 0，视为无断裂
    if t_p < 0 or t_p >= edge_mat.shape[0] or t_q < 0 or t_q >= edge_mat.shape[0]:
        return True
    alpha = float(edge_mat[t_p, t_q].item())
    return alpha <= gate_thresh

def propagation_weight_soft(tri_id_map, edge_mat, p_y, p_x, q_y, q_x):
    t_p = int(tri_id_map[p_y, p_x].item())
    t_q = int(tri_id_map[q_y, q_x].item())
    if t_p == t_q:
        return 1.0
    if t_p < 0 or t_p >= edge_mat.shape[0] or t_q < 0 or t_q >= edge_mat.shape[0]:
        return 1.0
    alpha = float(edge_mat[t_p, t_q].item())
    return 1.0 - alpha
