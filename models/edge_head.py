import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import _sample_map, check_tensor


class EdgeHead(nn.Module):
    """
    EdgeHead —— 边级断裂预测器（使用 tri 特征 + 深度差作为 edge_stat）
    输入:
      - feat: [B, C, Hf, Wf]  （backbone / FPN 特征）
      - img: [B, C_img, H, W] （原图，可为 None）
      - depth_map: [B,1,H,W] 或 [B,H,W] （当前网络的深度预测或GT）
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
    说明:
      - tri_feat_dim: 三角特征维度（masked pooling 后的线性投影维度），表示该三角在 backbone 特征空间的区域描述子；
      - tri_masks: 表示每个三角的像素集合（可用 cv2.fillPoly 在预处理阶段生成）；
      - 我们只使用 depth 差（|mean_depth(tri1) - mean_depth(tri2)|）作为 edge_stat（你要求暂不使用法向）。
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
        self.edge_stat_dim = 1
        mlp_in = tri_feat_dim * 2 + self.edge_stat_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(mlp_in, edge_mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(edge_mlp_hidden, edge_mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(edge_mlp_hidden, 1)
        )
        # 可学习缩放，用来调节 sigmoid 输入尺度（训练稳定）
        self.register_parameter("edge_scale", nn.Parameter(torch.tensor(1.0)))

    def forward(self, feat, img, depth_map, tri_infos):
        """
        EdgeHead.forward（兼容原始 tri_infos 格式）

        Args:
            - feat: [B, C, H, W]   (torch.Tensor)
            - depth_map: [B, 1, H, W] (torch.Tensor) - 推荐传入 depth_map.detach() 如果不想让 edgehead 影响 depth predictor
            - tri_infos: list length B, 每项为 dict:
                {
                    'batch_num_tri': int,三角形的数量
                    'centers_list': [B,n_tri,2] 每个三角形的质点，已经归一化处理
                    'vertices_list': [B,n_tri,3,2] 每个三角形的顶点，已进行归一化处理
                    'edges_list' :每个边的邻接面，
                    'edges_pixels': 每个边的像素（归一化）集合，后续需要引入作为一个特征传入mlp中
                    'boundary_local_idxs_per_batch': 存储着断裂边，也就是只有一个面的边
                }
        Returns:
            output_alphas_list: list length B, 每项是 torch.Tensor shape [E_b]，类型 float，位于 feat.device 上
            tri_infos：已经处理完毕的数据
        """
        # --------------------------------------------
        # 0) 清洗数据
        # --------------------------------------------
        # 很多时候是 Backbone 炸了导致这里接收到 NaN
        # if torch.isnan(feat).any() or torch.isinf(feat).any():
        #     print("Warning: Backbone features contain NaN/Inf. Cleaning...")
        #     feat = feat.clone()
        #     feat[torch.isnan(feat)] = 0.0
        #     feat[feat == float('inf')] = 0.0
        #     feat[feat == -float('inf')] = 0.0
        #     # 截断特征值，防止过大导致 MLP 爆炸
        #     feat = torch.clamp(feat, min=-100.0, max=100.0)

        # --------------------------------------------
        # 1) 基本准备：device / dims
        # --------------------------------------------
        device = feat.device
        B, C, H, W = feat.shape

        # 投影降低通道（和你类里定义的 self.proj 保持一致）
        feat_proj = self.proj(feat)  # [B, D, H, W]

        # --------------------------------------------
        # 2) 将已经处理好的数据传入进来,pad之后并传入torch中，目的是保证张量大小一致
        # --------------------------------------------
        batch_num_tri = tri_infos[0]['batch_num_tri']
        centers_list = tri_infos[0]['centers_list']
        vertices_list = tri_infos[0]['vertices_list']
        edges_list = tri_infos[0]['edges_list']
        boundary_local_idxs_per_batch = tri_infos[0]['boundary_local_idxs_per_batch']

        # 计算 batch 的 N_max (padding 到最大三角数)
        N_max = max(batch_num_tri) if len(batch_num_tri) > 0 else 0

        # 如果没有任何三角，直接返回空 list
        if N_max == 0:
            return [torch.zeros(0, device=device) for _ in range(B)]

        # pad centers/vertices 到 N_max，形成 tensors [B, N_max, 1, 2] 和 [B, N_max, 3, 2] 目的是后面进行一个统一计算
        centers_padded = []
        vertices_padded = []
        for b in range(B):
            n_tri = batch_num_tri[b]
            c = centers_list[b]  # [n_tri,2] 或 shape (0,2)
            v = vertices_list[b]  # [n_tri,3,2] 或 (0,3,2)

            if n_tri < N_max:
                # 进行一个填充，填充到三角形数量最多大小的张量
                # ym-issue 可能要进行一个修改效率有点低了
                # pad with zeros (zeros 对 grid_sample 等价于图像左上角；
                # 但是这些 padding 三角不会被 edges用到所以可以放心填充
                pad_c = torch.zeros((N_max - n_tri, 2), dtype=torch.float32)
                c = torch.cat([c, pad_c], dim=0)
                pad_v = torch.zeros((N_max - n_tri, 3, 2), dtype=torch.float32)
                v = torch.cat([v, pad_v], dim=0)

            # reshape to required shape for sampling: centers -> [N_max, 1, 2]; vertices -> [N_max, 3, 2]
            centers_padded.append(c.unsqueeze(1))  # [N_max,1,2]
            vertices_padded.append(v)  # [N_max,3,2]

        # stack batch -> [B, N_max, 1, 2], [B, N_max, 3, 2],放入显卡，进行特征提取
        batch_centers = torch.stack(centers_padded, dim=0).to(device=device)  # float
        batch_vertices = torch.stack(vertices_padded, dim=0).to(device=device)  # float

        # ym-modify 验证点数据是否有问题，没问题
        # if torch.isnan(batch_centers).any() or torch.isinf(batch_centers).any():
        #     print("Warning: centers features contain NaN/Inf. ")
        #     # 这种情况下通常无法修复，只能报错或填充0
        #     # 为了跑通，将其设为 0 (采样左上角)
        #     batch_centers[torch.isnan(batch_centers)] = 0.0
        #
        # if torch.isnan(batch_vertices).any() or torch.isinf(batch_vertices).any():
        #     print("Warning: vertices features contain NaN/Inf. ")
        #     # 这种情况下通常无法修复，只能报错或填充0
        #     # 为了跑通，将其设为 0 (采样左上角)
        #     batch_vertices[torch.isnan(batch_vertices)] = 0.0

        # --------------------------------------------
        # 3) 采样三角特征和三角深度（在统一尺度 和设备上）
        # --------------------------------------------
        # ym-issue 深度值有问题，有nan,对数据进行一个清理
        clean_depth_map = depth_map.clone()

        # 1. 替换 NaN 和 Inf
        # 深度图中 NaN 换成 0 (或其他安全值)，Inf 截断
        # 版本太低没有nan_to_num函数
        # clean_depth_map = torch.nan_to_num(clean_depth_map, nan=0.0, posinf=0.0, neginf=0.0)

        clean_depth_map[torch.isnan(clean_depth_map)] = 0.0
        clean_depth_map[clean_depth_map == float('inf')] = 0.0
        clean_depth_map[clean_depth_map == -float('inf')] = 0.0

        # 2. 再次 Clamp 保证数值稳定
        # 假设是物理深度，防止出现 -1e8 这种奇怪的负数
        clean_depth_map = torch.clamp(clean_depth_map, min=0.0, max=2000.0)

        check_tensor('clean_depth_map', clean_depth_map)
        # check_tensor('feat-map',feat_proj)

        tri_feats_map = _sample_map(feat_proj, batch_centers, batch_vertices)  # [B, N_max, D]
        # 深度采样 ym-issue 之后替换成共享边
        tri_depths_map = _sample_map(clean_depth_map, batch_centers, batch_vertices)  # [B, N_max, 1]

        # --------------------------------------------
        # 4) 扁平化并一次性 gather 边特征（Flatten strategy）
        # --------------------------------------------
        D = tri_feats_map.shape[-1]
        # 将三角面特征进行一个展平
        flat_feats = tri_feats_map.view(B * N_max, D)  # [Total_Tri, D]
        flat_depths = tri_depths_map.view(B * N_max, 1)  # [Total_Tri, 1]

        all_edges_global = []
        batch_edge_counts = []

        # 将不同组的三角id进行一个累加类似于 第一组trinum有500 第二组trinum的标记从500开始计算
        # 这样保证了不会访问到之前pad填充的zero的特征和深度

        for b in range(B):
            edges_b = edges_list[b]  # LongTensor [E_b, 2] or shape (0,2)
            if edges_b.numel() == 0:
                batch_edge_counts.append(0)
                continue
            edges_b = edges_b.to(device=device).long()
            # 注意：edges_b 中的索引是 local 0..(n_tri-1)
            global_offset = b * N_max
            global_edges = edges_b + global_offset  # broadcasting
            all_edges_global.append(global_edges)
            batch_edge_counts.append(global_edges.shape[0])

        # 如果没有任何边，返回空列表（每个样本对应空 tensor）
        if len(all_edges_global) == 0:
            return [torch.zeros(0, device=device) for _ in range(B)]

        # 拼接为 [Total_E, 2]，将所有批次的线段拼在一起
        all_edges_indices = torch.cat(all_edges_global, dim=0).long().to(device=device)

        # ym-need-modify 11.26 后续可能要将断裂概率求完损失函数再固定，或者是预测头给面id为-1设置一个空的深度
        # 这里不存在-1的id，只会存在两个id相同的情况，之前处理完毕，所以求特征差也是等于0

        # 分别取边的相邻两个三角面的编号

        idx1 = all_edges_indices[:, 0]
        idx2 = all_edges_indices[:, 1]

        # 取得三角面的特征和深度
        f1 = flat_feats[idx1]  # [Total_E, D]
        f2 = flat_feats[idx2]
        d1 = flat_depths[idx1]  # [Total_E, 1]
        d2 = flat_depths[idx2]

        # --------------------------------------------
        # 5) 构造 MLP 输入并预测 alpha
        # --------------------------------------------

        depth_diff = torch.abs(d1 - d2)  # [Total_E, 1]
        # 比如我们认为超过 1000 的差异和 10000 的差异对“断裂”来说是一样的
        depth_diff = torch.clamp(depth_diff, max=500.0)

        # ym-need-modify 需要加入边特征，为了更好的求断裂边概率
        mlp_input = torch.cat([f1, f2, depth_diff], dim=1)  # [Total_E, 2D+1]

        logits = self.edge_mlp(mlp_input).squeeze(1) * self.edge_scale
        all_alphas = torch.sigmoid(logits)  # [Total_E]

        # --------------------------------------------
        # 6) 目的是解决边对应一个面的情况，默认断裂，值为1，
        # 不能在求损失函数之前对需求求梯度的张量进行一个修改 11.26
        # --------------------------------------------

        # global_starts = []
        # acc = 0
        # for cnt in batch_edge_counts:
        #     global_starts.append(acc)
        #     acc += cnt
        #
        # # 计算这些断裂边在edge_total的位置
        # global_boundary_positions = []
        # for b_idx, local_idxs in enumerate(boundary_local_idxs_per_batch):
        #     if not local_idxs:
        #         continue
        #     start = global_starts[b_idx]
        #     for li in local_idxs:
        #         global_boundary_positions.append(start + int(li))
        #
        # # 将对应断裂边位置赋值为1
        # if len(global_boundary_positions) > 0:
        #     gb = torch.tensor(global_boundary_positions, dtype=torch.long, device=all_alphas.device)
        #     all_alphas[gb] = 1.0

        # --------------------------------------------
        # 7) 拆分回 per-sample list，并返回
        # --------------------------------------------
        # torch.split 需要一个 Python list 的 sizes
        alphas_split = torch.split(all_alphas, batch_edge_counts)
        output_alphas_list = []
        for s in alphas_split:
            if s.numel() == 0:
                output_alphas_list.append(torch.zeros(0, device=device))
            else:
                output_alphas_list.append(s.to(device=device).float())

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
