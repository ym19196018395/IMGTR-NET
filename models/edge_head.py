
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeHead(nn.Module):
    """
    EdgeHead —— 边级断裂预测器（使用 tri 特征 + 深度差作为 edge_stat）
    输入:
      - feat: [B, C, Hf, Wf]  （backbone / FPN 特征）
      - img: [B, C_img, H, W] （原图，可为 None）
      - depth_map: [B,1,H,W] 或 [B,H,W] （当前网络的深度预测或GT）
      - tri_infos: list 长度 B，每个元素为 dict:
           {
             'num_tri': N,
             'tri_masks': tensor [N, H, W] 或 list of N masks,  # 三角像素掩码
             'edges': [ {'tri_ids': (t1,t2), 'edge_pixels': [(y,x), ...]}, ... ]
             边的相邻面，边像素
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

    # todo:三角掩码图片大小问题要进行额外处理
    def _masked_pool_triangles(self, feat_map, tri_masks):
        """
        对单张图做 tri_masks 的 masked average pool
        聚合成一个向量tri_feat，类似于一个超像素
        feat_resized → [128, 512, 512] → feat_flat → [262144, 128]；
        tri_masks → [300, 262144] → tri_sum = [300, 128] → tri_mean = [300,128].
        Args:
          feat_map: [C', Hf, Wf]  单张图投影后的特征 (torch.Tensor)
          tri_masks: [N, H, W]    三角掩码 (torch.Tensor 或 list)
        Returns:
          tri_feats: [N, C']  每个三角的 pooled 特征
        注意：
          - 若 feat_map 分辨率与 tri_masks 分辨率不同，会把 feat_map 先上/下采样到 tri_masks 大小
        """
        C = feat_map.shape[0]
        # tri_masks 支持 list 或 tensor
        if isinstance(tri_masks, list):
            tri_masks = torch.stack([m.float() for m in tri_masks], dim=0)  # [N,H,W]
        else:
            tri_masks = tri_masks.float()

        N, Hm, Wm = tri_masks.shape
        # 把 feat_map resize 到 (Hm, Wm) ym-need-modify
        _,H,W =feat_map.shape
        # 如果三角掩码和特征大小不同，就进行一个适配
        if Hm != H or Wm != W:
            feat_map = F.interpolate(feat_map.unsqueeze(0), size=(Hm, Wm), mode='bilinear', align_corners=False)[0]  # [C,Hm,Wm]

        feat_flat = feat_map.view(C, -1).permute(1,0)  # [Hm*Wm, C]
        # ym-question对于一个像素点都没有的小三角形
        mask_flat = tri_masks.view(N, -1).float()          # [N, Hm*Wm]
        # tri_sum = mask_flat @ feat_flat  -> [N, C]
        # 每个三角在空间维度对 feature 做加权求和
        tri_sum = torch.matmul(mask_flat, feat_flat)       # [N, C]
        counts = mask_flat.sum(dim=1).clamp(min=1.0).unsqueeze(1)  # [N,1]
        tri_mean = tri_sum / counts                         # [N, C]
        return tri_mean  # [N, C]

    def _masked_pool_depth(self, depth_map, tri_masks):
        """
        对单张图的深度做 tri_masks 的平均，得到每个三角的 mean depth
        Args:
          depth_map: [1,H,W] 或 [H,W]
          tri_masks: [N,H,W]
        Returns:
          tri_depths: [N] tensor
        """
        if depth_map.dim() == 3:
            depth = depth_map[0]
        else:
            depth = depth_map
        if isinstance(tri_masks, list):
            tri_masks = torch.stack([m.float() for m in tri_masks], dim=0)
        N, Hm, Wm = tri_masks.shape
        # 把 depth_map resize 到 (Hm, Wm), 因为interpolate至少需要3d，要符合bchw的话需要变为四维

        H,W =depth.shape
        if Hm != H or Wm != W:
            depth = depth.unsqueeze(0).unsqueeze(0)
            depth = F.interpolate(depth,size=(Hm, Wm), mode='nearest')[0][0]

        depth_flat = depth.contiguous().view(-1)  # [HW]
        mask_flat = tri_masks.view(N, -1).float()  # [N, HW]
        tri_sum = torch.matmul(mask_flat, depth_flat)  # [N]
        counts = mask_flat.sum(dim=1).clamp(min=1.0)
        tri_mean = tri_sum / counts
        return tri_mean  # [N]

    def forward(self, feat, img, depth_map, tri_infos):
        """
        Args:
          feat: [B, C, Hf, Wf]
          img: [B, C_img, H, W] 或 None
          depth_map: [B,1,H,W] 或 [B,H,W]
          tri_infos: list 长度 B，每个元素为 dict（见上方说明）
        Returns:
          out_edge_alphas: list len=B，每个 tensor [E]
          out_edge_mats: list len=B，每个 tensor [num_tri, num_tri]
        """
        # batch-size
        B = feat.shape[0]
        device = feat.device
        # 先投影 feat 到 tri_feat_dim，通道改变而已
        feat_proj = self.proj(feat)  # [B, C', Hf, Wf]

        out_edge_alphas = []
        out_edge_mats = []

        for b in range(B):
            info = tri_infos[b]
            tri_masks = info['tri_masks']  # [N, H, W] 或 list

            edges = info['edges']          # list of {'tri_ids': (t1,t2), 'edge_pixels': [...]}
            num_tri = int(info['num_tri'])

            # 若 tri_masks 是 numpy array 转成 tensor
            if isinstance(tri_masks, list):
                # 允许 tri_masks 已经是 list of tensors
                tri_masks_t = [m.to(device) if torch.is_tensor(m) else torch.tensor(m, device=device) for m in tri_masks]
                tri_masks = torch.stack([m.float() for m in tri_masks_t], dim=0).to(device)  # [N,H,W]
            else:
                tri_masks = tri_masks.to(device)

            # pooled tri features [N, C']
            feat_proj_b = feat_proj[b]  # [C', Hf, Wf]
            tri_feats = self._masked_pool_triangles(feat_proj_b, tri_masks)  # [N, C']

            # pooled tri depths [N]
            depth_b = depth_map[b]
            tri_depths = self._masked_pool_depth(depth_b, tri_masks)  # [N]

            # 为每条 edge 构造输入并预测 alpha
            E = len(edges)
            if E == 0:
                out_edge_alphas.append(torch.zeros(0, device=device))
                out_edge_mats.append(torch.zeros((num_tri, num_tri), device=device))
                continue

            edge_feats = []
            for e in edges:
                t1, t2 = e['tri_ids']
                # 检查 tri id 合法性 如果一条边只对应一个三角形则pass
                if (t1 < 0 or t1 >= num_tri) or (t2 < 0 or t2 >= num_tri):
                    edge_feats.append(None)
                    continue
                f1 = tri_feats[t1]  # [C']
                f2 = tri_feats[t2]
                # 深度差（标量）
                d1 = tri_depths[t1]
                d2 = tri_depths[t2]
                depth_diff = torch.abs(d1 - d2).unsqueeze(0)  # [1]
                # 拼接特征：f1 || f2 || depth_diff
                feat_concat = torch.cat([f1, f2, depth_diff.to(f1.device)], dim=0)
                edge_feats.append(feat_concat)

            # MLP 预测 只计算有效值 ym-issue 在后面做loss的时候这个有没有问题
            valid_feats = [f for f in edge_feats if f is not None]
            if len(valid_feats) == 0:
                out_edge_alphas.append(torch.zeros(E, device=device))
                out_edge_mats.append(torch.zeros((num_tri, num_tri), device=device))
                continue
            feats_stack = torch.stack(valid_feats, dim=0)  # [E_valid, D]
            # 将其放入mlp里面，进行一个预测，并将其缩放到0-1区间
            logits = self.edge_mlp(feats_stack)[:, 0] * self.edge_scale  # [E_valid]
            # 用激活函数变成概率值
            alphas_valid = torch.sigmoid(logits)

            # 恢复到完整 edge 顺序（None->0）
            alphas = []
            idx = 0
            for f in edge_feats:
                if f is None:
                    alphas.append(torch.tensor(0.0, device=device))
                else:
                    alphas.append(alphas_valid[idx])
                    idx += 1
            alphas = torch.stack(alphas, dim=0)  # [E]
            out_edge_alphas.append(alphas)

            # 生成对称矩阵
            edge_mat = torch.zeros((num_tri, num_tri), device=device)
            for i_e, e in enumerate(edges):
                t1, t2 = e['tri_ids']
                if 0 <= t1 < num_tri and 0 <= t2 < num_tri:
                    edge_mat[t1, t2] = alphas[i_e]
                    edge_mat[t2, t1] = alphas[i_e]
                else: # 如果一条边只有一个三角面则跳过
                    continue
            out_edge_mats.append(edge_mat)

        return out_edge_alphas, out_edge_mats


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
