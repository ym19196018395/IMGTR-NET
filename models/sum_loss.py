import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeConsistencyLoss(nn.Module):
    """
    自监督 EdgeConsistencyLoss（BCE 形式），用于监督 edge alpha：
      - 基于每条边两侧三角的 mean depth 差（可选融合特征差）生成 soft-label（consistency_score）
      - 使用 BCE(pred_alpha, consistency_score) 作为主损失
      - 可选 smoothness 正则（邻边 alpha 应相似）
    兼容输入：
      - pred_alpha_list: list 长度 B，每项 tensor [E_b]（EdgeHead 返回）
      - depth_map: [B,1,H,W]（用于内部 pool tri_depths，除非你提供 tri_depths_list）
      - tri_infos_list: list 长度 B，每项 dict，需含：
            'num_tri': N,
            'tri_masks': tensor [N,Hc,Wc] 或 list of N masks (bool/0-1)
            'edges': list length E_b, 每项 {'tri_ids': (t1,t2), 'edge_pixels': optional}
      - feat_map: [B,C,H,W]（可选，用于计算 tri_feats）
      - tri_depths_list / tri_feats_list: 可选，若已在 EdgeHead 里计算则传入以复用
    返回：
      loss (scalar tensor), diagnostics (dict)
    """

    def __init__(self, depth_threshold=0.05, feat_weight=0.0, smooth_weight=0.0, device=None):
        """
        Args:
          depth_threshold: float，用于把 depth_diff 归一化到 [0,1]（除数）
          feat_weight: float in [0,1]，feat_score 的权重（0 则只用 depth）
          smooth_weight: float，smoothness 正则权重
          device: torch.device 或 None（自动选）
        """
        super().__init__()
        self.depth_threshold = float(depth_threshold)
        self.feat_weight = float(feat_weight)
        self.smooth_weight = float(smooth_weight)
        self.device = device if device is not None else (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    # ----------------- 辅助函数：从 tri_masks pool depth/feat -----------------
    def _pool_tri_depths_from_masks(self, depth_b, tri_masks):
        """
        depth_b: tensor [1,H,W] 或 [H,W]
        tri_masks: tensor [N,H,W] (0/1 or bool)
        返回 tri_depths tensor [N]
        """
        if depth_b.dim() == 3:
            depth_b = depth_b[0]  # -> [H,W]
        N = tri_masks.shape[0]
        # 将 tri_masks 和 depth_b 都转换到 float，并用向量化计算 sum/count
        masks_flat = tri_masks.view(N, -1).to(depth_b.device).float()  # [N, HW]
        depth_flat = depth_b.view(-1).to(depth_b.device).float()  # [HW]
        # tri_depth_sum = masks_flat @ depth_flat  -> [N]
        tri_sum = torch.matmul(masks_flat, depth_flat)  # [N]
        counts = masks_flat.sum(dim=1).clamp(min=1.0)  # [N]
        tri_mean = tri_sum / counts
        return tri_mean  # [N]

    def _pool_tri_feats_from_masks(self, feat_b, tri_masks):
        """
        feat_b: tensor [C,H,W]
        tri_masks: tensor [N,H,W]
        返回 tri_feats tensor [N, C]
        """
        C = feat_b.shape[0]
        N = tri_masks.shape[0]
        masks_flat = tri_masks.view(N, -1).to(feat_b.device).float()  # [N, HW]
        feat_flat = feat_b.view(C, -1).permute(1, 0)  # [HW, C]
        # tri_sum: [N, C] = masks_flat @ feat_flat  (via scatter-like matmul)
        # Using matmul: (N x HW) @ (HW x C) -> N x C
        tri_sum = torch.matmul(masks_flat, feat_flat)  # [N, C]
        counts = masks_flat.sum(dim=1).clamp(min=1.0).unsqueeze(1)  # [N,1]
        tri_mean = tri_sum / counts  # [N,C]
        return tri_mean

    # ----------------- 主接口 -----------------
    def forward(self,
                pred_alpha_list,
                depth_map=None,
                tri_infos_list=None,
                feat_map=None,
                tri_depths_list=None,
                tri_feats_list=None):
        """
        Args:
          pred_alpha_list: list len=B, each tensor [E_b] (pred alpha per edge)
          depth_map: tensor [B,1,H,W], optional if tri_depths_list provided
          tri_infos_list: list len=B, each dict with keys 'num_tri','tri_masks','edges'
          feat_map: optional [B,C,H,W]，若 feat_weight>0 且 tri_feats_list 未提供则需要
          tri_depths_list, tri_feats_list: optional lists of precomputed tri-level pools to avoid re-pooling

        Returns:
          (loss, diagnostics)
        """
        device = self.device
        batch_size = len(pred_alpha_list)
        total_bce = 0.0
        total_edges = 0
        smooth_losses = []
        all_consistency_means = []
        all_alpha_means = []

        # 遍历 batch（每张图 num_tri 不同，逐图处理）
        for b in range(batch_size):
            pred_alpha = pred_alpha_list[b].to(device).view(-1)  # [E]
            tri_infos = tri_infos_list[b]
            edges_info = tri_infos['edges']  # list length E_b, each {'tri_ids':(t1,t2), ...}
            E = len(edges_info)
            if E == 0:
                # 没有边，跳过
                smooth_losses.append(torch.tensor(0.0, device=device))
                continue

            # ---- 获取 tri_depths ----
            if tri_depths_list is not None and tri_depths_list[b] is not None:
                tri_depths = tri_depths_list[b].to(device)  # [N]
            else:
                # 必需有 depth_map 与 tri_masks
                assert depth_map is not None and tri_infos.get('tri_masks', None) is not None, \
                    "需要 depth_map + tri_masks 或 tri_depths_list"
                depth_b = depth_map[b].to(device)  # [1,H,W]
                tri_masks = tri_infos['tri_masks']
                # tri_masks 可能是 numpy 或 list 或 tensor
                if isinstance(tri_masks, list):
                    # list of [H,W] masks -> stack
                    tri_masks_t = torch.stack(
                        [torch.from_numpy(m).to(device) if isinstance(m, np.ndarray) else m.to(device) for m in
                         tri_masks], dim=0).float()
                else:
                    tri_masks_t = tri_masks.to(device).float()  # [N,Hc,Wc]
                tri_depths = self._pool_tri_depths_from_masks(depth_b, tri_masks_t)  # [N]

            # ---- 获取 tri_feats（若需要） ----
            if self.feat_weight > 0.0:
                if tri_feats_list is not None and tri_feats_list[b] is not None:
                    tri_feats = tri_feats_list[b].to(device)  # [N,C]
                else:
                    assert feat_map is not None and tri_infos.get('tri_masks', None) is not None, \
                        "需要 tri_feats_list 或 feat_map + tri_masks"
                    feat_b = feat_map[b].to(device)  # [C,H,W]
                    tri_masks = tri_infos['tri_masks']
                    if isinstance(tri_masks, list):
                        tri_masks_t = torch.stack(
                            [torch.from_numpy(m).to(device) if isinstance(m, np.ndarray) else m.to(device) for m in
                             tri_masks], dim=0).float()
                    else:
                        tri_masks_t = tri_masks.to(device).float()
                    tri_feats = self._pool_tri_feats_from_masks(feat_b, tri_masks_t)  # [N,C]
            else:
                tri_feats = None

            # ---- 构建 t1,t2 索引向量（vectorized） ----
            # edges_info 每项可能是 dict {'tri_ids':(t1,t2), ...} 或 tuple/list
            t1_list = []
            t2_list = []
            for e in edges_info:
                if isinstance(e, dict):
                    tid = e.get('tri_ids', None)
                    if tid is None:
                        raise ValueError("edges 条目必须包含 'tri_ids'")
                    t1_list.append(int(tid[0]));
                    t2_list.append(int(tid[1]))
                else:
                    # assume (t1,t2)
                    t1_list.append(int(e[0]));
                    t2_list.append(int(e[1]))
            # 存储着每个边对应的两个三角形，包括只有一个面的边，用来后面做差
            t1 = torch.tensor(t1_list, dtype=torch.long, device=device)  # [E]
            t2 = torch.tensor(t2_list, dtype=torch.long, device=device)  # [E]

            # ---- 计算 depth_diff -> consistency_score (depth component) ----
            d1 = tri_depths[t1]  # [E]
            d2 = tri_depths[t2]
            depth_diff = torch.abs(d1 - d2)  # [E]
            # 归一化到 [0,1]：除以 depth_threshold
            depth_score = torch.clamp(depth_diff / (self.depth_threshold + 1e-12), min=0.0, max=1.0)  # [E]

            # ---- 如果需要，计算 feat_score 并融合 ----
            if self.feat_weight > 0.0 and tri_feats is not None:
                f1 = tri_feats[t1]  # [E, C]
                f2 = tri_feats[t2]
                feat_dist = torch.norm(f1 - f2, p=2, dim=1)  # [E]
                # heuristic normalize (可替换为 dataset percentile)
                feat_score = torch.clamp(feat_dist / 10.0, min=0.0, max=1.0)  # [E]
                consistency_score = (1.0 - self.feat_weight) * depth_score + self.feat_weight * feat_score
            else:
                consistency_score = depth_score  # [E]

            # ---- 过滤无效的三角（若某些 tri 在 pool 时 counts == 0） ----
            # 若 tri_depths 中某 id 的 count 为 0（理论上 pool 时用 clamp(min=1) 已避免），但此处做保险：
            # 只保留 t1,t2 均在有效范围的边，处理一下只有一个三角面的边
            N_tri = tri_depths.shape[0]
            valid_mask_edges = (t1 >= 0) & (t1 < N_tri) & (t2 >= 0) & (t2 < N_tri)
            # 若希望进一步排除那些在 tri_masks 中像素数为0的三角，请在传入 tri_depths_list 处把这些 tri 标记或 tri_masks 中计算 counts 并传出 counts
            if not valid_mask_edges.all():
                valid_idx = torch.nonzero(valid_mask_edges).squeeze(1)
                if valid_idx.numel() == 0:
                    # 没有有效边
                    smooth_losses.append(torch.tensor(0.0, device=device))
                    continue
                # 筛除对应元素
                t1 = t1[valid_idx];
                t2 = t2[valid_idx]
                consistency_score = consistency_score[valid_idx]
                pred_alpha = pred_alpha[valid_idx]
                E = t1.shape[0]

            # ---- BCE between pred_alpha and consistency_score (soft label) ----
            # pred_alpha (来自 edgehead) 期望在 [0,1]，consistency_score 也是 [0,1]
            # 使用 reduction='sum' 然后除以 E 保持数值稳定，ym-issue，这里可能存在一个问题，不匹配问题
            num1 =pred_alpha.shape[0]
            num2 =consistency_score.shape[0]
            assert num1 is not num2, "预测头断裂概率和损失函数得分shape不匹配"
            bce = F.binary_cross_entropy(pred_alpha, consistency_score, reduction='sum')
            total_bce = total_bce + bce
            total_edges += E

            all_consistency_means.append(float(consistency_score.mean().detach().cpu().item()))
            all_alpha_means.append(float(pred_alpha.mean().detach().cpu().item()))

            # ---- smoothness 正则（对每张图计算） ----
            # 构建 tri -> incident edge indices 列表（python list），对每三角做 pairwise (ai-aj)^2 求和 (高效公式)
            # 注意：这里用简单实现，通常 N_tri 较大但稀疏邻接，成本可接受
            if(self.smooth_weight>0):
                tri_to_edges = [[] for _ in range(int(tri_depths.shape[0]))]
                # 使用原始 edges_info 顺序（如果我们之前删掉无效边，需要 remap）
                # 但为简单性，使用 t1,t2 与当前边索引范围 E
                for ei in range(E):
                    a = int(t1[ei].item());
                    b2 = int(t2[ei].item())
                    if a >= 0 and a < len(tri_to_edges):
                        tri_to_edges[a].append(ei)
                    if b2 >= 0 and b2 < len(tri_to_edges):
                        tri_to_edges[b2].append(ei)

                smooth_loss_img = torch.tensor(0.0, device=device)
                count_tri_with_pairs = 0
                for tri_edges in tri_to_edges:
                    L = len(tri_edges)
                    if L <= 1:
                        continue
                    idxs = torch.tensor(tri_edges, dtype=torch.long, device=device)
                    alphas = pred_alpha[idxs]  # [L]
                    # pairwise sum of squared differences: sum_{i<j} (ai-aj)^2 = 0.5*(L*sum(ai^2) - (sum ai)^2)
                    sum_sq = (alphas * alphas).sum()
                    sum_ = alphas.sum()
                    pair_sum = 0.5 * (L * sum_sq - sum_ * sum_)
                    smooth_loss_img = smooth_loss_img + pair_sum
                    count_tri_with_pairs += 1
                if count_tri_with_pairs > 0:
                    smooth_loss_img = smooth_loss_img / float(count_tri_with_pairs)
                else:
                    smooth_loss_img = torch.tensor(0.0, device=device)
                smooth_losses.append(smooth_loss_img)

        # end for b in batch

        # finalize BCE loss 平均化
        if total_edges == 0:
            loss_bce = torch.tensor(0.0, device=device)
        else:
            loss_bce = total_bce / float(total_edges)  # average per-edge

        # finalize smoothness
        if len(smooth_losses) > 0:
            smooth_loss_mean = torch.stack(smooth_losses).mean()
        else:
            smooth_loss_mean = torch.tensor(0.0, device=device)

        total_loss = loss_bce + self.smooth_weight * smooth_loss_mean

        diagnostics = {
            'loss_bce': float(loss_bce.detach().cpu().item()),
            'smooth_loss': float(smooth_loss_mean.detach().cpu().item()),
            'avg_consistency': float(sum(all_consistency_means) / max(len(all_consistency_means), 1)),
            'avg_alpha': float(sum(all_alpha_means) / max(len(all_alpha_means), 1)),
            'num_edges': int(total_edges)
        }
        return total_loss, diagnostics