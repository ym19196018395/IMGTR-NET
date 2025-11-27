import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import _sample_map


class EdgeConsistencyLoss(nn.Module):
    """
    自监督 EdgeConsistencyLoss（BCE 形式），用于监督 edge alpha：
      - 基于每条边两侧三角的 mean depth 差（可选融合特征差）生成 soft-label（consistency_score）
      - 使用 BCE(pred_alpha, consistency_score) 作为主损失
      - 可选 smoothness 正则（邻边 alpha 应相似）
    兼容输入：
      - pred_alpha_list: list 长度 B，每项 tensor [E_b]（EdgeHead 返回）
      - depth_map: [B,1,H,W]（用于内部 pool tri_depths，除非你提供 tri_depths_list）
      - tri_infos: list length B, 每项为 dict:
                {
                    'batch_num_tri': int,三角形的数量
                    'centroids': [B,n_tri,2] 每个三角形的质点，已经归一化处理
                    'vertices': [B,n_tri,3,2] 每个三角形的顶点，已进行归一化处理
                    'edges_list' :每个边的邻接面，
                    boundary_local_idxs_per_batch: 存储着断裂边，也就是只有一个面的边
                }
      - feat_map: [B,C,H,W]（可选，用于计算 tri_feats）
      - tri_depths_list / tri_feats_list: 可选，若已在 EdgeHead 里计算则传入以复用
    返回：
      loss (scalar tensor), diagnostics (dict)
    """

    def __init__(self, depth_threshold=0.10, feat_weight=0.0, smooth_weight=0.0, device=None,feat_scale=1.0, sparsity_weight=0.0):
        """
        Args:
            depth_threshold: 浮点，深度差归一化门限（用于 depth_diff_norm = clamp(depth_diff / depth_threshold, 0, 1)）
            smooth_weight: 平滑项权重（越大越平滑）
            feat_weight: 特征一致性在 combined target 中的权重（0.0 表示只用深度）
            feat_scale: 特征差距归一化因子（将 L2 距离 / feat_scale -> clamp 到 [0,1]）
            sparsity_weight: alpha 稀疏性权重（鼓励 alpha 小）
        """
        super().__init__()
        self.depth_threshold = float(depth_threshold)
        self.feat_weight = float(feat_weight)
        self.smooth_weight = float(smooth_weight)
        self.feat_scale = float(feat_scale)
        self.sparsity_weight = float(sparsity_weight)
        self.device = device if device is not None else (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    # ----------------- 主接口 -----------------
    def forward(self, pred_alphas_list, gt_depth_map, tri_infos, feat_map=None):
        """
        Args:
            - pred_alphas_list: list len=B, each tensor [E_b] (float 0..1)
            - gt_depth_map: tensor [B,1,H,W] (float, on device)
            - tri_infos: list length B, 每项为 dict:
                {
                    'batch_num_tri': int,三角形的数量
                    'centers_list': [B,n_tri,2] 每个三角形的质点，已经归一化处理
                    'vertices_list': [B,n_tri,3,2] 每个三角形的顶点，已进行归一化处理
                    'edges_list' :每个边的邻接面，
                    'edges_pixels': 每个边的像素（归一化）集合，后续需要引入作为一个特征传入mlp中
                    boundary_local_idxs_per_batch: 存储着断裂边，也就是只有一个面的边
                }
            - feat_map: optional tensor [B,C,H,W]，若给定将用于特征一致性项
        Returns:
            final_loss (tensor scalar), info dict 包含分项损失
        """
        device = gt_depth_map.device
        B = gt_depth_map.shape[0]
        _, _, H, W = gt_depth_map.shape

        # ------------------ 1. 从 tri_infos 提取预处理好的 batch 列表 ------------------
        # 你说数据放在 tri_infos[0] 中，所以直接读取
        info_batch = tri_infos[0] if isinstance(tri_infos, (list, tuple)) else tri_infos
        batch_num_tri = info_batch['batch_num_tri']  # list length B
        centers_list = info_batch['centers_list']  # list len B
        vertices_list = info_batch['vertices_list']  # list len B
        edges_list = info_batch['edges_list']  # list len B
        boundary_local_idxs_per_batch = info_batch.get('boundary_local_idxs_per_batch',
                                                       [[] for _ in range(B)])  # list len B

        # ------------------ 2. 统一 centers/vertices 到 tensor 并 pad 到 N_max ------------------
        # 假定 centers_list[b] 与 vertices_list[b] 已归一化到 [-1,1] 且为 numpy 或 torch tensor
        N_list = [int(x) for x in batch_num_tri]
        N_max = int(max(N_list)) if len(N_list) > 0 else 0
        if N_max == 0:
            zero = torch.tensor(0.0, device=device)
            return zero, {'bce': 0.0, 'smooth': 0.0, 'feat_cons': 0.0, 'sparsity': 0.0}

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

        batch_centers = torch.stack(centers_padded, dim=0).to(device=device).float()  # [B, N_max, 1, 2]
        batch_vertices = torch.stack(vertices_padded, dim=0).to(device=device).float()  # [B, N_max, 3, 2]

        # ------------------ 3. 使用相同的采样策略采样 GT 三角深度（保证一致性） ------------------
        # 依赖 sample_tri_attributes(map_tensor, centers, vertices) 函数 (与 EdgeHead 共用)
        gt_tri_depths = _sample_map(gt_depth_map,batch_centers, batch_vertices)  # [B, N_max, 1]

        # 若提供 feat_map，则同样采样三角特征
        if feat_map is not None:
            tri_feats = _sample_map(feat_map,batch_centers, batch_vertices)  # [B, N_max, C_feat]
        else:
            tri_feats = None

        # ------------------ 4. 将一个batch数据链接在一起方便计算，构建 all_edges_indices (global) & pred_alphas_flat ------------------
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
            zero = torch.tensor(0.0, device=device)
            return zero, {'bce': 0.0, 'smooth': 0.0, 'feat_cons': 0.0, 'sparsity': 0.0}

        # 拼接为 [Total_E, 2]，将所有批次的线段拼在一起
        all_edges_indices = torch.cat(all_edges_global, dim=0).long().to(device=device)

        # 对齐并拼接 pred_alphas_list（按 batch_edge_counts 顺序）
        pred_parts = []
        for b in range(B):
            cnt = batch_edge_counts[b]
            if cnt == 0:
                continue
            p = pred_alphas_list[b]
            if p is None or (isinstance(p, torch.Tensor) and p.numel() == 0):
                pred_parts.append(torch.zeros((cnt,), device=device))
            else:
                pred_parts.append(p.to(device=device).view(-1))
        if len(pred_parts) == 0:
            zero = torch.tensor(0.0, device=device)
            return zero, {'bce': 0.0, 'smooth': 0.0, 'feat_cons': 0.0, 'sparsity': 0.0}

        pred_alphas_flat = torch.cat(pred_parts, dim=0).unsqueeze(dim=1)  # [Total_E,1]


        if pred_alphas_flat.shape[0] != all_edges_indices.shape[0]:
            raise RuntimeError(
                f"[EdgeConsistencyLoss] pred_alphas_flat length ({pred_alphas_flat.shape[0]}) != target_score ({all_edges_indices.shape[0]})")


        # ------------------ 5. 从 gt_tri_depths / tri_feats 中 gather 两侧值 ------------------
        # 将三角面特征进行一个展平
        flat_gt = gt_tri_depths.view(B * N_max, 1)  # [B*N_max, 1]
        # 分别取边的相邻两个三角面的编号
        idx1 = all_edges_indices[:, 0]
        idx2 = all_edges_indices[:, 1]
        # 取得三角面的深度
        d1 = flat_gt[idx1] # [Total_E]
        d2 = flat_gt[idx2]
        depth_diff = torch.abs(d1 - d2)  # [Total_E]

        # depth consistency -> 归一化到 [0,1],ym-issue 有点问题全都是1 要么就是0
        depth_diff_norm = torch.clamp(depth_diff / (self.depth_threshold + 1e-6), 0.0, 1.0)  # [Total_E,1]

        # 特征一致性（若有）: L2 距离 -> 归一化到 [0,1]
        if tri_feats is not None:
            flat_feats = tri_feats.view(B * N_max, -1)  # [B*N_max, C]
            f1 = flat_feats[idx1]  # [Total_E, C]
            f2 = flat_feats[idx2]
            feat_dist = torch.norm(f1 - f2, p=2, dim=1)  # [Total_E]
            feat_dist_norm = torch.clamp(feat_dist / (self.feat_scale + 1e-6), 0.0, 1.0)
        else:
            feat_dist_norm = None

        # 合并成 soft target (depth + feat)，若没有 feat 则只用 depth
        if feat_dist_norm is not None:
            target_score = (depth_diff_norm * (1.0) + feat_dist_norm * (self.feat_weight)) / (1.0 + self.feat_weight)
        else:
            target_score = depth_diff_norm  # [Total_E] in [0,1]

        # ------------------ 6. 强制 boundary edges 的 target = 1.0 ------------------
        # 目的是解决边对应一个面的情况，默认断裂，值为1，让其进行学习
        # 计算每个 batch 的 global start index（按 batch_edge_counts）
        global_starts = []
        acc = 0
        for cnt in batch_edge_counts:
            global_starts.append(acc)
            acc += cnt

        # 计算这些断裂边在edge_total的位置
        global_boundary_positions = []
        for b_idx, local_idxs in enumerate(boundary_local_idxs_per_batch):
            if not local_idxs:
                continue
            start = global_starts[b_idx]
            for li in local_idxs:
                global_boundary_positions.append(start + int(li))

        # 将对应断裂边位置赋值为1
        if len(global_boundary_positions) > 0:
            gb = torch.tensor(global_boundary_positions, dtype=torch.long, device=device)
            target_score[gb] = 1.0

        # ------------------ 7. BCE 损失（pred_alphas_flat vs target_score） ------------------
        pred = pred_alphas_flat
        ts = target_score

        # 报数值统计 ym-need-delete
        print("DEBUG pred_alphas_flat: min={}, max={}, nan={}, inf={}".format(
            float(torch.min(pred).detach().cpu()), float(torch.max(pred).detach().cpu()),
            int(torch.isnan(pred).any()), int(torch.isinf(pred).any())
        ))
        print("DEBUG target_score: min={}, max={}, nan={}, inf={}".format(
            float(torch.min(ts).detach().cpu()), float(torch.max(ts).detach().cpu()),
            int(torch.isnan(ts).any()), int(torch.isinf(ts).any())
        ))

        # 如有异常，抛出更友好的错误
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            raise RuntimeError("pred_alphas_flat contains NaN or Inf. Check EdgeHead outputs / any in-place ops.")
        if torch.isnan(ts).any() or torch.isinf(ts).any():
            raise RuntimeError("target_score contains NaN or Inf. Check gt sampling / math.")
        # 检查超出区间
        if pred.min() < -1e-6 or pred.max() > 1.0 + 1e-6:
            print("WARN: pred outside [0,1], min/max:", float(pred.min()), float(pred.max()))

        loss_bce_sum = F.binary_cross_entropy(pred_alphas_flat, target_score.detach(), reduction='sum')
        loss_bce = loss_bce_sum / (pred_alphas_flat.numel() + 1e-8)

        # ------------------ 8. Smoothness 损失（基于三角-边关联） ------------------
        # 构建 tri -> incident edge indices 映射（使用 edges_list）
        smooth_loss=0.0
        if(self.smooth_weight>0.0):
            tri_to_edges = dict()  # key: global_tri_index (b*N_max + tri_local), value: list of global edge indices
            # global edge index enumeration: 0..Total_E-1 in the same order as all_edges_indices
            total_e = all_edges_indices.shape[0]
            for g_idx in range(total_e):
                t1 = int(all_edges_indices[g_idx, 0].item())
                t2 = int(all_edges_indices[g_idx, 1].item())
                # 对于边被记录成 (t,t)（boundary），仍然添加到该三角对应的 incident list
                if t1 not in tri_to_edges:
                    tri_to_edges[t1] = []
                if t2 not in tri_to_edges:
                    tri_to_edges[t2] = []
                tri_to_edges[t1].append(g_idx)
                if t2 != t1:
                    tri_to_edges[t2].append(g_idx)

            smooth_loss = torch.tensor(0.0, device=device)
            smooth_count = 0
            for tri_idx, edge_idxs in tri_to_edges.items():
                # 若一个三角 incident 边少于2 则跳过
                if len(edge_idxs) < 2:
                    continue
                # pairwise 差平方和（可优化，但此处直接计算）
                for i in range(len(edge_idxs)):
                    for j in range(i + 1, len(edge_idxs)):
                        e1 = edge_idxs[i];
                        e2 = edge_idxs[j]
                        diff = pred_alphas_flat[e1] - pred_alphas_flat[e2]
                        smooth_loss = smooth_loss + diff * diff
                        smooth_count += 1
            if smooth_count > 0:
                smooth_loss = smooth_loss / float(smooth_count)
            else:
                smooth_loss = torch.tensor(0.0, device=device)


        # ------------------ 9. Sparsity（可选） ------------------
        sparsity_loss = 0
        if self.sparsity_weight > 0.0:
            sparsity_loss=torch.mean(pred_alphas_flat)

        # ------------------ 10. 总损失合成 ----------------        -
        final_loss = loss_bce + self.smooth_weight * smooth_loss + self.sparsity_weight * sparsity_loss

        info = {
            'bce': loss_bce.item() if isinstance(loss_bce, torch.Tensor) else float(loss_bce),
            'smooth': smooth_loss.item() if isinstance(smooth_loss, torch.Tensor) else float(smooth_loss),
            'feat_cons': float(torch.mean(target_score).item()) if target_score.numel() > 0 else 0.0,
            'sparsity': float(sparsity_loss.item()) if isinstance(sparsity_loss, torch.Tensor) else float(
                sparsity_loss),
            'total': float(final_loss.item()) if isinstance(final_loss, torch.Tensor) else float(final_loss)
        }
        return final_loss, info



def _sanity_check_and_report(device, B, N_max, all_edges_indices, pred_alphas_flat, gt_tri_depths, target_score=None):
    # move small summary to cpu for printing (no heavy copy)
    try:
        # ensure long dtype and on cpu for inspection
        idx_cpu = all_edges_indices.detach().cpu().long()
    except Exception as e:
        raise RuntimeError("all_edges_indices 无法 detach/cpu: " + str(e))

    if idx_cpu.numel() == 0:
        return

    idx1 = idx_cpu[:,0].numpy()
    idx2 = idx_cpu[:,1].numpy()

    max_idx = max(int(idx1.max()) if idx1.size>0 else -1, int(idx2.max()) if idx2.size>0 else -1)
    min_idx = min(int(idx1.min()) if idx1.size>0 else 10**9, int(idx2.min()) if idx2.size>0 else 10**9)

    total_tri_flat = int(B * N_max)
    msg = f"[SANITY] all_edges count={idx_cpu.shape[0]}, idx range min={min_idx}, max={max_idx}, allowed 0..{total_tri_flat-1}"
    print(msg)

    if min_idx < 0 or max_idx >= total_tri_flat:
        # 输出更多上下文并 raise 明确错误（避免 device assert）
        bad1 = idx1[(idx1 < 0) | (idx1 >= total_tri_flat)] if idx1.size>0 else np.array([])
        bad2 = idx2[(idx2 < 0) | (idx2 >= total_tri_flat)] if idx2.size>0 else np.array([])
        raise IndexError(f"索引越界: found bad idxs in all_edges_indices. B*N_max={total_tri_flat}. bad1={bad1[:10]}, bad2={bad2[:10]}. "
                         "检查 edges_list, batch_edge_counts, tri_offsets 是否正确生成。")

    # 检查 pred_alphas_flat 长度与 all_edges_indices 行数是否一致
    if pred_alphas_flat is not None:
        try:
            pred_len = int(pred_alphas_flat.detach().cpu().numel())
        except:
            pred_len = -1
        if pred_len != idx_cpu.shape[0]:
            raise ValueError(f"pred_alphas_flat 长度 ({pred_len}) != all_edges 行数 ({idx_cpu.shape[0]}). "
                             "可能是拼接顺序或某些 batch 的 pred 是空导致不对齐。")

    # 检查 gt_tri_depths 是否包含 NaN/Inf
    try:
        gt_flat = gt_tri_depths.detach().cpu().view(-1)
        if not torch.isfinite(gt_flat).all():
            raise ValueError("gt_tri_depths 包含 NaN 或 Inf，检查采样或输入 depth_gt 是否有问题。")
    except Exception as e:
        print("无法检查 gt_tri_depths: ", e)

    # 检查 target_score 是否 finite 且在 [0,1]
    if target_score is not None:
        ts = target_score.detach().cpu()
        if not torch.isfinite(ts).all():
            raise ValueError("target_score 包含 NaN/Inf")
        if ts.min() < -1e-6 or ts.max() > 1.0001:
            print("WARN: target_score 超出 [0,1] 范围:", float(ts.min()), float(ts.max()))