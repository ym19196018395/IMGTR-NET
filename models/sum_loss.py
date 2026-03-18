import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import _sample_map


class EdgeLabelGenerator(nn.Module):
    """
    自监督 EdgeLabelGenerator（BCE 形式），用于监督 edge alpha：
    目标：利用 GT 深度图动态生成“断裂真值 (Ground Truth Label)”进行监督。
    逻辑流程：

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

    def __init__(self, depth_threshold=0.2, sparsity_weight=0.0):
        """
                Args:
                    depth_threshold (float): 判定断裂的深度阈值 (单位是m，超过0.2m就算断裂)。
                                             如果 GT 深度差大于此值，认为该边是断裂的 (Label=1)。
                    sparsity_weight (float): 稀疏正则权重 (可选)。
                """
        super().__init__()
        self.depth_threshold = float(depth_threshold)
        self.sparsity_weight = float(sparsity_weight)


    def _generate_gt_target(self, midpoints, gt_depth_map, tri_id_map, t1_ids, t2_ids):
        """
        基于 GT 深度图生成二值标签
        Returns:
            targets: [E] 0 or 1
            valid_mask: [E] 标记哪些边的 GT 数据是有效的
        """
        E = midpoints.shape[0]
        H, W = gt_depth_map.shape[-2:]
        device = midpoints.device

        # 1. 生成 3x3 采样网格
        dx = torch.linspace(-1, 1, 3, device=device) * (2.0 / W)
        dy = torch.linspace(-1, 1, 3, device=device) * (2.0 / H)
        grid_y, grid_x = torch.meshgrid(dy, dx, indexing='ij')
        offsets = torch.stack((grid_x, grid_y), dim=-1).view(1, 9, 2)

        # [E, 9, 2]
        sample_grid = midpoints + offsets

        # 2. 采样 GT Depth 和 ID Map
        # 技巧：reshape 为 [1, E*9, 1, 2] 进行一次性采样
        flat_grid = sample_grid.view(1, E * 9, 1, 2).unsqueeze(2)  # [1, E*9, 1, 2] fix dim
        # grid_sample 需要 grid 是 4D (N, H, W, 2)
        flat_grid = sample_grid.view(1, E * 9, 1, 2)

        # 采样深度 (Bilinear)
        # 注意：GT Depth 通常包含 0 (无效值)，采样后需要处理
        sampled_depth = F.grid_sample(gt_depth_map, flat_grid, mode='bilinear', align_corners=True,
                                      padding_mode='zeros')

        # 采样 ID (Nearest)
        sampled_id = F.grid_sample(tri_id_map, flat_grid, mode='nearest', align_corners=True, padding_mode='border')

        patches_depth = sampled_depth.view(E, 9)
        patches_id = sampled_id.view(E, 9)

        # 3. 分离区域 T1 和 T2
        # 判断 patch 中的像素属于哪个三角形
        mask_t1 = (torch.abs(patches_id - t1_ids.unsqueeze(1)) < 0.1).float()
        mask_t2 = (torch.abs(patches_id - t2_ids.unsqueeze(1)) < 0.1).float()

        # 4. 过滤无效 GT (GT=0 的位置不参与计算)
        valid_gt = (patches_depth > 1e-4).float()
        mask_t1 = mask_t1 * valid_gt
        mask_t2 = mask_t2 * valid_gt

        # 5. 计算均值
        sum_t1 = mask_t1.sum(dim=1)
        sum_t2 = mask_t2.sum(dim=1)

        # 有效性判断：只有当 T1 和 T2 区域都有至少一个有效 GT 像素时，这条边才有效
        # 否则无法判断深度差，应忽略此边
        edge_valid_mask = (sum_t1 > 0) & (sum_t2 > 0)

        mean_d1 = (patches_depth * mask_t1).sum(dim=1) / sum_t1.clamp(min=1.0)
        mean_d2 = (patches_depth * mask_t2).sum(dim=1) / sum_t2.clamp(min=1.0)

        # 6. 生成标签
        diff = torch.abs(mean_d1 - mean_d2)

        # 核心逻辑：如果深度差大于阈值，Label=1 (断裂)，否则 0
        targets = (diff > self.depth_threshold).float()

        return targets, edge_valid_mask

    # ----------------- 主接口 -----------------
    def forward(self, pred_alphas_list, gt_depth_map, tri_infos, tri_id_map):
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
                    'tri_edge_ids_list':每个三角形的边ID列表
                    'edges_midpoints': 边对应的中点已经归一化 List[B] of [E, 2]
                }
            - tri_id_map: [B, H, W] (Dense ID Map)
        Returns:
            final_loss (tensor scalar), info dict 包含分项损失
        """
        device = gt_depth_map.device
        B = len(pred_alphas_list)

        total_bce_loss = 0.0
        total_valid_edges = 0
        total_sparsity_loss = 0.0

        # 诊断统计
        diag_stats = {'pos_ratio': 0.0, 'valid_ratio': 0.0}

        output_alphas_list = []
        output_mask_list=[]
        for b in range(B):
            # 获取当前数据
            pred_alphas = pred_alphas_list[b]  # [E]
            if pred_alphas.numel() == 0:
                continue

            # 移动数据到 GPU
            current_edges = tri_infos[0]['edges_list'][b].to(device).long()
            current_vertices = tri_infos[0]['vertices_list'][b].to(device)

            # 准备单张图的 Map
            curr_gt_map = gt_depth_map[b].unsqueeze(0)  # [1, 1, H, W]
            curr_id_map = tri_id_map[b].unsqueeze(0).unsqueeze(0).float()  # [1, 1, H, W]

            # --- Step A: 复用定位逻辑 (计算中点) ---
            # 获取边两侧三角形的索引
            idx1 = current_edges[:, 0].long()
            idx2 = current_edges[:, 1].long()

            # 获取归一化中点
            midpoints = tri_infos[0]['edges_midpoints'][b].unsqueeze(1)  # [E_b, 1, 2]

            # --- Step B & C: 基于 GT 生成真值标签 ---
            # 传入 idx1, idx2 作为 t1_ids, t2_ids
            # targets: [E], valid_mask: [E]
            gt_targets, valid_mask = self._generate_gt_target(
                midpoints, curr_gt_map, curr_id_map, idx1, idx2
            )

            # 这里的 idx1 == idx2 表示这条边只有一个邻接面 (边界)
            is_boundary = (idx1 == idx2)  # [E] bool

            if is_boundary.any():
                # 1. 强制将边界边的 Target 设为 1 (断裂)
                # 注意：gt_targets 不需要梯度，所以直接修改是安全的
                gt_targets[is_boundary] = 1.0

                # 2. 强制认为边界边是“有效”的 (即使 GT 采样失败)
                # 因为边界本身就是一种极强的几何先验，不需要 GT 深度也能确定是断裂
                valid_mask = valid_mask | is_boundary

            output_alphas_list.append(gt_targets)
            output_mask_list.append(valid_mask)

        return output_alphas_list,output_mask_list


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



