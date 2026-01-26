import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import _sample_map


class EdgeConsistencyLoss(nn.Module):
    """
    自监督 EdgeConsistencyLoss（BCE 形式），用于监督 edge alpha：
    目标：利用 GT 深度图动态生成“断裂真值 (Ground Truth Label)”进行监督。
    逻辑流程：
      1. 计算每条边的几何中点。
      2. 在 GT 深度图上，围绕中点进行 3x3 局部采样。
      3. 利用 tri_id_map 区分采样点属于 T1 还是 T2。
      4. 计算 GT 深度差: diff = |Mean(T1) - Mean(T2)|。
      5. 生成标签: Target = 1 if diff > threshold else 0。
      6. 计算 BCE Loss。
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
            idx1 = current_edges[:, 0]
            idx2 = current_edges[:, 1]
            v1 = current_vertices[idx1]
            v2 = current_vertices[idx2]

            midpoints = self._compute_edge_midpoints(v1, v2)  # [E, 1, 2]

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
            # --- Step D: 计算 BCE Loss ---
            # 只在 GT 有效的边上计算 Loss
            if valid_mask.sum() > 0:
                valid_pred = pred_alphas[valid_mask]
                valid_target = gt_targets[valid_mask]

                # BCE Loss
                loss_bce = F.binary_cross_entropy(valid_pred, valid_target, reduction='sum')

                total_bce_loss += loss_bce
                total_valid_edges += valid_mask.sum().item()

                # 统计正样本比例 (断裂边的比例)
                diag_stats['pos_ratio'] += valid_target.sum().item()

            diag_stats['valid_ratio'] += valid_mask.sum().item()

            # --- Sparsity Loss (针对所有预测，无论 GT 是否有效) ---
            # 鼓励预测值整体偏向 0
            if self.sparsity_weight > 0:
                total_sparsity_loss += pred_alphas.mean()

        # 归一化 Loss
        if total_valid_edges > 0:
            final_bce = total_bce_loss / total_valid_edges
            diag_stats['pos_ratio'] /= total_valid_edges
        else:
            final_bce = torch.tensor(0.0, device=device)

        # Sparsity 平均
        final_sparsity = (total_sparsity_loss / B) * self.sparsity_weight

        total_loss = final_bce + final_sparsity

        info = {
            'bce': final_bce.item(),
            'sparsity': final_sparsity.item() if isinstance(final_sparsity, torch.Tensor) else 0.0,
            'total': total_loss.item(),
            'pos_ratio': diag_stats['pos_ratio'],  # 诊断：当前的 GT 阈值下，有多少比例的边被判定为断裂
            'valid_edges': total_valid_edges  # 诊断：有多少边成功采样到了 GT
        }

        return total_loss, info,output_alphas_list

    def compute_continuity_loss(self, planes, neighbor_indices, edge_probs, tri_vertices):
        """
        纯粹的连续性损失 (C0 Continuity Loss)
        只约束公共边的深度一致，不强求法向量一致 (允许有棱角，但不能裂开)

        Args:
            planes: [B, N, 4] (n, d)
            neighbor_indices: [B, N, 3] (3个邻居的索引)
            edge_probs: [B, N, 3] (3个边的边缘概率，来自EdgeHead)
            tri_vertices: [B, N, 3, 3] (三角形的3个顶点坐标, 顺序通常是 v0, v1, v2)
                           假设邻居0对应边 v0-v1, 邻居1对应 v1-v2... 需要根据你的Mesh结构确定
        """
        B, N, _ = planes.shape
        device = planes.device

        # 1. 获取邻居平面
        batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(-1, N, 3)
        neighbor_planes = planes[batch_idx, neighbor_indices]  # [B, N, 3, 4]

        # 2. 计算公共边中点 (Edge Midpoints)
        # 假设：
        # neighbor[:,:,0] 对应边 (v0, v1)
        # neighbor[:,:,1] 对应边 (v1, v2)
        # neighbor[:,:,2] 对应边 (v2, v0)
        # 必须确保这里的对应关系与你构建 neighbor_indices 时一致！

        v0 = tri_vertices[:, :, 0, :]
        v1 = tri_vertices[:, :, 1, :]
        v2 = tri_vertices[:, :, 2, :]

        # 计算三条边的中点
        mid_0 = (v0 + v1) / 2.0
        mid_1 = (v1 + v2) / 2.0
        mid_2 = (v2 + v0) / 2.0

        # [B, N, 3, 3] (3个中点的坐标)
        mid_points = torch.stack([mid_0, mid_1, mid_2], dim=2)

        # 3. 计算深度差异 (Discontinuity)
        # 当前平面在 3 个中点的深度: d = -n*x
        # planes.unsqueeze(2): [B, N, 1, 4]
        # n: [B, N, 1, 3]
        curr_n = planes.unsqueeze(2)[..., :3]
        # 计算出来的 d_val 应该等于 plane 的 d 参数，但这里我们用 n*x 计算“几何深度”
        # 实际上直接比较 Plane Equation 的残差更直接： n*x + d = 0

        # 更好的方法：
        # 如果点 x 在平面 A 上，则 n_A * x + d_A = 0
        # 连续性意味着：点 x (平面A的边缘) 也应该在平面 B 上，即 n_B * x + d_B ≈ 0

        # 我们用 Neighbor 的平面方程去测 Current 的边中点
        # 误差 = | n_neigh * mid_curr + d_neigh |
        # 如果连续，Current 的边中点也应该满足 Neighbor 的平面方程

        n_neigh = neighbor_planes[..., :3]  # [B, N, 3, 3]
        d_neigh = neighbor_planes[..., 3:]  # [B, N, 3, 1]

        # 计算点积: (n_x * x + n_y * y + n_z * z)
        # mid_points: [B, N, 3, 3]
        dot_val = torch.sum(n_neigh * mid_points, dim=-1, keepdim=True)  # [B, N, 3, 1]

        # 代入平面方程: | n*x + d |
        dist_error = torch.abs(dot_val + d_neigh)  # [B, N, 3, 1]

        # 计算深度不连续性 error: [B, N, 3]

        # 联动核心:
        # 如果 EdgeProb=1 (边界), weight=0 -> 允许不连续
        # 如果 EdgeProb=0 (平面), weight=1 -> 强制连续
        continuity_weight = torch.exp(-edge_probs)

        # Loss
        loss = (dist_error * continuity_weight).mean()

        # 还可以加一个正则项，防止 EdgeProb 全为 1 (模型为了降低 Loss 把所有地方都当成边界)
        # 鼓励 Edge 稀疏 (L1 Regularization)
        sparsity_loss = edge_probs.mean() * 0.1

        return loss + sparsity_loss

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