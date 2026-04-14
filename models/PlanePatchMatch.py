import torch
import torch.nn as nn
import torch.nn.functional as F
from mpmath import eye

from models.edge_head import EdgeHead
from utils import convert_edge_features_to_tri_format


class PlaneHypothesisGenerator(nn.Module):
    def __init__(self, num_hypotheses=5, perturbation_range=0.05):
        super().__init__()
        # 保证 K 至少为 2 (Best + FP)，否则随机逻辑会出错
        self.K = max(2, num_hypotheses)
        self.noise_scale = perturbation_range

    def forward(self, fitted_planes, depth_stage2_mean):
        """
        一次性生成 K 个假设：
        - 0: Best Guess (原始拟合)
        - 1: Fronto-Parallel (保底)
        - 2~K: Random Jitter (随机扰动)
        Args:
            fitted_planes: [B, N_tri, 4] (由 Fitter 生成的初始平面)
            depth_stage2_mean: [B, N_tri, 1] (每个三角形的平均深度，用于生成前向平行假设)
        Returns:
            hypotheses: [B, N_tri, K, 4]
        """
        B, N, _ = fitted_planes.shape
        device = fitted_planes.device

        # List 用于收集所有假设
        hypo_list = []

        # === 1. Hypothesis 0: 原始拟合平面 (The Best Guess) ===
        hypo_best = fitted_planes.unsqueeze(2)  # [B, N, 1, 4]
        hypo_list.append(hypo_best)

        # === 2. Hypothesis 1: 前向平行平面 (Fronto-Parallel) ===
        # n=[0,0,1], d=-mean_depth
        normal_fp = torch.zeros((B, N, 1, 3), device=device)
        normal_fp[..., 2] = 1.0
        d_fp = -depth_stage2_mean.view(B, N, 1, 1)  # 确保维度匹配
        hypo_fp = torch.cat([normal_fp, d_fp], dim=-1)  # [B, N, 1, 4]
        hypo_list.append(hypo_fp)

        # === 3. Hypothesis 2...K: 批量随机扰动 (Vectorized Jitter) ===
        num_random = self.K - 2

        if num_random > 0:
            # 基础数据扩展: [B, N, 1, 3] -> [B, N, num_random, 3]
            base_n = fitted_planes[:, :, :3].unsqueeze(2).expand(-1, -1, num_random, -1)
            base_d = fitted_planes[:, :, 3:].unsqueeze(2).expand(-1, -1, num_random, -1)

            # 一次性生成所有噪声
            rand_n = (torch.rand_like(base_n) - 0.5) * self.noise_scale
            # d 的扰动通常需要大一点 (比如乘 100 或者根据场景深度范围动态调整)
            rand_d = (torch.rand_like(base_d) - 0.5) * self.noise_scale * 50.0

            # 应用扰动
            new_n = F.normalize(base_n + rand_n, dim=-1)
            new_d = base_d + rand_d

            hypo_random = torch.cat([new_n, new_d], dim=-1)  # [B, N, num_random, 4]
            hypo_list.append(hypo_random)

        # === 4. 拼接所有假设 ===
        # 结果形状: [B, N, K, 4]
        hypotheses = torch.cat(hypo_list, dim=2)

        return hypotheses

class PlaneHomographyWarper(nn.Module):
    def __init__(self):
        super().__init__()

    def get_homography(self, plane_params, K_ref, K_src, R_rel, t_rel):
        """
        计算 Homography 矩阵
        配合 compute_costs 的 trick:
        - K_src 传入 Identity
        - R_rel 传入 rot @ K_ref
        - t_rel 传入 trans
        这样计算结果 H = rot - trans * n^T * inv(K_ref) / d，完全符合预期
        """
        B, C, H_img, W_img = plane_params.shape
        # device = plane_params.device # 未使用，可注释

        # 解析 n, d
        n = plane_params[:, :3, :, :]  # [B, 3, H, W]
        d = plane_params[:, 3:, :, :]  # [B, 1, H, W]

        # 1. 展平处理 [B, Pix, ...]
        n_flat = n.permute(0, 2, 3, 1).reshape(B, -1, 3, 1)  # [B, Pix, 3, 1]
        d_flat = d.permute(0, 2, 3, 1).reshape(B, -1, 1, 1)  # [B, Pix, 1, 1]

        # 2. 准备 R, t
        # 注意：这里的 R, t 已经在 compute_costs 里被扩展为 [B*K, 3, 3]
        R = R_rel.view(B, 1, 3, 3)
        t = t_rel.view(B, 1, 3, 1)

        # 3. 核心公式: M = R - (t @ n.T) / d
        # d + 1e-7 防止除零
        # 这中间已经乘出来了一个源视图的内参矩阵，所以内参矩阵设置为了单位阵，以防乘两次
        tn_T = torch.matmul(t, n_flat.transpose(-2, -1))
        M = R - (tn_T / (d_flat + 1e-7))  # [B, Pix, 3, 3]

        # 4. 前后乘内参
        K_src_expand = K_src.view(B, 1, 3, 3)
        K_ref_inv = torch.inverse(K_ref).view(B, 1, 3, 3)

        # H = K_src * M * K_ref^-1
        H_mat = torch.matmul(K_src_expand, torch.matmul(M, K_ref_inv))  # [B, Pix, 3, 3]

        return H_mat.view(B, H_img, W_img, 3, 3)

    def warp_feature(self, src_feature, H_mat):
        """
        利用 H 矩阵对 Src 特征进行 grid_sample
        优化点：增加了对相机背面点 (w < 0) 的处理
        """
        B, C, H, W = src_feature.shape
        device = src_feature.device

        # 1. 生成 Ref 图像的网格 (u, v, 1)
        y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        coords = torch.stack([x, y, torch.ones_like(x)], dim=-1).float()  # [H, W, 3]

        # [B, H, W, 3, 1]
        coords = coords.view(1, H, W, 3, 1).expand(B, -1, -1, -1, -1)

        # 2. 应用 H 变换: p_src = H @ p_ref
        # H_mat: [B, H, W, 3, 3]
        projected_coords = torch.matmul(H_mat, coords).squeeze(-1)  # [B, H, W, 3] (u', v', w')

        # 3. 透视除法与归一化
        u_prime = projected_coords[..., 0]
        v_prime = projected_coords[..., 1]
        w_prime = projected_coords[..., 2]

        # === 🛡️ 安全修正：处理相机背面的点 ===
        # 如果 w < 0，说明点在相机背面。如果不处理，除法后坐标可能是合法的(图像翻转)，导致采样错误特征。
        # 我们给它一个极小的正数 eps，或者将其坐标设为极大值(采样出界为0)
        valid_mask = w_prime > 1e-6

        # 这种写法既防止了除零，又把背面点(w<=0)除成了一个极大值/极小值，从而 grid_sample 采样到 0 (padding)
        w_prime = torch.where(valid_mask, w_prime, torch.ones_like(w_prime) * 1e-12)

        # 正常的透视除法
        u_norm = u_prime / w_prime
        v_norm = v_prime / w_prime

        # 如果是无效点(背面)，手动将其设为出界值 (例如 2.0，范围是 -1~1)
        # 这样 grid_sample 会填充 0
        u_norm = torch.where(valid_mask, u_norm, torch.tensor(10.0, device=device))
        v_norm = torch.where(valid_mask, v_norm, torch.tensor(10.0, device=device))

        # 4. 归一化到 [-1, 1] 用于 grid_sample
        # 公式: 2 * x / (W-1) - 1
        uv_grid = torch.stack([
            2.0 * u_norm / (W - 1) - 1.0,
            2.0 * v_norm / (H - 1) - 1.0
        ], dim=-1)  # [B, H, W, 2]

        # 5. 采样
        warped_feat = F.grid_sample(src_feature, uv_grid, align_corners=True, padding_mode='zeros')

        return warped_feat

class LearnedTrianglePropagator(nn.Module):
    def __init__(self, plane_dim=4, hidden_dim=64,feature_dim=16):
        """
        深度可微三角传播模块
        包含: Soft Gating, Attention Aggregation, MLP Refinement
        """
        super().__init__()
        self.feature_dim = feature_dim

        # 1. 初始编码器: 专管初始平面，供门控评估使用
        # 输入: Plane(4) + Cost(1) = 5 加入非线性激活函数，使其真正成为深度网络
        self.init_encoder = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        # 🌟 2. 精修编码器: 纯净感知聚合后的几何平面 (不要脏 Cost!)
        # 输入: Plane(4) = 4
        self.refine_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        # ym-add,新增一个自己的门控网络来计算自身权重，而不是固定为1
        # 修改：自身门控不再使用 Sigmoid，输出 Logit
        self.self_gate_net = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        # 初始化：bias=2.0，让自身初始权重偏高（保守起步，不要一开始就乱传播）
        nn.init.constant_(self.self_gate_net[2].bias, 2.0)
        nn.init.zeros_(self.self_gate_net[2].weight)

        # 3. 核心邻居门控网络 (Neighbor Gating): 决定传播多少邻居信息
        # 输入维度总计: hidden_dim*2 + 7
        # - self_hidden: hidden_dim
        # - neighbor_hidden: hidden_dim
        # - edge_prob: 1 (深度断裂概率)
        # - neighbor_cost: 1 (邻居置信度)
        # - plane_diff: 4 (几何参数差异)
        # - feat_dist: 1 (add 图像特征空间差异，网络的"眼睛")
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 7, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

        # 🌟 3. 终极残差头与 Modality Dropout
        # 输入: refine_hidden(H) + init_hidden(H) + current_costs(1) + F_curr(C)
        head_in_dim = hidden_dim * 2 + 1 + feature_dim
        self.plane_head = nn.Sequential(
            nn.Linear(head_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4)
        )

        # 你的神来之笔：极低概率 Dropout，防视觉依赖
        self.feat_dropout = nn.Dropout(p=0.15)

        # 残差网络零初始化
        nn.init.constant_(self.plane_head[2].bias, 0.0)
        nn.init.normal_(self.plane_head[2].weight, mean=0.0, std=0.001)

        # 定义一个缩放因子，用于压制 d 的巨大数值
        # 如果你的深度大多在几十到几百，100.0 是个好数值
        # todo：如果后续训练数据集有变化要进行更改
        self.d_scale_factor = 200

        # 🔥 新增：敞门初始化
        # 将 Bias 设为 2.0，Sigmoid(2.0) ≈ 0.88。
        # 让网络默认给予邻居极高的注意力权重，除非 EdgeHead 强行出示红牌！
        # nn.init.constant_(self.gate_net[2].bias, 2.0)
        # nn.init.normal_(self.gate_net[2].weight, mean=0.0, std=0.01)

    def forward(self, current_planes, current_costs, neighbor_indices,
                edge_probs, pixel_counts=None, ref_feature=None, centroids_norm=None):
        """
        Args:
            current_planes: [B, N, 4] 当前平面的 (nx, ny, nz, d)
            current_costs: [B, N, 1] 当前平面的光度匹配代价
            neighbor_indices: [B, N, 3] 邻居的索引
            edge_probs: [B, N, 3] EdgeHead 预测的 C0 物理断裂概率 (0 连通, 1 断裂)
            pixel_counts: [B, N] 三角形的像素个数
            ref_feature: [B, C, H, W] Stage 1 的高频图像特征图 (用于特征感知)
            centroids_norm: [B, N, 2] 三角形的归一化质心坐标 [-1, 1] (用于采样特征)
        Returns:
            final_planes: 传播精修后的最终平面 [B, N, 4]
        """
        B, N, _ = current_planes.shape
        device = current_planes.device

        # ==========================================
        # 1. 提取基础隐特征 & 分离缩放 d
        # ==========================================
        n_curr = current_planes[..., :3]
        d_curr_scaled = current_planes[..., 3:] / self.d_scale_factor
        scaled_planes = torch.cat([n_curr, d_curr_scaled], dim=-1)  # [B, N, 4]

        # 构造输入特征: [B, N, 5] -> 编码 -> [B, N, H]
        plane_feat = torch.cat([scaled_planes, current_costs], dim=-1)
        init_hidden = self.init_encoder(plane_feat)

        # ==========================================
        # 2. 🌟 特征感知提取 (为门控装上"眼睛")
        # ==========================================
        feat_dist = torch.zeros((B, N, 3, 1), device=device)  # 默认距离为0 (容错防崩溃)

        if ref_feature is not None and centroids_norm is not None:
            # grid_sample 采样质心处的图像特征
            grid = centroids_norm.view(B, N, 1, 2)
            # F_curr: [B, C, N, 1] -> [B, N, C]
            F_curr = F.grid_sample(ref_feature, grid, mode='bilinear', align_corners=True).squeeze(-1).permute(0, 2, 1)
            # L2 归一化，极大稳定训练
            F_curr = F.normalize(F_curr, p=2, dim=-1)

            # 提取邻居的特征 [B, N, 3, C]
            batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(-1, N, 3)
            F_neighbor = F_curr[batch_idx, neighbor_indices]

            # 计算 L2 距离的平方作为差异度量 [B, N, 3, 1]
            F_curr_exp = F_curr.unsqueeze(2)
            feat_dist = ((F_curr_exp - F_neighbor) ** 2).sum(dim=-1, keepdim=True)

        # ==========================================
        # 3. 收集邻居信息 & 计算几何差异
        # ==========================================
        batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(-1, N, 3)

        neighbor_hidden = init_hidden[batch_idx, neighbor_indices]  # [B, N, 3, H]
        neighbor_scaled_planes = scaled_planes[batch_idx, neighbor_indices]  # [B, N, 3, 4]
        neighbor_costs = current_costs[batch_idx, neighbor_indices]  # [B, N, 3, 1]

        self_scaled_expand = scaled_planes.unsqueeze(2)  # [B, N, 1, 4]
        plane_diff = neighbor_scaled_planes - self_scaled_expand  # [B, N, 3, 4]

        # ==========================================
        # 4. 双重门控计算 (Self-Gating & Neighbor-Gating)
        # ==========================================
        # --- 自身门控 (Logit) ---
        self_gate_input = torch.cat([init_hidden, current_costs], dim=-1)  # [B, N, H+1]
        self_weight_logits = self.self_gate_net(self_gate_input)  # [B, N, 1]

        # 物理封杀：如果一个三角形内部包含的像素少于 4 个，它绝对拟合不出合理的 3D 平面！
        if pixel_counts is not None:
            is_tiny = (pixel_counts < 4).unsqueeze(-1)
            self_weight_logits = self_weight_logits.masked_fill(is_tiny, -1e9)

        # --- 邻居门控 (Logit) ---
        self_hidden_expand = init_hidden.unsqueeze(2).expand(-1, -1, 3, -1)  # [B, N, 3, H]

        gate_input = torch.cat([
            self_hidden_expand,
            neighbor_hidden,
            edge_probs.unsqueeze(-1),
            neighbor_costs,
            plane_diff,
            feat_dist  # 图像颜色差异
        ], dim=-1)

        neighbor_weight_logits = self.gate_net(gate_input)  # [B, N, 3, 1]

        # 邻居 Logit Masking (物理断崖强行截断) todo：暂时不用开启
        # is_hard_break = (edge_probs > 0.9).unsqueeze(-1)
        # neighbor_weight_logits = neighbor_weight_logits.masked_fill(is_hard_break, -1e9)

        # ==========================================
        # 5. Masked Softmax 软传播融合 (Soft Aggregation)
        # ==========================================
        # 取指数：-1e9 会瞬间变成 0.0
        exp_self = torch.exp(self_weight_logits)  # [B, N, 1]
        exp_neighbor = torch.exp(neighbor_weight_logits)  # [B, N, 3, 1]

        # 算出总权重 (未加 1e-6 前)
        total_weight = exp_self + exp_neighbor.sum(dim=2)  # [B, N, 1]

        # 防团灭底线 (Dead-End Bypass)
        # 如果四个方向全被 Mask (-1e9)，total_weight 会接近 0
        is_dead_end = total_weight < 1e-5

        # 加上 1e-6 确保除法安全
        total_weight_safe = total_weight + 1e-6

        # 得到绝对干净的概率分布 [0, 1]
        self_weight = exp_self / total_weight_safe  # [B, N, 1]
        neighbor_weights = exp_neighbor / total_weight_safe.unsqueeze(2)  # [B, N, 3, 1]

        # 融合平面
        neighbor_planes = current_planes[batch_idx, neighbor_indices]  # [B, N, 3, 4]
        aggregated_planes = (neighbor_weights * neighbor_planes).sum(dim=2) + (self_weight * current_planes)

        # 团灭替换 (拒绝变 [0,0,0,0])
        # 如果陷入绝境，强行保持原样 (Fitter 里已经做过安全的 Fallback，保留它是最稳妥的)
        aggregated_planes = torch.where(is_dead_end, current_planes, aggregated_planes)

        # 对融合后的法向量重新归一化，防止向量长度坍塌
        agg_n_raw = aggregated_planes[..., :3]
        agg_d_raw = aggregated_planes[..., 3:]
        norm_scale = torch.norm(agg_n_raw, p=2, dim=-1, keepdim=True).clamp_min(1e-6)

        agg_n = agg_n_raw / norm_scale
        agg_d = agg_d_raw / norm_scale
        aggregated_planes_norm = torch.cat([agg_n, agg_d], dim=-1)

        # ==========================================
        # 6. 上下文感知残差精修 (Context-Aware Refinement) ym-modify-4.12
        # ==========================================
        agg_d_scaled = agg_d / self.d_scale_factor
        scaled_agg_planes = torch.cat([agg_n, agg_d_scaled], dim=-1)

        # A. 纯净编码融合后的几何平面 (剔除了已经失效的旧 Cost)
        refine_hidden = self.refine_encoder(scaled_agg_planes)  # [B, N, H]

        # B. 视觉特征 Dropout 处理
        if F_curr is not None:
            F_curr_processed = self.feat_dropout(F_curr)  # 你的核心思路落地
        else:
            F_curr_processed = torch.zeros((B, N, self.feature_dim), device=device)

        # C. 组装最强上下文记忆
        # 顺序必须严格对应 __init__ 里的 head_in_dim:
        # refine_hidden(融合几何) + init_hidden(历史记忆) + current_costs(初始评分) + F_curr(视觉特征)
        refine_input = torch.cat([
            refine_hidden,
            init_hidden,
            current_costs,
            F_curr_processed
        ], dim=-1)

        # 预测残差 $\Delta n$ 和 $\Delta d$
        delta_plane_scaled = self.plane_head(refine_input)

        delta_n = delta_plane_scaled[..., :3]
        delta_d = delta_plane_scaled[..., 3:] * self.d_scale_factor
        delta_plane = torch.cat([delta_n, delta_d], dim=-1)

        new_planes = aggregated_planes_norm + delta_plane

        # ==========================================
        # 7. 终极安全约束
        # ==========================================
        new_n_raw = new_planes[..., :3]
        new_d_raw = new_planes[..., 3:]
        final_norm_scale = torch.norm(new_n_raw, p=2, dim=-1, keepdim=True).clamp_min(1e-6)

        new_n_final = new_n_raw / final_norm_scale
        new_d_final = new_d_raw / final_norm_scale

        final_planes = torch.cat([new_n_final, new_d_final], dim=-1)

        return final_planes

    def compute_continuity_loss(self, planes, neighbor_indices, edge_probs,aligned_midpoints_norm, intrinsics,
                                H, W,depth_min,depth_max):
        """
           基于逆深度的 C0 连续性损失 (防坍缩修正版)

           Args:
               planes:                [B, N, 4]
               neighbor_indices:      [B, N, 3] — 每个三角形3条边的对面三角形ID
               edge_probs:            [B, N, 3] — EdgeHead 输出的断裂概率 (detach 后传入)
               aligned_midpoints_norm:[B, N, 3, 2] — 每条边的归一化中点坐标
               intrinsics:            [B, 3, 3]
               H, W:                  图像尺寸
           Returns:
               continuity_loss (scalar), diagnostic_dict
           """
        B, N, K, _ = aligned_midpoints_norm.shape  # K=3
        device = planes.device

        # ============================================================
        # Step 1: 像素坐标 + 齐次化
        # ============================================================
        px = (aligned_midpoints_norm[..., 0] + 1.0) / 2.0 * (W - 1)  # [B,N,3]
        py = (aligned_midpoints_norm[..., 1] + 1.0) / 2.0 * (H - 1)  # [B,N,3]
        ones = torch.ones_like(px)  # [B,N,3]

        # [B, N, 3, 3] — 最后一维是 [u, v, 1]
        uv_homo = torch.stack([px, py, ones], dim=-1)

        # ============================================================
        # Step 2: 正确反投影为射线方向
        # K_inv: [B,3,3] → [B,1,1,3,3] for broadcasting
        # ============================================================
        K_inv = torch.inverse(intrinsics)  # [B,3,3]
        K_inv_exp = K_inv[:, None, None, :, :]  # [B,1,1,3,3]
        uv_exp = uv_homo.unsqueeze(-1)  # [B,N,3,3,1]

        # K_inv @ [u,v,1]^T = ray_dir, shape: [B,N,3,3]
        rays = (K_inv_exp @ uv_exp).squeeze(-1)  # [B,N,3,3]

        # ============================================================
        # Step 3: 当前平面在中点的深度
        # planes: [B,N,4] → n:[B,N,3], d:[B,N,1]
        # ============================================================
        n_curr = planes[:, :, :3]  # [B,N,3]
        d_curr = planes[:, :, 3:]  # [B,N,1]

        # [B,N,1,3] × [B,N,3,3] → dot product → [B,N,3]
        n_curr_exp = n_curr.unsqueeze(2)  # [B,N,1,3]
        denom_curr = (n_curr_exp * rays).sum(dim=-1)  # [B,N,3]
        denom_curr = torch.where(torch.abs(denom_curr) < 1e-6, torch.tensor(1e-6, device=device), denom_curr)
        depth_curr = -d_curr / denom_curr  # [B,N,3]

        # ============================================================
        # Step 4: 邻居平面深度 (完全 detach，作为 target)
        # ============================================================
        batch_idx = torch.arange(B, device=device)[:, None, None].expand(B, N, K)
        # neighbor_planes: [B,N,3,4]
        neighbor_planes = planes[batch_idx, neighbor_indices].detach()

        n_neigh = neighbor_planes[..., :3]  # [B,N,3,3]
        d_neigh = neighbor_planes[..., 3:]  # [B,N,3,1]

        denom_neigh = (n_neigh * rays).sum(dim=-1, keepdim=True)  # [B,N,3,1]
        denom_neigh = torch.where(torch.abs(denom_neigh) < 1e-6, torch.tensor(1e-6, device=device), denom_neigh)
        depth_neigh = (-d_neigh / denom_neigh).squeeze(-1)  # [B,N,3]

        # ============================================================
        # Step 5: 深度范围保护 + 逆深度
        # ============================================================
        # depth_curr = depth_curr.clamp(min=depth_min, max=depth_max)
        # depth_neigh = depth_neigh.clamp(min=depth_min, max=depth_max)
        depth_curr = torch.abs(depth_curr)
        depth_neigh = torch.abs(depth_neigh)

        inv_curr = 1.0 / depth_curr  # [B,N,3]
        inv_neigh = 1.0 / depth_neigh  # [B,N,3]

        # ============================================================
        # Step 6: 尺度不变的相对深度误差 (Relative Depth Error)
        # 核心：彻底解决深远场景下的逆深度数值湮灭问题
        # ============================================================
        depth_diff = torch.abs(depth_curr - depth_neigh)
        depth_sum = depth_curr + depth_neigh + 1e-6

        # relative_error 范围永远在 [0, 1) 之间
        relative_error = depth_diff / depth_sum  # [B, N, 3]

        # ============================================================
        # Step 7: Edge 门控
        # 关键修正：
        #   (a) edge_probs 必须在调用前已 detach，不参与本损失的梯度
        #   (b) 用 (1 - prob) 而非 exp(-prob)，门控范围更清晰 [0,1]
        #   (c) 自环边（边界，idx==自身）权重强制为 0
        # ============================================================
        gate = (1.0 - edge_probs)  # [B,N,3], edge_probs 应在外部 detach 传入

        # 自环边不施加连续性惩罚（边界天然不连续）
        is_self_loop = (neighbor_indices == torch.arange(N, device=device)[None, :, None])
        gate = gate.masked_fill(is_self_loop, 0.0)

        # ============================================================
        # Step 8: 聚合 —— 避免空三角形导致除零
        # ============================================================
        # 赋予一个合理的 loss 权重系数（建议 10.0）
        # 这样算出来的 relative_error (一般在 0.01~0.1 级别)
        # 乘上 10 之后，能落在 0.1~1.0 之间，与 depth_loss 匹配
        continuity_scale = 10.0
        weighted_error = relative_error * gate * continuity_scale

        weight_sum = gate.sum().clamp(min=1.0)
        continuity_loss = weighted_error.sum() / weight_sum

        with torch.no_grad():
            print(f"[PLANE DIVERSITY]"
                  f" cont_loss_val={continuity_loss:.6f}"  # 法向量方差，如果接近0则全相同
                  f" rel_error_mean={relative_error.mean():.6f}"  # 偏移量方差
                  f" gate_mean={gate.mean():.6f}"
                  f" weight_sum={weight_sum:.6f}"
                  f" depth_curr_mean={depth_curr.mean():.4f}"
                  f" depth_neigh_mean={depth_neigh.mean():.4f}")

        return continuity_loss, 0.0

    def compute_smoothness_loss(self, planes, neighbor_indices, edge_probs,
                                centroids_norm, intrinsics, ref_feature,
                                H, W, sigma_F=0.5, lambda_ang=1.0):
        """
        V1.0 极简版光滑性约束 (特征双边驱动 + 纯 3D 物理几何能量)

        Args:
            planes:             [B, N, 4] 当前预测的物理平面参数 (n_x, n_y, n_z, d)
            neighbor_indices:   [B, N, 3] 每个三角形的邻居面 ID
            edge_probs:         [B, N, 3] EdgeHead 预测的断裂概率 (外部需 detach)
            centroids_norm:     [B, N, 2] 每个三角形的归一化质心坐标 [-1, 1]
            intrinsics:         [B, 3, 3] 相机内参
            ref_feature:        [B, C, H, W] 图像特征图
            H, W:               图像高宽
            sigma_F:            特征高斯核的标准差 (控制软阻断对颜色/纹理变化的敏感度)
            lambda_ang:         法向量夹角的惩罚权重 (控制面对折痕的惩罚力度)

        Returns:
            L_smooth (scalar)
        """
        B, N, _ = planes.shape
        device = planes.device

        # ============================================================
        # Step 1: 反投影计算 3D 物理质心 X_i
        # ============================================================
        # 1.1 质心转像素坐标
        u_px = (centroids_norm[..., 0] + 1.0) / 2.0 * (W - 1)  # [B, N]
        v_px = (centroids_norm[..., 1] + 1.0) / 2.0 * (H - 1)  # [B, N]
        uv_homo = torch.stack([u_px, v_px, torch.ones_like(u_px)], dim=-1)  # [B, N, 3]

        # 1.2 相机内参逆投影为射线方向
        K_inv = torch.inverse(intrinsics)  # [B, 3, 3]
        rays = torch.einsum('bij,bnj->bni', K_inv, uv_homo)  # [B, N, 3]

        # 1.3 射线与平面求交，得到 3D 坐标 X_i (深度 Z = -d / (n·ray))
        n_i = planes[..., :3]  # [B, N, 3]
        d_i = planes[..., 3:]  # [B, N, 1]

        denom = (n_i * rays).sum(dim=-1, keepdim=True)  # [B, N, 1]
        # eps = 1e-6
        # # 保证分母绝对值至少为 eps，且保留原始符号
        # denom_safe = torch.sign(denom) * torch.clamp(denom.abs(), min=eps)

        denom_safe = torch.where(denom.abs() < 1e-4, torch.full_like(denom, 1e-4), denom)
        Z_i = (-d_i / denom_safe).abs()  # [B, N, 1]

        X_i = rays * Z_i  # [B, N, 3]

        # ============================================================
        # Step 2: 收集邻居的几何信息 (X_j, n_j, d_j)
        # ============================================================
        batch_idx = torch.arange(B, device=device)[:, None, None].expand(B, N, 3)

        X_j = X_i[batch_idx, neighbor_indices]  # [B, N, 3, 3]
        n_j = n_i[batch_idx, neighbor_indices]  # [B, N, 3, 3]
        d_j = d_i[batch_idx, neighbor_indices]  # [B, N, 3, 1]

        # ============================================================
        # Step 3: 计算融合权重 W_ij = 硬阻断(EdgeHead) * 软阻断(特征亲和力)
        # ============================================================
        # 3.1 提取质心处的图像特征 F_i
        grid = centroids_norm.view(B, N, 1, 2)  # grid_sample 要求 [B, H_out, W_out, 2]
        F_i = F.grid_sample(ref_feature, grid, mode='bilinear', align_corners=True)  # [B, C, N, 1]
        F_i = F_i.squeeze(-1).permute(0, 2, 1)  # [B, N, C]

        # L2 归一化 (让平方距离等价于 2 - 2*CosineSimilarity，稳定超参)
        F_i = F.normalize(F_i, p=2, dim=-1)

        # 获取邻居特征 F_j
        F_j = F_i[batch_idx, neighbor_indices]  # [B, N, 3, C]

        # 3.2 软阻断：特征差异越大，权重越接近 0
        F_i_exp = F_i.unsqueeze(2)  # [B, N, 1, C]
        feat_dist_sq = ((F_i_exp - F_j) ** 2).sum(dim=-1)  # [B, N, 3]
        W_feat = torch.exp(-feat_dist_sq / (2.0 * sigma_F ** 2))  # [B, N, 3]

        # 3.3 硬阻断：1 - EdgeHead 断裂概率
        W_hard = (1.0 - edge_probs).clamp(min=0.0, max=1.0)  # [B, N, 3]

        # 3.4 最终亲和力权重
        W_ij = W_hard * W_feat

        # 3.5 边界屏蔽 (对于没有邻居的边，索引指向自己，将权重强置为 0)
        is_boundary = (neighbor_indices == torch.arange(N, device=device)[None, :, None])
        W_ij = W_ij.masked_fill(is_boundary, 0.0)

        # ============================================================
        # Step 4: 计算几何共面能量 E_geom(i, j)
        # ============================================================
        n_i_exp = n_i.unsqueeze(2)  # [B, N, 1, 3]
        d_i_exp = d_i.unsqueeze(2)  # [B, N, 1, 1]
        X_i_exp = X_i.unsqueeze(2)  # [B, N, 1, 3]

        # 4.1 相互“点到面”正交物理距离 (单位: 米)
        # 面 i 到 点 j 的距离: |n_i · X_j + d_i|
        dist_i_to_j = (n_i_exp * X_j).sum(dim=-1, keepdim=True) + d_i_exp  # [B, N, 3, 1]
        # 面 j 到 点 i 的距离: |n_j · X_i + d_j|
        dist_j_to_i = (n_j * X_i_exp).sum(dim=-1, keepdim=True) + d_j  # [B, N, 3, 1]

        E_dist = 0.5 * (dist_i_to_j.abs() + dist_j_to_i.abs()).squeeze(-1)  # [B, N, 3]

        # 4.2 法向夹角惩罚: 1 - cos(theta)
        cos_theta = (n_i_exp * n_j).sum(dim=-1)  # [B, N, 3]
        E_ang = 1.0 - cos_theta  # [B, N, 3]

        # 最终共面能量
        E_geom = E_dist + lambda_ang * E_ang  # [B, N, 3]

        # ============================================================
        # Step 5: 聚合加权 Loss
        # ============================================================
        weighted_energy = W_ij * E_geom
        weight_sum = W_ij.sum().clamp(min=1e-6)

        L_smooth = weighted_energy.sum() / weight_sum

        # --- 诊断打印 (观察量级，辅助调参) ---
        with torch.no_grad():
            print(f"[SMOOTH V1.0] L_smooth={L_smooth.item():.5f} | "
                  f"E_dist(m)={E_dist.mean().item():.4f} | "
                  f"E_ang={E_ang.mean().item():.4f} | "
                  f"W_feat={W_feat.mean().item():.3f} | "
                  f"W_hard={W_hard.mean().item():.3f}")

        return L_smooth

class SimilarityNet(nn.Module):
    def __init__(self, G):
        """
        Similarity Net: 将分组相关性 (Group-wise Correlation) 映射为匹配分数 (Score)

        Args:
            G: int, 分组数量 (Group number)，输入通道数
        """
        super(SimilarityNet, self).__init__()

        # Layer 1: G -> 16
        # 使用 1x1 卷积进行通道融合
        self.conv0 = nn.Conv2d(G, 16, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn0 = nn.BatchNorm2d(16)
        self.relu0 = nn.ReLU(inplace=True)

        # Layer 2: 16 -> 8
        self.conv1 = nn.Conv2d(16, 8, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(inplace=True)

        # Layer 3: 8 -> 1 (Output Score)
        # 输出 Logits，数值越大代表越相似
        self.similarity = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0, bias=False)

        # 🔥 关键：手动初始化权重
        # 确保初始状态下：输入越大 -> 输出越大 (Positive Correlation)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 1. 将卷积核权重设为小的正数 (例如 0.01)
                # 这样保证了 input * weight > 0，且保留了梯度传导能力
                nn.init.constant_(m.weight, 0.01)

                # 2. 将 Bias 设为 0
                # 防止初始 Bias 为负数导致整体分数偏移
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

            elif isinstance(m, nn.BatchNorm2d):
                # BN 层初始化为恒等映射
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        """
        Args:
            x: [B*K, G, H, W] - 分组相似度 (Group Correlation)
               通常数值范围在 -1 到 1 之间 (取决于特征归一化情况)

        Returns:
            score: [B*K, 1, H, W] - 匹配分数 (Logits)
                   数值越大表示置信度越高
        """
        # x: [B*K, G, H, W]
        out = self.conv0(x)
        out = self.bn0(out)
        out = self.relu0(out)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu1(out)

        # score: [B*K, 1, H, W]
        score = self.similarity(out)

        return score

class PlanePatchMatchModule(nn.Module):
    def __init__(self, num_hypotheses=3, G=8, feat_channels=16,propagator_iter=3):
        """
        Args:
            num_hypotheses: K (假设数量)
            G: Group Correlation 的组数 (默认8)
            feat_channels: 传入EdgeHead的通道数
        """
        super().__init__()

        self.num_hypotheses = num_hypotheses

        self.G=G

        # 核心组件
        self.warper = PlaneHomographyWarper()

        # 不需要 propa_conv, eval_conv, feature_weight_net
        # 不需要 depth_initialization (因为我们用 fitter 初始化)

        # 相似度计算网络 (类似于原版，但不需要 grid)
        # 这是一个简单的 1x1 Conv，把 G 组相关性映射为 1 个 Cost
        self.similarity_net = SimilarityNet(G=G)

        # 传播模块
        self.propagator=LearnedTrianglePropagator()

        # === 🔥 新增：EdgeHead作为内部模块 ===
        self.edge_head = EdgeHead(feat_channels)

        self.propagator_iter = propagator_iter


    def forward(self,fitter_module, depth_stage1, tri_infos, ref_feature, src_features,
                ref_proj, src_projs, intrinsics_s1,depth_min, depth_max, view_weights=None,
                neighbor_indices_batched=None,lambda_c=0.0, lambda_s=0.0):
        """
        Args:
            fitter_module: 实例化好的 DensePlaneFitter 对象
            depth_stage1: [B, 1, H/2, W/2] (Stage 1 深度)
            tri_infos: List[Dict], 包含 'tri_id_map' 和 'batch_num_tri'
            ref_feature: [B, C, H, W] (Stage 1 参考特征)
            src_features: List[[B, C, H, W]] (Stage 1 源特征列表)
            ref_proj: List[[B, 4, 4]] (参考图投影矩阵, 用于计算相对位姿)
            intrinsics_s1: 所有视图的内参矩阵
            src_projs: List[[B, 4, 4]] (源视图投影矩阵, 用于计算相对位姿)
            depth_min, depth_max: 深度范围
            view_weights: [B, N_view] (可选，视图权重)
            neighbor_indices_batched: 每个面的邻居索引 [B, N_tri, 3]

        Returns:
            depth_samples: List[[B, 1, H, W]] (这里只返回一项)
            score: [B, 1, H, W] (置信度)
            view_weights: [B, N_view] (返回传入的权重或None)
        """
        B, C, H, W = ref_feature.shape
        device = ref_feature.device
        self.fitter = fitter_module
        # 动态实例化 Visualizer 以适应当前 H, W
        visualizer = PlaneVisualizer(H, W, device)
        # ==========================================
        # 1. 数据准备 (Data Preparation)
        # ==========================================

        # 处理 tri_id_map: List[[1, H, W]] -> [B, H, W]
        # 步骤1：去掉每个Tensor中长度为1的维度（把[1, H, W]转成[H, W]）
        processed_list = [tensor.squeeze(0) for tensor in tri_infos[0]['tri_id_map']]
        # 步骤2：在第0维（batch维）堆叠，得到[B, H, W]
        tri_id_map = torch.stack(processed_list, dim=0)

        # 获取 batch 内最大的三角形数量
        max_tri_num = max(item['batch_num_tri'] for item in tri_infos)
        max_tri_num = max(max_tri_num)

        ref_intrinsics = intrinsics_s1[0]

        pixel_counts = []
        for b in range(B):
            counts = tri_infos[0]['tri_pixel_counts'][b].to(device)  # [N_b]
            pad_len = max_tri_num - counts.shape[0]
            if pad_len > 0:
                counts = F.pad(counts, (0, pad_len), value=1.0)
            pixel_counts.append(counts)

        # 把原始的像素数 Tensor
        pixel_counts_tensor = torch.stack(pixel_counts, dim=0)  # [B, N_max]

        centroids_norm = self.fitter.collate_centroids_norm(
            tri_infos[0]['centers_list'], device
        )

        # 在进入传播循环前，计算 pixel_costs
        # 将特征从计算图中剥离，保护 FeatureNet 不受 Stage 1 毒害
        # src_features_detached = [f.detach() for f in src_features]

        # ==========================================
        # 2. 拟合与生成 (Fitting & Generation)，根据stage2预测深度来拟合生成假设平面
        # ==========================================
        # 使用优化后的 get_plane_hypotheses 直接得到 [B, N, K, 4]
        # 其中 [:, :, 0, :] 是原始 SVD 拟合结果,[:, :, 1, :] 是前向平行, [:, :, 2-K:, :] 是随机扰动结果
        # todo:暂时不需要假设了，直接用拟合的结果,用了假设之后导致平面传播的一塌糊涂，很失败
        hypotheses = self.fitter.get_plane_hypotheses(
            depth_stage2=depth_stage1,
            tri_id_map=tri_id_map,
            intrinsics_s1=ref_intrinsics,
            max_num_triangles=max_tri_num
        )  # Output: [B, N_tri, 1, 4]

        # [B, N_tri, K, 4] -> [B, N_tri, 4] (取第0个假设,最佳平面的前3通道)
        # 进行一个可视化看看效果，拟合的初始平面
        # before_best_guess_planes = hypotheses[:, :, 0, :]  # [B, N_tri, 4]

        # 直接取出唯一的平面作为基底 (不需要 argmin)
        current_planes = hypotheses.squeeze(2)  # [B, N_tri, 4]

        # ==========================================
        # 3. 计算 SVD 基底的物理代价 (作为特征)
        # ==========================================
        pixel_hypotheses = self.map_tri_to_pixel(hypotheses, tri_id_map, H, W)

        # ✅ 注意这里直接传入 ref_feature，绝不 detach！梯度畅通无阻！
        pixel_costs = self.compute_costs(
            ref_feature, src_features, ref_proj, src_projs,
            pixel_hypotheses, view_weights=view_weights, ref_intrinsic=ref_intrinsics
        )
        # 聚合为三角形代价 [B, N_tri, 1]
        current_costs = self.aggregate_costs_per_triangle(pixel_costs, tri_id_map, max_tri_num)

        # ==========================================
        # 4. 预测物理断裂边 (EdgeHead)
        # ==========================================
        # 运行 EdgeHead 预测边缘  ym-need-modify 暂时不给其放梯度，
        edge_alphas = self.edge_head(
            feat=ref_feature.detach(),  # [B, C, H, W]
            tri_infos=tri_infos,
            tri_planes=current_planes.detach(), # [B, N_max, 4] 包含法向和距离
            intrinsics=ref_intrinsics,  # [B, 3, 3] 相机内参
            dense_depth=depth_stage1.detach() # dense深度
        )

        # 转换为三角形级别格式 [B, N_max, 3]
        edge_probs_tensor, aligned_midpoints_norm = convert_edge_features_to_tri_format(
            edge_alphas, tri_infos, max_tri_num, device
        )

        # 渲染没有经过传播的 深度图和法向量图 后续用
        no_prop_depth, no_propa_normal = visualizer.render_from_planes(
            current_planes.detach() ,  # 注意 detach，不传导梯度
            tri_id_map,
            ref_intrinsics,
            depth_range=(depth_min, depth_max)
        )

        # ==========================================
        # 6. 端到端神经融合传播 (Neural Soft Propagation)
        # ==========================================

        for iter_idx in range(self.propagator_iter):
            # 5.1 传播：MLP 综合平面、代价、特征、边缘，输出平滑后的新平面
            new_planes = self.propagator(
                current_planes=current_planes,
                current_costs=current_costs.detach(),  # 物理代价化身为特征引导 MLP
                neighbor_indices=neighbor_indices_batched,
                edge_probs=edge_probs_tensor.detach(),  # 阻断边缘头干扰
                ref_feature=ref_feature.detach(),  # 图像特征引导 MLP
                centroids_norm=centroids_norm,
                pixel_counts=pixel_counts_tensor # 每个三角形的个数
            )

            # 对传播进行一个保护
            new_planes = fitter_module.enforce_depth_hard_constraint(
                planes=new_planes,
                centroids_norm=centroids_norm,
                intrinsics=ref_intrinsics,
                depth_min=depth_min,
                depth_max=depth_max,
                H=H, W=W
            )

            # 5.2 重新评估新平面的代价 (极其重要：在此处设立绝对的梯度防火墙！)
            # 扩展维度以适配 compute_costs 接口: [B, N, 1, 4] (K=1)
            new_planes_k1 = new_planes.unsqueeze(2)

            # 将三角形平面广播到像素级
            pixel_hypo = self.map_tri_to_pixel(new_planes_k1, tri_id_map, H, W)

            # 计算像素代价
            pixel_costs_new = self.compute_costs(
                ref_feature, src_features, ref_proj, src_projs,
                pixel_hypo,
                view_weights=view_weights,
                ref_intrinsic=ref_intrinsics,
                is_debug=False
            )

            # 重新聚合成三角形级代价 -> [B, N, 1]
            current_costs = self.aggregate_costs_per_triangle(pixel_costs_new, tri_id_map, max_tri_num)

            # 5.3 状态更新，进入下一次迭代
            # 注意：因为是 Learned Propagator，我们直接相信它的更新（像 RNN 一样），而不进行 Argmin 判断
            current_planes = new_planes

        final_planes = current_planes

        # ==========================================
        # 7. 渲染与输出和计算损失
        # ==========================================
        continuity_loss, continuity_s_loss = 0.0, 0.0
        if lambda_c > 0.0:
            continuity_loss, continuity_s_loss = self.propagator.compute_continuity_loss(
                planes=final_planes,
                neighbor_indices=neighbor_indices_batched,
                edge_probs=edge_probs_tensor.detach(),  # 不让其受影响
                aligned_midpoints_norm=aligned_midpoints_norm,
                intrinsics=ref_intrinsics,
                H=H, W=W,
                depth_min=min(depth_min), depth_max=max(depth_max))

        # 计算光滑性损失
        smoothness_loss = 0.0
        if lambda_s > 0.0:
            smoothness_loss = self.propagator.compute_smoothness_loss(
                planes=final_planes,
                neighbor_indices=neighbor_indices_batched,
                edge_probs=edge_probs_tensor.detach(),  # 必须 detach
                centroids_norm=centroids_norm,  # 传入归一化质心 [B, N, 2]
                intrinsics=ref_intrinsics,  # 相机内参 [B, 3, 3]
                ref_feature=ref_feature.detach(),  # 原生特征图 [B, C, H, W]
                H=H, W=W,
                sigma_F=0.6,  # 可调：0.5 是 L2 归一化特征推荐值
                lambda_ang=4.0  # 可调：法向平滑的相对强度
            )

        # 准备 Mask (如果有 tri_id_map，可以生成 invalid_mask，没有则传 None)
        invalid_mask = (tri_id_map < 0)

        # 调用新函数直接渲染 ✅
        # depth_sample, normal_vis = visualizer.render_from_pixel_wise_planes(
        #     best_planes,
        #     ref_intrinsics,  # 这里假设 ref_proj 就是 intrinsics (如果是 [B,4,4] 需取 [:3,:3])
        #     (depth_min, depth_max),
        #     invalid_mask=invalid_mask
        # )

        # 将最终平面分别转化为深度图和法向量图
        # 将B,N,4 分别转化为,B,H,W,1 和B,H,W,3 可视化用
        final_depth, final_normal = visualizer.render_from_planes(
            final_planes,
            tri_id_map,
            ref_intrinsics,
            depth_range=(depth_min, depth_max))

        # 组装输出 (PatchMatchNet 通常需要 List 格式)
        # todo:一个是刚拟合完毕的，另外一个是经过传播处理的
        depth_samples = [no_prop_depth.detach(),final_depth]  # List[[B, 2, H, W]]
        # 法向量
        normal_samples= [no_propa_normal.detach(),final_normal]


        return (depth_samples, pixel_costs,
                view_weights,
                normal_samples,
                final_planes,
                edge_alphas,
                continuity_loss,smoothness_loss)

    # ==========================================
    # 辅助函数 (Placeholders)
    # ==========================================

    def map_tri_to_pixel(self, hypotheses, tri_id_map, H, W):
        """
        将三角形级假设广播到像素级
        Args:
            hypotheses: [B, N_tri, K, 4]
            tri_id_map: [B, H, W]
        Returns:
            pixel_hypotheses: [B, H, W, K, 4]
        """
        B, N_tri, K, C = hypotheses.shape
        device = hypotheses.device

        # 1. 安全处理无效 ID (-1)
        # 将 -1 (无效区域) 临时映射到 0，防止索引越界报错
        # 之后可以用 mask 再次处理，或者直接让它取第 0 个三角形的平面（通常无伤大雅，因为无效区域后续不参与计算）
        invalid_mask = (tri_id_map < 0)
        safe_id_map = tri_id_map.clone()
        safe_id_map[invalid_mask] = 0

        # 确保是 Long 类型
        safe_id_map = safe_id_map.long()

        # 2. 计算全局索引 (Global Indices)
        # 因为 hypotheses 是 [B, N, ...]，直接索引需要区分 batch
        # 我们构造一个偏移量: batch_idx * N_tri
        batch_offset = torch.arange(B, device=device) * N_tri
        batch_offset = batch_offset.view(B, 1, 1)  # [B, 1, 1] 用于广播

        # global_ids: 每个像素在 flattened hypotheses 中的绝对索引
        # shape: [B, H, W] -> view -> [B*H*W]
        global_ids = (safe_id_map + batch_offset).view(-1)

        # 3. 展平假设池以供索引
        # [B, N_tri, K, 4] -> [B * N_tri, K, 4]
        flat_hypotheses = hypotheses.view(-1, K, C)

        # 4. 查表 (Gather / Advanced Indexing)
        # [B*H*W, K, 4]
        pixel_hypotheses_flat = flat_hypotheses[global_ids]

        # 5. 恢复形状
        # [B, H, W, K, 4]
        pixel_hypotheses = pixel_hypotheses_flat.view(B, H, W, K, C)

        # (可选) 6. 处理无效区域
        # 如果你希望无效区域的平面参数是全0或者特定值，可以在这里处理
        # 例如: 将无效像素的假设全部置为 0
        # if invalid_mask.any():
        #     mask_expand = invalid_mask.unsqueeze(-1).unsqueeze(-1).expand_as(pixel_hypotheses)
        #     pixel_hypotheses[mask_expand] = 0.0

        return pixel_hypotheses

    def compute_costs(self, ref_feature, src_features, ref_proj, src_projs, current_hypotheses, view_weights,
                      ref_intrinsic,is_debug=True):
        """
        计算代价体积 (Cost Volume)
        Args:
            ref_feature: [B, C, H, W]
            src_features: List of [B, C, H, W]
            ref_proj: [B, 4, 4] (Ref 投影矩阵 P)
            src_projs: List of [B, 4, 4] (Src 投影矩阵 P)
            current_hypotheses: [B, H, W, K, 4] (像素级平面假设)
            view_weights: [B, Nview-1, H, W] (可选)
            ref_intrinsic: [B, 3, 3] (参考图内参 K, 必须提供以计算 Homography)
        Returns:
            costs: [B, H, W, K]
        """
        B, H, W, K, _ = current_hypotheses.shape
        C = ref_feature.shape[1]
        device = ref_feature.device

        # 1. 预处理平面参数
        # [B, H, W, K, 4] -> [B, K, 4, H, W] -> [B*K, 4, H, W]
        # 这样我们可以利用 Batch 并行计算所有假设的 Homography
        plane_params = current_hypotheses.permute(0, 3, 4, 1, 2).reshape(B * K, 4, H, W)

        # 2. 准备 Ref 特征 (Group 分组)
        # [B, C, H, W] -> [B, 1, G, C/G, H, W] -> Repeat K -> [B*K, G, C/G, H, W]
        ref_feat_grouped = ref_feature.view(B, self.G, C // self.G, H, W)
        ref_feat_expanded = ref_feat_grouped.unsqueeze(1).repeat(1, K, 1, 1, 1, 1).view(B * K, self.G, C // self.G, H,
                                                                                        W)

        # 3. 准备内参 K
        # 我们利用 get_homography 的数学特性：
        # H = K_src * (R - t * n^T / d) * K_ref_inv
        # 这里我们传入 K_src=Identity, K_ref=Real_K, R_rel=rot*K, t_rel=trans
        # 结果 H = I * (rot*K - trans * n^T / d) * K_inv = rot - trans * n^T * K_inv / d
        # 这正是我们需要的 Homography (因为 rot/trans 已经是投影空间的相对变换了)

        K_ref_expand = ref_intrinsic.unsqueeze(1).repeat(1, K, 1, 1).view(B * K, 3, 3)
        # 单位矩阵
        K_src_identity = torch.eye(3, device=device).view(1, 3, 3).expand(B * K, -1, -1)

        total_cost = 0

        # 🔥 修复 1：准备累加特征，而不是累加分数
        similarity_sum = 0.0
        weight_sum = 0.0

        # 遍历所有源视图
        for i, (src_feat, src_proj) in enumerate(zip(src_features, src_projs)):
            # --- A. 计算相对变换 (Projective Space) ---
            # M = P_src * P_ref^-1
            # rot (3x3) 对应 K' R K^-1
            # trans (3x1) 对应 K' t
            with torch.no_grad():
                proj_rel = torch.matmul(src_proj, torch.inverse(ref_proj))
                rot = proj_rel[:, :3, :3]  # [B, 3, 3]
                trans = proj_rel[:, :3, 3:4]  # [B, 3, 1]

                # 扩展到 [B*K, ...]
                rot_expand = rot.unsqueeze(1).repeat(1, K, 1, 1).view(B * K, 3, 3)
                trans_expand = trans.unsqueeze(1).repeat(1, K, 1, 1).view(B * K, 3, 1)

                # 构造传递给 warper 的 R_rel
                # Trick: 传入 rot @ K_ref，配合 get_homography 内部的 * K_ref_inv，正好抵消得到 rot
                R_rel_input = torch.matmul(rot_expand, K_ref_expand)

            # --- B. 计算 Homography ---
            # H: [B*K, H, W, 3, 3]
            H_mats = self.warper.get_homography(
                plane_params=plane_params,
                K_ref=K_ref_expand,
                K_src=K_src_identity,  # 这里设为 Identity
                R_rel=R_rel_input,
                t_rel=trans_expand
            )

            # --- C. 特征扭曲 (Warping) --- ✅
            # src_feat: [B, C, H, W] -> [B*K, C, H, W]
            src_feat_expand = src_feat.unsqueeze(1).repeat(1, K, 1, 1, 1).view(B * K, C, H, W)
            warped_src = self.warper.warp_feature(src_feat_expand, H_mats)

            del src_feat_expand, H_mats # 释放显存

            # --- D. 分组相关性 (Group Correlation) ---
            # [B*K, G, C/G, H, W]
            warped_src_grouped = warped_src.view(B * K, self.G, C // self.G, H, W)

            # 强行将特征向量的长度缩放为 1，将点积转化为余弦相似度 (Cosine Similarity)
            # 这样 similarity 的物理边界被死死锁在 [-1, 1] 之间，网络绝无作弊可能！
            warped_src_norm = F.normalize(warped_src_grouped, p=2, dim=2)
            ref_feat_norm = F.normalize(ref_feat_expanded, p=2, dim=2)

            # Similarity: [B*K, G, H, W]
            similarity = (warped_src_norm * ref_feat_norm).mean(dim=2)

            del warped_src, warped_src_grouped, warped_src_norm, ref_feat_norm  # 释放显存

            # =========================================================
            # 获取视图权重，并进行加权累加 (早期融合)
            # =========================================================
            if view_weights is not None:
                # vw: [B, 1, H, W] -> 扩展到 [B*K, 1, H, W]
                vw = view_weights[:, i:i + 1, :, :]
                vw_expand = vw.unsqueeze(1).repeat(1, K, 1, 1, 1).view(B * K, 1, H, W)
            else:
                # 如果没有 view_weights，默认为 1
                vw_expand = torch.ones((B * K, 1, H, W), device=device)

            # 累加 Similarity 特征和权重
            similarity_sum = similarity_sum + (similarity * vw_expand)
            weight_sum = weight_sum + vw_expand


        # =========================================================
        # 权重归一化 (防止 Softmax 尺度爆炸)
        # =========================================================
        # 除以总权重，得到平均相似度特征
        # 加 1e-6 防止完全被遮挡的像素产生除零错误
        similarity_fused = similarity_sum / (weight_sum + 1e-6)

        # [B*K, 1, H, W]
        # todo：暂时不用学习型，先用直接型
        # score_fused = self.similarity_net(similarity_fused)
        score_fused = similarity_fused.mean(dim=1, keepdim=True)

        # 转换为 Cost (越小越好)
        cost_fused = -score_fused

        # 还原形状 [B*K, 1, H, W] -> [B, K, H, W] -> [B, H, W, K]
        total_cost = cost_fused.view(B, K, H, W).permute(0, 2, 3, 1)

        return total_cost

    def aggregate_costs_per_triangle(self, pixel_costs, tri_id_map, max_num_tri):
        """
        将像素级的代价聚合为三角形级的代价 (Triangle-wise Aggregation)
        Args:
            pixel_costs: [B, H, W, K] (每个像素对 K 个假设的代价)
            tri_id_map: [B, H, W] (每个像素属于哪个三角形，-1 表示无效)
            max_num_tri: int (最大三角形数量)
        Returns:
            tri_costs: [B, N_tri, K] (每个三角形对 K 个假设的平均代价)
        """
        B, H, W, K = pixel_costs.shape
        device = pixel_costs.device

        # 1. 展平数据
        # pixel_costs 来自 permute，内存不连续，必须用 reshape
        flat_costs = pixel_costs.reshape(B, -1, K) # [B, Pixels, K]
        flat_ids = tri_id_map.reshape(B, -1)  # [B, Pixels]

        # 2. 生成有效 Mask
        valid_mask = (flat_ids >= 0)

        # ================ 安全检查与修正 (Fix Index Out of Bounds) ym-modify 2.13

        # 过滤掉超过 max_num_tri 的 ID (防止爆显存/越界)
        # 如果 tri_id >= max_num_tri，scatter 会直接报错
        inside_range_mask = (flat_ids < max_num_tri)

        # 合并 Mask
        valid_mask = valid_mask & inside_range_mask

        # (可选) 打印 Debug 信息，确认是否发生越界
        max_id = flat_ids.max().item()
        if max_id >= max_num_tri:
            print(f"[Warning] Max Tri ID {max_id} exceeds limit {max_num_tri}! Clipping...")

        # 3. 准备 Scatter 用的全局索引 (Offset Trick)
        # 将 Batch 维度折叠进三角形 ID：Global_ID = Batch_ID * Max_Tri + Local_Tri_ID
        batch_offset = (torch.arange(B, device=device) * max_num_tri).view(B, 1)

        # 为了计算索引安全，先把无效的 -1 变成 0 (反正后面会被 mask 过滤掉)
        safe_ids = flat_ids.clone()
        safe_ids[~valid_mask] = 0

        # 计算全局唯一索引 [B, Pixels] -> [Total_Pixels]
        global_ids = (safe_ids + batch_offset).view(-1)

        # 4. 提取有效数据 (Filtering)
        # 只保留 mask 为 True 的像素数据，减少计算量
        flat_mask = valid_mask.reshape(-1)

        valid_global_ids = global_ids[flat_mask]  # [Valid_Pixels]
        flat_costs_all = flat_costs.reshape(-1, K)
        valid_costs = flat_costs_all[flat_mask]  # [Valid_Pixels, K]

        # 5. 聚合 (Scatter Add)
        total_bins = B * max_num_tri

        # 准备输出容器 (扁平化)
        flat_tri_sum = torch.zeros(total_bins, K, device=device)
        flat_tri_counts = torch.zeros(total_bins, 1, device=device)

        # 扩展索引以匹配 K 维度: [N] -> [N, K]
        idx_expand = valid_global_ids.unsqueeze(1).expand(-1, K)

        # 累加 Cost
        flat_tri_sum.scatter_add_(0, idx_expand, valid_costs)

        # 累加计数 (Count)
        # 只需要对一列进行计数
        flat_tri_counts.scatter_add_(0, valid_global_ids.unsqueeze(1),
                                     torch.ones_like(valid_global_ids.unsqueeze(1), dtype=torch.float32))

        # 6. 求平均与异常处理 (关键补全!)

        # 找出有效的三角形 (Count > 0)
        has_pixels = (flat_tri_counts > 0)  # [Total_Bins, 1]

        # 计算平均 Cost
        # 加上 1e-6 防止除零，但对于 count=0 的情况，结果依然接近 0
        flat_tri_mean = flat_tri_sum / (flat_tri_counts + 1e-6)

        # 🔥 补全逻辑：将没有像素覆盖的三角形 Cost 设为最大值
        # 这样 Argmin 就绝对不会选中它们
        # 使用 100.0 或者更大的数，取决于你的 Cost 范围 (通常 Cost < 1.0)
        max_cost_value = 1

        # 如果 has_pixels 为 False，则赋值为 max_cost_value
        # 这一步非常关键，否则空三角形 Cost=0 会导致错误的“完美匹配”
        flat_tri_mean = torch.where(has_pixels, flat_tri_mean, torch.tensor(max_cost_value, device=device))

        # 7. 恢复形状
        tri_costs = flat_tri_mean.view(B, max_num_tri, K)

        return tri_costs

class DensePlaneFitter(nn.Module)   :
    def __init__(self, height_s1, width_s1, device, num_hypotheses=3, perturbation_range=0.05, depth_max=None,depth_min=None):
        """
        拟合平面，并根据拟合平面生成假设
        Args:
            height_s1, width_s1: Stage 1 的分辨率
            num_hypotheses: K (建议 3~5)
            perturbation_range: 扰动噪声范围

        """
        super().__init__()
        if depth_max is None:
            depth_max = []
        self.H = height_s1
        self.W = width_s1
        self.device = device
        self.K = max(2, num_hypotheses)  # 保证至少有 Best + FP
        self.noise_scale = perturbation_range

        # 这是对特别小的三角形进行的一个保底处理
        self.depth_max = depth_max
        self.depth_min = depth_min

        # 1. 生成 Stage 1 的像素坐标网格 (u, v)
        y_range = torch.arange(0, self.H, dtype=torch.float32, device=device)
        x_range = torch.arange(0, self.W, dtype=torch.float32, device=device)
        self.grid_y, self.grid_x = torch.meshgrid(y_range, x_range,indexing='ij')  # [H, W]

        # 2. 生成用于 grid_sample 的归一化坐标 (针对 Stage 2)
        # 归一化到 [-1, 1]
        self.norm_u = (self.grid_x / (self.W - 1)) * 2.0 - 1.0
        self.norm_v = (self.grid_y / (self.H - 1)) * 2.0 - 1.0
        self.sampling_grid = torch.stack((self.norm_u, self.norm_v), dim=-1)  # [H, W, 2]


    def By_SVD_Plane(self, depth_stage2, tri_id_map, intrinsics_s1, max_num_triangles):
        """
        Args:
            depth_stage2: [B, 1, H_s2, W_s2] (Stage 2 深度图)
            tri_id_map:   [B, H_s1, W_s1] (Stage 1 三角形索引图，值域 0~N-1, -1为无效)，每个位置存的是“它是第几个三角形”
            intrinsics_s1:[B, 3, 3] (Stage 1 内参)
            max_num_triangles: int (最大三角形数，用于分配 Tensor 大小)

        Returns:
            plane_params: [B, N_tri, 4] (nx, ny, nz, d)
        """
        B = depth_stage2.shape[0]
        device = depth_stage2.device
        # ==========================================
        # Step 1: 投影与采样 (Project & Sample)
        # ==========================================

        # 扩展 grid 以匹配 Batch
        # grid: [B, H, W, 2]
        grid_batch = self.sampling_grid.unsqueeze(0).expand(B, -1, -1, -1)

        # 使用 grid_sample 从 Stage 2 采样深度
        # 这一步实现了 "将 Stage 1 坐标投影到 Stage 2 并获取深度"
        # align_corners=True 对应我们上面的 -1~1 归一化逻辑
        sampled_depth = F.grid_sample(
            depth_stage2,
            grid_batch,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )  # [B, 1, H_s1, W_s1]

        # 检查是否全部无效
        if sampled_depth.max() < 1e-4:
            return self._get_fallback_planes(B, max_num_triangles, device)

        # ==========================================
        # Step 2: 反投影生成点云 (Back-Projection)，将深度图(2.5D) 变成 点云(3D)
        # ==========================================

        # 展平数据以便于 Scatter 操作
        depth_flat = sampled_depth.view(B, -1)  # [B, N_pix]
        u_flat = self.grid_x.reshape(1, -1).expand(B, -1)
        v_flat = self.grid_y.reshape(1, -1).expand(B, -1)

        fx = intrinsics_s1[:, 0, 0].unsqueeze(1)  # [B, 1]
        fy = intrinsics_s1[:, 1, 1].unsqueeze(1)
        cx = intrinsics_s1[:, 0, 2].unsqueeze(1)
        cy = intrinsics_s1[:, 1, 2].unsqueeze(1)

        # 计算 3D 坐标,通过得到一个Z=1时像素投影点的光线方向，乘上深度，得到其三维坐标
        # Ux_normal*d
        X = (u_flat - cx) * depth_flat / fx
        Y = (v_flat - cy) * depth_flat / fy
        Z = depth_flat

        # points_flat: [B*N_pix, 3]
        # 把 B 个样本的所有像素拼在一起，变成一个巨大的点云列表
        points_flat = torch.stack([X, Y, Z], dim=2).view(-1, 3)

        # ==========================================
        # Step 3: 并行 SVD 拟合 (Scatter SVD)
        # ==========================================

        # 处理 Batch ID 偏移，将所有 Batch 的三角形拉通处理
        # tri_id_map: [B, H, W] -> [B*N_pix]
        tri_ids_flat = tri_id_map.view(-1)

        batch_ids = torch.arange(B, device=self.device).unsqueeze(1).expand(-1, self.H * self.W).reshape(-1)

        # 全局唯一三角形id,格式是像素图展平,每个每个像素点代表着三角形的编号 ID: batch_idx * max_tri + local_tri_id
        global_tri_ids = batch_ids * max_num_triangles + tri_ids_flat

        # 过滤无效像素 (tri_id == -1 或 深度要求＞0)
        valid_mask = (tri_ids_flat >= 0) & (points_flat[:, 2] > 1e-4)

        if valid_mask.sum() == 0:
            return self._get_fallback_planes(B, max_num_triangles, device)

        # 保留有效点云和有效每个三角形包含像素的编号 是基于像素图
        valid_points = points_flat[valid_mask]
        valid_ids = global_tri_ids[valid_mask]
        # 总共有多少个三角形，基于三角形的
        total_bins = B * max_num_triangles

        # === 修复 1: 强制使用 float64 (Double) 进行累加 ===
        # float32 在计算平方和时极其容易溢出或丢失精度
        valid_points_64 = valid_points.double()

        # 确保 indices 不越界 (这是 scatter 崩溃的常见原因)
        if valid_ids.max() >= total_bins:
            print(f"❌ [Error] global_tri_ids Out of Bounds!")
            return self._get_fallback_planes(B, max_num_triangles, device)

        # --- 3.1 统计一阶矩 (Sum P) 和 计数 (Count) ---
        # scatter_add_的作用：对于 valid_ids 中的每个id，把ones中对应位置的值加到counts[id]上。
        # 效果等同于：if counts[id]==id,counts[id] += 1
        ones = torch.ones_like(valid_ids, dtype=torch.float64)
        counts = torch.zeros(total_bins, device=self.device,dtype=torch.float64)
        # --- 统计量 1: Count (每个三角形有多少个点 N) ---
        counts.scatter_add_(0, valid_ids, ones)

        sum_P = torch.zeros(total_bins, 3, device=self.device,dtype=torch.float64)
        # --- 统计量 2: Sum P (坐标和)
        # sum_P：一排空邮箱（每个邮箱代表一个三角形）。
        # valid_points：一堆信件（每个信件里写着 XYZ 坐标）。
        # valid_ids：信封上写的地址（ID 几就投进几号邮箱）。
        sum_P.index_add_(0, valid_ids, valid_points_64)

        # --- 3.2 统计二阶矩 (Sum PP^T) ---
        # 协方差矩阵是 3x3 对称矩阵：
        # [ xx  xy  xz ]
        # [ xy  yy  yz ]
        # [ xz  yz  zz ]
        # 我们只需要算上三角的 6 个分量：xx, xy, xz, yy, yz, zz
        pt_x, pt_y, pt_z = valid_points_64[:, 0], valid_points_64[:, 1], valid_points_64[:, 2]

        sum_XX = torch.zeros(total_bins, device=self.device,dtype=torch.float64).scatter_add_(0, valid_ids, pt_x * pt_x)
        sum_XY = torch.zeros(total_bins, device=self.device,dtype=torch.float64).scatter_add_(0, valid_ids, pt_x * pt_y)
        sum_XZ = torch.zeros(total_bins, device=self.device,dtype=torch.float64).scatter_add_(0, valid_ids, pt_x * pt_z)
        sum_YY = torch.zeros(total_bins, device=self.device,dtype=torch.float64).scatter_add_(0, valid_ids, pt_y * pt_y)
        sum_YZ = torch.zeros(total_bins, device=self.device,dtype=torch.float64).scatter_add_(0, valid_ids, pt_y * pt_z)
        sum_ZZ = torch.zeros(total_bins, device=self.device,dtype=torch.float64).scatter_add_(0, valid_ids, pt_z * pt_z)

        # --- 3.3 构建协方差矩阵 ---
        # 防止除零
        safe_counts = counts.clamp(min=3)  # 至少3个点才能拟合

        # 质心 centroid = sum_P / N
        centroids = sum_P / safe_counts.unsqueeze(1)

        # 组合 sum_PP^T 矩阵 [Total, 3, 3]协方差矩阵
        sum_PPt = torch.stack([
            sum_XX, sum_XY, sum_XZ,
            sum_XY, sum_YY, sum_YZ,
            sum_XZ, sum_YZ, sum_ZZ
        ], dim=1).reshape(total_bins, 3, 3)

        # Covariance= sum((p-mu)(p-mu)^T) = sum_PPt - N * 质心 * 质心^T
        # 这里计算未归一化的 Scatter Matrix 即可，特征向量方向一样
        center_correction = torch.bmm(sum_P.unsqueeze(2), sum_P.unsqueeze(1)) / safe_counts.view(-1, 1, 1)
        covariance = sum_PPt - center_correction

        # === 修复 2: 防止协方差矩阵出现 NaN (例如 Inf-Inf 导致) ===
        # 即使前面 clamp 了深度，数值计算仍可能产生极其微小的负数(应该为0)或 NaN
        covariance = torch.nan_to_num(covariance, nan=0.0)

        # --- 3.4 特征值分解求解法向量 ---
        # 求解 Cov * n = lambda * n
        # eigh 适用于实对称矩阵，速度快且稳定
        try:
            vals, vecs = torch.linalg.eigh(covariance)

            # 取最小特征值对应的特征向量
            normals = vecs[:, :, 0].float()  # [Total, 3]
            centroids = centroids.float()


        except RuntimeError as e:
            print(f"⚠️ SVD Failed: {e}")
            normals = torch.zeros(total_bins, 3, device=self.device, dtype=torch.float32)
            normals[:, 2] = 1.0
            centroids = centroids.float()

        # ==========================================================
        # 强制法向量指向相机 (Orient Normals)
        # ==========================================================

        # 方法：计算法向量与“相机-质心”视线的点积
        # 在相机坐标系下，相机原点是 (0,0,0)，所以视线向量就是 centroids 本身
        # 我们希望 normal 指向相机，即 normal 与 centroids 的夹角 > 90度 (点积 < 0)
        # 或者 normal 指向外部 (背离相机)，即 normal 与 centroids 的夹角 < 90度 (点积 > 0)

        # 通常 MVS 定义法向量指向物体外部（即指向相机）。
        # 所以我们需要 dot(n, v) < 0。如果 dot > 0，就翻转 n。

        # 1. 计算点积 (Batch Dot Product)
        # raw_normals: [N, 3], centroids: [N, 3]
        dot_product = torch.sum(normals * centroids, dim=1, keepdim=True)  # [N, 1]

        # 2. 判断方向并翻转
        # 如果点积 > 0 (同向)，说明法向量指向了屏幕里面（背离相机），需要翻转
        # sign: 正数变 -1 (翻转)，负数变 1 (保持)
        flip_mask = -torch.sign(dot_product)

        # 处理 0 的情况 (极少见，垂直于视线)
        flip_mask[flip_mask == 0] = 1.0

        # 3. 应用翻转
        normals = normals * flip_mask


        # ==========================================================
        # 深度计算修正
        # ==========================================================
        # 必须使用翻转后的 normals 来计算 d
        # n * x + d = 0  =>  d = - n * centroid
        d_vals = -torch.sum(normals * centroids, dim=1, keepdim=True)

        # --- 3.6 异常处理 ---
        # 对于点数不足的三角形
        invalid_tris = (counts < 3)

        if invalid_tris.any():
            # 1. 法向量设为指向相机 (0, 0, -1) 或者 (0, 0, 1) 取决于你的坐标系定义
            # 假设: n=[0,0,-1], d=depth_max -> -1*z + d = 0 -> z = d
            normals[invalid_tris] = torch.tensor([0.0, 0.0, -1.0], device=self.device)

            # 2. d 值直接设为最大深度 (将平面推到最远背景处)
            # 这样它的 Cost 会非常高，第一轮传播就会被邻居覆盖掉
            d_vals[invalid_tris, 0] = min(self.depth_max)


        # 组合结果
        plane_params = torch.cat([normals, d_vals], dim=1)

        return plane_params.view(B, max_num_triangles, 4)

    def By_SVD_Plane_grad(self, depth_stage2, tri_id_map, intrinsics_s1, max_num_triangles):
        """
        可微 SVD 拟合 + 假设生成
        Args:
            depth_stage2: [B, 1, H_s2, W_s2] (Stage 2 深度图)
            tri_id_map:   [B, H_s1, W_s1] (Stage 1 三角形索引图，值域 0~N-1, -1为无效)，每个位置存的是“它是第几个三角形”
            intrinsics_s1:[B, 3, 3] (Stage 1 内参)
            max_num_triangles: int (最大三角形数，用于分配 Tensor 大小)

        Returns:
            hypotheses: [B, N_tri, K, 4] (nx, ny, nz, d)
        Modified: 使用 torch.svd_lowrank 近似求解平面法向量。
        原理: 提取协方差矩阵的前两个主成分 (q=2)，计算其叉积得到法向量。

        Returns:

        """
        B = depth_stage2.shape[0]
        device = depth_stage2.device

        # ==========================================
        # Step 1: 投影与采样 (保持不变)
        # ==========================================
        grid_batch = self.sampling_grid.unsqueeze(0).expand(B, -1, -1, -1)
        sampled_depth = F.grid_sample(
            depth_stage2, grid_batch, mode='bilinear', padding_mode='border', align_corners=True
        )

        if sampled_depth.max() < 1e-4:
            return self._get_fallback_planes(B, max_num_triangles, device)

        # ==========================================
        # Step 2: 反投影生成点云 (保持不变)
        # ==========================================
        depth_flat = sampled_depth.view(B, -1)
        u_flat = self.grid_x.reshape(1, -1).expand(B, -1)
        v_flat = self.grid_y.reshape(1, -1).expand(B, -1)

        fx = intrinsics_s1[:, 0, 0].unsqueeze(1)
        fy = intrinsics_s1[:, 1, 1].unsqueeze(1)
        cx = intrinsics_s1[:, 0, 2].unsqueeze(1)
        cy = intrinsics_s1[:, 1, 2].unsqueeze(1)

        X = (u_flat - cx) * depth_flat / fx
        Y = (v_flat - cy) * depth_flat / fy
        Z = depth_flat

        points_flat = torch.stack([X, Y, Z], dim=2).view(-1, 3)

        # ==========================================
        # Step 3: 并行 SVD_LowRank 拟合
        # ==========================================
        tri_ids_flat = tri_id_map.view(-1)
        batch_ids = torch.arange(B, device=self.device).unsqueeze(1).expand(-1, self.H * self.W).reshape(-1)
        global_tri_ids = batch_ids * max_num_triangles + tri_ids_flat

        # 过滤无效像素
        valid_mask = (tri_ids_flat >= 0) & (points_flat[:, 2] > 1e-4)

        if valid_mask.sum() == 0:
            return self._get_fallback_planes(B, max_num_triangles, device)

        valid_points = points_flat[valid_mask]
        valid_ids = global_tri_ids[valid_mask]
        total_bins = B * max_num_triangles

        # 强制使用 float64 (Double)
        valid_points_64 = valid_points.double()
        pt_x, pt_y, pt_z = valid_points_64[:, 0], valid_points_64[:, 1], valid_points_64[:, 2]

        if valid_ids.max() >= total_bins:
            valid_ids = torch.clamp(valid_ids, max=total_bins - 1)

        # ----------------------------------------------------------------
        # 3.1 统计协方差矩阵元素
        # ----------------------------------------------------------------
        # 基础统计量
        sum_P = torch.zeros(total_bins, 3, device=self.device, dtype=torch.float64)
        sum_P.index_add_(0, valid_ids, valid_points_64)  # Sum P

        # 二阶统计量
        sum_xx = torch.zeros(total_bins, device=self.device, dtype=torch.float64).scatter_add_(0, valid_ids, pt_x ** 2)
        sum_xy = torch.zeros(total_bins, device=self.device, dtype=torch.float64).scatter_add_(0, valid_ids,
                                                                                               pt_x * pt_y)
        sum_xz = torch.zeros(total_bins, device=self.device, dtype=torch.float64).scatter_add_(0, valid_ids,
                                                                                               pt_x * pt_z)
        sum_yy = torch.zeros(total_bins, device=self.device, dtype=torch.float64).scatter_add_(0, valid_ids, pt_y ** 2)
        sum_yz = torch.zeros(total_bins, device=self.device, dtype=torch.float64).scatter_add_(0, valid_ids,
                                                                                               pt_y * pt_z)
        sum_zz = torch.zeros(total_bins, device=self.device, dtype=torch.float64).scatter_add_(0, valid_ids, pt_z ** 2)

        # 计数 N
        ones = torch.ones_like(valid_ids, dtype=torch.float64)
        counts = torch.zeros(total_bins, device=self.device, dtype=torch.float64)
        counts.scatter_add_(0, valid_ids, ones)

        # ----------------------------------------------------------------
        # 3.2 构建协方差矩阵 (Covariance Matrix)
        # ----------------------------------------------------------------
        safe_counts = counts.clamp(min=3)
        centroids = sum_P / safe_counts.unsqueeze(1)  # [Total, 3]

        sum_PPt = torch.stack([
            sum_xx, sum_xy, sum_xz,
            sum_xy, sum_yy, sum_yz,
            sum_xz, sum_yz, sum_zz
        ], dim=1).reshape(total_bins, 3, 3)

        # Cov = Sum(PP^t) - N * mu * mu^T
        center_correction = torch.bmm(sum_P.unsqueeze(2), sum_P.unsqueeze(1)) / safe_counts.view(-1, 1, 1)
        covariance = sum_PPt - center_correction

        # 数值稳定性处理 (防止 svd 不收敛)
        covariance = torch.nan_to_num(covariance, nan=0.0)

        # ----------------------------------------------------------------
        # 3.3 使用 torch.svd_lowrank 近似求解
        # ----------------------------------------------------------------
        try:
            # 关键点：我们取 q=2 (平面上的两个主方向)，而不是 q=1
            # niter=2 即可，3x3 矩阵收敛极快
            # U, S, V = torch.svd_lowrank(A, q=2)
            # V 的形状是 [Total, 3, 2]，列向量是奇异向量
            _, _, V = torch.svd_lowrank(covariance.float(), q=2, niter=2)

            # 取出两个主方向向量
            v1 = V[:, :, 0]  # [Total, 3]
            v2 = V[:, :, 1]  # [Total, 3]

            # 两个主方向的叉积即为垂直于平面的法向量
            # n = v1 x v2
            normals = torch.cross(v1, v2, dim=1)

            # 归一化 (防止叉积结果模长不为1)
            normals = F.normalize(normals, dim=1)
            centroids = centroids.float()

        except RuntimeError as e:
            print(f"⚠️ SVD LowRank Failed: {e}")
            normals = torch.zeros(total_bins, 3, device=self.device, dtype=torch.float32)
            normals[:, 2] = 1.0
            centroids = centroids.float()

        # ==========================================================
        # 3.4 法向量定向与距离计算 (保持不变)
        # ==========================================================

        # 检测方向：法向量应指向相机 (与视线夹角 > 90度, 点积 < 0)
        # 视线向量 ≈ centroid (相机在原点 0,0,0)
        dot_product = torch.sum(normals * centroids, dim=1, keepdim=True)

        # 翻转 mask
        flip_mask = -torch.sign(dot_product)
        flip_mask[flip_mask == 0] = 1.0

        normals = normals * flip_mask

        # 计算 d: n * x + d = 0  =>  d = - n * centroid
        d_vals = -torch.sum(normals * centroids, dim=1, keepdim=True)

        # ==========================================================
        # 3.5 异常处理 (Fallbacks)
        # ==========================================================
        invalid_tris = (counts < 3)
        if invalid_tris.any():
            fallback_z = centroids[invalid_tris, 2]
            fallback_z[fallback_z == 0] = 10
            normals[invalid_tris] = torch.tensor([0.0, 0.0, -1.0], device=self.device)
            d_vals[invalid_tris, 0] = -fallback_z
            print("===========三角形点数≤3========================")

        # 组合结果
        plane_params = torch.cat([normals, d_vals], dim=1)

        return plane_params.view(B, max_num_triangles, 4)

    def get_plane_hypotheses(self, depth_stage2, tri_id_map, intrinsics_s1, max_num_triangles):
        """
        可微 SVD 拟合 + 假设生成
        Args:
            depth_stage2: [B, 1, H_s2, W_s2] (Stage 2 深度图)
            tri_id_map:   [B, H_s1, W_s1] (Stage 1 三角形索引图，值域 0~N-1, -1为无效)，每个位置存的是“它是第几个三角形”
            intrinsics_s1:[B, 3, 3] (Stage 1 内参)
            max_num_triangles: int (最大三角形数，用于分配 Tensor 大小)

        Returns:
            hypotheses: [B, N_tri, K, 4] (nx, ny, nz, d)
        Modified: 使用 torch.svd_lowrank 近似求解平面法向量。
        原理: 提取协方差矩阵的前两个主成分 (q=2)，计算其叉积得到法向量。

        """
        B = depth_stage2.shape[0]
        device = depth_stage2.device

        # ==========================================
        # Step 1: 投影与采样 (保持不变)
        # ==========================================
        grid_batch = self.sampling_grid.unsqueeze(0).expand(B, -1, -1, -1)
        sampled_depth = F.grid_sample(
            depth_stage2, grid_batch, mode='bilinear', padding_mode='border', align_corners=True
        )

        if sampled_depth.max() < 1e-4:
            return self._get_fallback_planes(B, max_num_triangles, device)

        # ==========================================
        # Step 2: 反投影生成点云 (保持不变)
        # ==========================================
        depth_flat = sampled_depth.view(B, -1)
        u_flat = self.grid_x.reshape(1, -1).expand(B, -1)
        v_flat = self.grid_y.reshape(1, -1).expand(B, -1)

        fx = intrinsics_s1[:, 0, 0].unsqueeze(1)
        fy = intrinsics_s1[:, 1, 1].unsqueeze(1)
        cx = intrinsics_s1[:, 0, 2].unsqueeze(1)
        cy = intrinsics_s1[:, 1, 2].unsqueeze(1)

        X = (u_flat - cx) * depth_flat / fx
        Y = (v_flat - cy) * depth_flat / fy
        Z = depth_flat

        points_flat = torch.stack([X, Y, Z], dim=2).view(-1, 3)

        # ==========================================
        # Step 3: 并行 SVD_LowRank 拟合
        # ==========================================
        tri_ids_flat = tri_id_map.view(-1)
        batch_ids = torch.arange(B, device=self.device).unsqueeze(1).expand(-1, self.H * self.W).reshape(-1)
        global_tri_ids = batch_ids * max_num_triangles + tri_ids_flat

        # 过滤无效像素
        valid_mask = (tri_ids_flat >= 0)

        if valid_mask.sum() == 0:
            return self._get_fallback_planes(B, max_num_triangles, device)

        valid_points = points_flat[valid_mask]
        valid_ids = global_tri_ids[valid_mask]
        total_bins = B * max_num_triangles

        # 强制使用 float64 (Double)
        valid_points_64 = valid_points.double()
        pt_x, pt_y, pt_z = valid_points_64[:, 0], valid_points_64[:, 1], valid_points_64[:, 2]

        if valid_ids.max() >= total_bins:
            valid_ids = torch.clamp(valid_ids, max=total_bins - 1)

        # ----------------------------------------------------------------
        # 3.1 统计协方差矩阵元素
        # ----------------------------------------------------------------
        # 基础统计量
        sum_P = torch.zeros(total_bins, 3, device=self.device, dtype=torch.float64)
        sum_P.index_add_(0, valid_ids, valid_points_64)  # Sum P

        # 二阶统计量
        sum_xx = torch.zeros(total_bins, device=self.device, dtype=torch.float64).scatter_add_(0, valid_ids, pt_x ** 2)
        sum_xy = torch.zeros(total_bins, device=self.device, dtype=torch.float64).scatter_add_(0, valid_ids,
                                                                                               pt_x * pt_y)
        sum_xz = torch.zeros(total_bins, device=self.device, dtype=torch.float64).scatter_add_(0, valid_ids,
                                                                                               pt_x * pt_z)
        sum_yy = torch.zeros(total_bins, device=self.device, dtype=torch.float64).scatter_add_(0, valid_ids, pt_y ** 2)
        sum_yz = torch.zeros(total_bins, device=self.device, dtype=torch.float64).scatter_add_(0, valid_ids,
                                                                                               pt_y * pt_z)
        sum_zz = torch.zeros(total_bins, device=self.device, dtype=torch.float64).scatter_add_(0, valid_ids, pt_z ** 2)

        # 计数 N
        ones = torch.ones_like(valid_ids, dtype=torch.float64)
        counts = torch.zeros(total_bins, device=self.device, dtype=torch.float64)
        counts.scatter_add_(0, valid_ids, ones)

        # ----------------------------------------------------------------
        # 3.2 构建协方差矩阵 (Covariance Matrix)
        # ----------------------------------------------------------------

        # 即使 counts < 3，我们也需要计算质心 (作为前向平行平面的深度)
        # clamp(min=1) 防止除零，对于 count=0 的情况，centroid 会是 0 (sum_P也是0)

        safe_counts = counts.clamp(min=3)
        centroids = sum_P / safe_counts.unsqueeze(1)  # [Total, 3]

        # 初始化 normals 和 d_vals
        # 默认全部初始化为 [0, 0, 1] (前向平行)
        normals = torch.zeros(total_bins, 3, device=self.device, dtype=torch.float32)
        normals[:, 2] = 1.0

        sum_PPt = torch.stack([
            sum_xx, sum_xy, sum_xz,
            sum_xy, sum_yy, sum_yz,
            sum_xz, sum_yz, sum_zz
        ], dim=1).reshape(total_bins, 3, 3)

        # Cov = Sum(PP^t) - N * mu * mu^T
        center_correction = torch.bmm(sum_P.unsqueeze(2), sum_P.unsqueeze(1)) / safe_counts.view(-1, 1, 1)
        covariance = sum_PPt - center_correction

        # 数值稳定性处理 (防止 svd 不收敛)
        covariance = torch.nan_to_num(covariance, nan=0.0)

        # ----------------------------------------------------------------
        # 🔥 核心修复：SVD 改为 EIGH (对称特征分解) 🔥
        # ----------------------------------------------------------------

        # 1. 识别有效拟合的三角形
        valid_fit_mask = (counts > 3)

        # 2. 正则化: 增加 eps 使得特征值分离
        # 增大 eps 到 1e-5，对于 float32 来说 1e-6 可能太小
        eps = 1e-4
        # 构造扰动向量 [eps, 10*eps, 100*eps]
        # 这样即使 covariance 全为 0，特征值也会被强行拉开差距
        perturb_vec = torch.tensor([1.0, 10.0, 100.0], device=device, dtype=covariance.dtype) * eps
        perturb_matrix = torch.diag(perturb_vec).unsqueeze(0)  # [1, 3, 3]

        # 加到原来的协方差矩阵上
        covariance = covariance + perturb_matrix

        # 3. 梯度防火墙 (Gradient Firewall)
        # 任何无效的三角形，强行把协方差矩阵设为 Identity
        # I 的特征值是 1,1,1 (虽然重复，但我们后续会处理)，或者设为 diag(1, 2, 3) 避免特征值重复

        # 构造一个特征值绝对不重复的安全矩阵: diag(100, 10, 1)
        # 这样最小特征向量明确是 Z 轴 [0,0,1]
        safe_matrix = torch.diag(torch.tensor([100.0, 10.0, 1.0], device=device, dtype=covariance.dtype)).unsqueeze(0)

        mask_expand = valid_fit_mask.view(-1, 1, 1).expand_as(covariance)
        covariance_safe = torch.where(mask_expand, covariance, safe_matrix.expand_as(covariance))

        # ----------------------------------------------------------------
        # 3.3 使用 torch.linalg.eigh 求解
        # ----------------------------------------------------------------
        try:
            # eigh 专门用于 Hermitian (对称) 矩阵
            # 我们强制使用 double (float64) 精度来计算梯度，防止下溢出
            vals, vecs = torch.linalg.eigh(covariance_safe.double())

            # 转换回 float32
            vecs = vecs.float()

            # 在平面拟合中，法向量对应 **最小** 的特征值
            # 因为 eigh 是升序排列，所以取 index 0
            normals = vecs[:, :, 0]  # [Total, 3]

            # 归一化 (理论上 eigh 出来的已经是归一化的，但为了保险)
            normals = F.normalize(normals, dim=1)
            centroids = centroids.float()

        except RuntimeError as e:
            print(f"⚠️ SVD LowRank Failed: {e}")
            normals = torch.zeros(total_bins, 3, device=self.device, dtype=torch.float32)
            normals[:, 2] = 1.0
            centroids = centroids.float()

        # ==========================================================
        # 3.4 法向量定向与距离计算 (保持不变)
        # ==========================================================

        # 检测方向：法向量应指向相机 (与视线夹角 > 90度, 点积 < 0)
        # 视线向量 ≈ centroid (相机在原点 0,0,0)
        dot_product = torch.sum(normals * centroids, dim=1, keepdim=True)

        # 翻转 mask
        flip_mask = -torch.sign(dot_product)
        flip_mask[flip_mask == 0] = 1.0

        normals = normals * flip_mask

        # 计算 d: n * x + d = 0  =>  d = - n * centroid
        d_vals = -torch.sum(normals * centroids, dim=1, keepdim=True)

        # ==========================================================
        # 3.5 🔥 新增：基于物理射线的异常三角平面过滤与替换 🔥
        # ==========================================================
        # 我们利用质心处的精确物理射线来检验拟合出的平面是否“爆炸”

        with torch.no_grad():  # 校验掩码的生成不需要梯度
            # 1. 构造从相机中心指向质心的射线方向
            # centroid 本身就是相机坐标系下的 3D 点，所以它就是未归一化的射线方向
            rays = centroids.clone()  # [Total, 3]

            # 2. 射线与拟合平面求交，计算交点深度
            # Z = -d * ray_z / (n \cdot ray)
            denom = torch.sum(normals * rays, dim=1, keepdim=True)  # [Total, 1]
            denom_safe = torch.where(denom.abs() < 1e-4, torch.full_like(denom, 1e-4), denom)

            # ray_z 就是 centroids[:, 2:3]
            depth_est = (-d_vals * rays[:, 2:3] / denom_safe).abs()  # [Total, 1]

            # 3. 判断是否为异常平面 (包含点数不足的情况)
            # 兼容深度极值的类型
            #  核心修复：兼容 depth_min/max 是 [B] 形状的 Tensor
            if isinstance(self.depth_min, torch.Tensor):
                # 将 [B] 扩展并 reshape 为 [Total, 1] 以对齐 depth_est
                d_min_val = self.depth_min.view(B, 1).expand(B, max_num_triangles).reshape(total_bins, 1)
                d_max_val = self.depth_max.view(B, 1).expand(B, max_num_triangles).reshape(total_bins, 1)
            else:
                # 兼容外部传入的是 list 或标量的情况
                d_min_val = min(self.depth_min) if isinstance(self.depth_min, list) else float(self.depth_min)
                d_max_val = max(self.depth_max) if isinstance(self.depth_max, list) else float(self.depth_max)

            invalid_tris = (counts <= 3).unsqueeze(1)  # 点数不足，必错 [Total, 1]
            is_bad_geom = (depth_est < d_min_val * 0.5) | \
                          (depth_est > d_max_val * 2.0) | \
                          (~torch.isfinite(depth_est))  # 几何爆炸 [Total, 1]

            # 如果法向量的 Z 分量绝对值太小 (< 0.1)，说明平面几乎与视线平行，这在城市场景中极度危险
            is_bad_normal = (normals[:, 2:3].abs() < 0.1)

            is_invalid = invalid_tris | is_bad_geom | is_bad_normal # [Total, 1]

            # (可选) 打印过滤日志
            bad_ratio = is_invalid.float().mean()
            if bad_ratio > 0.05:
                print(f"[HYPO GEN] SVD 拟合拦截异常平面: {bad_ratio * 100:.2f}%")

        # 4. 准备安全的 Fallback 平面 (前向平行平面)
        # 遵守网络约定：法向强制指向相机 [0, 0, -1]
        fallback_n = torch.zeros_like(normals)
        fallback_n[:, 2] = -1.0  # [Total, 3]

        # 截距：因为 nz 是 -1，所以 d = +Z
        safe_z = centroids[:, 2:3].clone()  # [Total, 1]
        safe_z = torch.where(counts.unsqueeze(1) < 3, d_max_val, safe_z)
        fallback_d = safe_z  # [Total, 1]  <-- 注意这里没有负号了！

        # 5. 使用 torch.where 执行安全的 Out-of-place 替换 (保持梯度连贯)
        normals = torch.where(is_invalid.expand_as(normals), fallback_n, normals)
        d_vals = torch.where(is_invalid, fallback_d, d_vals)

        # [Hypothesis 0] 原始拟合结果 [B, N, 1, 4]
        hypo_0 = torch.cat([normals, d_vals], dim=1).view(B, max_num_triangles, 1, 4)

        # ==========================================
        # 不需要生成假设
        # ==========================================
        hypo_list = [hypo_0]

        # 最终拼接 [B, N, K, 4]
        # 直接返回唯一的假设 (K=1)，不再进行任何随机扰动拼凑！
        # [B, N, 1, 4]
        hypotheses = torch.cat([normals, d_vals], dim=1).view(B, max_num_triangles, 1, 4)

        return hypotheses

    def By_LLS_Plane(self, depth_stage2, tri_id_map, intrinsics_s1, max_num_triangles):
        """
        Modified: 使用线性最小二乘法 (Linear Least Squares) 替代 SVD 进行平面拟合。
        Solving: Z = aX + bY + c  =>  aX + bY - Z + c = 0
        """
        B = depth_stage2.shape[0]
        device = depth_stage2.device

        # ==========================================
        # Step 1: 投影与采样 (保持不变)
        # ==========================================
        grid_batch = self.sampling_grid.unsqueeze(0).expand(B, -1, -1, -1)

        sampled_depth = F.grid_sample(
            depth_stage2,
            grid_batch,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )

        if sampled_depth.max() < 1e-4:
            return self._get_fallback_planes(B, max_num_triangles, device)

        # ==========================================
        # Step 2: 反投影生成点云 (保持不变)
        # ==========================================
        depth_flat = sampled_depth.view(B, -1)
        u_flat = self.grid_x.reshape(1, -1).expand(B, -1)
        v_flat = self.grid_y.reshape(1, -1).expand(B, -1)

        fx = intrinsics_s1[:, 0, 0].unsqueeze(1)
        fy = intrinsics_s1[:, 1, 1].unsqueeze(1)
        cx = intrinsics_s1[:, 0, 2].unsqueeze(1)
        cy = intrinsics_s1[:, 1, 2].unsqueeze(1)

        X = (u_flat - cx) * depth_flat / fx
        Y = (v_flat - cy) * depth_flat / fy
        Z = depth_flat

        points_flat = torch.stack([X, Y, Z], dim=2).view(-1, 3)

        # ==========================================
        # Step 3: 并行线性最小二乘拟合 (Parallel LLS)
        # ==========================================
        tri_ids_flat = tri_id_map.view(-1)
        batch_ids = torch.arange(B, device=self.device).unsqueeze(1).expand(-1, self.H * self.W).reshape(-1)
        global_tri_ids = batch_ids * max_num_triangles + tri_ids_flat

        # 过滤无效像素
        valid_mask = (tri_ids_flat >= 0) & (points_flat[:, 2] > 1e-4)

        if valid_mask.sum() == 0:
            return self._get_fallback_planes(B, max_num_triangles, device)

        valid_points = points_flat[valid_mask]
        valid_ids = global_tri_ids[valid_mask]
        total_bins = B * max_num_triangles

        # 强制使用 float64 (Double) 以保证 ATA 矩阵求逆的精度
        valid_points_64 = valid_points.double()
        pt_x, pt_y, pt_z = valid_points_64[:, 0], valid_points_64[:, 1], valid_points_64[:, 2]

        if valid_ids.max() >= total_bins:
            # 简单的越界保护
            valid_ids = torch.clamp(valid_ids, max=total_bins - 1)

        # ----------------------------------------------------------------
        # 3.1 统计矩阵元素 (Scatter Add)
        # 构建线性方程组 Ax = b，其中 A=[x, y, 1], b=[z]
        # 需要计算 A^T A 和 A^T b
        # A^T A = [[sum_xx, sum_xy, sum_x],
        #          [sum_xy, sum_yy, sum_y],
        #          [sum_x,  sum_y,  count]]
        # ----------------------------------------------------------------

        # 预分配零张量
        zeros_t = torch.zeros(total_bins, device=self.device, dtype=torch.float64)

        # 基础统计量
        sum_x = zeros_t.clone().scatter_add_(0, valid_ids, pt_x)
        sum_y = zeros_t.clone().scatter_add_(0, valid_ids, pt_y)
        sum_z = zeros_t.clone().scatter_add_(0, valid_ids, pt_z)  # 用于 A^T b

        # 二阶统计量
        sum_xx = zeros_t.clone().scatter_add_(0, valid_ids, pt_x * pt_x)
        sum_xy = zeros_t.clone().scatter_add_(0, valid_ids, pt_x * pt_y)
        sum_yy = zeros_t.clone().scatter_add_(0, valid_ids, pt_y * pt_y)
        sum_xz = zeros_t.clone().scatter_add_(0, valid_ids, pt_x * pt_z)  # 用于 A^T b
        sum_yz = zeros_t.clone().scatter_add_(0, valid_ids, pt_y * pt_z)  # 用于 A^T b

        # 计数 N
        ones = torch.ones_like(valid_ids, dtype=torch.float64)
        counts = torch.zeros(total_bins, device=self.device, dtype=torch.float64)
        counts.scatter_add_(0, valid_ids, ones)

        # ----------------------------------------------------------------
        # 3.2 组装矩阵并求解
        # ----------------------------------------------------------------

        # 构造 ATA: [Total, 3, 3]
        ATA = torch.stack([
            sum_xx, sum_xy, sum_x,
            sum_xy, sum_yy, sum_y,
            sum_x, sum_y, counts
        ], dim=1).reshape(total_bins, 3, 3)

        # 构造 ATb: [Total, 3, 1]
        ATb = torch.stack([sum_xz, sum_yz, sum_z], dim=1).reshape(total_bins, 3, 1)

        # 正则化：对角线加 epsilon 防止奇异矩阵 (不可逆)
        eps = 1e-6
        eye = torch.eye(3, device=self.device, dtype=torch.float64).unsqueeze(0)
        ATA = ATA + eye * eps

        # 求解线性方程组
        try:
            # solution shape: [Total, 3, 1] -> [a, b, c]^T
            # torch.linalg.solve 比 inverse 更快更准且梯度稳定
            solution = torch.linalg.solve(ATA, ATb)

            a = solution[:, 0, 0]
            b = solution[:, 1, 0]
            c = solution[:, 2, 0]

            # ----------------------------------------------------------------
            # 3.3 转换参数为 (n, d)
            # 平面方程: z = ax + by + c  =>  ax + by - z + c = 0
            # 法向量 raw_n = (a, b, -1)
            # ----------------------------------------------------------------

            # 构造未归一化法向量 [Total, 3]
            minus_ones = -torch.ones_like(a)
            raw_normals = torch.stack([a, b, minus_ones], dim=1).float()

            # 归一化
            n_norm = torch.norm(raw_normals, dim=1, keepdim=True)
            normals = raw_normals / (n_norm + 1e-8)

            normals = -normals

            # 计算 d
            # 标准方程: n_normalized * X + d = 0
            # 我们的方程: raw_n * X + c = 0
            # 两边同除以 norm => (raw_n/norm) * X + (c/norm) = 0
            # 所以 d = c / norm
            d_vals = c.float().unsqueeze(1) / (n_norm + 1e-8)

        except RuntimeError as e:
            print(f"⚠️ Linear Solve Failed: {e}")
            return self._get_fallback_planes(B, max_num_triangles, device)

        # ==========================================================
        # 定向修正 (Orient Normals)
        # ==========================================================
        # 计算质心 (用于判断法向量方向)
        safe_counts = counts.clamp(min=1).view(-1, 1)
        centroids = torch.stack([sum_x, sum_y, sum_z], dim=1).float() / safe_counts.float()

        # 计算点积 check
        # MVS中通常希望法向量指向相机（或与视线夹角锐角）
        # 这里 n 的 z 分量已经是负的了 (a, b, -1)，大部分情况是自动对齐的
        # 但为了鲁棒性，依然执行几何检测
        # dot_product = torch.sum(normals * centroids, dim=1, keepdim=True)
        #
        # dot_product = -dot_product
        #
        # # 如果 dot > 0 (同向)，说明法向量指向了屏幕内，需要翻转
        # flip_mask = -torch.sign(dot_product)
        # flip_mask[flip_mask == 0] = 1.0
        #
        # normals = normals * flip_mask
        # d_vals = d_vals * flip_mask  # 注意：d 也要跟着翻转，保持平面方程成立

        # ==========================================================
        # 异常处理 (Fallbacks)
        # ==========================================================
        # 对于点数极少 (<3) 的三角形，解不可靠，使用 fallback
        invalid_tris = (counts < 3)
        if invalid_tris.any():
            fallback_z = centroids[invalid_tris, 2]
            fallback_z[fallback_z == 0] = 10.0

            # 填充默认值：垂直于Z轴的平面
            normals[invalid_tris] = torch.tensor([0.0, 0.0, 1.0], device=self.device)
            d_vals[invalid_tris, 0] = -fallback_z

        # 组合结果
        plane_params = torch.cat([normals, d_vals], dim=1)

        return plane_params.view(B, max_num_triangles, 4)

    def get_gt_planes_by_3points(self, depth_gt, tri_id_map, intrinsics, max_num_triangles):
        """
        快速利用 GT 深度图，通过三点法解析求解平面参数。
        策略：在每个三角形内取 [First, Middle, Last] 三个像素点计算法向量。

        Args:
            depth_gt: [B, 1, H, W] (Stage 1 分辨率的真值深度图)
            tri_id_map: [B, H, W] (三角形 ID 图)
            intrinsics: [B, 3, 3]
            max_num_triangles: int

        Returns:
            plane_params: [B, N_tri, 4] (nx, ny, nz, d)
        """
        B, _, H, W = depth_gt.shape
        device = depth_gt.device
        total_bins = B * max_num_triangles

        # ==========================================
        # 1. 数据准备 (Flatten & ID Offset)
        # ==========================================
        # 展平数据
        depth_flat = depth_gt.view(-1)  # [B*H*W]
        tri_ids_flat = tri_id_map.view(-1)  # [B*H*W]

        # 生成 Batch 偏移的全局三角形 ID
        batch_ids = torch.arange(B, device=device).unsqueeze(1).expand(-1, H * W).reshape(-1)
        global_tri_ids = batch_ids * max_num_triangles + tri_ids_flat

        # 筛选有效像素: ID有效 且 GT深度有效
        valid_mask = (tri_ids_flat >= 0) & (depth_flat > 1e-4)

        valid_depth = depth_flat[valid_mask]
        valid_global_ids = global_tri_ids[valid_mask]

        # 我们还需要像素的 u, v 坐标来反投影
        u_flat = self.grid_x.reshape(1, -1).expand(B, -1).reshape(-1)
        v_flat = self.grid_y.reshape(1, -1).expand(B, -1).reshape(-1)

        valid_u = u_flat[valid_mask]
        valid_v = v_flat[valid_mask]
        valid_batch_idx = batch_ids[valid_mask]  # 记录每个像素属于哪个batch，用于查内参

        # ==========================================
        # 2. 并行分组与采样 (Sorting & Indexing)
        # ==========================================
        # 对 ID 进行排序，这样同一个三角形的像素就会聚在一起
        sorted_ids, sort_idx = torch.sort(valid_global_ids)

        # 找到每个三角形 ID 的唯一值和出现次数 (Counts)
        unique_ids, counts = torch.unique_consecutive(sorted_ids, return_counts=True)

        # 过滤掉点数少于 3 个的三角形 (无法构成平面)
        valid_tri_mask = counts >= 3

        final_tri_ids = unique_ids[valid_tri_mask]  # [K] 有效的三角形ID
        final_counts = counts[valid_tri_mask]  # [K] 每个三角形的点数

        # --- 核心：找到每个三角形的 Start Index ---
        # unique_consecutive 不直接返回 start index，我们需要算 cumsum
        # 这里的 cumsum 是针对排序后的数组的
        ends = torch.cumsum(counts, dim=0)
        starts = torch.cat([torch.zeros(1, device=device, dtype=torch.long), ends[:-1]])

        # 应用 mask，只保留有效三角形的 start 和 count
        tri_starts = starts[valid_tri_mask]

        # --- 核心：采样三个点 (首、中、尾) ---
        # 这种采样方式能最大程度保证三点在空间上拉开距离，避免共线
        idx_a_local = tri_starts
        idx_b_local = tri_starts + (final_counts // 2)
        idx_c_local = tri_starts + final_counts - 1

        # 映射回原始数据的索引 (通过 sort_idx)
        # 这里的 idx_a 是 valid_points 数组中的索引
        idx_a = sort_idx[idx_a_local]
        idx_b = sort_idx[idx_b_local]
        idx_c = sort_idx[idx_c_local]

        # ==========================================
        # 3. 反投影为 3D 点 (Back-Projection)
        # ==========================================
        def get_3d_points(indices):
            z = valid_depth[indices]
            u = valid_u[indices]
            v = valid_v[indices]
            b_idx = valid_batch_idx[indices]

            # 获取对应的内参
            fx = intrinsics[b_idx, 0, 0]
            fy = intrinsics[b_idx, 1, 1]
            cx = intrinsics[b_idx, 0, 2]
            cy = intrinsics[b_idx, 1, 2]

            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            return torch.stack([x, y, z], dim=1)  # [K, 3]

        Pa = get_3d_points(idx_a)
        Pb = get_3d_points(idx_b)
        Pc = get_3d_points(idx_c)

        # ==========================================
        # 4. 向量叉乘求法向量 (Vector Math)
        # ==========================================
        # 向量 AB 和 AC
        V1 = Pb - Pa
        V2 = Pc - Pa

        # 叉乘: N = V1 x V2
        normals = torch.cross(V1, V2, dim=1)

        # 归一化
        norm_len = torch.norm(normals, dim=1, keepdim=True)
        # 防止除零 (虽然很难发生，因为我们选了首尾点)
        normals = normals / (norm_len + 1e-8)

        # --- 强制法向量指向相机 (Z > 0 修正) ---
        # 视线向量约为 Pa (从原点到点)
        # 我们希望 dot(n, view) < 0 => 法向量迎着视线
        # 或者简单地：MVS 中通常定义法向量指向 Z 负 (迎着相机) 或 Z 正 (远离相机)
        # 这里为了和你之前的代码一致，我们用 "点积检测"
        dot_prod = torch.sum(normals * Pa, dim=1, keepdim=True)
        dot_prod=-dot_prod
        # 如果同向 (dot>0)，说明指向屏幕内，需要翻转指向屏幕外(相机)
        flip = -torch.sign(dot_prod)
        flip[flip == 0] = 1.0
        normals = normals * flip

        # ==========================================
        # 5. 计算 d 并 填充结果
        # ==========================================
        # d = - n * p
        d_vals = -torch.sum(normals * Pa, dim=1, keepdim=True)

        # --- 填回 [B * Max_Tri, 4] 的大张量 ---
        # 初始化默认平面 (0, 0, 1, -10)
        all_planes = torch.zeros(total_bins, 4, device=device)
        all_planes[:, 2] = 1.0
        all_planes[:, 3] = -10.0  # 默认深度

        # 将计算好的值填入对应的 Global ID 位置
        # final_tri_ids 记录了我们计算的是哪些三角形
        all_planes[final_tri_ids, :3] = normals
        all_planes[final_tri_ids, 3:] = d_vals

        return all_planes.view(B, max_num_triangles, 4)

    def _get_fallback_planes(self, B, max_tri, device):
        """辅助函数：全挂了的时候返回默认平面"""
        normals = torch.tensor([0.0, 0.0, 1.0], device=device).view(1, 1, 3).expand(B, max_tri, -1)
        d_vals = torch.tensor([-10.0], device=device).view(1, 1, 1).expand(B, max_tri, -1)
        return torch.cat([normals, d_vals], dim=-1)

    def collate_centroids_norm(self,centroids_norm_list, device):
        """
        centroids_norm_list: List[B] of Tensor [N_b, 2]
        返回: [B, N_max, 2]，不足 N_max 的位置用 0 填充
        """
        B = len(centroids_norm_list)
        N_max = max(c.shape[0] for c in centroids_norm_list)

        out = torch.zeros(B, N_max, 2, device=device)
        for b, c in enumerate(centroids_norm_list):
            N_b = c.shape[0]
            out[b, :N_b, :] = c.to(device)

        return out  # [B, N_max, 2]

    def enforce_depth_hard_constraint(self,planes, centroids_norm, intrinsics, depth_min, depth_max, H, W):
        """
        对传播后的平面进行基于真实物理射线的深度硬截断保护。
        利用广播机制完美兼容深度极值为 Tensor 或标量的情况。

        Args:
            planes: [B, N, 4] 需要校验的平面参数
            centroids_norm: [B, N, 2] 三角形质心的归一化坐标 [-1, 1]
            intrinsics: [B, 3, 3] 相机内参
            depth_min/max: Tensor [B] 或标量/列表，场景的深度极值
            H, W: 图像高宽

        Returns:
            planes_clean: [B, N, 4] 剔除爆炸参数后的安全平面
        """
        B, N, _ = planes.shape
        device = planes.device

        # ==========================================
        # 1. 坐标还原与射线计算
        # ==========================================
        u_px = (centroids_norm[..., 0] + 1.0) / 2.0 * (W - 1)  # [B, N]
        v_px = (centroids_norm[..., 1] + 1.0) / 2.0 * (H - 1)  # [B, N]
        uv_homo = torch.stack([u_px, v_px, torch.ones_like(u_px)], dim=-1)  # [B, N, 3]

        K_inv = torch.inverse(intrinsics)  # [B, 3, 3]
        rays = torch.einsum('bij,bnj->bni', K_inv, uv_homo)  # [B, N, 3]

        # ==========================================
        # 2. 精确计算平面在质心处的物理深度
        # ==========================================
        n = planes[..., :3]  # [B, N, 3]
        d = planes[..., 3:]  # [B, N, 1]

        denom = (n * rays).sum(dim=-1, keepdim=True)  # [B, N, 1]
        denom_safe = torch.where(denom.abs() < 1e-4, torch.full_like(denom, 1e-4), denom)
        depth_est = (-d / denom_safe).abs()  # [B, N, 1]

        # ==========================================
        # 3. 动态处理边界极值与掩码判定
        # ==========================================
        if isinstance(depth_min, torch.Tensor):
            d_min_val = depth_min.view(-1, 1, 1)  # [B, 1, 1] 方便广播
            d_max_val = depth_max.view(-1, 1, 1)
        else:
            # 兼容 list 或单个数字
            d_min_val = min(depth_min) if isinstance(depth_min, list) else float(depth_min)
            d_max_val = max(depth_max) if isinstance(depth_max, list) else float(depth_max)

        # 查杀异常值：容忍度 0.5倍 ~ 2.0倍
        is_bad = (depth_est < d_min_val * 0.5) | \
                 (depth_est > d_max_val * 2.0) | \
                 (~torch.isfinite(depth_est))  # [B, N, 1]

        # ==========================================
        # 4. 构造安全前向平行平面 (规避 full_like 报错)
        # ==========================================
        replace_depth = (d_min_val + d_max_val) / 2.0

        fp_n = torch.zeros_like(n)
        fp_n[..., 2] = -1.0  # 🔥 修改 1：强制法向指向相机 (z 分量为 -1)

        # 核心：因为方程是 n*P + d = 0，代入 n=[0,0,-1] 和 Z=replace_depth
        # 得到 -replace_depth + d = 0，即 d = +replace_depth
        fp_d = torch.zeros_like(d) + replace_depth  # 🔥 修改 2：d 取正值 (去掉减号)

        fp_planes = torch.cat([fp_n, fp_d], dim=-1)  # [B, N, 4]

        # ==========================================
        # 5. 执行替换与防断流
        # ==========================================
        planes_clean = torch.where(is_bad.expand_as(planes), fp_planes, planes)

        # 诊断输出（可注释掉，仅在极值爆发时提醒）
        with torch.no_grad():
            bad_ratio = is_bad.float().mean()
            if bad_ratio > 0.05:
                print(f"[HARD CONSTRAINT] 警告: 拦截了 {bad_ratio * 100:.2f}% 的数值崩溃平面！")

        return planes_clean

    def filter_invalid_planes(self,planes, depth_min, depth_max,
                              centroids_norm, intrinsics, H, W):
        """
        修复svd拟合的平面
        planes:          [B, N, 4]
        centroids_norm:  [B, N, 2]  归一化坐标 [-1, 1]（来自 tri_infos）
        intrinsics:      [B, 3, 3]
        depth_min/max:   [B] 或 float
        """
        B, N, _ = planes.shape
        device = planes.device

        # [-1,1] → 像素坐标
        u_px = (centroids_norm[..., 0] + 1.0) / 2.0 * (W - 1)  # [B,N]
        v_px = (centroids_norm[..., 1] + 1.0) / 2.0 * (H - 1)  # [B,N]
        ones = torch.ones_like(u_px)
        uv_homo = torch.stack([u_px, v_px, ones], dim=-1)  # [B,N,3]

        # 用 K_inv 得到射线方向（与 compute_continuity_loss 完全一致）
        K_inv = torch.inverse(intrinsics)  # [B,3,3]
        rays = torch.einsum('bij,bnj->bni', K_inv, uv_homo)  # [B,N,3]

        # 计算深度
        n = planes[..., :3]  # [B,N,3]
        d = planes[..., 3:]  # [B,N,1]

        denom = (n * rays).sum(dim=-1, keepdim=True)  # [B,N,1]
        denom_safe = torch.where(denom.abs() < 1e-4,
                                 torch.full_like(denom, 1e-4), denom)
        depth_est = (-d / denom_safe).abs()  # [B,N,1]

        # 异常掩码
        d_min = depth_min.view(B, 1, 1) if isinstance(depth_min, torch.Tensor) \
            else torch.full((B, 1, 1), depth_min, device=device)
        d_max = depth_max.view(B, 1, 1) if isinstance(depth_max, torch.Tensor) \
            else torch.full((B, 1, 1), depth_max, device=device)

        is_invalid = (depth_est < d_min * 0.5) | \
                     (depth_est > d_max * 2.0) | \
                     (~torch.isfinite(depth_est))

        # 替换为前向平行平面
        valid_depth = depth_est.masked_fill(is_invalid, float('nan'))
        batch_mean = valid_depth.nanmean(dim=1, keepdim=True)
        fallback = (d_min + d_max) / 2.0
        batch_mean = torch.where(torch.isnan(batch_mean), fallback, batch_mean)

        fp_planes = torch.zeros_like(planes)
        fp_planes[..., 2] = 1.0
        fp_planes[..., 3:] = -batch_mean.expand(B, N, 1)

        planes_clean = torch.where(is_invalid.expand_as(planes), fp_planes, planes)

        with torch.no_grad():
            ratio = is_invalid.float().mean()
            if ratio > 0.005:
                print(f"[FILTER] 异常平面 {ratio * 100:.2f}%")

        return planes_clean, is_invalid.squeeze(-1)

class PlaneVisualizer:
    def __init__(self, height, width, device):
        self.H = height
        self.W = width
        self.device = device

        # 预先生成像素网格 (u, v)
        y_range = torch.arange(0, height, dtype=torch.float32, device=device)
        x_range = torch.arange(0, width, dtype=torch.float32, device=device)
        # indexing='ij' 确保行优先
        self.grid_y, self.grid_x = torch.meshgrid(y_range, x_range,indexing='ij')

    def render_from_planes(self, plane_params, tri_id_map, intrinsics,depth_range):
        """
        利用平面参数渲染深度图和法向量图。
        并确保法向量图的颜色风格与 compute_normal_map_torch 一致

        Args:
            plane_params: [B, N_tri, 4] (nx, ny, nz, d)
            tri_id_map:   [B, H, W] (值域 0~N-1, -1为无效)
            intrinsics:   [B, 3, 3]

        Returns:
            depth_map:  [B, 1, H, W]
            normal_map: [B, 3, H, W] (颜色化法向量, 0~1, 无效区域为0)
        """
        B, N_tri, _ = plane_params.shape

        # 1. 检查平面数量
        if N_tri == 0:
            print("{}为0=======================".format(N_tri))
            return 0

        # ==========================================
        # 1. 像素级平面参数映射 (Sparse to Dense)
        # ==========================================

        # 处理 -1 的无效区域：
        # 技巧：我们将 -1 替换为 0 (指向第0个平面)，稍后用 mask 清零
        invalid_mask = (tri_id_map < 0)
        safe_id_map = tri_id_map.clone()
        safe_id_map[invalid_mask] = 0

        # 展平 ID Map 以便 gather
        # flat_ids: [B, H*W]
        flat_ids = safe_id_map.view(B, -1)

        # 扩展 plane_params 用于 gather: [B, N, 4]
        # Batch 偏移量 + view (-1)
        batch_offset = torch.arange(B, device=self.device) * N_tri
        batch_offset = batch_offset.view(B, 1)

        # global_ids: [B, H*W] -> [B*H*W]
        global_ids = (flat_ids + batch_offset).view(-1)

        # 把 plane_params 展平为 [B*N, 4]
        flat_planes = plane_params.view(-1, 4)

        # 核心操作：一次性查表
        # pixel_planes: [B*H*W, 4]
        # === 此时如果 N_tri > 0，这里就不会越界了 ===
        pixel_planes_flat = flat_planes[global_ids]

        # 恢复形状 [B, H, W, 4]
        pixel_planes = pixel_planes_flat.view(B, self.H, self.W, 4)

        # 分离 n 和 d
        n_map = pixel_planes[..., :3]
        d_map = pixel_planes[..., 3]

        # ==========================================
        # 2. 生成法向量图，并且对x，y进行一个归一化
        # ==========================================
        normal_vis = n_map.permute(0, 3, 1, 2).clone()
        # === 关键修改：将 [-1, 1] 映射到 [0, 1] ===
        # 这一步是为了让颜色跟你之前的函数保持一致
        # (x+1)/2: -1->0, 0->0.5, 1->1

        # 视觉友好可视化 (偏蓝) -> 用于 Tensorboard 展示
        # 我们临时翻转 Z 轴用于显示，虽然物理上它是负的
        normal_vis[:, 2, :, :] = -normal_vis[:, 2, :, :]  # 翻转 Z 用于显示
        # 专门针对于目前这个 svd 拟合 ym-modify
        normal_vis[:, 1, :, :] = -normal_vis[:, 1, :, :]  # 翻转 Y 用于显示
        normal_vis = (normal_vis + 1.0) / 2.0

        mask_expand = invalid_mask.unsqueeze(1).expand(-1, 3, -1, -1)
        normal_vis[mask_expand] = 0.0

        # ==========================================
        # 3. 生成深度图 (Ray-Plane Intersection)
        # ==========================================

        fx = intrinsics[:, 0, 0].view(B, 1, 1)
        fy = intrinsics[:, 1, 1].view(B, 1, 1)
        cx = intrinsics[:, 0, 2].view(B, 1, 1)
        cy = intrinsics[:, 1, 2].view(B, 1, 1)

        u_grid = self.grid_x.unsqueeze(0).expand(B, -1, -1)
        v_grid = self.grid_y.unsqueeze(0).expand(B, -1, -1)

        u_bar = (u_grid - cx) / fx
        v_bar = (v_grid - cy) / fy

        dot_product = (n_map[..., 0] * u_bar) + \
                      (n_map[..., 1] * v_bar) + \
                      (n_map[..., 2] * 1.0)

        valid_dot = torch.abs(dot_product) > 0.01  # [B,H,W]
        dot_product = torch.where(
            valid_dot,
            dot_product,
            torch.sign(dot_product + 1e-10) * 0.01  # 保留符号，限制最小绝对值
        )

        depth_map = (-d_map / dot_product).abs()


        # dot_product[torch.abs(dot_product) < 1e-6] = 1e-6
        #
        # depth_map = -d_map / dot_product
        #
        # # 解释：无论是平面法向反了(n -> -n)，还是点积反了，
        # # 我们都知道物体肯定在相机前面，所以直接要由距离的模长。
        # depth_map = depth_map.abs()

        # ==========================================
        # 4. 数值截断 (Clamp) - 解决 900/160 问题
        # ==========================================
        if depth_range is not None:
            min_d, max_d = depth_range

            # 如果 min_d 是 Tensor (例如 [B])，强制转为 [B, 1, 1] 以匹配 [B, H, W]
            if isinstance(min_d, torch.Tensor):
                if min_d.ndim == 1:
                    min_d = min_d.view(-1, 1, 1)
                min_d = min_d.to(depth_map.device)  # 确保设备一致

            if isinstance(max_d, torch.Tensor):
                if max_d.ndim == 1:
                    max_d = max_d.view(-1, 1, 1)
                max_d = max_d.to(depth_map.device)

            depth_map = torch.clamp(depth_map, min=min_d, max=max_d)

            # ── 修复：min_d 可能是 [B,1,1] tensor，不能直接 float() ──
            # 用 expand_as 广播成与 depth_map 相同形状，再用 where 填充
            if isinstance(min_d, torch.Tensor):
                fill_val = min_d.expand_as(depth_map)
            else:
                fill_val = torch.full_like(depth_map, float(min_d))

            depth_map = torch.where(valid_dot, depth_map, fill_val)
        else:
            # 即使没有给定范围，也要限制一下无穷大
            depth_map = torch.clamp(depth_map, max=600.0)

        print(f"Depth Max: {depth_map.max().item()}, Dot Min: {dot_product.abs().min().item()}")
        # ym-needmodify
        fill_value = depth_map.mean()

        if invalid_mask is not None:
            # ✅ 使用 torch.where (非原地操作)
            # 逻辑：如果 mask 为 True，取 fill_value；否则保持 depth_map 原值
            # 这会创建一个全新的 tensor，不会破坏原来的 depth_map，梯度可以正常回传
            depth_map = torch.where(invalid_mask, fill_value, depth_map)

        depth_map = depth_map.unsqueeze(1)

        return depth_map, normal_vis

    def render_from_pixel_wise_planes(self, pixel_planes, intrinsics, depth_range, invalid_mask=None):
        """
        [新版渲染函数] 从像素级平面参数 [B, H, W, 4] 直接渲染深度图和法向量图。
        替代原有的 Sparse-to-Dense 过程，直接处理 Dense 输入，且保持可视化风格一致。

        Args:
            pixel_planes: [B, H, W, 4] (nx, ny, nz, d)
            intrinsics:   [B, 3, 3] 相机内参
            depth_range:  (min_d, max_d), 支持 float 或 Tensor([B])
            invalid_mask: [B, H, W] (可选，指定无效区域，如 tri_id_map < 0 的区域)

        Returns:
            depth_map:  [B, 1, H, W]
            normal_vis: [B, 3, H, W] (颜色化法向量, 0~1)
        """
        B, H, W, _ = pixel_planes.shape
        device = pixel_planes.device

        # 分离 n 和 d
        n_map = pixel_planes[..., :3]  # [B, H, W, 3]
        d_map = pixel_planes[..., 3]  # [B, H, W]

        # ==========================================
        # 1. 生成法向量图 (保持与 compute_normal_map_torch 一致的风格)
        # ==========================================
        # [B, H, W, 3] -> [B, 3, H, W]
        normal_vis = n_map.permute(0, 3, 1, 2).clone()

        # === 视觉友好可视化处理 (完全复用你的逻辑) ===
        # 翻转 Z (物理上指向相机为负，显示改为正，偏蓝)
        normal_vis[:, 2, :, :] = -normal_vis[:, 2, :, :]
        # 翻转 Y (针对 SVD 拟合的特殊修改 ym-modify)
        normal_vis[:, 1, :, :] = -normal_vis[:, 1, :, :]

        # 映射到 [0, 1]
        normal_vis = (normal_vis + 1.0) / 2.0

        # 处理 Mask (背景置黑)
        if invalid_mask is not None:
            mask_expand = invalid_mask.unsqueeze(1).expand(-1, 3, -1, -1)
            normal_vis[mask_expand] = 0.0

        # ==========================================
        # 2. 生成深度图 (Ray-Plane Intersection)
        # ==========================================
        fx = intrinsics[:, 0, 0].view(B, 1, 1)
        fy = intrinsics[:, 1, 1].view(B, 1, 1)
        cx = intrinsics[:, 0, 2].view(B, 1, 1)
        cy = intrinsics[:, 1, 2].view(B, 1, 1)

        # 动态构建网格 (确保尺寸匹配)
        if not hasattr(self, 'grid_x') or self.grid_x.shape != (H, W):
            y_range = torch.arange(0, H, dtype=torch.float32, device=device)
            x_range = torch.arange(0, W, dtype=torch.float32, device=device)
            self.grid_y, self.grid_x = torch.meshgrid(y_range, x_range, indexing='ij')

        u_grid = self.grid_x.unsqueeze(0).expand(B, -1, -1)
        v_grid = self.grid_y.unsqueeze(0).expand(B, -1, -1)

        u_bar = (u_grid - cx) / fx
        v_bar = (v_grid - cy) / fy

        # dot = n_x * u_bar + n_y * v_bar + n_z * 1
        dot_product = (n_map[..., 0] * u_bar) + \
                      (n_map[..., 1] * v_bar) + \
                      (n_map[..., 2] * 1.0)

        # 防止除零
        dot_product[torch.abs(dot_product) < 1e-6] = 1e-6

        # depth = -d / dot
        depth_map = -d_map / dot_product
        depth_map = depth_map.abs()  # 确保正深度

        # ==========================================
        # 3. 数值截断 (Clamp)
        # ==========================================
        if depth_range is not None:
            min_d, max_d = depth_range

            # 兼容 Tensor 类型 [B] -> [B, 1, 1]
            if isinstance(min_d, torch.Tensor):
                if min_d.ndim == 1: min_d = min_d.view(-1, 1, 1)
                min_d = min_d.to(device)

            if isinstance(max_d, torch.Tensor):
                if max_d.ndim == 1: max_d = max_d.view(-1, 1, 1)
                max_d = max_d.to(device)

            depth_map = torch.clamp(depth_map, min=min_d, max=max_d)
        else:
            depth_map = torch.clamp(depth_map, max=600.0)

        fill_value = depth_map.max().detach()

        if invalid_mask is not None:
            # ✅ 使用 torch.where (非原地操作)
            # 逻辑：如果 mask 为 True，取 fill_value；否则保持 depth_map 原值
            # 这会创建一个全新的 tensor，不会破坏原来的 depth_map，梯度可以正常回传
            depth_map = torch.where(invalid_mask, fill_value, depth_map)

        depth_map = depth_map.unsqueeze(1)  # [B, 1, H, W]

        return depth_map, normal_vis

