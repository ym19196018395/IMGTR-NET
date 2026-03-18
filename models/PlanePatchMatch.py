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
    def __init__(self, plane_dim=4, hidden_dim=64):
        """
        深度可微三角传播模块
        包含: Soft Gating, Attention Aggregation, MLP Refinement
        """
        super().__init__()

        # 1. 特征提取器: 从平面参数提取特征
        # 输入: Plane(4) + Cost(1) = 5 加入非线性激活函数，使其真正成为深度网络
        self.encoder = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        # 2. 门控网络 (Gating Network): 决定传播多少信息
        # hidden*2 (self+neigh)
        # + edge(1)
        # + neighbor_cost(1) (作为邻居的自我置信度)
        # + plane_diff(4) (几何参数差异)
        # = hidden*2 + 6
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1 + 1 + 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.plane_head = nn.Linear(hidden_dim, 4)

        # 依然保持你的优秀习惯：残差网络零初始化
        # nn.init.constant_(self.plane_head.bias, 0.0)
        # nn.init.normal_(self.plane_head.weight, mean=0.0, std=0.001)

        nn.init.zeros_(self.plane_head.weight)
        nn.init.zeros_(self.plane_head.bias)

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
                edge_probs,ref_feature=None):
        """
        Args:
            current_planes: [B, N, 4]
            current_costs: [B, N, 1] (注意维度)
            neighbor_indices: [B, N, 3]
            edge_probs: [B, N, 3] (来自 EdgeHead, 0~1, 越大表示越阻断)
        return:
            final_planes:最终得到的传播平面 [B, N, 4]
        """

        B, N, _ = current_planes.shape
        device = current_planes.device

        # ==========================================
        # 1. 准备数据: 缩放 d 防止量纲爆炸
        # ==========================================
        # === 分离并缩放 d ===
        n_curr = current_planes[..., :3]
        d_curr_scaled = current_planes[..., 3:] / self.d_scale_factor
        scaled_planes = torch.cat([n_curr, d_curr_scaled], dim=-1)

        # 构造输入特征: [Plane, Cost]
        plane_feat = torch.cat([scaled_planes, current_costs], dim=-1)  # [B, N, 5]

        # 编码特征: [B, N, H]
        hidden = self.encoder(plane_feat)

        # 获取邻居的特征
        batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(-1, N, 3)
        # [B, N, 3, H]
        neighbor_hidden = hidden[batch_idx, neighbor_indices]
        # [B, N, 3, 4]
        # 计算几何差异时也必须用 scaled_planes!
        neighbor_scaled_planes = scaled_planes[batch_idx, neighbor_indices]

        # 邻居 Cost 仅作为置信度
        # current_costs: [B, N, 1] -> neighbor_costs: [B, N, 3, 1]
        neighbor_costs = current_costs[batch_idx, neighbor_indices]  # [B, N, 3, 1]

        # 你的平面参数是 (n, d)。如果两个三角形共面，它们的 (n, d) 应该极其相似。
        # 我们让网络看到这个差异，网络就能学会："如果差异很小，且中间没 Edge，那大概率是同一个大平面，权重给高点"
        # self_planes_expand: [B, N, 1, 4]
        self_scaled_expand = scaled_planes.unsqueeze(2)

        # plane_diff: [B, N, 3, 4]
        plane_diff = neighbor_scaled_planes - self_scaled_expand

        # ==========================================
        # 2. 软门控 (Soft Gating) - 替代硬截断
        # ==========================================
        # 我们希望: EdgeProb 越小 (连通), Weight 越大

        # 构造门控输入: [Self_Hidden, Neighbor_Hidden, Edge_Prob]
        # Self 扩展: [B, N, 1, H] -> [B, N, 3, H]
        self_hidden_expand = hidden.unsqueeze(2).expand(-1, -1, 3, -1)

        # Concat: [B, N, 3, H*2 + 1]
        # [Self_H, Neighbor_H, Edge_Prob, Neighbor_Cost, plane_Diff]
        gate_input = torch.cat([
            self_hidden_expand,  # 我是谁
            neighbor_hidden,  # 邻居是谁
            edge_probs.unsqueeze(-1),  # 有墙吗
            neighbor_costs,  # 邻居自信吗
            plane_diff  # 几何上我们像吗
        ], dim=-1)

        # 计算注意力权重 (Attention Weights)
        # [B, N, 3, 1]
        raw_weights = self.gate_net(gate_input)

        # 结合 EdgeProb 的物理约束 (如果 EdgeProb=1, 强制权重为0)
        # 这是一个 "Hard Constraint via Soft Mechanism"
        physics_guidance = 1.0 - edge_probs.unsqueeze(-1)  # [B, N, 3, 1]
        final_weights = raw_weights * physics_guidance


        # ==========================================
        # 3. 软传播 (Weighted Aggregation)
        # ==========================================
        # 聚合邻居平面: Sum(w_i * p_i)
        # 聚合真实的平面参数 (这里用真实值聚合，保证物理意义正确)
        neighbor_planes = current_planes[batch_idx, neighbor_indices]
        # [B, N, 3, 1] * [B, N, 3, 4] -> sum(dim=2) -> [B, N, 4]
        weighted_neighbor_planes = (final_weights * neighbor_planes).sum(dim=2)
        sum_weights = final_weights.sum(dim=2) + 1e-6

        # 混合: (Sum_W * Neighbors + 1.0 * Self) / (Sum_W + 1.0)
        # 这种混合方式保证了数值稳定性
        aggregated_planes = (weighted_neighbor_planes + current_planes) / (sum_weights + 1.0)

        # 在加入残差前，必须让被平均拉短的法向量归一化一下，防止自身平面值因为聚合过于小了
        agg_n_raw = aggregated_planes[..., :3]
        agg_d_raw = aggregated_planes[..., 3:]

        # 计算平均后法向量的真实长度 (加 clamp 防止除零)
        norm_scale = torch.norm(agg_n_raw, p=2, dim=-1, keepdim=True).clamp_min(1e-6)

        # n 和 d 必须同时除以这个长度！保持平面方程物理意义不变！
        agg_n = agg_n_raw / norm_scale
        agg_d = agg_d_raw / norm_scale

        aggregated_planes_norm = torch.cat([agg_n, agg_d], dim=-1)

        # ==========================================
        # 4. MLP Refinement (Deep Learning Part)
        # ==========================================
        # 再次缩放以喂给网络
        agg_d_scaled = agg_d / self.d_scale_factor
        scaled_agg_planes = torch.cat([agg_n, agg_d_scaled], dim=-1)

        agg_feat = torch.cat([scaled_agg_planes, current_costs], dim=-1)
        agg_hidden = self.encoder(agg_feat)

        # 网络输出的 d_delta 是相对缩小尺度的
        delta_plane_scaled = self.plane_head(agg_hidden)

        # 把 d_delta 放大回真实尺度
        delta_n = delta_plane_scaled[..., :3]
        delta_d = delta_plane_scaled[..., 3:] * self.d_scale_factor
        delta_plane = torch.cat([delta_n, delta_d], dim=-1)

        # 更新真实的、未缩放的平面
        new_planes = aggregated_planes_norm + delta_plane

        # 最终输出强制保障几何合法性
        new_n_raw = new_planes[..., :3]
        new_d_raw = new_planes[..., 3:]

        final_norm_scale = torch.norm(new_n_raw, p=2, dim=-1, keepdim=True).clamp_min(1e-6)

        new_n_final = new_n_raw / final_norm_scale
        new_d_final = new_d_raw / final_norm_scale

        final_planes = torch.cat([new_n_final, new_d_final], dim=-1)

        return final_planes

    def compute_continuity_loss(self, planes, neighbor_indices, edge_probs,aligned_midpoints_norm, intrinsics,H, W):
        """
        纯粹的连续性损失 (C0 Continuity Loss) - 包含坐标反投影

        Args:
            planes: [B, N, 4] (n, d) 当前平面参数 (Camera Space)
            neighbor_indices: [B, N, 3] 邻居索引
            edge_probs: [B, N, 3] 边缘概率
            aligned_midpoints_norm: [B, N, 3, 2] 完美对齐的边中点坐标，值域 [-1, 1]
            intrinsics: [B, 3, 3] 相机内参矩阵
        """
        B, N, _, _ = aligned_midpoints_norm.shape
        device = planes.device

        # ==========================================
        # 0. 解除归一化: [-1, 1] -> 像素坐标 [0, W-1] / [0, H-1]
        # ==========================================
        midpoints_uv = torch.zeros_like(aligned_midpoints_norm)
        midpoints_uv[..., 0] = (aligned_midpoints_norm[..., 0] + 1.0) / 2.0 * (W - 1)
        midpoints_uv[..., 1] = (aligned_midpoints_norm[..., 1] + 1.0) / 2.0 * (H - 1)

        # ==========================================
        # 1. 将像素坐标 (UV) 反投影回相机坐标 (XYZ)
        # ==========================================

        # 1.1 构造齐次像素坐标 [B, N, 3, 3] -> (u, v, 1)
        ones = torch.ones(B, N, 3, 1, device=device)
        uv_homo = torch.cat([midpoints_uv, ones], dim=-1)

        # 1.2 计算归一化射线方向 (Ray Direction)
        # K_inv * uv
        # intrinsics: [B, 3, 3] -> [B, 1, 3, 3] -> inv
        K_inv = torch.inverse(intrinsics).unsqueeze(1)  # [B, 1, 3, 3]

        # 矩阵乘法: K_inv @ uv_homo.T
        # uv_homo: [B, N, 3, 3] -> permute -> [B, N, 3, 3, 1]
        # 为了方便计算，我们把 dim=2 (3个顶点) 和 dim=1 (N个三角) 合并处理或者直接广播
        # 这里使用 Einstein Summation 更加清晰:
        # B: batch, n: num_tri, v: num_vert, i/j: matrix dims
        rays = torch.einsum('blij,bnvj->bnvi', K_inv, uv_homo)  # [B, N, 3, 3]

        # 1.3 利用平面方程求解深度 Z
        # 平面方程: n * X + d = 0  =>  n * (Z * ray) + d = 0
        # => Z * (n * ray) = -d
        # => Z = -d / (n * ray)

        # planes: [B, N, 4] -> n:[B, N, 1, 3], d:[B, N, 1, 1]
        n_curr = planes[:, :, :3].unsqueeze(2)  # [B, N, 1, 3]
        d_curr = planes[:, :, 3:].unsqueeze(2)  # [B, N, 1, 1]

        # 分母: n * ray
        denom = torch.sum(n_curr * rays, dim=-1, keepdim=True)  # [B, N, 3, 1]

        # 防止除零 (加上极小值)
        denom = torch.where(torch.abs(denom) < 1e-6, torch.tensor(1e-6, device=device), denom)

        depth = -d_curr / denom  # [B, N, 3, 1] (每个顶点的深度)

        # 1.4 恢复相机坐标 XYZ
        # XYZ = depth * ray
        mid_points = rays * depth  # [B, N, 3, 3] (Camera Space XYZ)

        # ==========================================
        # 2. 计算连续性损失 (Discontinuity Loss)
        # ==========================================

        # 获取邻居平面
        batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(-1, N, 3)
        neighbor_planes = planes[batch_idx, neighbor_indices]  # [B, N, 3, 4]

        n_neigh = neighbor_planes[..., :3]  # [B, N, 3, 3]
        d_neigh = neighbor_planes[..., 3:]  # [B, N, 3, 1]

        # 这里的逻辑：
        # 我们用当前的平面恢复了 3D 点 (mid_points)，这说明 mid_points 一定完全满足当前平面方程。
        # 现在的 Loss 是看这些点是否 *也满足* 邻居的平面方程。

        # 代入邻居平面方程: | n_neigh * X_mid + d_neigh |
        # mid_points: [B, N, 3, 3]
        dot_val = torch.sum(n_neigh * mid_points, dim=-1, keepdim=True)  # [B, N, 3, 1]

        # 距离误差
        dist_error = torch.abs(dot_val + d_neigh)  # [B, N, 3, 1]
        dist_error = dist_error.squeeze(-1)  # [B, N, 3]

        # ==========================================
        # 3. Edge 门控与 Loss 聚合
        # ==========================================

        # 权重: EdgeProb越大(边界)，权重越小
        continuity_weight = torch.exp(-edge_probs)  # [B, N, 3]

        # 主 Loss，连续性损失 (受 EdgeProb 控制)
        loss = (dist_error * continuity_weight).mean()

        # 正则项: 防止 EdgeProb 偷懒全输出 1
        sparsity_loss = edge_probs.mean()

        return loss,sparsity_loss

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
    def __init__(self, num_hypotheses=3, G=8, feat_channels=16,propagator_iter=2):
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


    def forward(self,fitter_module, depth_stage2, tri_infos, ref_feature, src_features,
                ref_proj, src_projs, intrinsics_s1,depth_min, depth_max, view_weights=None,
                neighbor_indices_batched=None):
        """
        Args:
            fitter_module: 实例化好的 DensePlaneFitter 对象
            depth_stage2: [B, 1, H/4, W/4] (Stage 2 深度)
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

        # ==========================================
        # 2. 拟合与生成 (Fitting & Generation)，根据stage2预测深度来拟合生成假设平面
        # ==========================================
        # 使用优化后的 get_plane_hypotheses 直接得到 [B, N, K, 4]
        # 其中 [:, :, 0, :] 是原始 SVD 拟合结果,[:, :, 1, :] 是前向平行, [:, :, 2-K:, :] 是随机扰动结果
        # todo:暂时不需要假设了，直接用拟合的结果,用了假设之后导致平面传播的一塌糊涂，很失败
        hypotheses = self.fitter.get_plane_hypotheses(
            depth_stage2=depth_stage2,
            tri_id_map=tri_id_map,
            intrinsics_s1=ref_intrinsics,
            max_num_triangles=max_tri_num
        )  # Output: [B, N_tri, K, 4]

        # [B, N_tri, K, 4] -> [B, N_tri, 4] (取第0个假设,最佳平面的前3通道)
        # 进行一个可视化看看效果，拟合的初始平面
        before_best_guess_planes = hypotheses[:, :, 0, :]  # [B, N_tri, 4]

        # ==========================================
        # 3. 广播 (Broadcasting: Triangle -> Pixel)
        # ==========================================

        # 将三角形级的假设映射到像素级 pixel_hypotheses: [B, H, W, K, 4]
        pixel_hypotheses = self.map_tri_to_pixel(hypotheses, tri_id_map, H, W)

        current_hypotheses = pixel_hypotheses

        # ==========================================
        # 4. 代价计算 (Cost Computation)
        # ==========================================
        # 计算所有假设的代价,计算了每个像素点的代价，但是后面会转化成一个个三角形所以不影响
        # costs: [B, H, W, K]
        pixel_costs = self.compute_costs(
            ref_feature, src_features,
            ref_proj, src_projs,
            current_hypotheses,
            None,
            ref_intrinsic=ref_intrinsics
        )

        # 🔥 修改：基于三角形聚合 Cost 并选择 (Triangle-wise Selection)
        # 不在基于单个像素了，保证一个整体的出现

        # 1. 聚合 Cost: [B, H, W, K] -> [B, N_tri, K]
        tri_costs = self.aggregate_costs_per_triangle(pixel_costs, tri_id_map, max_tri_num)

        # 2. 三角形级选择 (Triangle-wise Argmin)
        # best_tri_idx: [B, N_tri] (每个三角形选择了第几个假设 0~K-1)
        best_tri_idx = torch.argmin(tri_costs, dim=2)

        # 3. 提取最佳平面参数
        # hypotheses: [B, N_tri, K, 4]
        # 我们需要根据 best_tri_idx 从 K 个假设中 Gather 出最好的那个

        # 构造 Gather Index: [B, N_tri, 1, 4]
        gather_idx = best_tri_idx.view(B, max_tri_num, 1, 1).expand(-1, -1, 1, 4)

        # best_planes: [B, N_tri, 4] (更新后的三角形平面)
        # ym-debug 为了验证是否是聚合导致测试出问题
        # best_planes = torch.gather(hypotheses, 2, gather_idx).squeeze(2)

        best_planes = before_best_guess_planes

        # 渲染没有经过传播的 深度图和法向量图 后续用
        no_prop_depth, no_propa_normal = visualizer.render_from_planes(
            best_planes.detach(),  # 注意 detach，不传导梯度
            tri_id_map,
            ref_intrinsics,
            depth_range=(depth_min, depth_max)
        )

        # ==========================================
        # 5. 通过边预测头得到边的断裂概率
        # ==========================================

        # 运行 EdgeHead 预测边缘  ym-need-modify 暂时不给其放梯度，
        edge_alphas = self.edge_head(
            feat=ref_feature.detach(),  # [B, C, H, W]
            tri_infos=tri_infos,
            tri_planes=best_planes.detach(),  # [B, N_max, 4] 包含法向和距离
            intrinsics=ref_intrinsics  # [B, 3, 3] 相机内参
        )

        # 转换为三角形级别格式 [B, N_max, 3]
        edge_probs_tensor, aligned_midpoints_norm = convert_edge_features_to_tri_format(
            edge_alphas, tri_infos, max_tri_num, device
        )

        # ==========================================
        # 6. 传播 (Propagation)
        # ==========================================
        current_planes = best_planes
        # torch.min(dim=2) 会同时返回最小值(values)和对应索引(indices)
        current_costs = tri_costs.min(dim=2)[0].unsqueeze(-1) # [B, N, 1]

        # === 暴力测试：强行抹除所有边缘，让传播彻底放飞自我 ===
        # edge_probs_for_prop = torch.zeros_like(edge_probs_tensor)

        for iter_idx in range(self.propagator_iter):
            # 6.1 传播 得到新平面
            new_planes = self.propagator(
                current_planes=current_planes,
                current_costs=current_costs,
                neighbor_indices=neighbor_indices_batched,
                edge_probs=edge_probs_tensor.detach() # 暂时不让边预测头互相影响
            )

            # 6.2 重新评估新平面的代价 (极其重要：在此处设立绝对的梯度防火墙！)
            # 使用 torch.no_grad() 彻底阻断这一整块代码的梯度图构建
            with torch.no_grad():
                # 扩展维度以适配 compute_costs 接口: [B, N, 1, 4] (K=1)
                new_planes_k1 = new_planes.unsqueeze(2)

                # 将三角形平面广播到像素级
                pixel_hypo = self.map_tri_to_pixel(new_planes_k1, tri_id_map, H, W)

                # 计算像素代价
                pixel_costs_new = self.compute_costs(
                    ref_feature, src_features, ref_proj, src_projs,
                    pixel_hypo, None, ref_intrinsics, is_debug=False
                )

                # 重新聚合成三角形级代价 -> [B, N, 1]
                tri_costs_new = self.aggregate_costs_per_triangle(pixel_costs_new, tri_id_map, max_tri_num)

            # 6.3 状态更新，进入下一次迭代
            # 注意：因为是 Learned Propagator，我们直接相信它的更新（像 RNN 一样），而不进行 Argmin 判断
            current_planes = new_planes
            current_costs = tri_costs_new

        final_planes = current_planes

        # ==========================================
        # 7. 渲染与输出和计算损失
        # ==========================================

        continuity_loss, continuity_s_loss = 0,0
        # continuity_loss,continuity_s_loss = self.propagator.compute_continuity_loss(
        #     planes=final_planes,
        #     neighbor_indices=neighbor_indices_batched,
        #     edge_probs=edge_probs_tensor.detach(),  # 不让其受影响
        #     aligned_midpoints_norm=aligned_midpoints_norm,
        #     intrinsics=ref_intrinsics,
        #     H=H,W=W)

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
            final_planes.detach(),
            tri_id_map,
            ref_intrinsics,
            depth_range=(depth_min, depth_max))

        # 组装输出 (PatchMatchNet 通常需要 List 格式)
        # todo:一个是刚拟合完毕的，另外一个是经过传播处理的
        depth_samples = [no_prop_depth,final_depth]  # List[[B, 2, H, W]]
        # 法向量
        normal_samples= [no_propa_normal,final_normal]

        # Score (Confidence) = -min_cost
        score = -torch.min(pixel_costs, dim=3)[0].unsqueeze(1)  # [B, 1, H, W]

        return (depth_samples, score,
                view_weights,
                normal_samples,
                final_planes,
                edge_alphas,
                continuity_loss,continuity_s_loss)

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
            # Similarity: [B*K, G, H, W]
            similarity = (warped_src_grouped * ref_feat_expanded).mean(dim=2)

            del warped_src, warped_src_grouped  # 释放显存

            # ================== debug专用 ==================================
            # if i == 0:
            #     # similarity: [B*K, G, H, W] -> Mean over G -> [B*K, 1, H, W]
            #     raw_sim = similarity.mean(dim=1).view(B, K, H, W)
            #
            #     raw_svd = raw_sim[:, 0, ...].mean().item()
            #     raw_fp = raw_sim[:, 1, ...].mean().item()
            #     raw_rnd = raw_sim[:, 2, ...].mean().item()
            #     print(f"✅ [Check Input] Raw SVD: {raw_svd:.4f} | FP: {raw_fp:.4f} | Rnd: {raw_rnd:.4f}")

            # todo:暂时不用学习型
            # similarity 是 [B*K, G, H, W]，我们在 G 维度取平均
            score_i = similarity.mean(dim=1, keepdim=True)  # [B*K, 1, H, W]


            # --- E. 计算代价 (Cost Regression) --- ❌
            # similarity_net: [B*K, G, H, W] -> [B*K, 1, H, W]
            # cost 越小越好，correlation 越大越好，所以取负

            # score_i = self.similarity_net(similarity)
            cost_i = -score_i  # [B*K, 1, H, W]

            # ============debug专用========================
            if i == 0 and is_debug:  # 只看第一个源视图
                # Reshape 回 [B, K, 1, H, W]
                temp_score = score_i.view(B, K, 1, H, W)
                score_0 = temp_score[:, 0, ...].mean().item()
                score_1 = temp_score[:, 1, ...].mean().item()
                score_2 = temp_score[:, 2, ...].mean().item()
                score_3 = temp_score[:, 3, ...].mean().item()
                print(f"\n[Debug] Raw Similarity Mean | Hypo 0 (SVD): {score_0:.4f} | Hypo 1 (FP): {score_1:.4f}"
                      f"Hypo 2 (SVD): {score_2:.4f} | Hypo 3 (FP): {score_3:.4f}")

            # ========== 特征扭曲debug =====================================
            # if i == 0 :  # 你需要自己加个计数器或者只跑一个 batch
            #     import torchvision.utils as vutils
            #     # Reshape 为 [B, K, C, H, W]
            #     debug_warp = warped_src.view(B, K, C, H, W)
            #
            #     # 取 Hypo 0 (SVD) 和 Hypo 1 (FP)
            #     # 归一化到 0-1 以便显示
            #     img_0 = debug_warp[0, 0, :3].detach().cpu()  # 取前3个通道当RGB
            #     img_1 = debug_warp[0, 2, :3].detach().cpu()
            #
            #     # 归一化
            #     img_0 = (img_0 - img_0.min()) / (img_0.max() - img_0.min())
            #     img_1 = (img_1 - img_1.min()) / (img_1.max() - img_1.min())
            #
            #     vutils.save_image(img_0, "debug_warp_svd.png")
            #     vutils.save_image(img_1, "debug_warp_fp.png")
            #     print("📸 已保存 debug_warp_svd.png 和 debug_warp_fp.png")

            # --- F. 应用视图权重 (View Weights) ---
            if view_weights is not None:
                # view_weights: [B, Nview-1, H, W] -> 取第 i 个 -> [B, 1, H, W]
                # 扩展到 K
                vw = view_weights[:, i:i + 1, :, :].unsqueeze(1).repeat(1, K, 1, 1, 1).view(B * K, 1, H, W)
                cost_i = cost_i * vw

            total_cost = total_cost + cost_i

        # 还原形状 [B*K, 1, H, W] -> [B, K, H, W] -> [B, H, W, K]
        total_cost = total_cost.view(B, K, H, W).permute(0, 2, 3, 1)

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
    def __init__(self, height_s1, width_s1, device, num_hypotheses=3, perturbation_range=0.05, depth_max=None):
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
        self.depth_max = depth_max # 这是对特别小的三角形进行的一个保底处理

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
        valid_fit_mask = (counts >= 3)

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
        # 3.5 异常三角平面处理 (Fallbacks) - 修复梯度报错版
        # 我们不使用 if invalid_tris.any() 来做 in-place 修改,而是始终使用 torch.where 生成一个新的 tensor
        # ==========================================================
        invalid_tris = (counts < 3)

        # if invalid_tris.any():
        #     # 1. 法向量设为指向相机 (0, 0, -1) 或者 (0, 0, 1) 取决于你的坐标系定义
        #     # 假设: n=[0,0,-1], d=depth_max -> -1*z + d = 0 -> z = d
        #     normals[invalid_tris] = torch.tensor([0.0, 0.0, -1.0], device=self.device)
        #
        #     # 2. d 值直接设为最大深度 (将平面推到最远背景处)
        #     # 这样它的 Cost 会非常高，第一轮传播就会被邻居覆盖掉
        #     d_vals[invalid_tris, 0] = min(self.depth_max)

        # 1. 准备 Fallback 的值
        # 扩展成和 normals 一样的形状
        fallback_n = torch.tensor([0.0, 0.0, -1.0], device=device, dtype=normals.dtype)
        fallback_n = fallback_n.view(1, 3).expand_as(normals)

        # 2. 准备 Mask
        # invalid_tris 是 [Total], 需要变成 [Total, 3] 才能用于 where
        mask_n = invalid_tris.unsqueeze(1).expand_as(normals)

        # 3. ✅ 使用 torch.where (Out-of-place)
        # 如果 mask 为 True，取 fallback，否则保持原样
        # 这会创建一个全新的 tensor，旧的 normals 保留用于梯度计算
        normals = torch.where(mask_n, fallback_n, normals)

        # --- 处理 d_vals ---
        ## 1. 准备 Fallback d
        max_d = min(self.depth_max) # 给一个保底值
        fallback_d = torch.full_like(d_vals, max_d)

        ## 2. 准备 Mask [Total, 1]
        mask_d = invalid_tris.unsqueeze(1)

        ## 3. ✅ 使用 torch.where
        d_vals = torch.where(mask_d, fallback_d, d_vals)


        # [Hypothesis 0] 原始拟合结果 [B, N, 1, 4]
        hypo_0 = torch.cat([normals, d_vals], dim=1).view(B, max_num_triangles, 1, 4)

        # ==========================================
        # 4 生成其他假设 (Hypothesis Expansion)
        # ==========================================
        hypo_list = [hypo_0]

        # ==========[Hypothesis 1] 前向平行 (Fronto-Parallel)=====
        # 利用现成的 centroids[:, 2] (平均深度)

        # mean_depth = centroids[:, 2].view(B, max_num_triangles, 1, 1)
        #
        # normal_fp = torch.zeros((B, max_num_triangles, 1, 3), device=device)
        # normal_fp[..., 2] = -1.0
        # d_fp = mean_depth  # d = z
        #
        # hypo_fp = torch.cat([normal_fp, d_fp], dim=-1)  # [B, N, 1, 4]
        #
        # # todo:用了前向平行平面，如果不用需要将这里注释，并且self.K - 2 变为 self.K - 1
        # hypo_list.append(hypo_fp)

        # ==========[Hypothesis 2+] 随机扰动 (Vectorized Jitter)======
        num_random = self.K - 1
        if num_random > 0:
            # 扩展基础平面 [B, N, 1, 4] -> [B, N, num_rnd, 4]
            base_n = hypo_0[..., :3].expand(-1, -1, num_random, -1)
            base_d = hypo_0[..., 3:].expand(-1, -1, num_random, -1)

            # 生成噪声
            rand_n = (torch.rand_like(base_n) - 0.5) * self.noise_scale * 2.0
            # d 的扰动范围需要大一点
            rand_d = (torch.rand_like(base_d) - 0.5) * self.noise_scale

            # 应用噪声
            new_n = F.normalize(base_n + rand_n, dim=-1)
            new_d = base_d + rand_d

            hypo_random = torch.cat([new_n, new_d], dim=-1)
            hypo_list.append(hypo_random)

        # 最终拼接 [B, N, K, 4]
        hypotheses = torch.cat(hypo_list, dim=2)

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

        dot_product[torch.abs(dot_product) < 1e-6] = 1e-6

        depth_map = -d_map / dot_product

        # 解释：无论是平面法向反了(n -> -n)，还是点积反了，
        # 我们都知道物体肯定在相机前面，所以直接要由距离的模长。
        depth_map = depth_map.abs()

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

            # 现在无论是 float 还是 Tensor[B,1,1]，clamp 都能处理
            depth_map = torch.clamp(depth_map, min=min_d, max=max_d)
        else:
            # 即使没有给定范围，也要限制一下无穷大
            depth_map = torch.clamp(depth_map, max=600.0)

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

