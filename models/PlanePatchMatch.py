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

class DoubleDecoupledTrianglePropagator(nn.Module):
    def __init__(self, plane_dim=4, hidden_dim=64,feature_dim=16):
        """
            双解耦三角形几何拓扑网络 (Double-Decoupled Triangle Propagator)
            前向完全双解耦：
            - 深度流 (GNN_Z): 只认空间几何对齐与匹配代价，计算专属传播系数
            - 法向流 (GNN_N): 只认高频特征流形与表面置信度，计算专属平滑系数
            出口端合并：复活释放 Z 残差头，通过质心射线投影完美反推闭环
        """
        super().__init__()
        self.feature_dim = feature_dim

        # 1. 基础多模态特征编码器 (Plane + Cost)
        self.init_encoder = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        # 2. 精修后特征感知编码器 (只读纯几何，屏蔽脏Cost)
        self.refine_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        # 3. 🎯 核心升级：自身权重门控解耦拆分为双通道，彻底隔离 Z 与 N 的语义干扰
        self.self_gate_net_z = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.self_gate_net_n = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 偏置保守初始化，确保训练初期信任初始拟合大面
        nn.init.constant_(self.self_gate_net_z[2].bias, 2.0)
        nn.init.zeros_(self.self_gate_net_z[2].weight)
        nn.init.constant_(self.self_gate_net_n[2].bias, 2.0)
        nn.init.zeros_(self.self_gate_net_n[2].weight)

        # 4. 【精准修改点：矩阵扩容】深度传播专属门控 (GNN_Z): 全量多模态特征总线接入
        # 输入维度完美解锁: self_hidden(H) + neighbor_hidden(H) + plane_diff(4) + edge_prob(1) + neighbor_cost(1) + feat_dist(1) + self_W(1) + neighbor_W(1) = 2H + 9
        self.gate_net_z = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 9, hidden_dim),  # 👈 从 2H + 7 扩容为 2H + 9
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

        # 5. 【精准修改点：矩阵扩容】法向传播专属门控 (GNN_N): 全量多模态特征总线接入
        # 输入维度完美解锁: self_hidden(H) + neighbor_hidden(H) + plane_diff(4) + edge_prob(1) + neighbor_cost(1) + feat_dist(1) + self_W(1) + neighbor_W(1) = 2H + 9
        self.gate_net_n = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 9, hidden_dim),  # 👈 从 2H + 7 扩容为 2H + 9
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

        # 6. 复活的 Z 残差微调头 (The Magic Polisher)
        head_in_dim = hidden_dim * 2 + 1 + feature_dim
        self.plane_head = nn.Sequential(
            nn.Linear(head_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)  # 输出解耦的绝对尺度位移 delta_Z
        )

        # 7. 【修复记忆断流：矩阵扩容】可学习内生置信度刷新头 (Confidence Predict Head)
        # 输入特征完美解锁: init_hidden(H维) + current_costs(1维) + delta_cost(1维) + 上一步W_plane_tri先验(1维) + W_raw_anchor(1维) = H + 4
        self.confidence_predict_head = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim // 2),  # 👈 扩容为 + 4，支持物理锚点并网
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.feat_dropout = nn.Dropout(p=0.15)
        nn.init.constant_(self.plane_head[2].bias, 0.0)
        nn.init.normal_(self.plane_head[2].weight, mean=0.0, std=0.01)

    def forward(self, current_planes, current_costs, neighbor_indices,
                edge_probs, rays_centroids, depth_max,prev_costs,W_plane_tri,
                pixel_counts=None, ref_feature=None, centroids_norm=None, temperature=0.2, 
                W_raw_anchor=None, cross_costs=None):
        """
        前向双解耦核心计算图流动
        Args:
            current_planes:    [B, N, 4] 三角面片参数 (nx, ny, nz, d)
            current_costs:     [B, N, 1] 原始拟合光度匹配代价
            neighbor_indices:  [B, N, 3] 拓扑邻居矩阵索引
            edge_probs:        [B, N, 3] 物理断裂概率
            rays_centroids:    [B, N, 3] 三角形中心视锥射线
            depth_max:         场景最大深度裁剪边界
            prev_costs:        [B, N, 1] 💾 核心增量：上一轮迭代的历史光度代价快照 (用于计算变异度 delta_cost)
            W_plane_tri:       核心解耦注入：由外部过滤系统计算出的当前三角形平面置信度 [B, N, 1]
            pixel_counts:      每个三角形包含的密集像素计数
            ref_feature:       图像级 Stage 1 语义感知特征
            centroids_norm:    归一化三角形质心
            temperature:       可微退火控温系数 (从1.0渐变到0.1，逼近硬性选择)
            W_raw_anchor:      刚性追加尾部入参
        Returns:
            final_planes:      传递精修后的下一代面片方程 [B, N, 4]
            W_plane_tri_learn: 【Learn轨】带完整求导梯度的置信度张量 [B, N, 1]，送入 L_conf 进行硬核监督
        """
        B, N, _ = current_planes.shape
        device = current_planes.device
        F_curr = None

        # =====================================================================
        # 1. 特征域归一化与隐空间映射
        # =====================================================================
        if isinstance(depth_max, torch.Tensor):
            d_max_val = depth_max.view(-1, 1, 1)
        else:
            d_max_val = float(depth_max)

        n_curr = current_planes[..., :3]
        d_curr_scaled = current_planes[..., 3:] / d_max_val
        scaled_planes = torch.cat([n_curr, d_curr_scaled], dim=-1)  # [B, N, 4]

        plane_feat = torch.cat([scaled_planes, current_costs], dim=-1)
        init_hidden = self.init_encoder(plane_feat)  # [B, N, H]

        # =====================================================================
        # 2. 图像特征感知采样 (采样中心高频梯度)
        # =====================================================================
        feat_dist = torch.zeros((B, N, 3, 1), device=device)
        if ref_feature is not None and centroids_norm is not None:
            grid = centroids_norm.view(B, N, 1, 2)
            F_curr = F.grid_sample(ref_feature, grid, mode='bilinear', align_corners=True).squeeze(-1).permute(0, 2, 1)
            F_curr = F.normalize(F_curr, p=2, dim=-1)

            batch_idx_exp = torch.arange(B, device=device).view(B, 1, 1).expand(-1, N, 3)
            F_neighbor = F_curr[batch_idx_exp, neighbor_indices]
            F_curr_exp = F_curr.unsqueeze(2)
            feat_dist = ((F_curr_exp - F_neighbor) ** 2).sum(dim=-1, keepdim=True)  # [B, N, 3, 1]


        # =====================================================================
        # 3. 收集图网络邻居异构上下文
        # =====================================================================
        batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(-1, N, 3)

        neighbor_hidden = init_hidden[batch_idx, neighbor_indices]  # [B, N, 3, H]
        neighbor_scaled_planes = scaled_planes[batch_idx, neighbor_indices]  # [B, N, 3, 4]
        
        # [修改点]：如果提供了 cross_costs，则使用真实的客场代价！
        if cross_costs is not None:
            neighbor_costs = cross_costs.unsqueeze(-1) # [B, N, 3, 1]
        else:
            neighbor_costs = current_costs[batch_idx, neighbor_indices]  # [B, N, 3, 1]


        self_scaled_expand = scaled_planes.unsqueeze(2)  # [B, N, 1, 4]
        plane_diff = neighbor_scaled_planes - self_scaled_expand  # [B, N, 3, 4]

        # =====================================================================
        # 时空门控自愈置信度循环更新总线 (置信度图神经网络传导)
        # =====================================================================

        # 💾 计算 Transient 动态匹配代价变异量量纲
        if prev_costs is not None:
            delta_cost = torch.abs(current_costs - prev_costs)
        else:
            delta_cost = torch.zeros_like(current_costs)

        if W_plane_tri is None:
            # 实验一的优良血统：第 0 轮冷启动，由于法向尚未熨平，强制沿用初始真理铁锚自保
            W_plane_tri_learn = W_raw_anchor.clone()
            W_plane_tri_reg = W_raw_anchor.clone()
        else:
            if W_plane_tri.dim() == 2:
                W_plane_tri = W_plane_tri.unsqueeze(-1)  # 强制升维至 [B, N, 1]
                
            # 物理锚点维度对齐与保底防御
            if W_raw_anchor is not None:
                if W_raw_anchor.dim() == 2:
                    W_raw_anchor_in = W_raw_anchor.unsqueeze(-1)
                else:
                    W_raw_anchor_in = W_raw_anchor
            else:
                W_raw_anchor_in = torch.ones_like(W_plane_tri)

            # 串联宏观先验特征、实时匹配代价、收敛速度探测器、上一步W_plane_tri先验、以及物理冷启动初始置信度
            # 此时特征量纲严格对齐 [B, N, H + 4]，完美接入扩容后的 Linear 矩阵！
            # 💡 【核心阻断：阻断梯度倒流】使用 init_hidden.detach() 切断置信度 Loss 对前向平面参数的扭曲反噬
            W_update_input = torch.cat([init_hidden.detach(), current_costs, delta_cost, W_plane_tri, W_raw_anchor_in], dim=-1)
            W_plane_tri_raw = self.confidence_predict_head(W_update_input)

            # 🎯 【残差修正模式】：以上一轮置信度为基准，神经网络只学习 [-0.3, 0.3] 内的修正量，实现时序累加
            delta_W = torch.tanh(W_plane_tri_raw) * 0.3
            W_plane_tri_learn = torch.clamp(W_plane_tri + delta_W, min=1e-6, max=1.0)  # 🧾 【Learn轨】
            W_plane_tri_reg = W_plane_tri_learn.detach()  # 🛡️ 【Reg轨】
  
        # =====================================================================
        # 4. 双重完全解耦门控计算 (自卫分配与邻居流形熔断)
        # =====================================================================
        # 还原 self_gate_input 维度 (依然是 hidden_dim + 1)，不改网络结构
        self_gate_input = torch.cat([init_hidden, current_costs], dim=-1)

        # 4.a 拆分为两个独立的自身 Logit 预测通路
        self_weight_logits_z = self.self_gate_net_z(self_gate_input)  # [B, N, 1]
        self_weight_logits_n = self.self_gate_net_n(self_gate_input)  # [B, N, 1]

        # 物理面积过小拦截
        if pixel_counts is not None:
            is_tiny = (pixel_counts < 4).unsqueeze(-1)
            self_weight_logits_z = self_weight_logits_z.masked_fill(is_tiny, -1e9)
            self_weight_logits_n = self_weight_logits_n.masked_fill(is_tiny, -1e9)

        self_hidden_expand = init_hidden.unsqueeze(2).expand(-1, -1, 3, -1)

        # 提取自身置信度与邻居置信度做门控输入
        W_self_expand = W_plane_tri_reg.unsqueeze(2).expand(-1, -1, 3, -1)
        neighbor_W_conf = W_plane_tri_reg[batch_idx, neighbor_indices]

        # 4.b 深度通道邻居 Logits 预测
        # 将 [隐特征、完整的 4维 plane_diff、断裂概率、邻居代价、高频视觉距离、自身置信度、邻居置信度] 无损打包
        # 形状完全契合扩容后的 [B, N, 3, 2H + 9]
        gate_input_z = torch.cat([
            self_hidden_expand, neighbor_hidden,
            plane_diff, edge_probs.unsqueeze(-1), neighbor_costs, feat_dist,
            W_self_expand, neighbor_W_conf
        ], dim=-1)
        neighbor_logits_z = self.gate_net_z(gate_input_z)  # [B, N, 3, 1]

        # 4.c 🔥 【精准更替：满血版法向门控输入】
        # 让法向流和深度流享有完全同等的“知情权”，同样通过全模态总线过滤边缘与曲面拉扯
        # 形状完全契合扩容后的 [B, N, 3, 2H + 9]
        gate_input_n = torch.cat([
            self_hidden_expand, neighbor_hidden,
            plane_diff, edge_probs.unsqueeze(-1), neighbor_costs, feat_dist,
            W_self_expand, neighbor_W_conf
        ], dim=-1)
        neighbor_logits_n = self.gate_net_n(gate_input_n)  # [B, N, 3, 1]

        # # 4.d 降下流形断路器：拒绝接受低置信度曲面邻居的代数污染
        # is_curved_neighbor = (neighbor_W_conf < 0.2)
        # neighbor_logits_z = neighbor_logits_z.masked_fill(is_curved_neighbor, -1e9)
        # neighbor_logits_n = neighbor_logits_n.masked_fill(is_curved_neighbor, -1e9)
        #
        # # 4.e 🎯 修复白墙强力劫持：白墙刚性霸权【只能恩赐给法向流】，深度流保留宏观平滑传导权
        # is_perfect_wall = (W_plane_tri > 0.85).float()
        # self_weight_logits_n = self_weight_logits_n + (is_perfect_wall * 8.0)

        # =====================================================================
        # 5. 🛡️ 核心锁定二：数值稳定版可微退火采样软路由 (彻底根除 inf/NaN 崩溃)
        # =====================================================================
        # 🎯 核心修复：强制将自身 Logits 升维成 4 维 [B, N, 1, 1]，确保与 neighbor_logits 的 4 维空间绝对右对齐！
        # 我们架设保底减速带，强行维持 Softmax 消息流在测试集上的高流动性，消灭面片微观碎裂化
        # 与外层余弦退火 [0.55, 1.0] 对齐，避免硬地板 0.5 抵消尾盘控温
        safe_temp = max(float(temperature), 0.55)

        # 强制将自身 Logits 升维成 4 维 [B, N, 1, 1]，确保空间绝对右对齐
        self_logits_z_4d = self_weight_logits_z.unsqueeze(-1)
        self_logits_n_4d = self_weight_logits_n.unsqueeze(-1)

        # 5.a 稳定化深度流 Z 轴分配 (全程套用平滑安全控温)
        max_logits_z = torch.max(self_logits_z_4d, neighbor_logits_z.max(dim=2, keepdim=True).values)
        exp_self_z = torch.exp((self_logits_z_4d - max_logits_z) / safe_temp)
        exp_neigh_z = torch.exp((neighbor_logits_z - max_logits_z) / safe_temp)

        # 先做原生累加和判断，优先且正确地执行 1e-5 图断点熔断
        raw_sum_z = exp_self_z + exp_neigh_z.sum(dim=2, keepdim=True)
        is_dead_end_z = (raw_sum_z < 1e-5).squeeze(-1).squeeze(-1)
        total_w_z_safe = raw_sum_z + 1e-6

        self_weight_z = exp_self_z / total_w_z_safe
        neighbor_weights_z = exp_neigh_z / total_w_z_safe

        # 5.b 稳定化法向流 N 轴分配
        max_logits_n = torch.max(self_logits_n_4d, neighbor_logits_n.max(dim=2, keepdim=True).values)
        exp_self_n = torch.exp((self_logits_n_4d - max_logits_n) / safe_temp)
        exp_neigh_n = torch.exp((neighbor_logits_n - max_logits_n) / safe_temp)

        raw_sum_n = exp_self_n + exp_neigh_n.sum(dim=2, keepdim=True)
        is_dead_end_n = (raw_sum_n < 1e-5).squeeze(-1).squeeze(-1)
        total_w_n_safe = raw_sum_n + 1e-6

        self_weight_n = exp_self_n / total_w_n_safe
        neighbor_weights_n = exp_neigh_n / total_w_n_safe

        # =====================================================================
        # 6. 完全解耦物理流形重组
        # =====================================================================
        # 将四维的单平面权重降维回三维 [B, N, 1]，完美匹配 3D 物理乘法
        self_weight_z_3d = self_weight_z.squeeze(-1)
        neighbor_weights_z_3d = neighbor_weights_z.squeeze(-1)  # [B, N, 3]

        self_weight_n_3d = self_weight_n.squeeze(-1)
        neighbor_weights_n_3d = neighbor_weights_n.squeeze(-1)  # [B, N, 3]

        d_curr = current_planes[..., 3:]
        neighbor_planes = current_planes[batch_idx, neighbor_indices]
        n_neigh = neighbor_planes[..., :3]
        d_neigh = neighbor_planes[..., 3:]

        # 6.a 熨平切空间朝向场 (法向独立图传播)
        agg_n_raw = (neighbor_weights_n * n_neigh).sum(dim=2) + (self_weight_n_3d * n_curr)
        agg_n = F.normalize(agg_n_raw, p=2, dim=-1)

        # 6.b 物理计算当前几何绝对深度
        denom_curr = (n_curr * rays_centroids).sum(dim=-1, keepdim=True)
        denom_curr_safe = torch.where(denom_curr.abs() < 1e-4, torch.sign(denom_curr + 1e-10) * 1e-4, denom_curr)
        Z_curr = (-d_curr / denom_curr_safe).abs()
        # 核心修复：将 min 裁剪边界也张量化，使其与 max=d_max_val 保持绝对同构对齐
        min_bound_tensor = torch.full_like(d_max_val, 1e-2)
        Z_curr = torch.clamp(Z_curr, min=min_bound_tensor, max=d_max_val)

        # 6.c 物理投影邻居深度场至当前三角形视锥射线
        rays_exp = rays_centroids.unsqueeze(2)
        denom_neigh = (n_neigh * rays_exp).sum(dim=-1, keepdim=True)
        denom_neigh_safe = torch.where(denom_neigh.abs() < 1e-4, torch.sign(denom_neigh + 1e-10) * 1e-4, denom_neigh)
        Z_neigh_to_me_raw = -d_neigh / denom_neigh_safe

        # 🎯 核心锁定三：废除危险的 .abs()，对相机后方负深度及超视锥鬼影强制用自身 Z_curr 实施拦截隔离
        d_max_aligned = d_max_val.view(B, 1, 1, 1).expand(B, N, 3, 1)
        min_bound_aligned = min_bound_tensor.view(B, 1, 1, 1).expand(B, N, 3, 1)

        is_invalid_proj = (Z_neigh_to_me_raw <= min_bound_aligned) | \
                          (Z_neigh_to_me_raw > d_max_aligned) | \
                          (denom_neigh.abs() < 1e-3)

        Z_neigh_to_me = torch.where(is_invalid_proj, Z_curr.unsqueeze(2), Z_neigh_to_me_raw)

        # 深度场独立融合 (深度独立图传播)
        Z_agg = (neighbor_weights_z * Z_neigh_to_me).sum(dim=2) + (self_weight_z_3d * Z_curr)

        # 级联隔离保底
        agg_n = torch.where(is_dead_end_n.unsqueeze(-1), n_curr, agg_n)
        Z_agg = torch.where(is_dead_end_z.unsqueeze(-1), Z_curr, Z_agg)

        # =====================================================================
        # 7. 残差头恢复释放与出口刚性反推打包
        # =====================================================================
        Z_agg_scaled = Z_agg / d_max_val
        refine_input_planes = torch.cat([agg_n, Z_agg_scaled], dim=-1)
        refine_hidden = self.refine_encoder(refine_input_planes)

        if F_curr is not None:
            F_curr_processed = self.feat_dropout(F_curr)
        else:
            F_curr_processed = torch.zeros((B, N, self.feature_dim), device=device)

        refine_input = torch.cat([refine_hidden, init_hidden, current_costs, F_curr_processed], dim=-1)
        delta_Z_raw = self.plane_head(refine_input)

        max_shift_Z = d_max_val * 0.05
        delta_Z = torch.tanh(delta_Z_raw) * max_shift_Z

        # 执行刚性约束闭环更新
        n_new = agg_n
        Z_new = (Z_agg + delta_Z).clamp(min=min_bound_tensor.view(B, 1, 1), max=d_max_val)

        # 质心物理射线反解出唯一的闭环截距 d (捍卫大面根基)
        X_new = rays_centroids * Z_new
        d_new = -(n_new * X_new).sum(dim=-1, keepdim=True)

        final_planes = torch.cat([n_new, d_new], dim=-1)
        return final_planes,W_plane_tri_learn

    def compute_continuity_loss(self, planes, neighbor_indices, edge_probs,aligned_midpoints_norm, intrinsics,
                                H, W,depth_min,depth_max, W_plane_tri=None):
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

        # 🪐 引入置信度引导的动态连续性增强 (P0 修复量纲版)
        if W_plane_tri is not None:
            # 直接使用原生未极化置信度
            W_plane_pure = W_plane_tri.squeeze(-1).detach()
            W_i = W_plane_pure.unsqueeze(2)
            batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(-1, N, 3)
            W_j = W_plane_pure[batch_idx, neighbor_indices]
            # 采用 min 门控，只有双侧都是高置信度时才实施强连续性约束
            W_edge_plane = torch.min(W_i, W_j)
            plane_boost_ratio = 2.0
            dynamic_scale = continuity_scale * (1.0 + plane_boost_ratio * W_edge_plane)
        else:
            dynamic_scale = continuity_scale

        weighted_error = relative_error * gate * dynamic_scale

        # 🚨 P0 核心修复：分母只由 gate 决定，去掉 dynamic_scale，使 scale 真正生效而不被稀释！
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
                                centroids_norm, intrinsics, ref_feature,rays_centroids,
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
        rays = rays_centroids  # [B, N, 3]

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

    def compute_smoothness_loss_v2(self, planes, neighbor_indices, edge_probs,
                                   vertices_norm, pixel_counts, intrinsics, ref_feature,
                                   H, W,W_plane_tri=None, sigma_F=0.5, lambda_ang=6.0):
        """
        V2.0 终极版光滑性约束 (多点面片 Point-to-Plane + 面积感知正则化)

        Args:
            planes:             [B, N, 4] 当前预测的物理平面参数 (n_x, n_y, n_z, d)
            neighbor_indices:   [B, N, 3] 每个三角形的邻居面 ID
            edge_probs:         [B, N, 3] EdgeHead 预测的断裂概率 (外部需 detach)
            vertices_norm:      [B, N, 3, 2] 每个三角形 3 个顶点的归一化坐标 [-1, 1]
            pixel_counts:       [B, N] 每个三角形在原图的像素个数 (面积)
            intrinsics:         [B, 3, 3] 相机内参
            ref_feature:        [B, C, H, W] 图像特征图
            H, W:               图像高宽
            sigma_F:            特征高斯核的标准差
            lambda_ang:         法向量夹角的惩罚权重

        Returns:
            L_smooth (scalar)
        """
        B, N, _, _ = vertices_norm.shape
        device = planes.device

        # ============================================================
        # Step 1: 面积感知权重 (Asymmetric Clipping 算法)
        # 目标：小面片保底(0.5)，大面片压制上限(3.0)，无需二次归一化破坏语义
        # ============================================================
        pixel_counts_f = pixel_counts.float()
        mean_counts = pixel_counts_f.mean(dim=1, keepdim=True).clamp(min=1.0)
        area_weights = (pixel_counts_f / mean_counts).clamp(min=0.5, max=1.0)  # [B, N]

        # ============================================================
        # Step 2: 提取三顶点的 3D 射线 (Rays for 3 Vertices)
        # ============================================================
        # [B, N, 3(个顶点), 2(u,v)] -> 像素坐标
        px = (vertices_norm[..., 0] + 1.0) / 2.0 * (W - 1)
        py = (vertices_norm[..., 1] + 1.0) / 2.0 * (H - 1)
        uv_homo = torch.stack([px, py, torch.ones_like(px)], dim=-1)  # [B, N, 3, 3]

        K_inv = torch.inverse(intrinsics)  # [B, 3, 3]
        # einsum 魔法：批量将 3 个顶点的齐次坐标转化为 3D 射线方向
        rays_v = torch.einsum('bij,bnvj->bnvi', K_inv, uv_homo)  # [B, N, 3(顶点), 3(xyz)]

        # ============================================================
        # Step 3: 将射线与平面求交，得到 3D 物理顶点 X_v
        # ============================================================
        n_i = planes[..., :3]  # [B, N, 3]
        d_i = planes[..., 3:]  # [B, N, 1]

        # 计算分母: n · ray 得到一个单位向量的
        denom_v = torch.einsum('bni,bnvi->bnv', n_i, rays_v)  # [B, N, 3(顶点)]
        denom_v_safe = torch.where(denom_v.abs() < 1e-4, torch.full_like(denom_v, 1e-4), denom_v)

        # 强行给 d_i 增加一维，使其变为 [B, N, 1, 1]
        d_i_exp = d_i.unsqueeze(2)

        # 深度 Z_v 和 3D 坐标 X_v
        Z_v = (-d_i_exp / denom_v_safe.unsqueeze(-1)).abs()  # [B, N, 3(顶点), 1]
        X_v = rays_v * Z_v  # [B, N, 3(顶点), 3(xyz)] 当前平面上真实的 3 个三维顶点

        # ============================================================
        # Step 4: 收集邻居的几何信息
        # ============================================================
        batch_idx = torch.arange(B, device=device)[:, None, None].expand(B, N, 3)

        # 邻居的法向和截距
        n_j = n_i[batch_idx, neighbor_indices]  # [B, N, 3(邻居), 3]
        d_j = d_i[batch_idx, neighbor_indices]  # [B, N, 3(邻居), 1]

        # 邻居的 3 个物理顶点
        X_v_neigh = X_v[batch_idx, neighbor_indices]  # [B, N, 3(邻居), 3(顶点), 3(xyz)]

        # ============================================================
        # Step 5: 计算极其严苛的 Point-to-Plane 对称距离 (单向目标切断防爆版)
        # ============================================================

        # 5.1 邻居的 3 个顶点到当前平面(i) 的距离: |n_i · X_neigh + d_i|
        n_i_exp = n_i.unsqueeze(2).unsqueeze(2)  # [B, N, 1, 1, 3]
        d_i_exp = d_i.unsqueeze(2).unsqueeze(2)  # [B, N, 1, 1, 1]

        # 🚨 终极防爆：强行切断邻居顶点的梯度！
        # 让 n_i 和 d_i 去主动拟合固定的空间点，绝不让梯度穿透邻居的透视除法！
        X_v_neigh_detached = X_v_neigh.detach()  # [B, N, 3(邻居), 3(顶点), 3(xyz)]

        dist_neigh_to_self = (n_i_exp * X_v_neigh_detached).sum(dim=-1, keepdim=True) + d_i_exp
        E_dist_n2s = dist_neigh_to_self.abs().mean(dim=3).squeeze(-1)  # [B, N, 3(邻居)]

        # 5.2 当前的 3 个顶点到邻居平面(j) 的距离: |n_j · X_self + d_j|
        n_j_exp = n_j.unsqueeze(3)  # [B, N, 3(邻居), 1, 3]
        d_j_exp = d_j.unsqueeze(3)  # [B, N, 3(邻居), 1, 1]

        # 🚨 同理防爆：切断当前顶点的梯度！
        # X_v: [B, N, 3(顶点), 3(xyz)] -> 扩展后: [B, N, 1, 3(顶点), 3(xyz)]
        X_v_self_detached = X_v.unsqueeze(2).detach()

        dist_self_to_neigh = (n_j_exp * X_v_self_detached).sum(dim=-1, keepdim=True) + d_j_exp
        E_dist_s2n = dist_self_to_neigh.abs().mean(dim=3).squeeze(-1)  # [B, N, 3(邻居)]

        # 对称物理距离 (此时梯度绝对干净平稳，只允许微调法向和截距)
        E_dist = 0.5 * (E_dist_n2s + E_dist_s2n)  # [B, N, 3]

        # ============================================================
        # Step 6: 角度惩罚 & 最终能量
        # ============================================================
        cos_theta = (n_i.unsqueeze(2) * n_j).sum(dim=-1)  # [B, N, 3]
        E_ang = 1.0 - cos_theta  # [B, N, 3]

        E_geom = E_dist + lambda_ang * E_ang  # [B, N, 3]

        # ============================================================
        # Step 7: 计算双门控权重 W_ij (EdgeHead + Feature Affinity)
        # ============================================================
        # 这里提取质心特征 (代码简化，你需要传入质心 centroids_norm)
        # 取平均顶点坐标作为近似质心用于采样特征
        centroids_approx = vertices_norm.mean(dim=2)  # [B, N, 2]
        grid = centroids_approx.view(B, N, 1, 2)

        F_i = F.grid_sample(
            ref_feature,
            grid,
            mode='bilinear',
            align_corners=True,
            padding_mode='border'  # <-- 加上这里！
        ).squeeze(-1).permute(0, 2, 1)

        F_i = F.normalize(F_i, p=2, dim=-1)

        F_j = F_i[batch_idx, neighbor_indices]
        feat_dist_sq = ((F_i.unsqueeze(2) - F_j) ** 2).sum(dim=-1)
        W_feat = torch.exp(-feat_dist_sq / (2.0 * sigma_F ** 2))

        W_hard = (1.0 - edge_probs).clamp(min=0.0, max=1.0)
        W_ij = W_hard * W_feat

        # 边界屏蔽
        is_boundary = (neighbor_indices == torch.arange(N, device=device)[None, :, None])
        W_ij = W_ij.masked_fill(is_boundary, 0.0)

        # ============================================================
        # Step 8: 融入面积特权，计算加权 Loss
        # ============================================================
        W_final = W_ij * area_weights.unsqueeze(2)
        # 🔥 新增：双向木桶效应隔离
        if W_plane_tri is not None:
            # 1. 自适应维度规范化：不管外面传进来的是 [B, N] 还是 [B, N, 1]，统一 squeeze 成纯净的一维特征轴
            W_plane_pure = W_plane_tri.squeeze(-1)  # 刚性锁死为 [B, N]

            # 2. 自身置信度在末尾升维，以适配拓扑邻居广播形态
            W_i = W_plane_pure.unsqueeze(2)  # [B, N, 1]

            # 3. 高级索引查表抽取邻居置信度
            batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(-1, N, 3)
            W_j = W_plane_pure[batch_idx, neighbor_indices]  # 严格对齐至 [B, N, 3]，彻底封锁末尾伪维度的干扰

            # 4. 取双向连通度的最小值：任何一个邻居是曲面，刚性熔断此边的平滑势能
            W_edge_confidence = torch.min(W_i, W_j)  # [B, N, 3]

            # 5. 纯净的三维张量空间点乘对撞：[B, N, 3] * [B, N, 3]，量纲完美契合
            W_final = W_ij * area_weights.unsqueeze(2)
        if W_plane_tri is not None:
            # 彻底开除极化操作，直接使用原生未极化的连续置信度 W_plane_tri 且强制 .detach()
            W_plane_pure = W_plane_tri.squeeze(-1).detach()  # [B, N]
            W_i = W_plane_pure.unsqueeze(2)  # [B, N, 1]
            batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(-1, N, 3)
            W_j = W_plane_pure[batch_idx, neighbor_indices]  # [B, N, 3]

            # 1. 提取基础木桶约束以保护好平面
            W_edge_min = torch.min(W_i, W_j)  # [B, N, 3]

            # 2. 👑 Base + Boost min 黄金公式：0.3 保底维系曲面 TV 渐变，其余 0.7 弹性留给大平墙
            base_smooth = 0.3
            W_gating = base_smooth + (1.0 - base_smooth) * W_edge_min  # [B, N, 3]

            # 叠乘门控
            W_final = W_final * W_gating

        # 物理距离 + 角度惩罚
        weighted_energy = W_final * E_geom

        weight_sum = W_final.sum().clamp(min=1e-6)
        L_smooth = weighted_energy.sum() / weight_sum

        # --- 诊断打印 ---
        with torch.no_grad():
            print(f"[SMOOTH V2.0] L_smooth={L_smooth.item():.5f} | "
                  f"E_dist(m)={E_dist.mean().item():.4f} | "
                  f"Area_W_Max={area_weights.max().item():.2f}")

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
        self.propagator=DoubleDecoupledTrianglePropagator()

        # === 🔥 新增：EdgeHead作为内部模块 ===
        self.edge_head = EdgeHead(feat_channels)

        self.propagator_iter = propagator_iter

    def forward(self,fitter_module, depth_stage1, tri_infos, ref_feature, src_features,
                ref_proj, src_projs, intrinsics_s1,depth_min, depth_max, view_weights=None,
                neighbor_indices_batched=None,lambda_c=0.0, lambda_s=0.0,current_temp=0.0):
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

        # 三角形质心坐标[B, N_max, 2]
        centroids_norm = self.fitter.collate_centroids_norm(
            tri_infos[0]['centers_list'], device
        )

        # 三角形顶点归一化坐标 Tensor
        vertices_norm = self.fitter.collate_vertices_norm(
            tri_infos[0]['vertices_list'], device
        )

        # 三角形质心射线 rays_centroids (质心射线)
        u_px = (centroids_norm[..., 0] + 1.0) / 2.0 * (W - 1)  # [B, N]
        v_px = (centroids_norm[..., 1] + 1.0) / 2.0 * (H - 1)  # [B, N]
        uv_homo = torch.stack([u_px, v_px, torch.ones_like(u_px)], dim=-1)  # [B, N, 3]

        K_inv = torch.inverse(ref_intrinsics)  # [B, 3, 3]
        rays_centroids = torch.einsum('bij,bnj->bni', K_inv, uv_homo)  # [B, N, 3]

        # 在进入传播循环前，计算 pixel_costs
        # 将特征从计算图中剥离，保护 FeatureNet 不受 Stage 1 毒害
        # src_features_detached = [f.detach() for f in src_features]

        # ==========================================
        # 2. 拟合与生成 (Fitting & Generation)，根据stage2预测深度来拟合生成假设平面
        # ==========================================

        hypotheses,surface_var = self.fitter.get_plane_hypotheses(
            depth_stage2=depth_stage1,
            tri_id_map=tri_id_map,
            intrinsics_s1=ref_intrinsics,
            max_num_triangles=max_tri_num
        )  # Output: [B, N_tri, 1, 1]

        # [B, N_tri, K, 1] -> [B, N_tri, 4] (取第0个假设,最佳平面的前3通道)
        # 进行一个可视化看看效果，拟合的初始平面
        # before_best_guess_planes = hypotheses[:, :, 0, :]  # [B, N_tri, 4]

        # 直接取出唯一的平面作为基底 (不需要 argmin)
        # current_planes is selected after K-wise photometric cost aggregation.

        # ==========================================
        # 3. 计算 SVD 基底的物理代价,多假设密集并网 · 稀疏网格硬选择
        # ==========================================
        # 3.1 将稀疏三角形级多假设池广播查表至密集像素空间阵列 -> [B, H, W, K, 4]
        pixel_hypotheses = self.map_tri_to_pixel(hypotheses, tri_id_map, H, W)

        # 3.2 运行端到端可微重采样，解算出全假设、全像素的匹配代价特征图 -> [B, H, W, K]
        pixel_costs, pixel_costs_raw, cost_variance = self.compute_costs(
            ref_feature, src_features, ref_proj, src_projs,
            pixel_hypotheses, view_weights=view_weights, ref_intrinsic=ref_intrinsics,
            is_debug_diag=True
        )
        self.pixel_costs_raw = pixel_costs_raw
        self.cost_variance = cost_variance
        
        # 聚合为三角形代价 [B, N_tri, 1]
        tri_costs_volume = self.aggregate_costs_per_triangle(pixel_costs, tri_id_map, max_tri_num)

        # 【硬核硬选择】：在假设维度直接取最小光度代价对应的“黄金轨号”索引 -> [B, N_tri]
        best_k = tri_costs_volume.argmin(dim=2)  # [B, N_tri]

        gather_idx = best_k.view(B, max_tri_num, 1, 1).expand(-1, -1, 1, 4)
        current_planes = torch.gather(hypotheses, 2, gather_idx).squeeze(2)  # [B, N_tri, 4]
        # 同步抽取三角形级获胜轨道的自发光度代价缓存 -> [B, N_tri, 1]
        current_costs = torch.gather(tri_costs_volume, 2, best_k.unsqueeze(-1))  # [B, N_tri, 1]

        safe_tri_id_map = tri_id_map.long().clamp(min=0, max=max_tri_num - 1)
        batch_lookup = torch.arange(B, device=device).view(B, 1, 1).expand(B, H, W)
        best_k_pixel = best_k[batch_lookup, safe_tri_id_map].unsqueeze(-1)
        pixel_costs = torch.gather(pixel_costs, -1, best_k_pixel)  # [B, H, W, 1]

        # 渲染没有经过传播的 深度图和法向量图 后续用
        no_prop_depth, no_propa_normal = visualizer.render_from_planes(
            current_planes.detach() ,  # 注意 detach，不传导梯度
            tri_id_map,
            ref_intrinsics,
            depth_range=(depth_min, depth_max)
        )

        # 生成三角形的平面置信度
        W_plane_tri = self._compute_plane_confidence(
            ref_feature, depth_stage1, no_prop_depth, surface_var, tri_id_map, max_tri_num,
            current_planes=current_planes, rays_centroids=rays_centroids,
        )

        # ==========================================
        # 4. 预测物理断裂边 (EdgeHead) 在传播中第一轮预测
        # ==========================================

        # 全 0 的 Tensor 意味着没有任何人工边缘阻断，纯靠特征距离(feat_dist)去平滑
        edge_probs_tensor = torch.zeros(B, max_tri_num, 3, device=device)

        # 循环外：提取 O(1) 的不变图像与稠密深度特征

        static_edge_feats = self.edge_head.extract_static_features(
            feat=ref_feature.detach(),
            tri_infos=tri_infos,
            dense_depth=depth_stage1.detach(),
            intrinsics=ref_intrinsics  # 👈 刚性丢入相机参数，完成 3D 绝对位移空间解算
        )

        # 边断裂概率输出后续算loss
        edge_alpha = None

        # 初始化上一轮代价缓存 (第一轮没有历史，设为 None)
        prev_costs = None

        W_raw_anchor = W_plane_tri.clone().detach() # 👈 进循环前，原地备份最纯净的物理冷启动初始置信度
        # ==========================================
        # 5. 端到端神经融合传播 (Neural Soft Propagation)
        # ==========================================

        for iter_idx in range(self.propagator_iter):

            # 极速动态边缘更新 (O(N) 仅计算几何与 MLP)
            edge_alpha = self.edge_head.dynamic_forward(
                    static_feats_list=static_edge_feats,
                    tri_infos=tri_infos,
                    tri_planes=current_planes.detach(),  # 👈 永远使用最新修正的平面
                    intrinsics=ref_intrinsics,
                    H=H, W=W,
                    W_plane_tri=W_plane_tri.detach()  # 👈 断开传播，纯粹作为特征输入
            )

            # 转换为三角形级别格式 [B, N_max, 3]
            edge_probs_tensor, aligned_midpoints_norm, aligned_endpoints_norm = convert_edge_features_to_tri_format(
                    edge_alpha, tri_infos, max_tri_num, device)

            # 取消第一轮断流冷启动：每一轮都使用当前 W_plane_tri 参与置信度门控更新。
            W_gating_input = W_plane_tri

            # ------ 【新增：Cross-Cost 零成本验证逻辑 (全量像素评估版)】 ------
            # 1. 提取邻居平面
            _B, _N, _ = current_planes.shape
            _batch_idx = torch.arange(_B, device=device).view(_B, 1, 1).expand(-1, _N, 3)
            neighbor_planes = current_planes[_batch_idx, neighbor_indices_batched] # [B, N, 3, 4]
            
            # 2. 将邻居平面广播到密集像素阵列 [B, H, W, 3, 4]
            pixel_hypotheses_cross = self.map_tri_to_pixel(neighbor_planes, tri_id_map, H, W)
            
            # 3. 运行端到端光度代价计算 (客场验证) - 🛡️ 必须加上 no_grad 防止 OOM！
            with torch.no_grad():
                cross_costs_pixel = self.compute_costs(
                    ref_feature, src_features, ref_proj, src_projs,
                    pixel_hypotheses_cross, view_weights=view_weights, ref_intrinsic=ref_intrinsics,
                    is_debug_diag=False
                ) # [B, H, W, 3]
            
            # 4. 聚合并提纯到面片维度 [B, N, 3]
            cross_costs_flat = self.aggregate_costs_per_triangle(cross_costs_pixel, tri_id_map, max_tri_num)
            # ----------------------------------------------------

            # 5.1 ym-modify 传播：全新双解耦网络，完美注入 W_plane_tr，在传播中不断更新w_plane
            new_planes,W_plane_tri_learn = self.propagator(
                current_planes=current_planes,
                current_costs=current_costs.detach(),
                neighbor_indices=neighbor_indices_batched,
                edge_probs=edge_probs_tensor.detach(),
                ref_feature=ref_feature,
                rays_centroids=rays_centroids,
                depth_max=depth_max,
                prev_costs=prev_costs,
                W_plane_tri=W_gating_input,
                centroids_norm=centroids_norm,
                pixel_counts=pixel_counts_tensor,
                temperature=current_temp,  # 可根据当前训练的 Epoch 动态退火压低
                W_raw_anchor=W_raw_anchor,  # 刚性投递，防止置信度头多轮迭代后神经失忆
                cross_costs=cross_costs_flat # 传入 Cross-Cost
            )

            # 将本轮的代价封存，作为下一轮的“历史代价”
            prev_costs = current_costs.detach()

            # 取消第一轮 anchor 恢复逻辑：每一轮都使用本轮预测的置信度更新结果。
            W_plane_tri = W_plane_tri_learn

            # # 对传播进行一个保护
            new_planes = fitter_module.enforce_depth_hard_constraint(
                planes=new_planes,
                centroids_norm=centroids_norm,
                intrinsics=ref_intrinsics,
                depth_min=depth_min,
                depth_max=depth_max,
                H=H, W=W
            )

            # 5.2 重新评估新平面的代价
            if iter_idx < self.propagator_iter - 1:  # 💡 性能榨取：最后一轮不需要重复计算代价，直接省掉 33% 采样耗时！
                new_planes_k1 = new_planes.unsqueeze(2)
                pixel_hypo = self.map_tri_to_pixel(new_planes_k1, tri_id_map, H, W)

                with torch.no_grad():  # 🛡️ 降下绝对梯度防火墙，封杀可微重采样的非连续毛刺
                    pixel_costs_new = self.compute_costs(
                        ref_feature, src_features, ref_proj, src_projs,
                        pixel_hypo,
                        view_weights=view_weights,
                        ref_intrinsic=ref_intrinsics,
                        is_debug=False
                    ).detach()
                current_costs = self.aggregate_costs_per_triangle(pixel_costs_new, tri_id_map, max_tri_num).detach()

            # 5.3 状态更新，进入下一次迭代
            # 注意：因为是 Learned Propagator，我们直接相信它的更新（像 RNN 一样），而不进行 Argmin 判断
            current_planes = new_planes

        final_planes = current_planes
        # =====================================================================
        # 👑 【核心重构：外层全标度刚性上界饱和铁闸】~
        # =====================================================================
        # 物理因果：在这里对置信度降下全管线统一的 SmoothStep 极化法案！
        # 设定上界 theta_high = 0.80（刚性严苛对齐，深度误差需锁定在 1.20 米内）。
        # 强行注入 .detach() 锁定，阻断下游光滑性损失（L_smoothness）逆向洗劫置信度更新头。
        th_low, th_high = 0.05, 0.70
        W_pure = W_plane_tri.detach().squeeze(-1)                       # 刚性挤压退化的 [B, N, 1] 至 [B, N]
        x_norm = torch.clamp((W_pure - th_low) / (th_high - th_low + 1e-8), 0.0, 1.0)

        # 降下三阶 Hermite 极化算子，强制将高于 0.80 的不确定性拉满至严格的 1.0 铁板
        W_tri_polarized = ((norm_val := x_norm) ** 2 * (3.0 - 2.0 * norm_val)).detach()
        W_plane_tri_polarized = W_tri_polarized.unsqueeze(-1)  # 升维 [B, N, 1] 供下游光滑性损失平稳查表    

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
                depth_min=min(depth_min), depth_max=max(depth_max),
                W_plane_tri=W_plane_tri.detach()) # 🚨 传入原生未极化的 W_plane_tri

        # 计算光滑性损失
        smoothness_loss = 0.0

        if lambda_s > 0.0:
            # smoothness_loss = self.propagator.compute_smoothness_loss(
            #     planes=final_planes,
            #     neighbor_indices=neighbor_indices_batched,
            #     edge_probs=edge_probs_tensor.detach(),  # 必须 detach
            #     centroids_norm=centroids_norm,  # 传入归一化质心 [B, N, 2]
            #     intrinsics=ref_intrinsics,  # 相机内参 [B, 3, 3]
            #     ref_feature=ref_feature.detach(),  # 原生特征图 [B, C, H, W]
            #     rays_centroids=rays_centroids,
            #     H=H, W=W,
            #     sigma_F=0.4,  # 可调：0.5 是 L2 归一化特征推荐值
            #     lambda_ang=8.0  # 可调：法向平滑的相对强度
            # )

            smoothness_loss = self.propagator.compute_smoothness_loss_v2(
                planes=final_planes,
                neighbor_indices=neighbor_indices_batched,
                edge_probs=edge_probs_tensor.detach(),  # 必须 detach
                vertices_norm=vertices_norm,  # 传入三角形三个顶点
                pixel_counts=pixel_counts_tensor,  # 每个三角形的大小
                intrinsics=ref_intrinsics,  # 相机内参 [B, 3, 3]
                ref_feature=ref_feature.detach(),  # 原生特征图 [B, C, H, W]
                W_plane_tri=W_plane_tri.detach(),  # 🚨 传入原生未极化的 W_plane_tri
                H=H, W=W,
                sigma_F=0.4,  # 可调：0.5 是 L2 归一化特征推荐值
                lambda_ang=5.0  # 可调：法向平滑的相对强度
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

        # 将三角形置信度转化为全分辨率稠密灰度图
        W_plane_pixel_out = self._scatter_triangle_to_pixel(W_plane_tri, tri_id_map)
        # 传播前：_compute_plane_confidence 冷启动锚点（与循环内 W_raw_anchor 一致）
        W_plane_pixel_init = self._scatter_triangle_to_pixel(W_raw_anchor, tri_id_map)

        return (depth_samples, pixel_costs,
                view_weights,
                normal_samples,
                final_planes,
                edge_alpha,
                continuity_loss,smoothness_loss,
                W_plane_pixel_out, W_plane_tri, W_plane_tri_polarized,
                pixel_counts_tensor,
                W_plane_pixel_init, W_raw_anchor)  # 更新前/后对比用

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
                      ref_intrinsic, is_debug=True, is_debug_diag=False):
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
        view_scores = []

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

            # --- D. todo:分组相关性 (Group Correlation) ---
            # [B*K, G, C/G, H, W]
            warped_src_grouped = warped_src.view(B * K, self.G, C // self.G, H, W)

            # 强行将特征向量的长度缩放为 1，将点积转化为余弦相似度 (Cosine Similarity)
            # 这样 similarity 的物理边界被死死锁在 [-1, 1] 之间，网络绝无作弊可能！
            warped_src_norm = F.normalize(warped_src_grouped, p=2, dim=2)
            ref_feat_norm = F.normalize(ref_feat_expanded, p=2, dim=2)

            # Similarity: [B*K, G, H, W]
            similarity = (warped_src_norm * ref_feat_norm).mean(dim=2)

            # 收集每个视角的分数并分组求平均，形状为 [B*K, 1, H, W]
            view_scores.append(similarity.mean(dim=1, keepdim=True))

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

        if is_debug_diag:
            # 1. 未加权的平均代价：[B*K, Nview-1, H, W] -> mean -> [B*K, 1, H, W]
            all_scores = torch.cat(view_scores, dim=1)
            raw_score = all_scores.mean(dim=1, keepdim=True)
            raw_cost_fused = -raw_score
            raw_cost = raw_cost_fused.view(B, K, H, W).permute(0, 2, 3, 1)

            # 2. 各视图相似度的方差 (无偏估计)：[B*K, 1, H, W] -> 还原形状 [B, H, W, K]
            variance_fused = all_scores.var(dim=1, keepdim=True, unbiased=False)
            cost_variance = variance_fused.view(B, K, H, W).permute(0, 2, 3, 1)

            return total_cost, raw_cost, cost_variance

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

    def _compute_plane_confidence(self, ref_feature, depth_stage1, no_prop_depth, surface_var, tri_id_map,
                                max_tri_num, current_planes, rays_centroids):
        """
        将几何冲突、高频纹理梯度与 SVD 表面变异度熔炼为三角形级别的平面置信度 W_plane.
        所有计算运行在 torch.no_grad() 下，绝对隔绝反向传播梯度。
        """
        with torch.no_grad():
            B, _, H, W = depth_stage1.shape
            device = depth_stage1.device

            # 1. 检查空间分辨率对齐，封杀隐式广播核爆
            if ref_feature.shape[2:] != depth_stage1.shape[2:]:
                depth_stage1 = F.interpolate(depth_stage1, size=ref_feature.shape[2:], mode='bilinear',
                                             align_corners=True)
            if no_prop_depth.shape[2:] != ref_feature.shape[2:]:
                no_prop_depth = F.interpolate(no_prop_depth, size=ref_feature.shape[2:], mode='bilinear',
                                              align_corners=True)

            # 2. 提取特征高频梯度 (Texture Gradient)
            feat_norm = F.normalize(ref_feature.detach(), p=2, dim=1)
            grad_x = torch.abs(feat_norm[:, :, :, :-1] - feat_norm[:, :, :, 1:])
            grad_x = F.pad(grad_x, (0, 1, 0, 0))
            grad_y = torch.abs(feat_norm[:, :, :-1, :] - feat_norm[:, :, 1:, :])
            grad_y = F.pad(grad_y, (0, 0, 0, 1))

            T_feat = (grad_x + grad_y).mean(dim=1, keepdim=True)

            # 过滤视觉低频噪点
            noise_floor = 0.02
            T_feat_clean = F.relu(T_feat - noise_floor)
            T_feat_scaled = torch.tanh(T_feat_clean * 1.5)

            # =====================================================================
            # 3. 倾斜流形自适应视线补偿 (Geometric Conflict)
            # =====================================================================
            # n_curr = current_planes[..., :3]  # [B, N, 3]
            # denom_curr = (n_curr * rays_centroids).sum(dim=-1, keepdim=True)  # [B, N, 1]

            # # 物理级三角形面片视线夹角余弦值计算
            # cos_tri = denom_curr.abs().clamp(min=0.1, max=1.0).squeeze(-1)  # [B, N]

            # # 将密集格式一阶拉平，强制执行 .long() 显式转型防御，杜绝 C++ 内部指针发散
            # flat_tri_id = tri_id_map.view(B, -1).long()  # [B, H*W]

            # # 空间边界实体点合规检查掩码
            # valid_px = (flat_tri_id >= 0) & (flat_tri_id < max_tri_num)
            # safe_idx = flat_tri_id.clamp(0, max_tri_num - 1)

            # # 执行视点不变高通量查表，每个像素精准带回所属三角形的真实纯净 cos 值
            # cos_gathered = torch.gather(cos_tri, 1, safe_idx)  # [B, H*W]

            # # 虚空无主背景处强制用 1.0 保底进行物理绝缘，不释放任何错误几何拉扯
            # cos_gathered = torch.where(valid_px, cos_gathered, torch.ones_like(cos_gathered))

            # # 重新规整形态，完美降维输出 [B, 1, H, W] 标准张量，与 depth 场绝对同构！
            # cos_theta = cos_gathered.view(B, 1, H, W)

            # todo：暂时不用svd不干净的法向量进行一个计算
            C_geo_abs = torch.abs(depth_stage1.detach() - no_prop_depth.detach())
            C_geo_norm = (C_geo_abs / (depth_stage1.detach() + 1e-6))

            # 4. 熔炼像素级非平面惩罚项
            pixel_curve_penalty = C_geo_norm * T_feat_scaled

            # 5. 像素级聚合至稀疏三角形网格 [B, N]
            tri_curve_penalty = self.scatter_penalty_to_triangle(
                pixel_curve_penalty.permute(0, 2, 3, 1),
                tri_id_map,
                max_tri_num
            )

            # 6. 引入 Surface Variation 表面粗糙度方差
            sv_scaled = torch.tanh(surface_var * 10.0)  # 进一步放宽粗糙度敏感底线

            # 降下刚性惩罚重锤
            alpha_conf = 10.0  # 几何冲突惩罚全面加码！
            beta_conf = 10.0   # 表面粗糙度惩罚全面加码！

            total_penalty = alpha_conf * tri_curve_penalty + beta_conf * sv_scaled
            W_raw = torch.exp(-total_penalty)

            # ==========================================================
            # 2. 🚀 工业级多项式死区门控区间重组
            # ==========================================================
            theta_low = 0.20   # 适当抬高截断底线，让黄绿过渡区直接掉落死区
            theta_high = 0.80  # 刚性还原高门槛，只有真正的 0.069m 级别绝对平整面才配拿到高置信度！

            # 线性映射并强制收拢至凸空间
            x = torch.clamp((W_raw - theta_low) / (theta_high - theta_low + 1e-8), 0.0, 1.0)

            # 经典一阶连续 Hermite 插值 (SmoothStep)，保障边界偏导数连续，绝不静默泄露跳变伪梯度
            W_plane_final = (x ** 2) * (3.0 - 2.0 * x)

        # 强制加上退化维度防护，输出标准的 [B, N, 1] 拓扑
        return W_plane_final.unsqueeze(-1) if W_plane_final.dim() == 2 else W_plane_final

    def scatter_penalty_to_triangle(self, pixel_penalty, tri_id_map, max_num_tri):
        """
        将像素级惩罚项（值域[0,+∞)）聚合为三角形级别的算术平均值。
        
        专门用于 _compute_plane_confidence 中的 pixel_curve_penalty 聚合，
        与 aggregate_costs_per_triangle 的软过滤逻辑完全隔离。
    
        Args:
            pixel_penalty: [B, H, W, 1]  像素级惩罚项，值域[0,+∞)
            tri_id_map:    [B, H, W]     三角形ID映射
            max_num_tri:   int           最大三角形数量
    
        Returns:
            tri_penalty:   [B, N_tri]    三角形级别算术平均惩罚值
        """
        B, H, W, K = pixel_penalty.shape
        device = pixel_penalty.device

        # 1. 展平
        flat_penalty = pixel_penalty.reshape(B, -1, K)  # [B, H*W, 1]
        flat_ids     = tri_id_map.reshape(B, -1)         # [B, H*W]

        # 2. 有效mask
        valid_mask = (flat_ids >= 0) & (flat_ids < max_num_tri)

        # 3. Scatter索引
        batch_offset = (torch.arange(B, device=device) * max_num_tri).view(B, 1)
        safe_ids     = flat_ids.clone()
        safe_ids[~valid_mask] = 0
        global_ids   = (safe_ids + batch_offset).view(-1)

        flat_mask        = valid_mask.reshape(-1)
        valid_global_ids = global_ids[flat_mask]
        valid_penalty    = flat_penalty.reshape(-1, K)[flat_mask]  # [V, 1]

        # 4. Scatter累加（算术平均，分母=N）
        total_bins    = B * max_num_tri
        flat_sum      = torch.zeros(total_bins, K, device=device)
        flat_counts   = torch.zeros(total_bins, 1, device=device)

        idx_expand = valid_global_ids.unsqueeze(1).expand(-1, K)
        flat_sum.scatter_add_(0, idx_expand, valid_penalty)
        flat_counts.scatter_add_(
            0,
            valid_global_ids.unsqueeze(1),
            torch.ones(valid_global_ids.shape[0], 1, device=device)
        )

        # 5. 均值，无像素的三角形设为0（无惩罚）
        has_pixels  = (flat_counts > 0)
        flat_mean   = flat_sum / (flat_counts + 1e-6)
        flat_mean   = torch.where(has_pixels, flat_mean, torch.zeros_like(flat_mean))

        # 输出 [B, N_tri]（squeeze掉K=1的维度）
        return flat_mean.view(B, max_num_tri, K).squeeze(-1)

    def _scatter_triangle_to_pixel(self, tri_values, tri_id_map):
        """
        将三角形级别的属性 [B, N_tri] 反向重投影回稠密像素空间 [B, 1, H, W]
        """
        B, H, W = tri_id_map.shape
        device = tri_id_map.device

        # 1. 确保输入是 [B, N_tri, 1] 架构，方便 gather 算子对齐
        if tri_values.ndim == 2:
            tri_values = tri_values.unsqueeze(-1)
        C = tri_values.shape[-1]

        # 2. 展平拓扑索引图并扩展通道维度 -> [B, H*W, C]
        tri_id_flat = tri_id_map.view(B, -1, 1).expand(-1, -1, C)

        # 🛡️ 边界防爆：提取有效三角形掩码（屏蔽背景的 -1）
        valid_mask = tri_id_flat >= 0

        # 3. 建立安全索引：将负值（-1）暂时强制指向 0 号三角形，防止 gather 越界核爆
        safe_index = torch.where(valid_mask, tri_id_flat, torch.zeros_like(tri_id_flat))

        # 4. 执行多维反向计算图映射
        dense_flat = torch.gather(tri_values, dim=1, index=safe_index.long())

        # 🛡️ 物理置零：将原本为 -1 的无效区域彻底抹碎为 0.0（黑边）
        dense_flat = torch.where(valid_mask, dense_flat, torch.zeros_like(dense_flat))

        # 5. 重塑回标准的 TensorBoard 图像张量格式 [B, C, H, W]
        dense_map = dense_flat.permute(0, 2, 1).view(B, C, H, W)
        return dense_map


class DensePlaneFitter(nn.Module)   :
    def __init__(self, height_s1, width_s1, device, num_hypotheses=7, perturbation_range=0.05, depth_max=None,depth_min=None):
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

    def _get_plane_hypotheses_multi(self, depth_stage2, tri_id_map, intrinsics_s1, max_num_triangles):
        """
        【各项异性自适应多假设池生成大底】
        物理因果：并行规约平权 OLS 与拉普拉斯 WLS 双轨协方差。利用 eigh 特征正交基张成 3D 切空间流形。
                 采用周期性模算子确保 [+e1, -e1, +e2, -e2] 对称正交探索，由表面变异度动态控温。
        量纲边界：输出 hypotheses [B, N_tri, K, 4], surface_variation [B, N_tri]
        """
        B = depth_stage2.shape[0]
        device = depth_stage2.device
        total_bins = B * max_num_triangles
        K = self.K

        # 🪐 内部匿名函数：动态展开近远景视差剪裁张量边界，防浮点下溢
        def depth_bounds_flat():
            if isinstance(self.depth_min, torch.Tensor):
                d_min = self.depth_min.to(device).float().view(B, 1).expand(B, max_num_triangles).reshape(total_bins, 1)
                d_max = self.depth_max.to(device).float().view(B, 1).expand(B, max_num_triangles).reshape(total_bins, 1)
            else:
                min_scalar = 0.1 if self.depth_min is None else (
                    float(min(self.depth_min)) if isinstance(self.depth_min, (list, tuple)) else float(self.depth_min)
                )
                max_scalar = 10.0 if self.depth_max is None or self.depth_max == [] else (
                    float(max(self.depth_max)) if isinstance(self.depth_max, (list, tuple)) else float(self.depth_max)
                )
                d_min = torch.full((total_bins, 1), min_scalar, device=device, dtype=torch.float32)
                d_max = torch.full((total_bins, 1), max_scalar, device=device, dtype=torch.float32)
            return d_min, d_max

        # 🪐 内部匿名函数：退化简并网格全平行刚性熔断防火墙
        def fallback_pool():
            _, d_max = depth_bounds_flat()
            fallback_n = torch.zeros(total_bins, K, 3, device=device, dtype=torch.float32)
            fallback_n[..., 2] = -1.0
            fallback_d = d_max.view(total_bins, 1, 1).expand(-1, K, -1).contiguous()
            pool = torch.cat([fallback_n, fallback_d], dim=-1)
            surface = torch.full((total_bins,), 0.333, device=device, dtype=torch.float32)
            return pool.view(B, max_num_triangles, K, 4), surface.view(B, max_num_triangles)

        # =====================================================================
        # Step 0: 全分辨率密集二阶拉普拉斯算子清洗（各向异性防护纵深）
        # =====================================================================
        with torch.no_grad():
            depth_stage2_clean = torch.nan_to_num(depth_stage2, nan=0.0, posinf=0.0, neginf=0.0)
            padded_depth = F.pad(depth_stage2_clean, (1, 1, 1, 1), mode='replicate')
            dense_laplacian = torch.abs(
                padded_depth[:, :, 2:, 1:-1] + padded_depth[:, :, :-2, 1:-1] +
                padded_depth[:, :, 1:-1, 2:] + padded_depth[:, :, 1:-1, :-2] -
                4.0 * padded_depth[:, :, 1:-1, 1:-1]
            )

        # =====================================================================
        # Step 1: 三角网格拓扑亚像素投影采样
        # =====================================================================
        grid_batch = self.sampling_grid.unsqueeze(0).expand(B, -1, -1, -1)
        sampled_depth = F.grid_sample(
            depth_stage2_clean, grid_batch, mode='bilinear', padding_mode='border', align_corners=True
        )
        sampled_laplacian = F.grid_sample(
            dense_laplacian, grid_batch, mode='bilinear', padding_mode='border', align_corners=True
        )

        if sampled_depth.max() < 1e-4:
            return fallback_pool()

        # =====================================================================
        # Step 2: 反投影逆向解算 3D 密集点云场
        # =====================================================================
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

        # =====================================================================
        # Step 3: 并行去中心化一阶/二阶规约总线（OLS与WLS双轨并行）
        # =====================================================================

        tri_ids_flat = tri_id_map.view(-1)
        batch_ids = torch.arange(B, device=device).unsqueeze(1).expand(-1, self.H * self.W).reshape(-1)
        global_tri_ids = batch_ids * max_num_triangles + tri_ids_flat

        is_finite_points = torch.isfinite(points_flat).all(dim=1)
        valid_mask = (tri_ids_flat >= 0) & (points_flat[:, 2] > 1e-4) & is_finite_points
        if valid_mask.sum() == 0:
            return fallback_pool()

        valid_points = points_flat[valid_mask]
        valid_ids = global_tri_ids[valid_mask].long().clamp(min=0, max=total_bins - 1)
        valid_points_64 = valid_points.double()
        pt_x, pt_y, pt_z = valid_points_64[:, 0], valid_points_64[:, 1], valid_points_64[:, 2]

        # 🪐 算软门控：放宽大分辨率温标线至 4% 融合 Sigmoid 核，绞杀悬崖滑坡点
        laplacian_flat_raw = sampled_laplacian.view(-1)[valid_mask].double()
        adaptive_tau_lap = (pt_z * 0.04).clamp(min=0.15, max=2.0)
        valid_geo_weights = torch.sigmoid(-16.0 * (laplacian_flat_raw - adaptive_tau_lap))
        valid_geo_weights = torch.clamp(valid_geo_weights, min=1e-3, max=1.0)

        # 寄存计数器分母
        ones_v = torch.ones_like(valid_ids, dtype=torch.float64)
        counts = torch.zeros(total_bins, device=device, dtype=torch.float64)
        counts.scatter_add_(0, valid_ids, ones_v)
        counts_for_cov = torch.zeros(total_bins, device=device, dtype=torch.float64)
        counts_for_cov.scatter_add_(0, valid_ids, valid_geo_weights)

        # 面积覆盖率门限互锁
        weight_coverage = counts_for_cov / counts.clamp(min=1.0)
        low_coverage_mask = (weight_coverage < 0.30) & (counts > 5.0)
        valid_fit_mask = (counts > 3.0) & (~low_coverage_mask)

        # 🚀 收集轨道 A 分子统计量（纯平权 OLS）
        sum_P_ols = torch.zeros(total_bins, 3, device=device, dtype=torch.float64)
        sum_P_ols.index_add_(0, valid_ids, valid_points_64)
        sum_xx_ols = torch.zeros(total_bins, device=device, dtype=torch.float64).scatter_add_(0, valid_ids, pt_x * pt_x)
        sum_xy_ols = torch.zeros(total_bins, device=device, dtype=torch.float64).scatter_add_(0, valid_ids, pt_x * pt_y)
        sum_xz_ols = torch.zeros(total_bins, device=device, dtype=torch.float64).scatter_add_(0, valid_ids, pt_x * pt_z)
        sum_yy_ols = torch.zeros(total_bins, device=device, dtype=torch.float64).scatter_add_(0, valid_ids, pt_y * pt_y)
        sum_yz_ols = torch.zeros(total_bins, device=device, dtype=torch.float64).scatter_add_(0, valid_ids, pt_y * pt_z)
        sum_zz_ols = torch.zeros(total_bins, device=device, dtype=torch.float64).scatter_add_(0, valid_ids, pt_z * pt_z)

        # 🚀 收集轨道 B 分子统计量（各项异性 WLS）
        weighted_points = valid_points_64 * valid_geo_weights.unsqueeze(1)
        sum_P_wls = torch.zeros(total_bins, 3, device=device, dtype=torch.float64)
        sum_P_wls.index_add_(0, valid_ids, weighted_points)
        pt_x_w = pt_x * valid_geo_weights
        pt_y_w = pt_y * valid_geo_weights
        pt_z_w = pt_z * valid_geo_weights
        sum_xx_wls = torch.zeros(total_bins, device=device, dtype=torch.float64).scatter_add_(0, valid_ids, pt_x * pt_x_w)
        sum_xy_wls = torch.zeros(total_bins, device=device, dtype=torch.float64).scatter_add_(0, valid_ids, pt_x * pt_y_w)
        sum_xz_wls = torch.zeros(total_bins, device=device, dtype=torch.float64).scatter_add_(0, valid_ids, pt_x * pt_z_w)
        sum_yy_wls = torch.zeros(total_bins, device=device, dtype=torch.float64).scatter_add_(0, valid_ids, pt_y * pt_y_w)
        sum_yz_wls = torch.zeros(total_bins, device=device, dtype=torch.float64).scatter_add_(0, valid_ids, pt_y * pt_z_w)
        sum_zz_wls = torch.zeros(total_bins, device=device, dtype=torch.float64).scatter_add_(0, valid_ids, pt_z * pt_z_w)

        # 解算解析去中心化重心
        centroids_ols = torch.nan_to_num(sum_P_ols / counts.clamp(min=1.0).unsqueeze(1), nan=0.0)
        centroids_wls = torch.nan_to_num(sum_P_wls / counts_for_cov.clamp(min=1e-6).unsqueeze(1), nan=0.0)
        centroids_wls = torch.where(valid_fit_mask.unsqueeze(1), centroids_wls, centroids_ols)

        # 严格同构去中心化协方差改正
        sum_PPt_ols = torch.stack([
            sum_xx_ols, sum_xy_ols, sum_xz_ols,
            sum_xy_ols, sum_yy_ols, sum_yz_ols,
            sum_xz_ols, sum_yz_ols, sum_zz_ols
        ], dim=1).reshape(total_bins, 3, 3)
        covariance_ols = sum_PPt_ols - torch.bmm(
            sum_P_ols.unsqueeze(2), sum_P_ols.unsqueeze(1)
        ) / counts.clamp(min=1.0).view(-1, 1, 1)

        sum_PPt_wls = torch.stack([
            sum_xx_wls, sum_xy_wls, sum_xz_wls,
            sum_xy_wls, sum_yy_wls, sum_yz_wls,
            sum_xz_wls, sum_yz_wls, sum_zz_wls
        ], dim=1).reshape(total_bins, 3, 3)
        covariance_wls = sum_PPt_wls - torch.bmm(
            sum_P_wls.unsqueeze(2), sum_P_wls.unsqueeze(1)
        ) / counts_for_cov.clamp(min=1e-6).view(-1, 1, 1)

        # 刚性正则化对角摄动，拉开奇异值，保护后向可微求导
        covariance_ols = torch.nan_to_num(covariance_ols, nan=0.0)
        covariance_wls = torch.nan_to_num(covariance_wls, nan=0.0)
        perturb_matrix = torch.diag(
            torch.tensor([1.0, 10.0, 100.0], device=device, dtype=torch.float64)
        ).unsqueeze(0) * 1e-4
        covariance_ols = covariance_ols + perturb_matrix
        covariance_wls = covariance_wls + perturb_matrix

        safe_matrix = torch.diag(
            torch.tensor([100.0, 10.0, 1.0], device=device, dtype=torch.float64)
        ).unsqueeze(0)
        covariance_ols_safe = torch.where(
            (counts > 3.0).view(-1, 1, 1).expand_as(covariance_ols),
            covariance_ols,
            safe_matrix.expand_as(covariance_ols)
        )
        covariance_wls_safe = torch.where(
            valid_fit_mask.view(-1, 1, 1).expand_as(covariance_wls),
            covariance_wls,
            safe_matrix.expand_as(covariance_wls)
        )

        hypotheses_pool = torch.zeros(total_bins, K, 4, device=device, dtype=torch.float32)
        surface_variation_out = torch.full((total_bins,), 0.333, device=device, dtype=torch.float32)

        # =====================================================================
        # Step 4: 高维谱分解求导与对称自旋十字星假设池繁衍
        # =====================================================================
        try:
            vals_ols, vecs_ols = torch.linalg.eigh(covariance_ols_safe.double())
            vals_wls, vecs_wls = torch.linalg.eigh(covariance_wls_safe.double())
            vals_wls, vecs_ols, vecs_wls = (
                vals_wls.float(),
                vecs_ols.float(),
                vecs_wls.float(),
            )

            # 提取无量纲局部表面粗糙度变差作为神经自适应旋钮
            surface_variation = (vals_wls[:, 0] / (vals_wls.sum(dim=1) + 1e-6)).detach()

            # 基础正交基底提取
            n_ols_base = F.normalize(vecs_ols[:, :, 0], dim=1)  # OLS 全局趋势面法线
            n_wls_base = F.normalize(vecs_wls[:, :, 0], dim=1)  # WLS 拉普拉斯精雕面法线
            e_tangent_1 = F.normalize(vecs_wls[:, :, 2], dim=1)  # 主延展切向量
            e_tangent_2 = F.normalize(vecs_wls[:, :, 1], dim=1)  # 次延展切向量

            centroids_ols_f = centroids_ols.float()
            centroids_wls_f = centroids_wls.float()

            for k_idx in range(K):
                if k_idx == 0:
                    # 👑 轨道 0：全局 trends 锚点解 (OLS) - 保全平整大面主权
                    n_curr = n_ols_base
                    c_curr = centroids_ols_f
                elif k_idx == 1:
                    # 👑 轨道 1：各项异性精雕边缘解 (拉普拉斯加权 WLS) - 姚敏火线修正！
                    n_curr = n_wls_base
                    c_curr = centroids_wls_f
                elif k_idx == 2:
                    # 👑 轨道 2：刚性前向平行保底面 - 锁死盲区高程突变
                    n_curr = torch.zeros_like(n_wls_base)
                    n_curr[:, 2] = -1.0
                    c_curr = centroids_ols_f
                else:
                    # 👑 轨道 3 ~ K-1：切空间周期自旋探索轨。
                    # 🪐【Bug 修复核心】：利用周期模算子 4 步循环，完美闭环 [+e1, -e1, +e2, -e2] 对称十字星繁衍！
                    local_idx = k_idx - 3
                    axis_type = local_idx % 4

                    if axis_type == 0:
                        axis, sign = e_tangent_1, 1.0
                    elif axis_type == 1:
                        axis, sign = e_tangent_1, -1.0
                    elif axis_type == 2:
                        axis, sign = e_tangent_2, 1.0
                    else:
                        axis, sign = e_tangent_2, -1.0

                    # 伴随扩展圈数自发线性放大探索半径系数
                    round_num = local_idx // 4
                    multiplier = 1.0 + round_num * 0.5
                    scale_factor = torch.clamp(
                        surface_variation * 6.0 * multiplier, min=0.0, max=0.20
                    ).unsqueeze(1)

                    n_curr = n_wls_base + sign * scale_factor * axis
                    c_curr = centroids_wls_f

                # 刚性重归一化并进行相机光心定向对齐
                n_curr = F.normalize(n_curr, dim=1)
                dot_p = torch.sum(n_curr * c_curr, dim=1, keepdim=True)
                flip_m = -torch.sign(dot_p)
                flip_m[flip_m == 0] = 1.0
                n_curr = n_curr * flip_m

                # 反解水密级绝对连续截距 d
                d_curr = -torch.sum(n_curr * c_curr, dim=1, keepdim=True)
                hypotheses_pool[:, k_idx, :3] = n_curr
                hypotheses_pool[:, k_idx, 3:4] = d_curr

            surface_variation_out = surface_variation

        except RuntimeError as e:
            print(f"[HYPO GEN] 谱矩阵简并报错，启动全平行熔断保底: {e}")
            safe_z = centroids_ols[:, 2:3].float().clamp(min=0.1)
            hypotheses_pool[:, :, :3] = 0.0
            hypotheses_pool[:, :, 2] = -1.0
            hypotheses_pool[:, :, 3:4] = safe_z.unsqueeze(1).expand(-1, K, -1)

        # =====================================================================
        # Step 5: 基于物理射线的几何异常一元交叉检验过滤与最终返回
        # =====================================================================
        with torch.no_grad():
            d_min_val, d_max_val = depth_bounds_flat()
            n_test = hypotheses_pool[:, 0, :3]
            d_test = hypotheses_pool[:, 0, 3:4]
            rays = centroids_ols.float()
            denom = torch.sum(n_test * rays, dim=1, keepdim=True)
            denom_safe = torch.where(
                denom.abs() < 1e-4, torch.sign(denom + 1e-10) * 1e-4, denom
            )

            # 🛡️ 稳健性加固：增加 .abs() 防御层，封杀由于极端符号震荡导致的几何误判
            depth_est = (-d_test * rays[:, 2:3] / denom_safe).abs()

            invalid_tris = (counts <= 3).unsqueeze(1)
            is_bad_geom = (
                (depth_est < d_min_val * 0.5)
                | (depth_est > d_max_val * 2.0)
                | (~torch.isfinite(depth_est))
            )
            is_invalid = invalid_tris | is_bad_geom | low_coverage_mask.unsqueeze(1)

        # 执行替换
        fallback_n = torch.zeros_like(hypotheses_pool[:, 0, :3])
        fallback_n[:, 2] = -1.0
        _, d_max_val = depth_bounds_flat()
        safe_z = centroids_ols[:, 2:3].float().clone()
        safe_z = torch.where(
            (safe_z <= 0.01) | (counts.unsqueeze(1) < 3), d_max_val, safe_z
        )
        fallback_hypo = (
            torch.cat([fallback_n, safe_z], dim=1).unsqueeze(1).expand(-1, K, -1)
        )

        hypotheses_pool = torch.where(
            is_invalid.unsqueeze(2).expand_as(hypotheses_pool),
            fallback_hypo,
            hypotheses_pool,
        )
        surface_variation_out = torch.where(
            is_invalid.squeeze(-1),
            torch.tensor(0.333, device=device, dtype=torch.float32),
            surface_variation_out,
        )

        return hypotheses_pool.view(
            B, max_num_triangles, K, 4
        ), surface_variation_out.view(B, max_num_triangles)

    def get_plane_hypotheses(self, depth_stage2, tri_id_map, intrinsics_s1, max_num_triangles):
        return self._get_plane_hypotheses_multi(depth_stage2, tri_id_map, intrinsics_s1, max_num_triangles)
       

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

    def collate_vertices_norm(self, vertices_norm_list, device):
        B = len(vertices_norm_list)
        N_max = max(v.shape[0] for v in vertices_norm_list)
        out = torch.zeros(B, N_max, 3, 2, device=device)
        for b, v in enumerate(vertices_norm_list):
            N_b = v.shape[0]
            out[b, :N_b, :, :] = v.to(device)
        return out

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
        利用平面参数渲染深度图和法向量图,为的是可视化
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
        # 4. 数值截断 (Clamp) - 解决 900/160 问题 todo:深度的最大最小值是否对应同一批次
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


