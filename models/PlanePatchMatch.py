import torch
import torch.nn as nn
import torch.nn.functional as F
from mpmath import eye


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
        为每个像素/三角形计算单应性矩阵 H
        Args:
            plane_params: [B, 4, H, W] (像素级平面参数) 或 [B, N, 4]
            K_ref, K_src: [B, 3, 3]
            R_rel, t_rel: [B, 3, 3], [B, 3] (Ref to Src 的相对位姿)
        Returns:
            H: [B, H, W, 3, 3] (每个像素一个 H)
        """
        B, C, H_img, W_img = plane_params.shape
        device = plane_params.device

        # 解析 n, d
        n = plane_params[:, :3, :, :]  # [B, 3, H, W]
        d = plane_params[:, 3:, :, :]  # [B, 1, H, W]

        # 1. 计算中间项 K_src * (R - t*n^T / d) * K_ref_inv
        # 为了高效，我们将 H, W 拉平处理
        n_flat = n.permute(0, 2, 3, 1).reshape(B, -1, 3, 1)  # [B, Pix, 3, 1]
        d_flat = d.permute(0, 2, 3, 1).reshape(B, -1, 1, 1)  # [B, Pix, 1, 1]

        # 准备 R, t
        R = R_rel.view(B, 1, 3, 3)
        t = t_rel.view(B, 1, 3, 1)

        # 核心公式: M = R - (t @ n.T) / d
        # t @ n.T -> [B, 1, 3, 1] @ [B, Pix, 1, 3] -> [B, Pix, 3, 3]
        tn_T = torch.matmul(t, n_flat.transpose(-2, -1))
        M = R - (tn_T / (d_flat + 1e-7))  # [B, Pix, 3, 3]

        # 前后乘内参
        K_src_expand = K_src.view(B, 1, 3, 3)
        K_ref_inv = torch.inverse(K_ref).view(B, 1, 3, 3)

        H_mat = torch.matmul(K_src_expand, torch.matmul(M, K_ref_inv))  # [B, Pix, 3, 3]

        return H_mat.view(B, H_img, W_img, 3, 3)

    def warp_feature(self, src_feature, H_mat):
        """
        利用 H 矩阵对 Src 特征进行 grid_sample
        Args:
            src_feature: [B, C, H, W]
            H_mat: [B, H, W, 3, 3]
        """
        B, C, H, W = src_feature.shape
        device = src_feature.device

        # 生成 Ref 图像的归一化坐标 grid
        y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        coords = torch.stack([x, y, torch.ones_like(x)], dim=-1).float()  # [H, W, 3]
        coords = coords.view(1, H, W, 3, 1).expand(B, -1, -1, -1, -1)  # [B, H, W, 3, 1]

        # 应用 H 变换: p_src = H @ p_ref
        # H_mat: [B, H, W, 3, 3]
        projected_coords = torch.matmul(H_mat, coords).squeeze(-1)  # [B, H, W, 3]

        # 归一化齐次坐标 (u, v, w) -> (u/w, v/w)
        w = projected_coords[..., 2:3]
        uv = projected_coords[..., :2] / (w + 1e-7)

        # 归一化到 [-1, 1]
        uv_norm = torch.zeros_like(uv)
        uv_norm[..., 0] = 2 * uv[..., 0] / (W - 1) - 1
        uv_norm[..., 1] = 2 * uv[..., 1] / (H - 1) - 1

        # 采样
        warped_feat = F.grid_sample(src_feature, uv_norm, align_corners=True, padding_mode='zeros')

        return warped_feat


class PlanePatchMatchModule(nn.Module):
    def __init__(self, fitter_module, num_hypotheses=3, G=8):
        """
        Args:
            fitter_module: 实例化好的 DensePlaneFitter 对象
            num_hypotheses: K (假设数量)
            G: Group Correlation 的组数 (默认8)
        """
        super().__init__()
        self.fitter = fitter_module
        self.num_hypotheses = num_hypotheses

        # 核心组件
        self.warper = PlaneHomographyWarper()

        # 复用 PatchMatchNet 的 Evaluation 模块 (用于计算特征相似度)
        # 假设 Evaluation 类可用，若未导入需从 patchmatch.py 导入
        # self.evaluation = Evaluation(G=G, stage=1)
        # 暂时用一个占位符或你需要确保 Evaluation 类在上下文中可用
        self.evaluation = None

    def forward(self, depth_stage2, tri_infos, ref_feature, src_features,
                ref_proj, src_projs, intrinsics_s1,depth_min, depth_max, view_weights=None):
        """
        Args:
            depth_stage2: [B, 1, H/4, W/4] (Stage 2 深度)
            tri_infos: List[Dict], 包含 'tri_id_map' 和 'batch_num_tri'
            ref_feature: [B, C, H, W] (Stage 1 参考特征)
            src_features: List[[B, C, H, W]] (Stage 1 源特征列表)
            ref_proj: List[[B, 4, 4]] (参考图投影矩阵, 用于计算相对位姿)
            intrinsics_s1: 所有视图的内参矩阵
            src_projs: List[[B, 4, 4]] (源视图投影矩阵, 用于计算相对位姿)
            depth_min, depth_max: 深度范围
            view_weights: [B, N_view] (可选，视图权重)

        Returns:
            depth_samples: List[[B, 1, H, W]] (这里只返回一项)
            score: [B, 1, H, W] (置信度)
            view_weights: [B, N_view] (返回传入的权重或None)
        """
        B, C, H, W = ref_feature.shape
        device = ref_feature.device

        # ==========================================
        # 1. 数据准备 (Data Preparation)
        # ==========================================
        # 处理 tri_id_map: List[[1, H, W]] -> [B, H, W]
        # 对应源代码中的 stack 逻辑
        if isinstance(tri_infos[0]['tri_id_map'], list):
            # 兼容处理
            processed_list = [t.squeeze(0) for t in tri_infos[0]['tri_id_map']]
            tri_id_map = torch.stack(processed_list, dim=0).to(device)
        else:
            tri_id_map = tri_infos[0]['tri_id_map'].to(device)  # 假设已经是 Tensor

        # 获取 batch 内最大的三角形数量
        # 注意：tri_infos 结构可能比较复杂，这里按你提供的逻辑获取
        # 假设 tri_infos[0] 包含了所有 batch 的信息
        if 'batch_num_tri' in tri_infos[0]:
            batch_num_tri_list = tri_infos[0]['batch_num_tri']
            # 如果是 tensor 转 list
            if isinstance(batch_num_tri_list, torch.Tensor):
                batch_num_tri_list = batch_num_tri_list.tolist()
            max_num_tri = max(batch_num_tri_list)
        else:
            # 备用逻辑
            max_num_tri = 2000  # 默认值或报错

        # ==========================================
        # 2. 拟合与生成 (Fitting & Generation)
        # ==========================================
        # 使用优化后的 get_plane_hypotheses 直接得到 [B, N, K, 4]
        # 它内部包含了 By_SVD_Plane (Hypo 0) 和 Hypothesis Expansion (Hypo 1~K)
        hypotheses = self.fitter.get_plane_hypotheses(
            depth_stage2=depth_stage2,
            tri_id_map=tri_id_map,
            intrinsics_s1=ref_proj,
            max_num_triangles=max_num_tri
        )  # Output: [B, N_tri, K, 4]

        # ==========================================
        # 3. 广播 (Broadcasting: Triangle -> Pixel)
        # ==========================================
        # 将三角形级的假设映射到像素级
        # pixel_hypotheses: [B, H, W, K, 4]
        pixel_hypotheses = self.map_tri_to_pixel(hypotheses, tri_id_map, H, W)

        current_hypotheses = pixel_hypotheses

        # ==========================================
        # 4. 传播 (Propagation) - 暂时跳过
        # ==========================================
        # pass

        # ==========================================
        # 5. 代价计算 (Cost Computation)
        # ==========================================
        # 计算所有假设的代价
        # costs: [B, H, W, K]
        costs = self.compute_costs(
            ref_feature, src_features,
            ref_proj, src_projs,
            current_hypotheses,
            view_weights
        )

        # ==========================================
        # 6. 评估与选择 (Evaluation & Selection)
        # ==========================================
        # 简化版: Hard Argmin (直接取 Cost 最小的平面)
        # best_idx: [B, H, W]
        best_idx = torch.argmin(costs, dim=3)

        # Gather 最佳平面参数
        # [B, H, W, K, 4] -> [B, H, W, 4]
        # 需要构造 gather 索引
        gather_idx = best_idx.unsqueeze(3).unsqueeze(4).expand(-1, -1, -1, 1, 4)
        best_planes = torch.gather(current_hypotheses, 3, gather_idx).squeeze(3)  # [B, H, W, 4]

        # ==========================================
        # 7. 渲染与输出 (Rendering)
        # ==========================================
        # 动态实例化 Visualizer 以适应当前 H, W
        visualizer = PlaneVisualizer(H, W, device)

        # 渲染深度图 (这里直接用像素级平面参数渲染，不需要再传 tri_id_map 了，或者复用接口)
        # 注意：render_from_planes 原本接收 [B, N, 4]，但我们现在有 [B, H, W, 4]
        # 为了复用 visualizer，我们可以写一个专门针对 pixel-wise plane 的渲染函数，
        # 或者在这里直接用公式算：depth = -d / (n * K^-1 * uv)
        # 为了简单，假设 visualizer 有个 render_pixel_wise 或者我们手动算一下：

        # 手动快速渲染 (参考 Visualizer 逻辑)
        # n: [B, H, W, 3], d: [B, H, W, 1]
        n_map = best_planes[..., :3]
        d_map = best_planes[..., 3:]

        # 构建坐标网格
        y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        # [B, H, W]
        fx = ref_proj[:, 0, 0].view(B, 1, 1)
        fy = ref_proj[:, 1, 1].view(B, 1, 1)
        cx = ref_proj[:, 0, 2].view(B, 1, 1)
        cy = ref_proj[:, 1, 2].view(B, 1, 1)

        u_bar = (x.unsqueeze(0) - cx) / fx
        v_bar = (y.unsqueeze(0) - cy) / fy

        dot_product = (n_map[..., 0] * u_bar) + (n_map[..., 1] * v_bar) + (n_map[..., 2] * 1.0)
        dot_product = dot_product.clamp(min=1e-6)  # 防止除零（注意符号）

        # 正常情况下 dot_product 应该是负的(指向相机)，这里取绝对值简化
        depth_sample = (-d_map.squeeze(-1) / dot_product).abs()

        # Clamp
        if isinstance(depth_max, torch.Tensor):
            depth_max = depth_max.to(device).view(B, 1, 1)
        depth_sample = torch.clamp(depth_sample, min=depth_min, max=depth_max)

        # 组装输出
        depth_samples = [depth_sample.unsqueeze(1)]  # List[[B, 1, H, W]]

        # Score (Confidence) = -min_cost
        score = -torch.min(costs, dim=3)[0].unsqueeze(1)  # [B, 1, H, W]

        return depth_samples, score, view_weights

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
        # 待实现
        pass

    def compute_costs(self, ref_feature, src_features, ref_proj, src_projs, current_hypotheses, view_weights):
        """
        计算代价体积
        Returns:
            costs: [B, H, W, K]
        """
        # 待实现
        pass

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

        # [Hypothesis 1] 前向平行 (Fronto-Parallel)
        # 利用现成的 centroids[:, 2] (平均深度)
        # n=[0,0,1], d = -mean_depth
        mean_depth = centroids[:, 2].view(B, max_num_triangles, 1, 1)

        normal_fp = torch.zeros((B, max_num_triangles, 1, 3), device=device)
        normal_fp[..., 2] = -1.0
        d_fp = -mean_depth  # d = -z

        hypo_fp = torch.cat([normal_fp, d_fp], dim=-1)  # [B, N, 1, 4]
        hypo_list.append(hypo_fp)

        # [Hypothesis 2+] 随机扰动 (Vectorized Jitter)
        num_random = self.K - 2
        if num_random > 0:
            # 扩展基础平面 [B, N, 1, 4] -> [B, N, num_rnd, 4]
            base_n = hypo_0[..., :3].expand(-1, -1, num_random, -1)
            base_d = hypo_0[..., 3:].expand(-1, -1, num_random, -1)

            # 生成噪声
            rand_n = (torch.rand_like(base_n) - 0.5) * self.noise_scale
            # d 的扰动范围需要大一点
            rand_d = (torch.rand_like(base_d) - 0.5) * self.noise_scale * 50.0

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

        # # 方案 A: 物理正确可视化
        # normal_vis = (normal_vis + 1.0) / 2.0

        # 方案 B: 视觉友好可视化 (偏蓝) -> 用于 Tensorboard 展示
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

        # ym-needmodify，暂时进行一个修改本来是0.0
        depth_map[invalid_mask] = depth_map.max()

        depth_map = depth_map.unsqueeze(1)

        return depth_map, normal_vis


