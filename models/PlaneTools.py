import torch
import torch.nn as nn
import torch.nn.functional as F


class PlaneHypothesisGenerator(nn.Module):
    def __init__(self, num_hypotheses=5, perturbation_range=0.1):
        super().__init__()
        self.K = num_hypotheses
        self.noise_scale = perturbation_range

    def forward(self, fitted_planes, depth_stage2_mean):
        """
        Args:
            fitted_planes: [B, N_tri, 4] (由 Fitter 生成的初始平面)
            depth_stage2_mean: [B, N_tri, 1] (每个三角形的平均深度，用于生成前向平行假设)
        Returns:
            hypotheses: [B, N_tri, K, 4]
        """
        B, N, _ = fitted_planes.shape
        device = fitted_planes.device

        # 1. Hypothesis 0: 原始拟合平面 (Best Guess)
        hypo_0 = fitted_planes.unsqueeze(2)  # [B, N, 1, 4]

        # 2. Hypothesis 1: 前向平行平面 (Fronto-Parallel Backup)
        # n = [0,0,1], d = -mean_depth
        normal_fp = torch.zeros((B, N, 1, 3), device=device)
        normal_fp[..., 2] = 1.0
        d_fp = -depth_stage2_mean.unsqueeze(2)  # [B, N, 1, 1]
        hypo_1 = torch.cat([normal_fp, d_fp], dim=-1)

        # 3. Hypothesis 2~K: 随机扰动 (Random Perturbation)
        # 对法向量 n 和距离 d 加噪声
        num_random = self.K - 2
        if num_random > 0:
            rand_n = (torch.rand((B, N, num_random, 3), device=device) - 0.5) * self.noise_scale
            rand_d = (torch.rand((B, N, num_random, 1), device=device) - 0.5) * self.noise_scale * 10.0  # d 的范围通常较大

            base_n = fitted_planes[:, :, :3].unsqueeze(2)
            base_d = fitted_planes[:, :, 3:].unsqueeze(2)

            new_n = F.normalize(base_n + rand_n, dim=-1)
            new_d = base_d + rand_d
            hypo_random = torch.cat([new_n, new_d], dim=-1)

            # 拼接所有假设
            hypotheses = torch.cat([hypo_0, hypo_1, hypo_random], dim=2)
        else:
            hypotheses = torch.cat([hypo_0, hypo_1], dim=2)

        return hypotheses  # [B, N, K, 4]


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


class DensePlaneFitter:
    def __init__(self, height_s1, width_s1, device):
        """
        Args:
            height_s1, width_s1: Stage 1 的分辨率
        """
        self.H = height_s1
        self.W = width_s1
        self.device = device

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

            # # === 🛠️ 修复 3: 检查 SVD 输出是否含有 NaN ===
            # # try-except 抓不到 NaN 结果，必须手动查
            # if torch.isnan(normals).any():
            #     # print("⚠️ Warning: NaNs detected in normals after SVD. Fixing...")
            #     nan_mask = torch.isnan(normals).any(dim=1)
            #     normals[nan_mask] = torch.tensor([0.0, 0.0, 1.0], device=device)  # 默认朝前

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

        dot_product = -dot_product

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
            # 获取无效三角形对应的质心Z值
            fallback_z = centroids[invalid_tris, 2]
            # 防止深度为0导致后续除零
            fallback_z[fallback_z == 0] = 10.0

            # 兜底策略：使用 Hypothesis 2 (前向平行)
            normals[invalid_tris] = torch.tensor([0.0, 0.0, 1.0], device=self.device)
            d_vals[invalid_tris, 0] = -fallback_z


        # === 🛠️ 修复 4: 最终输出前的 NaN 检查 ===
        # 确保万无一失，防止任何 NaN 导致可视化全黑
        # if torch.isnan(normals).any() or torch.isnan(d_vals).any():
        #     print("⚠️ Final check: NaNs found in plane params. Cleaning.")
        #     normals = torch.nan_to_num(normals, nan=0.0)
        #     d_vals = torch.nan_to_num(d_vals, nan=-10.0)
        #     # 确保 normals 模长不为0
        #     norm_len = torch.norm(normals, dim=1, keepdim=True)
        #     zero_len = norm_len < 1e-6
        #     normals[zero_len.squeeze()] = torch.tensor([0.0, 0.0, 1.0], device=device)

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
        depth_map[invalid_mask] = depth_map.min()

        depth_map = depth_map.unsqueeze(1)

        return depth_map, normal_vis


