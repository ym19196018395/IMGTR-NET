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
        self.grid_y, self.grid_x = torch.meshgrid(y_range, x_range)  # [H, W]

        # 2. 生成用于 grid_sample 的归一化坐标 (针对 Stage 2)
        # 归一化到 [-1, 1]
        self.norm_u = (self.grid_x / (self.W - 1)) * 2.0 - 1.0
        self.norm_v = (self.grid_y / (self.H - 1)) * 2.0 - 1.0
        self.sampling_grid = torch.stack((self.norm_u, self.norm_v), dim=-1)  # [H, W, 2]

    def forward(self, depth_stage2, tri_id_map, intrinsics_s1, max_num_triangles):
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

        # --- 🛡️ 诊断 3: 检查深度值是否有效 ---
        if sampled_depth.max() < 1e-4:
            print(f"❌ [Fatal Error] Sampled Depth 几乎全为 0 (Max={sampled_depth.max().item():.6f})！")
            print("   -> 请检查 Stage 2 的输出深度是否正常，或者 grid_sample 的坐标是否正确。")
            return self._get_fallback_planes(B, max_num_triangles, depth_stage2.device)

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

        # --- 🛡️ 诊断 4: 检查有效像素数量 ---
        if valid_mask.sum() == 0:
            print("❌ [Fatal Error] valid_mask 全是 False！没有一个像素同时满足 ID>=0 和 Depth>0。")
            # 打印一下两边的状态
            print(f"   -> Valid IDs count: {(tri_ids_flat >= 0).sum().item()}")
            print(f"   -> Valid Depth count: {(points_flat[:, 2] > 1e-4).sum().item()}")
            return self._get_fallback_planes(B, max_num_triangles, depth_stage2.device)

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
            print(f"❌ [Error] global_tri_ids 越界！Max ID={valid_ids.max()}, Total Bins={total_bins}")
            print("   -> 请检查传入的 max_num_triangles 是否小于实际存在的三角形 ID。")
            return self._get_fallback_planes(B, max_num_triangles, depth_stage2.device)

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

        # --- 3.4 特征值分解求解法向量 ---
        # 求解 Cov * n = lambda * n
        # eigh 适用于实对称矩阵，速度快且稳定
        try:
            # torch.symeig (旧版) 或 torch.linalg.eigh (新版)
            # 假设你用的是 1.7+，用 symeig
            vals, vecs = torch.symeig(covariance, eigenvectors=True)

            # 取最小特征值对应的特征向量
            normals = vecs[:, :, 0].float()  # [Total, 3]
            centroids = centroids.float()

        except RuntimeError as e:
            print(f"⚠️ SVD Failed: {e}")
            normals = torch.zeros(total_bins, 3, device=self.device, dtype=torch.float32)
            normals[:, 2] = 1.0
            centroids = centroids.float()

        # ==========================================================
        # 统一法向量方向 (Orient Normals)
        # ==========================================================
        # 你的兜底策略是 (0, 0, 1)，说明你希望法向量指向 +Z 方向（通常是相机前方）
        # 我们检查每个法向量的 Z 分量。如果 n_z < 0，说明它指向了 -Z，翻转它！

        # 1. 获取 Z 分量
        nz = normals[:, 2:3]  # [Total, 1]

        # 2. 计算翻转系数：如果 nz < 0 则为 -1，否则为 1
        # 这里的 1e-6 是防止 0 的情况
        flip_mask = torch.sign(nz + 1e-6)

        # 3. 翻转法向量
        # 如果原来是 [0, 0, -1] -> 乘 -1 -> [0, 0, 1] (正确)
        # 如果原来是 [0, 0, 1]  -> 乘 1  -> [0, 0, 1] (保持)
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

        # 组合结果
        plane_params = torch.cat([normals, d_vals], dim=1)

        return plane_params.view(B, max_num_triangles, 4)

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
        self.grid_y, self.grid_x = torch.meshgrid(y_range, x_range)

    def render_from_planes(self, plane_params, tri_id_map, intrinsics):
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

        # === 🛡️ 修复 2: 确保使用的是内参 K，而不是投影矩阵 P ===
        # Proj = K [R|t] -> 4x4
        # Intrinsics = K -> 3x3
        if intrinsics.shape[-1] == 4:
            raise ValueError("错误：你传入了 4x4 的投影矩阵 (ref_proj)，这里需要 3x3 的内参矩阵 (intrinsics)！")

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

        depth_map[invalid_mask] = 0

        depth_map = depth_map.unsqueeze(1)

        return depth_map, normal_vis


