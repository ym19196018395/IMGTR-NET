import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import _sample_map, check_tensor


class EdgeHead(nn.Module):
    """
    EdgeHead —— 终极版边级断裂预测器 (Analytical Plane-based)
    升级特性：
      - 彻底抛弃 3x3 离散像素采样，免疫狭长三角形的特征崩溃
      - 基于平面方程 (n, d) 直接求解边中点的解析深度差，无限分辨率
      - 单点采样 Backbone 图像特征，保留 CNN 原生感受野
    输入:
      - feat: [B, C, Hf, Wf]  （backbone / FPN 特征）
      - img: [B, C_img, H, W] （原图，可为 None）
      - depth_map: [B,1,H,W] 或 [B,H,W] （当前网络的深度预测）
      - tri_infos: list length B, 每项为 dict:
                {
                    'batch_num_tri': int,三角形的数量
                    'centroids': [B,n_tri,2] 每个三角形的质点，已经归一化处理
                    'vertices': [B,n_tri,3,2] 每个三角形的顶点，已进行归一化处理
                    'edges_list' :每个边的邻接面，
                     boundary_local_idxs_per_batch: 存储着断裂边，也就是只有一个面的边
                    'tri_edge_ids_list':每个三角形的边ID列表
                    'edges_midpoints': 边对应的中点已经归一化 List[B] of [E, 2]
                }
    输出:
      - per_image_edges_alpha: list len=B，每个 tensor [E]，E = edges 数（按 edges 列表顺序）
      - per_image_edge_mat: list len=B，每个 tensor [num_tri, num_tri] 对称矩阵（不邻接处为0）
    """
    def __init__(self, feat_channels, tri_feat_dim=128, edge_mlp_hidden=128):
        """
        Args:
          feat_channels: backbone 特征通道数 C
          tri_feat_dim: 三角特征维度（masked pooling 后投影输出）
          edge_mlp_hidden: edge MLP 隐藏层大小
        """
        super().__init__()
        # 减少后续 pooling 与 MLP 的输入维度，降低参数与计算，同时让后续 tri_feats 维度固定，便于 MLP 设计
        # 用 1x1 conv 把 feat 投影到 tri_feat_dim，方便后续 pooling 与 MLP 输入维度一致
        self.proj = nn.Conv2d(feat_channels, tri_feat_dim, kernel_size=1)
        # edge_stat_dim = 1（只有 depth_diff）

        # 定义 MLP 输入维度
        # 输入组成:
        #   - Depth Diff (1 dim)
        #   - Normals (3+3=6 dims) -> T1 normal, T2 normal
        #   - Edge Feature (tri_feat_dim) -> 中点处的图像特征
        self.edge_stat_dim = 1 + 6
        mlp_in = tri_feat_dim + self.edge_stat_dim

        self.edge_mlp = nn.Sequential(
            nn.Linear(mlp_in, edge_mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(edge_mlp_hidden, edge_mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(edge_mlp_hidden, 1)
        )

        # 可学习缩放，用来调节 sigmoid 输入尺度（训练稳定）
        self.register_parameter("edge_scale", nn.Parameter(torch.tensor(10.0)))

        # ==========================================
        # 🔥 新增：闭门初始化 (Closed-gate Initialization)
        # ==========================================
        # 因为后面有 logits = mlp_out * self.edge_scale (10.0)
        # 我们把 bias 设为 0.2，那么初始 logits 就是 0.2 * 10 = 2.0
        # sigmoid(2.0) ≈ 0.88，这意味着一开始 88% 的概率被认为是边缘（阻断传播）
        # nn.init.constant_(self.edge_mlp[4].bias, 0.2)
        # nn.init.normal_(self.edge_mlp[4].weight, mean=0.0, std=0.01)


    def _compute_analytical_depth_diff(self, midpoints_norm, planes_t1, planes_t2, intrinsics, H, W):
        """
        使用平面方程精确计算边中点两侧的深度差 (无限分辨率，无视三角形大小)
        Args:
            midpoints_norm: [E, 2] 归一化坐标 [-1, 1]
            planes_t1: [E, 4] T1 的平面参数 (n, d)
            planes_t2: [E, 4] T2 的平面参数 (n, d)
            intrinsics: [3, 3] 当前视角的相机内参
            H, W: 图像高宽
        Returns:
            depth_diff: [E, 1]
        """
        E = midpoints_norm.shape[0]
        device = midpoints_norm.device

        # 1. 解除归一化: [-1, 1] -> 像素坐标 [0, W-1], [0, H-1]
        midpoints_uv = torch.zeros_like(midpoints_norm)
        midpoints_uv[:, 0] = (midpoints_norm[:, 0] + 1.0) / 2.0 * (W - 1)
        midpoints_uv[:, 1] = (midpoints_norm[:, 1] + 1.0) / 2.0 * (H - 1)

        # 2. 构造齐次坐标并反投影为射线方向 (Ray)
        ones = torch.ones(E, 1, device=device)
        uv_homo = torch.cat([midpoints_uv, ones], dim=-1)  # [E, 3]

        K_inv = torch.inverse(intrinsics)  # [3, 3]
        rays = torch.matmul(K_inv, uv_homo.unsqueeze(-1)).squeeze(-1)  # [E, 3]

        # 3. 提取平面参数 n 和 d
        n1, d1 = planes_t1[:, :3], planes_t1[:, 3:]
        n2, d2 = planes_t2[:, :3], planes_t2[:, 3:]

        # 4. 分别用 T1 和 T2 的平面方程计算该射线上的深度 Z
        # Z = -d / (n * ray)
        denom1 = torch.sum(n1 * rays, dim=-1, keepdim=True)
        denom1 = torch.where(torch.abs(denom1) < 1e-6, torch.tensor(1e-6, device=device), denom1)
        depth1 = -d1 / denom1  # [E, 1]

        denom2 = torch.sum(n2 * rays, dim=-1, keepdim=True)
        denom2 = torch.where(torch.abs(denom2) < 1e-6, torch.tensor(1e-6, device=device), denom2)
        depth2 = -d2 / denom2  # [E, 1]

        # 5. 获得绝对精确的深度差
        depth_diff = torch.abs(depth1 - depth2)  # [E, 1]

        return depth_diff

    def forward(self, feat, tri_infos, tri_planes, intrinsics):
        """
        EdgeHead.forward（兼容原始 tri_infos 格式）

        Args:
            - feat: [B, C, H, W]   (torch.Tensor)
            - tri_infos: list length B, 每项为 dict:
                {
                    'batch_num_tri': int,三角形的数量
                    'centers_list': [B,n_tri,2] 每个三角形的质点，已经归一化处理
                    'vertices_list': [B,n_tri,3,2] 每个三角形的顶点，已进行归一化处理
                    'edges_list' :每个边的邻接面，
                    'edges_pixels': 每个边的像素（归一化）集合，后续需要引入作为一个特征传入mlp中
                    'boundary_local_idxs_per_batch': 存储着断裂边，也就是只有一个面的边
                }
            - tri_planes: [B, N_max, 4] (当前拟合出的平面参数, n 和 d)
            - intrinsics: [B, 3, 3] (相机内参矩阵)
        Returns:
            output_alphas_list: list length B, 每项是 torch.Tensor shape [E_b]，类型 float，位于 feat.device 上
            tri_infos：已经处理完毕的数据
        """
        device = feat.device
        B, C, H, W = feat.shape

        feat_proj = self.proj(feat)  # [B, D, H, W]
        output_alphas_list = []
        # todo:后面改成并行运算
        for b in range(B):
            current_edges = tri_infos[0]['edges_list'][b].to(device)  # [E, 2]
            current_planes = tri_planes[b].to(device)  # [N_max, 4]
            curr_intrinsics = intrinsics[b].to(device) if intrinsics.dim() == 3 else intrinsics.to(device)

            curr_feat_map = feat_proj[b].unsqueeze(0)  # [1, D, H, W]

            num_edges = current_edges.shape[0]
            if num_edges == 0:
                output_alphas_list.append(torch.zeros(0, device=device))
                continue

            # --- Step A: 定位边中点和相关面 ---
            idx1 = current_edges[:, 0].long()
            idx2 = current_edges[:, 1].long()
            midpoints_norm = tri_infos[0]['edges_midpoints'][b]  # [E_b, 2]

            # --- Step B: 解析法计算深度差 (Analytical Depth Diff) ---
            planes_t1 = current_planes[idx1]  # [E, 4]
            planes_t2 = current_planes[idx2]  # [E, 4]

            depth_diff = self._compute_analytical_depth_diff(
                midpoints_norm, planes_t1, planes_t2, curr_intrinsics, H, W
            )
            depth_diff = torch.clamp(depth_diff, max=500.0)

            # --- Step C: 整合法向量 ---
            n1 = planes_t1[:, :3]
            n2 = planes_t2[:, :3]
            normal_feat = torch.cat([n1, n2], dim=1)  # [E, 6]

            # --- Step D: 单点采样边特征 ---
            # 直接提取中点这 1 个点的特征，利用 CNN 自带感受野，避免越界污染
            flat_midpoints = midpoints_norm.unsqueeze(0).unsqueeze(2)  # [1, E, 1, 2]
            edge_feats = F.grid_sample(
                curr_feat_map, flat_midpoints, align_corners=True
            ).view(feat_proj.shape[1], num_edges).permute(1, 0)  # [E, D]

            # --- Step E: MLP 预测 ---
            mlp_input = torch.cat([edge_feats, normal_feat, depth_diff], dim=1)  # [E, D+6+1]

            logits = self.edge_mlp(mlp_input).squeeze(1) * self.edge_scale
            alphas = torch.sigmoid(logits)  # [E]

            # 处理边界边 (self-loop edges, idx1 == idx2)
            # 这种情况下通常认为是边界，设为 1
            is_boundary = (idx1 == idx2)
            if is_boundary.any():
                # === 🛠️ 修复：先 clone 再修改，或者是创建一个新变量 ===
                # alphas.clone() 会创建一个新的 Tensor，但共享梯度历史。
                # 修改 clone 后的对象不会影响 SigmoidBackward 需要的原始数据。

                alphas = alphas.clone()
                alphas[is_boundary] = 1.0

            output_alphas_list.append(alphas)

        return output_alphas_list


class EdgeLabelGenerator:
    '''
    通过跨边双点正交采样，来判断边是否断裂，专门针对于whudata数据集倾斜摄影
    '''
    def __init__(self, relative_threshold=0.05, pixel_step=1.0):
        # relative_threshold: 允许的深度突变占总起伏的比例 (例如 5%)
        self.relative_threshold = relative_threshold
        self.pixel_step = pixel_step

    def __call__(self, pred_alphas_list, gt_depth_map, tri_infos, depth_min, depth_max):
        B = len(pred_alphas_list)
        device = gt_depth_map.device
        _, _, H0, W0 = gt_depth_map.shape

        gt_labels_list = []
        valid_mask_list = []

        # 计算在 [-1, 1] 归一化坐标系下的单像素步长
        step_u = self.pixel_step * (2.0 / W0)
        step_v = self.pixel_step * (2.0 / H0)
        step_tensor = torch.tensor([step_u, step_v], device=device).view(1, 2)  # [1, 2]

        # 计算当前场景的有效起伏范围 (例如 530 - 500 = 30)
        # 支持传入的可能是 Tensor 也可能是 float
        d_min_val = min(depth_min) if isinstance(depth_min, list) else float(depth_min)
        d_max_val = max(depth_max) if isinstance(depth_max, list) else float(depth_max)
        scene_depth_range = max(d_max_val - d_min_val, 1.0)  # 防止除零

        for b in range(B):
            pred_alphas = pred_alphas_list[b]
            E = pred_alphas.shape[0]

            if E == 0:
                gt_labels_list.append(torch.empty(0, device=device))
                valid_mask_list.append(torch.empty(0, dtype=torch.bool, device=device))
                continue

            # 1. 获取端点与中点
            # endpoints: [E, 2, 2] (V1, V2)
            endpoints = tri_infos[0]['edges_endpoints'][b].to(device)
            v1 = endpoints[:, 0, :]  # [E, 2]
            v2 = endpoints[:, 1, :]  # [E, 2]

            midpoints = (v1 + v2) / 2.0  # [E, 2]

            # 2. 计算正交法向量 (Orthogonal Vector)
            edge_vec = v2 - v1  # [E, 2]
            # 顺时针/逆时针旋转90度: (x, y) -> (-y, x)
            ortho_vec = torch.stack([-edge_vec[:, 1], edge_vec[:, 0]], dim=-1)
            ortho_vec = F.normalize(ortho_vec, p=2, dim=-1)  # 归一化为单位向量 [E, 2]

            # 3. 生成探测点 (Probe Points)
            scaled_ortho = ortho_vec * step_tensor  # [E, 2]
            probe_left = midpoints + scaled_ortho
            probe_right = midpoints - scaled_ortho

            # 限制在 [-1, 1] 范围内，防止越界
            probe_left = probe_left.clamp(-1.0, 1.0)
            probe_right = probe_right.clamp(-1.0, 1.0)

            # 4. 采样 GT 深度
            depth_b = gt_depth_map[b:b + 1]  # [1, 1, H0, W0]

            # grid_sample 需要 shape [1, E, 1, 2]
            depth_left = F.grid_sample(
                depth_b, probe_left.view(1, E, 1, 2),
                align_corners=True, padding_mode='border'
            ).squeeze()  # [E]

            depth_right = F.grid_sample(
                depth_b, probe_right.view(1, E, 1, 2),
                align_corners=True, padding_mode='border'
            ).squeeze()  # [E]

            # 5. 判定断裂与有效性
            depth_diff = torch.abs(depth_left - depth_right)
            # 除以场景的有效深度范围，而不是绝对深度之和
            # 引入相对深度阈值
            relative_diff = depth_diff / scene_depth_range

            # 断裂判定：突变高度超过了场景总起伏的 5%
            is_break = relative_diff > self.relative_threshold
            labels = is_break.float()  # [E]

            # 有效掩码：探测的两个点都必须有 GT 深度 (排除边界黑边干扰)
            valid_mask = (depth_left > 1e-4) & (depth_right > 1e-4)

            # --- 强制边界边断裂 ---
            # 如果这条边只属于一个三角形 (边界边)，必须强制为断裂
            current_edges = tri_infos[0]['edges_list'][b].to(device)
            is_boundary = (current_edges[:, 0] == current_edges[:, 1])
            labels[is_boundary] = 1.0
            # 边界边不需要探测点必须有效，强制视为有效
            valid_mask[is_boundary] = True

            gt_labels_list.append(labels)
            valid_mask_list.append(valid_mask)

        return gt_labels_list, valid_mask_list

