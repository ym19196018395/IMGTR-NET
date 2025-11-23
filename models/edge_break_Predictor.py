# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
#
#
# class EdgeFeatureExtractor(nn.Module):
#     """
#     从三角形和边的信息中提取特征
#     用于判断边是否为断裂
#     """
#
#     def __init__(self, feat_dim=32):
#         super(EdgeFeatureExtractor, self).__init__()
#         self.feat_dim = feat_dim
#
#     def forward(self, depth_map, normal_map, feat_map, tri_id_map, edge_list):
#         """
#         Args:
#             depth_map: (B, 1, H, W) 深度图
#             normal_map: (B, 3, H, W) 法向图
#             feat_map: (B, C, H, W) 骨干特征
#             tri_id_map: (B, H, W) 三角形ID映射
#             edge_list: (num_edges, 2) 边连接的两个三角形ID
#
#         Returns:
#             edge_features: (B, edge_feat_dim, num_edges)
#         """
#         batch_size = depth_map.shape, [object Object],
#         num_edges = edge_list.shape, [object Object],
#         device = depth_map.device
#
#         # 初始化边特征张量
#         edge_features_list = []
#
#         for edge_idx in range(num_edges):
#             tri_id_1, tri_id_2 = edge_list[edge_idx]
#
#             # 找到属于两个三角形的像素掩码
#             mask_1 = (tri_id_map == tri_id_1)
#             mask_2 = (tri_id_map == tri_id_2)
#
#             if mask_1.sum() > 0 and mask_2.sum() > 0:
#                 # ===== 1. 深度特征 =====
#                 depth_1 = depth_map[mask_1].mean()
#                 depth_2 = depth_map[mask_2].mean()
#                 depth_diff = torch.abs(depth_1 - depth_2)
#                 depth_std_1 = depth_map[mask_1].std()
#                 depth_std_2 = depth_map[mask_2].std()
#
#                 # ===== 2. 法向特征 =====
#                 normal_1 = normal_map[:, mask_1].mean(dim=-1)  # (B, 3)
#                 normal_2 = normal_map[:, mask_2].mean(dim=-1)  # (B, 3)
#                 normal_diff = torch.norm(normal_1 - normal_2, dim=0)
#                 normal_dot = torch.sum(normal_1 * normal_2, dim=0)  # 夹角余弦
#
#                 # ===== 3. 特征相似性 =====
#                 feat_1 = feat_map[:, :, mask_1].mean(dim=-1)  # (B, C)
#                 feat_2 = feat_map[:, :, mask_2].mean(dim=-1)  # (B, C)
#                 feat_diff = torch.norm(feat_1 - feat_2, dim=1)
#                 feat_sim = F.cosine_similarity(feat_1, feat_2, dim=1)
#
#                 # ===== 4. 颜色特征 =====
#                 color_1 = depth_map[mask_1].mean()  # 这里用深度代替，实际应该用RGB
#                 color_2 = depth_map[mask_2].mean()
#
#                 # ===== 5. 组合特征 =====
#                 edge_feat = torch.stack([
#                     depth_diff,  # 0: 深度差
#                     depth_std_1,  # 1: 三角形1深度方差
#                     depth_std_2,  # 2: 三角形2深度方差
#                     normal_diff,  # 3: 法向差
#                     normal_dot,  # 4: 法向夹角余弦
#                     feat_diff,  # 5: 特征差
#                     feat_sim,  # 6: 特征相似度
#                 ], dim=0)  # (7,)
#
#                 edge_features_list.append(edge_feat)
#             else:
#                 # 如果边的一侧没有像素，使用默认特征
#                 edge_features_list.append(torch.zeros(7, device=device))
#
#         # 堆叠所有边的特征
#         edge_features = torch.stack(edge_features_list, dim=-1)  # (7, num_edges)
#         edge_features = edge_features.unsqueeze(0).expand(batch_size, -1, -1)  # (B, 7, num_edges)
#
#         return edge_features
#
#     class EdgeBreakPredictor(nn.Module):
#         """
#         基于边特征预测边断裂概率
#         输出: alpha ∈ [0, 1]，表示边是否为断裂
#         """
#
#         def __init__(self, in_channels=7, hidden_channels=64):
#             super(EdgeBreakPredictor, self).__init__()
#
#             # 边特征编码
#             self.edge_encoder = nn.Sequential(
#                 nn.Linear(in_channels, hidden_channels),
#                 nn.BatchNorm1d(hidden_channels),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(hidden_channels, hidden_channels),
#                 nn.BatchNorm1d(hidden_channels),
#                 nn.ReLU(inplace=True),
#             )
#
#             # 断裂判决头
#             self.break_head = nn.Sequential(
#                 nn.Linear(hidden_channels, hidden_channels // 2),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(hidden_channels // 2, 1),
#                 nn.Sigmoid()  # 输出 [0, 1]
#             )
#
#         def forward(self, edge_features):
#             """
#             Args:
#                 edge_features: (B, feat_dim, num_edges)
#
#             Returns:
#                 alpha: (B, num_edges) 边断裂概率
#             """
#             batch_size, feat_dim, num_edges = edge_features.shape
#
#             # 转换为 (B*num_edges, feat_dim)
#             edge_feat_flat = edge_features.permute(0, 2, 1).reshape(-1, feat_dim)
#
#             # 编码边特征
#             encoded = self.edge_encoder(edge_feat_flat)  # (B*num_edges, hidden)
#
#             # 预测断裂概率
#             alpha_flat = self.break_head(encoded)  # (B*num_edges, 1)
#             alpha = alpha_flat.reshape(batch_size, num_edges)  # (B, num_edges)
#
#             return alpha
#
#
# class EdgeConsistencyLoss(nn.Module):
#     """
#     自监督损失：根据深度一致性学习边断裂
#     无需外部gt_alpha标签
#     """
#
#     def __init__(self, depth_threshold=0.05, smooth_weight=0.1):
#         super(EdgeConsistencyLoss, self).__init__()
#         self.depth_threshold = depth_threshold
#         self.smooth_weight = smooth_weight
#
#     def forward(self, pred_alpha, depth_map, tri_id_map, edge_list, feat_map=None):
#         """
#         Args:
#             pred_alpha: (B, num_edges) 预测的断裂概率
#             depth_map: (B, 1, H, W) 深度图
#             tri_id_map: (B, H, W) 三角形ID
#             edge_list: (num_edges, 2) 边连接
#             normal_map: (B, 3, H, W) 法向图（可选）
#             feat_map: (B, C, H, W) 特征图（可选）
#
#         Returns:
#             loss: 标量损失
#         """
#         batch_size = depth_map.shape, [object Object],
#         num_edges = edge_list.shape, [object Object],
#         device = depth_map.device
#
#         total_loss = 0.0
#
#         # ===== 遍历每条边 =====
#         for edge_idx in range(num_edges):
#             tri_id_1, tri_id_2 = edge_list[edge_idx]
#
#             # 找到两个三角形的像素
#             mask_1 = (tri_id_map == tri_id_1)
#             mask_2 = (tri_id_map == tri_id_2)
#
#             if mask_1.sum() > 0 and mask_2.sum() > 0:
#                 # ===== 计算边的一致性度量 =====
#
#                 # 1. 深度一致性
#                 depth_1 = depth_map[mask_1].mean()
#                 depth_2 = depth_map[mask_2].mean()
#                 depth_diff = torch.abs(depth_1 - depth_2)
#
#                 # 归一化深度差 [0, 1]
#                 depth_diff_norm = torch.clamp(depth_diff / (self.depth_threshold + 1e-6), 0, 1)
#
#                 consistency_score = depth_diff_norm
#
#
#                 # 3. 特征一致性（如果提供）
#                 if feat_map is not None:
#                     feat_1 = feat_map[:, :, mask_1].mean(dim=-1)
#                     feat_2 = feat_map[:, :, mask_2].mean(dim=-1)
#                     feat_dist = torch.norm(feat_1 - feat_2, dim=0)
#                     feat_consistency = torch.clamp(feat_dist / 10.0, 0, 1)
#                     consistency_score = (consistency_score + feat_consistency) / 2
#
#                 # ===== 自监督损失 =====
#                 # 如果consistency_score高（不一致），alpha应该高（断裂）
#                 # 如果consistency_score低（一致），alpha应该低（连续）
#
#                 alpha_edge = pred_alpha[0, edge_idx]
#
#                 # 交叉熵式损失
#                 break_loss = -torch.log(alpha_edge + 1e-6) * consistency_score - \
#                              torch.log(1 - alpha_edge + 1e-6) * (1 - consistency_score)
#
#                 total_loss += break_loss
#
#         # 平均损失
#         total_loss = total_loss / max(num_edges, 1)
#
#         # ===== 平滑性约束（可选） =====
#         # 相邻边的断裂概率应该相似
#         if self.smooth_weight > 0:
#             smooth_loss = self._compute_smoothness(pred_alpha, edge_list)
#             total_loss = total_loss + self.smooth_weight * smooth_loss
#
#         return total_loss
#
#     def _compute_smoothness(self, pred_alpha, edge_list):
#         """
#         相邻三角形的断裂概率应该相似
#         """
#         num_edges = pred_alpha.shape, [object Object],
#         smooth_loss = 0.0
#
#         # 构建三角形邻接关系
#         tri_neighbors = {}
#         for edge_idx, (tri_id_1, tri_id_2) in enumerate(edge_list):
#             if tri_id_1 not in tri_neighbors:
#                 tri_neighbors[tri_id_1] = []
#             if tri_id_2 not in tri_neighbors:
#                 tri_neighbors[tri_id_2] = []
#             tri_neighbors[tri_id_1].append((tri_id_2, edge_idx))
#             tri_neighbors[tri_id_2].append((tri_id_1, edge_idx))
#
#         # 计算相邻边的断裂概率差
#         for tri_id, neighbors in tri_neighbors.items():
#             for i in range(len(neighbors)):
#                 for j in range(i + 1, len(neighbors)):
#                     edge_idx_1 = neighbors[i], [object Object],
#                     edge_idx_2 = neighbors[j], [object Object],
#                     alpha_1 = pred_alpha[0, edge_idx_1]
#                     alpha_2 = pred_alpha[0, edge_idx_2]
#                     smooth_loss += (alpha_1 - alpha_2) ** 2
#
#         return smooth_loss / max(len(tri_neighbors), 1)
#
#
# class EdgeAwarePropagation(nn.Module):
#     """
#     在PatchMatchNet的传播阶段中应用边断裂约束
#     """
#
#     def __init__(self):
#         super(EdgeAwarePropagation, self).__init__()
#
#     def forward(self, depth_map, pred_alpha, edge_list, tri_id_map,
#                 cost_volume=None, num_iterations=3):
#         """
#         Args:
#             depth_map: (B, 1, H, W) 当前深度图
#             pred_alpha: (B, num_edges) 边断裂概率
#             edge_list: (num_edges, 2) 边连接
#             tri_id_map: (B, H, W) 三角形ID
#             cost_volume: (B, D, H, W) 代价体（可选）
#             num_iterations: 传播迭代次数
#
#         Returns:
#             refined_depth: (B, 1, H, W) 细化后的深度图
#         """
#         batch_size, _, height, width = depth_map.shape
#         device = depth_map.device
#
#         # 创建边界掩码：标记哪些像素对之间是断裂边
#         break_mask = self._build_break_mask(
#             pred_alpha, edge_list, tri_id_map, height, width, device
#         )  # (B, H, W)
#
#         refined_depth = depth_map.clone()
#
#         # ===== 多次迭代传播 =====
#         for iter_idx in range(num_iterations):
#             # 1. 随机搜索（保持原始）
#             random_depth = self._random_search(refined_depth, cost_volume)
#
#             # 2. 受断裂约束的传播
#             propagated_depth = self._constrained_propagation(
#                 refined_depth, random_depth, break_mask, cost_volume
#             )
#
#             # 3. 评估和更新
#             refined_depth = self._evaluate_and_update(
#                 refined_depth, propagated_depth, cost_volume
#             )
#
#         return refined_depth
#
#     def _build_break_mask(self, pred_alpha, edge_list, tri_id_map,
#                           height, width, device):
#         """
#         构建边界掩码：标记断裂边处的像素
#
#         Returns:
#             break_mask: (B, H, W) 1表示在断裂边附近，0表示可以传播
#         """
#         batch_size = pred_alpha.shape, [object Object],
#         break_mask = torch.zeros(batch_size, height, width, device=device)
#
#         for edge_idx, (tri_id_1, tri_id_2) in enumerate(edge_list):
#             alpha = pred_alpha[0, edge_idx]  # 断裂概率
#
#             if alpha > 0.5:  # 判断为断裂
#                 # 标记这条边
#                 mask_1 = (tri_id_map == tri_id_1)
#                 mask_2 = (tri_id_map == tri_id_2)
#
#                 # 在两个三角形的交界处标记
#                 break_mask[mask_1 | mask_2] = alpha
#
#         return break_mask
#
#     def _constrained_propagation(self, current_depth, random_depth,
#                                  break_mask, cost_volume):
#         """
#         受断裂约束的传播：
#         - 在break_mask=0的区域：正常传播（使用邻域信息）
#         - 在break_mask=1的区域：使用随机搜索结果
#         """
#         batch_size, _, height, width = current_depth.shape
#         device = current_depth.device
#
#         # 初始化传播结果
#         propagated = current_depth.clone()
#
#         # 四个方向的传播
#         directions = [
#             (0, 1),  # 右
#             (0, -1),  # 左
#             (1, 0),  # 下
#             (-1, 0)  # 上
#         ]
#
#         for dy, dx in directions:
#             # 创建偏移后的深度图
#             depth_shifted = torch.roll(current_depth, shifts=(dy, dx), dims=(2, 3))
#             mask_shifted = torch.roll(break_mask, shifts=(dy, dx), dims=(1, 2))
#
#             # 权重：如果边界处有断裂，权重为0
#             weight = 1.0 - break_mask  # (B, H, W)
#
#             # 加权融合
#             propagated = propagated * (1 - weight.unsqueeze(1)) + \
#                          depth_shifted * weight.unsqueeze(1)
#
#         return propagated
#
#     def _random_search(self, depth_map, cost_volume):
#         """
#         简化的随机搜索（实际应该参考PatchMatchNet实现）
#         """
#         batch_size, _, height, width = depth_map.shape
#         device = depth_map.device
#
#         # 生成随机深度扰动
#         random_delta = torch.randn_like(depth_map) * 0.1
#         random_depth = depth_map + random_delta
#
#         return random_depth
#
#     def _evaluate_and_update(self, current_depth, candidate_depth, cost_volume):
#         """
#         评估候选深度并更新
#         """
#         # 这里应该计算代价并选择更优的深度
#         # 简化版本：直接返回候选深度
#         return candidate_depth
#
#
# class PatchMatchNet_EdgeAware(nn.Module):
#     """
#     集成边断裂预测的PatchMatchNet
#     """
#
#     def __init__(self, patchmatchnet_model):
#         super(PatchMatchNet_EdgeAware, self).__init__()
#         self.patchmatchnet = patchmatchnet_model
#
#         # 边断裂预测模块
#         self.edge_feature_extractor = EdgeFeatureExtractor(feat_dim=32)
#         self.edge_predictor = EdgeBreakPredictor(in_channels=7, hidden_channels=64)
#         self.edge_propagation = EdgeAwarePropagation()
#
#         # 损失函数
#         self.edge_consistency_loss = EdgeConsistencyLoss(
#             depth_threshold=0.05, smooth_weight=0.1
#         )
#
#     def forward(self, ref_img, src_imgs, intrinsics, extrinsics,
#                 tri_id_map, edge_list, depth_init=None, training=False):
#         """
#         Args:
#             ref_img: (B, 3, H, W)
#             src_imgs: (B, N, 3, H, W)
#             intrinsics: (B, 3, 3)
#             extrinsics: (B, N, 4, 4)
#             tri_id_map: (B, H, W)
#             edge_list: (num_edges, 2)
#             depth_init: (B, 1, H, W) 可选
#             training: 是否为训练模式
#
#         Returns:
#             depth_map: (B, 1, H, W)
#             alpha_pred: (B, num_edges)
#             loss: (标量) 仅在training=True时返回
#         """
#
#         # ===== Step 1: 运行原始PatchMatchNet =====
#         depth_map, normal_map = self.patchmatchnet(
#             ref_img, src_imgs, intrinsics, extrinsics, depth_init
#         )
#
#         # ===== Step 2: 提取中间特征 =====
#         # 需要修改PatchMatchNet以返回特征，这里假设可以获取
#         feat_map = self.patchmatchnet.get_features(ref_img)  # (B, C, H, W)
#
#         # ===== Step 3: 提取边特征 =====
#         edge_features = self.edge_feature_extractor(
#             depth_map, normal_map, feat_map, tri_id_map, edge_list
#         )  # (B, feat_dim, num_edges)
#
#         # ===== Step 4: 预测边断裂 =====
#         alpha_pred = self.edge_predictor(edge_features)  # (B, num_edges)
#
#         # ===== Step 5: 受约束的传播细化 =====
#         depth_refined = self.edge_propagation(
#             depth_map, alpha_pred, edge_list, tri_id_map,
#             cost_volume=None, num_iterations=3
#         )
#
#         # ===== Step 6: 计算损失（训练时） =====
#         loss = None
#         if training:
#             loss = self.edge_consistency_loss(
#                 alpha_pred, depth_refined, tri_id_map, edge_list,
#                 normal_map=normal_map, feat_map=feat_map
#             )
#
#         return depth_refined, alpha_pred, loss
#
#
# class TrainingPipeline:
#     """
#     完整的训练流程
#     """
#
#     def __init__(self, model, device='cuda', lr=1e-4):
#         self.model = model.to(device)
#         self.device = device
#
#         # 损失函数
#         self.photometric_loss = nn.L1Loss()
#
#         # 优化器
#         self.optimizer = torch.optim.Adam(
#             self.model.parameters(), lr=lr
#         )
#
#     def train_step(self, batch_data):
#         """
#         单个训练步骤
#         """
#         ref_img = batch_data['ref_img'].to(self.device)
#         src_imgs = batch_data['src_imgs'].to(self.device)
#         intrinsics = batch_data['intrinsics'].to(self.device)
#         extrinsics = batch_data['extrinsics'].to(self.device)
#         tri_id_map = batch_data['tri_id_map'].to(self.device)
#         edge_list = batch_data['edge_list'].to(self.device)
#         gt_depth = batch_data['gt_depth'].to(self.device)  # 真实深度（如果有）
#
#         # ===== Forward Pass =====
#         depth_pred, alpha_pred, edge_loss = self.model(
#             ref_img, src_imgs, intrinsics, extrinsics,
#             tri_id_map, edge_list, training=True
#         )
#
#         # ===== 计算损失 =====
#         losses = {}
#
#         # 1. 深度损失（如果有gt_depth）
#         if gt_depth is not None:
#             depth_loss = self.photometric_loss(depth_pred, gt_depth)
#             losses['depth_loss'] = depth_loss
#         else:
#             depth_loss = 0.0
#
#         # 2. 边断裂自监督损失
#         edge_consistency_loss = edge_loss
#         losses['edge_loss'] = edge_consistency_loss
#
#         # 3. 总损失
#         total_loss = depth_loss + 0.5 * edge_consistency_loss
#         losses['total_loss'] = total_loss
#
#         # ===== 反向传播 =====
#         self.optimizer.zero_grad()
#         total_loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
#         self.optimizer.step()
#
# # 假设: model produces pred_depth (B,1,Hc,Wc), feat, and model.edge_head returns alpha_list, edge_mats, tri_depths_list
# pred = model(images)
# pred_depth = pred['depth']              # [B,1,Hc,Wc]
# feat = pred['feat']                    # backbone feature (use appropriate scale)
# alpha_list, edge_mats_list, tri_depths_list = model.edge_head(feat, images, pred_depth, tri_infos)
#
# # 1) pseudo labels from GT tri_depths (GT depth must be pooled to tri)
# tri_depths_gt_list = pooled_tri_depths_from_gt(depth_gt, tri_infos)  # list per image
# pseudo_labels_list = []
# for b in range(B):
#     t1 = edge_list_list[b][:,0]; t2 = edge_list_list[b][:,1]
#     d1 = tri_depths_gt_list[b][t1]; d2 = tri_depths_gt_list[b][t2]
#     d_diff = torch.abs(d1 - d2)
#     pseudo = torch.sigmoid(k*(d_diff - tau))
#     pseudo_labels_list.append(pseudo)
#
# # 2) EdgeConsistencyLoss (BCE)
# pred_alpha_flat = torch.cat([a.view(-1) for a in alpha_list])
# pseudo_flat = torch.cat([p.view(-1).to(pred_alpha_flat.device) for p in pseudo_labels_list])
# L_alpha_sup = F.binary_cross_entropy(pred_alpha_flat, pseudo_flat)
#
# # 3) continuity loss (uses pred_depth and edge_mats_list)
# L_cont, diag = continuity_loss_edge(pred_depth, tri_id_map_list, edge_mats_list,
#                                    mask=valid_mask, lambda_cont=1.0, lambda_sparsity=0.0)
# # Note: function signature may already multiply lambda_cont internally; adjust accordingly.
#
# # 4) depth GT supervision
# L_depth = depth_loss(pred_depth, depth_gt)  # L1 or smooth L1
#
# # 5) total
# loss = L_depth + lambda_alpha * L_alpha_sup + lambda_cont * L_cont + lambda_sparsity * mean_alpha
#
# # backward & step (optimizer includes depth & edgehead params if joint training)
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()
#
