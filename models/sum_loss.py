import torch
import argparse
import os
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F

import math
from utils import compute_normal_map_torch

def compute_edge_supervision_loss(
        edge_label_generator,
        pred_alphas_list,
        gt_depth_map,
        tri_infos,
        tri_id_map
):
    """
    封装边预测头 (EdgeHead) 的监督 Loss 计算全流程。
    包含: 伪标签生成 -> Mask 过滤 -> 加权 BCE -> 稀疏性惩罚统计。
    """
    # 1. 生成 Stage 1 伪真值标签和有效掩码
    gt_stage1_list, valid_mask_list = edge_label_generator(
        pred_alphas_list=pred_alphas_list,
        gt_depth_map=gt_depth_map,
        tri_infos=tri_infos,
        tri_id_map=tri_id_map
    )

    B = len(pred_alphas_list)
    device = gt_depth_map.device  # 提前获取 device，避免后续报错

    loss_alpha_sup = 0.0
    total_sparsity_loss = 0.0

    valid_bce_count = 0  # 记录有多少张图成功计算了 BCE
    valid_sparsity_count = 0  # 记录有多少张图成功计算了稀疏性

    for b in range(B):
        pred_alphas = pred_alphas_list[b]

        # 如果当前图没有任何边，直接跳过
        if pred_alphas.numel() == 0:
            continue

        valid_mask = valid_mask_list[b]
        gt_targets = gt_stage1_list[b]

        # --- 统计稀疏性 (只针对有边的图) ---
        total_sparsity_loss += pred_alphas.mean()
        valid_sparsity_count += 1

        # 如果当前图没有任何有效的 GT 边，跳过 BCE 计算
        if valid_mask.sum() == 0:
            continue

        # --- 计算加权 BCE ---
        valid_pred = pred_alphas[valid_mask]
        valid_target = gt_targets[valid_mask]

        loss_alpha_sup += compute_weighted_bce_core(valid_pred, valid_target)
        valid_bce_count += 1

    # ==========================================
    # 聚合求平均 (防止除 0 并保持计算图连通)
    # ==========================================

    # 1. BCE Loss 平均
    if valid_bce_count > 0:
        loss_alpha_sup = loss_alpha_sup / valid_bce_count
    else:
        # 🔥 关键修复：加上 requires_grad=True
        # 万一某个极端的 Batch 全被跳过了，直接返回 0.0 会导致 total_loss.backward() 找不到梯度图崩溃
        loss_alpha_sup = torch.tensor(0.0, device=device, requires_grad=True)

    # 2. 稀疏性 Loss 平均 (修复了你原来的 Bug)
    if valid_sparsity_count > 0:
        loss_sparsity = total_sparsity_loss / valid_sparsity_count
    else:
        loss_sparsity = torch.tensor(0.0, device=device, requires_grad=True)

    return loss_alpha_sup, loss_sparsity,gt_stage1_list


def compute_edge_supervision_loss_new(
        alphas_list,     # [B] 方案A中唯一一次预测的边断裂概率 (0~1)
        gt_stage0_list,  # [B] 绝对真值 (纯净的 0 或 1)
        valid_mask_list  # [B] 有效掩码
):
    """
    方案 A 专属：单次精准深层监督 Loss 计算。
    只接收 Iter 1 在干净平面上预测出的高质量边缘概率。
    """
    B = len(alphas_list)
    device = alphas_list[0].device if B > 0 else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    total_loss_alpha = 0.0
    total_sparsity = 0.0

    valid_count = 0

    for b in range(B):
        alphas = alphas_list[b]
        gt_target = gt_stage0_list[b]
        valid_mask = valid_mask_list[b]

        # 异常跳过
        if alphas.numel() == 0 or valid_mask.sum() == 0:
            continue

        # 1. 稀疏性惩罚 (直接对概率求平均)
        total_sparsity += alphas.mean()

        # 2. 提取有效区域的概率和真值
        valid_preds = alphas[valid_mask]
        valid_target = gt_target[valid_mask]

        # 3. 直接调用你的防爆 BCE 核心 (绝不再做 Sigmoid！)
        total_loss_alpha += compute_weighted_bce_core(valid_preds, valid_target)

        valid_count += 1

    # ==========================================
    # 4. 聚合计算
    # ==========================================
    if valid_count > 0:
        total_loss_alpha = total_loss_alpha / valid_count
        total_sparsity = total_sparsity / valid_count
    else:
        # 防崩溃保底机制
        total_loss_alpha = torch.tensor(0.0, device=device, requires_grad=True)
        total_sparsity = torch.tensor(0.0, device=device, requires_grad=True)

    return total_loss_alpha, total_sparsity

def get_dynamic_loss_weights(global_step, total_steps):
    """
    动态计算 Loss 权重
    """
    # 进度比例: 0.0 -> 1.0
    progress = global_step / total_steps

    # ==========================================
    # 1. 边缘监督权重 (alpha_sup) - 逐渐衰减
    # ==========================================
    # 起始权重为 10.0，随着训练进行，指数衰减到 1.0 甚至 0.1
    # 我们这里设计一个衰减曲线：前期保持较高，中后期迅速下降
    alpha_start = 10.0
    alpha_end = 1.0

    # 使用余弦退火曲线，过渡极其平滑
    # 当 progress=0 时为 1，progress=1 时为 0
    decay_factor = 0.5 * (1.0 + math.cos(math.pi * progress))

    weight_alpha = alpha_end + (alpha_start - alpha_end) * decay_factor

    # ==========================================
    # 2. 连续性约束权重 (continuity) - 逐渐增强 或 保持稳定
    # ==========================================
    # 刚开始拟合的平面很乱，强行做连续性约束会导致网络崩溃
    # 所以让它从 0.01 慢慢涨到 0.3，接管后期的边缘学习
    cont_start = 0.01
    cont_end = 0.3

    # 线性增长
    weight_cont = cont_start + (cont_end - cont_start) * progress

    # ==========================================
    # 3. 稀疏性惩罚 (sparsity) - 必须始终保持一定强度防止作弊
    # ==========================================
    # 随着 continuity 权重上升，作弊的收益变大，所以惩罚也要跟上
    weight_spars = 1.0 + 1.0 * progress  # 从 1.0 涨到 2.0

    return weight_alpha, weight_cont, weight_spars


def compute_weighted_bce_core(preds, targets):
    """
    纯粹的加权 BCE 计算核心 (防爆版)
    输入必须是经过 valid_mask 筛选后的 1D tensor
    """
    preds = torch.clamp(preds, 1e-6, 1.0 - 1e-6)
    num_pos = (targets > 0.5).sum().float()
    num_neg = (targets <= 0.5).sum().float()

    if num_pos < 1.0:
        return F.binary_cross_entropy(preds, targets)

    pos_weight = torch.clamp(num_neg / num_pos, min=1.0, max=40.0)

    loss_pos = - pos_weight * targets * torch.log(preds)
    loss_neg = - (1.0 - targets) * torch.log(1.0 - preds)
    return (loss_pos + loss_neg).mean()


def compute_pixel_cost_margin_loss(no_prop_depth, gt_depth, pixel_costs, tri_id_map,
                                   good_rel_thresh=0.01, bad_rel_thresh=0.05):
    """
    [完全体] 像素级 Cost Margin Loss：尺度不变 & 样本均衡 & 边界截断 & 网格掩码保护

    Args:
        no_prop_depth:   [B, 1, H, W] SVD 初值深度 (务必 detach!)
        gt_depth:        [B, 1, H, W] GT 深度
        pixel_costs:     [B, H, W, 1] 模型输出的代价 (L2归一化点积取负)
        tri_id_map:      [B, H, W] Stage 1 的三角形 ID 图，用于过滤无三角形的无效背景 (-1)
        good_rel_thresh: 好平面相对误差阈值 (默认 1%)
        bad_rel_thresh:  坏平面相对误差阈值 (默认 5%)
    """
    if pixel_costs.dim() == 4 and pixel_costs.shape[-1] == 1:
        pixel_costs = pixel_costs.permute(0, 3, 1, 2)  # [B, 1, H, W]

    # ==========================================
    # 1. 终极有效掩码 (剔除天空、无效 GT、无效网格、SVD 爆炸区)
    # ==========================================
    # 深度有效性
    valid_depth_mask = (gt_depth > 1e-3) & (no_prop_depth > 1e-3)
    # 网格有效性 (排除 tri_id_map < 0 的背景或天空)
    valid_tri_mask = (tri_id_map >= 0).unsqueeze(1)  # [B, 1, H, W]

    valid_mask = valid_depth_mask & valid_tri_mask

    # ==========================================
    # 2. 计算相对深度误差 (Scale-Invariant)
    # ==========================================
    rel_error = torch.abs(no_prop_depth - gt_depth) / (gt_depth + 1e-6)

    # ==========================================
    # 3. 划分好坏阵营
    # ==========================================
    mask_good = valid_mask & (rel_error <= good_rel_thresh)
    mask_bad = valid_mask & (rel_error >= bad_rel_thresh)

    # ==========================================
    # 4. 引入 Margin 思想 (单侧惩罚)
    # ==========================================
    loss_good = 0.0
    loss_bad = 0.0

    if mask_good.sum() > 0:
        # 只惩罚那些 "明明是好几何，但 Cost 却大于 -0.8" 的像素
        costs_good = pixel_costs[mask_good]
        # F.relu 惩罚超出 -0.8 的部分
        loss_good = F.relu(costs_good - (-0.8)).mean()

    if mask_bad.sum() > 0:
        # 只惩罚那些 "明明是坏几何，但 Cost 却居然小于 -0.2 (作弊匹配上了)" 的像素
        costs_bad = pixel_costs[mask_bad]
        # 惩罚低于 -0.2 的部分
        loss_bad = F.relu((-0.2) - costs_bad).mean()

    # ==========================================
    # 5. 正负样本解耦相加 (Hard Negative Mining)
    # ==========================================
    # 给坏平面(通常是建筑边缘)更大的权重 2.0，逼迫 FeatureNet 去抠边缘！
    total_loss = loss_good + 2.0 * loss_bad

    # 打印监控日志 (用于观察特征网的训练状态)
    with torch.no_grad():
        good_ratio = mask_good.float().mean().item()
        bad_ratio = mask_bad.float().mean().item()
        # 避免训练初期大量刷屏，你可以加上 if 限制输出频率
        print(f"[Cost Loss] Good: {good_ratio:.3f} | Bad: {bad_ratio:.3f} | "
              f"L_good: {loss_good.item() if isinstance(loss_good, torch.Tensor) else 0:.4f} | "
              f"L_bad: {loss_bad.item() if isinstance(loss_bad, torch.Tensor) else 0:.4f}")

    return total_loss

def compute_normal_cosine_loss(final_planes, tri_id_map, gt_normals_math_s0, depth_stage_1):
    """
    计算 Stage 1 预测平面与 Stage 0 GT 法向量之间的余弦相似度损失。

    Args:
        final_planes: [B, N, 4] Stage 1 的平面参数
        tri_id_map: [B, H, W] Stage 1 的三角形 ID 图
        gt_normals_math_s0: [B, 3, H0, W0] Stage 0 的数学真值法向量 [-1, 1]
        depth_stage_1: [B, 1, H, W] Stage 1 的预测/GT 深度，用于过滤无效背景

    Returns:
        normal_loss: 标量 Loss
    """
    B, N, _ = final_planes.shape
    _, H, W = tri_id_map.shape
    device = final_planes.device

    # =======================================================
    # 步骤 A: 对 Stage 0 的 GT 法向量进行正确的下采样
    # =======================================================
    # 必须使用 bilinear 插值，防止法向量出现最近邻的“马赛克锯齿”
    gt_normals_s1 = F.interpolate(
        gt_normals_math_s0,
        size=(H, W),  # 直接对齐目标尺寸，比 scale_factor 更安全
        mode='bilinear',
        align_corners=False
    )
    # 🌟 极其关键：插值后向量长度缩水，必须重新 L2 归一化
    gt_normals_s1 = F.normalize(gt_normals_s1, p=2, dim=1)

    # =======================================================
    # 步骤 B: 高效提取 Stage 1 预测的“纯数学”法向量
    # =======================================================
    # 1. 取出预测的法向量 n [B, N, 3]，并归一化
    pred_tri_normals = final_planes[..., :3]
    pred_tri_normals = F.normalize(pred_tri_normals, p=2, dim=-1)

    # 2. 处理无效区域 ID (将 -1 暂时替换为 0 以防查表越界)
    safe_id_map = tri_id_map.clone()
    invalid_mask = (safe_id_map < 0)
    safe_id_map[invalid_mask] = 0

    # 3. 极其高效的查表渲染 (替代复杂的 for 循环)
    batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, H, W)
    pred_pixel_normals = pred_tri_normals[batch_idx, safe_id_map]  # [B, H, W, 3]
    pred_pixel_normals = pred_pixel_normals.permute(0, 3, 1, 2)  # [B, 3, H, W]

    # =======================================================
    # 步骤 C: 计算余弦相似度损失
    # =======================================================
    # 1. 计算点积 sum(N_pred * N_gt)
    cos_sim = torch.sum(pred_pixel_normals * gt_normals_s1, dim=1, keepdim=True)  # [B, 1, H, W]

    # 2. 转换为 Loss (1.0 - cos_sim)，完全同向时 Loss 为 0
    loss_map = 1.0 - cos_sim

    # 3. 构建有效 mask (去除原来 tri_id_map < 0 的区域，以及深度为 0 的背景)
    valid_mask = (~invalid_mask.unsqueeze(1)) & (depth_stage_1 > 1e-4)

    # 4. 最终法向损失
    normal_loss = loss_map[valid_mask].mean()

    # 容错：如果整张图全部无效（极少见），返回 0 梯度
    if torch.isnan(normal_loss):
        normal_loss = torch.tensor(0.0, device=device, requires_grad=True)

    return normal_loss

def compute_normal_cosine_loss_s0(N_pred, N_gt, valid_mask):
    """
    【架构师特供】Stage 0 全分辨率法向量纯净余弦损失
    在全分辨率空间直接进行像素级张量对撞，剥离了一切查表与插值的架构负担。

    Args:
        N_pred:     [B, 3, H, W] Stage 0 抛光网络预测出的最终法向量 (例如 outputs['refined_normal'])
        N_gt:       [B, 3, H, W] Stage 0 原分辨率数学真值法向量 [-1, 1]
        valid_mask: [B, 1, H, W] 有效像素掩码 (通常使用数据集自带的 mask['stage_0'] > 0.5)

    Returns:
        normal_loss_s0: 标量 Loss
    """
    # =======================================================
    # 1. 架构级防御：强制 L2 归一化 (防爆锁死)
    # 虽然 N_pred 在内部做过归一化，但为了防止外部梯度扰动，
    # Loss 算子必须在入口处自己再上一次保险！
    # =======================================================
    N_pred_norm = F.normalize(N_pred, p=2, dim=1)
    N_gt_norm = F.normalize(N_gt, p=2, dim=1)

    # =======================================================
    # 2. 纯粹的代数点积 (Cosine Similarity)
    # 向量完全同向时 dot = 1.0，完全反向时 dot = -1.0
    # =======================================================
    cos_sim = torch.sum(N_pred_norm * N_gt_norm, dim=1, keepdim=True)  # [B, 1, H, W]

    # =======================================================
    # 3. 极化并计算 Loss 均值
    # 1.0 - cos_sim 使得完全同向时 Loss = 0.0
    # =======================================================
    loss_map = 1.0 - cos_sim

    # 4. 掩码物理拦截：只对真实的建筑/物体计算梯度，背景虚空不提供监督
    if valid_mask.dtype != torch.bool:
        mask_bool = valid_mask > 0.5
    else:
        mask_bool = valid_mask

        # 确保掩码维度与 loss_map [B, 1, H, W] 能够广播对齐
    if mask_bool.dim() == 3:  # 如果是 [B, H, W]
        mask_bool = mask_bool.unsqueeze(1)

    relevant_loss = loss_map[mask_bool]

    # 6. 极端情况兜底机制：如果整张图都被掩码遮蔽，返回安全的 0 梯度
    if relevant_loss.numel() == 0:
        return torch.tensor(0.0, device=N_pred.device, requires_grad=True)

    return relevant_loss.mean()


def compute_gated_dnc_loss(Z_pixel, N_pixel, W_plane_pixel, tri_id_map, intrinsics, valid_mask=None, eps=1e-6):
    """
        【首席审计官终极校准版】有源门控密集深度-法向一致性损失函数
        - 拓扑边界隔离（Boundary Masking）：利用 tri_id_map 空间差分，彻底切除网格分段常数导致的悬崖求导奇点
        - 空间流形钝化（Low-pass Smoothing）：内置 3x3 低通滤波器，熨平晶格多视图匹配的微观量化锯齿
        - 神经主权保护（Gradients Firewall）：对输入的 W_plane_pixel 强制注入 .detach()，阻断多任务博弈内战
    """
    B, C_z, H, W = Z_pixel.shape
    device = Z_pixel.device

    # =====================================================================
    # 1. 🛡️ 密集矩阵拓扑排查防御总线 (彻底绞杀隐式广播与通道混乱)
    # =====================================================================
    if Z_pixel.shape[1] != 1 and Z_pixel.shape[-1] == 1:
        Z_pixel = Z_pixel.permute(0, 3, 1, 2)

    # 【🎯 修正点一：闭环法向 Channel-Last 防御，防止 H==3 的极限边界情况混淆】
    if N_pixel.dim() == 4 and N_pixel.shape[-1] == 3 and N_pixel.shape[1] != 3:
        N_pixel = N_pixel.permute(0, 3, 1, 2)

    if W_plane_pixel.shape[1] != 1 and W_plane_pixel.shape[-1] == 1:
        W_plane_pixel = W_plane_pixel.permute(0, 3, 1, 2)

    # 自适应规范化整型拓扑图 tri_id_map 的张量形态为 [B, 1, H, W]
    if tri_id_map.dim() == 3:
        tri_id_map = tri_id_map.unsqueeze(1)
    elif tri_id_map.shape[-1] == 1 and tri_id_map.dim() == 4:
        tri_id_map = tri_id_map.permute(0, 3, 1, 2)

    # 🛡️ 建立梯度图防火墙：斩断有源门控的后向内战，禁止 Loss 逆向篡改 Stage 1 置信度特征权重
    W_plane_pixel = W_plane_pixel.detach()

    # =====================================================================
    # 2. 🚀 空间流形消毒与低通钝化
    # =====================================================================
    Z_pixel_clean = torch.nan_to_num(Z_pixel, nan=0.0, posinf=0.0, neginf=0.0)
    Z_smoothed = F.avg_pool2d(Z_pixel_clean, kernel_size=3, stride=1, padding=1)

    # 3. 构造密集相机反投影射线场
    y, x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    coords = torch.stack([x, y, torch.ones_like(x)], dim=0).float().to(device)  # [3, H, W]
    coords = coords.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, 3, H, W]

    inv_K = torch.linalg.inv(intrinsics)  # [B, 3, 3]
    flat_coords = coords.view(B, 3, -1)
    rays = torch.bmm(inv_K, flat_coords).view(B, 3, H, W)  # [B, 3, H, W]

    # 物理反投影得到钝化后的密集 3D 空间流形点云
    P = rays * Z_smoothed  # [B, 3, H, W]

    # =====================================================================
    # 4. 施加连续空间前向差分算子，推导表面局部单位切向量场
    # =====================================================================
    P_u_next = F.pad(P[:, :, :, 1:], (0, 1, 0, 0), mode='replicate')
    P_v_next = F.pad(P[:, :, 1:, :], (0, 0, 0, 1), mode='replicate')

    t_u = P_u_next - P  # 切向量 \partial P / \partial u
    t_v = P_v_next - P  # 切向量 \partial P / \partial v

    # 注入微观物理安全偏置 eps，封杀绝对平坦平面处的零模长导数黑洞
    t_u_norm = t_u / (torch.norm(t_u, p=2, dim=1, keepdim=True) + eps)
    t_v_norm = t_v / (torch.norm(t_v, p=2, dim=1, keepdim=True) + eps)

    # 5. 切平面与预测法向量进行像素级物理正交惩罚
    dot_u = torch.abs(torch.sum(t_u_norm * N_pixel, dim=1, keepdim=True))
    dot_v = torch.abs(torch.sum(t_v_norm * N_pixel, dim=1, keepdim=True))
    loss_dnc_map = dot_u + dot_v  # [B, 1, H, W] 一致性异常地图

    # 6. 有源置信度门控联合对撞
    gated_loss_map = loss_dnc_map * W_plane_pixel

    # =====================================================================
    # 7. 🎯【核心重构点】拓扑边界查表隔离锁 (彻底放逐分段常数带来的接缝断层悬崖)
    # =====================================================================
    tri_id_float = tri_id_map.float()
    tri_id_shift_x = F.pad(tri_id_float[:, :, :, 1:], (0, 1, 0, 0), mode='replicate')
    tri_id_shift_y = F.pad(tri_id_float[:, :, 1:, :], (0, 0, 0, 1), mode='replicate')

    # 判定规则：只要相邻两晶格对应的三角形 ID 不同，说明差分算子越过了断崖悬崖，切向量不可靠，标记为拓扑边界
    is_boundary = (tri_id_float != tri_id_shift_x) | (tri_id_float != tri_id_shift_y)  # [B, 1, H, W] (Bool)

    # 8. 综合密集掩码裁切总线
    edge_mask = torch.ones_like(Z_pixel, dtype=torch.bool)
    edge_mask[:, :, :, -1] = False
    edge_mask[:, :, -1, :] = False

    # 排除差分边界，同时通过按位与非（& ~is_boundary）从源头上干净、彻底地剥离接缝毒素像素
    final_valid = edge_mask & (~is_boundary)

    if valid_mask is not None:
        if valid_mask.dim() == 3:
            valid_mask = valid_mask.unsqueeze(1)
        final_valid = final_valid & (valid_mask > 0.5)

    relevant_loss = gated_loss_map[final_valid]

    # 无效区域自卫保底
    if relevant_loss.numel() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return relevant_loss.mean()

def compute_confidence_supervision_loss(pixel_depth_pred, pixel_depth_gt, pixel_normal_pred, tri_id_map, W_pred, progress,
                                        depth_range=None, valid_mask=None, eps=1e-7, intrinsics=None, pixel_counts=None,
                                        tri_conf_gt=None, is_gt_planar=None):
    """
    【首席审判者·动态重构自愈完全体·几何辅助版】不确定性软标签回归损失函数
    
    1. 动态真值主轨回归 (W_GT_dynamic)：
       - 投影深度误差正交清洗：使用预测法线与视线射线的余弦因子，将绝对深度差映射为视点不变的表面正交距离。
       - Cauchy 映射：将三角形平均正交误差 E_tri 转换为动态置信度真值，随前向重建精度实时协变。
       - 纯 L1 损失监督主干，消除量纲炸弹与对抗梯度。
    2. 静态几何特定区间辅助监督 (L_aux)：
       - 对 tri_conf_gt (W_GT_static) 极值区进行区间锁定。
       - 在 >= 0.8 (绝对大平面) 区，惩罚 W_pred < 0.8 的部分；
       - 在 <= 0.2 (绝对曲折/噪点) 且点数 >= 3 区，惩罚 W_pred > 0.2 的部分；
       - 彻底放过 0.2 ~ 0.8 灰区，解除光度-几何冲突，大图置信度强力去噪。
    """
    device = pixel_depth_pred.device
    B, _, H, W = pixel_depth_pred.shape
    num_triangles = W_pred.shape[1]

    # =====================================================================
    # 1. 🛡️ 防御性维度形态排查与 Channel-First 强行对齐
    # =====================================================================
    if pixel_depth_pred.shape[1] != 1 and pixel_depth_pred.shape[-1] == 1:
        pixel_depth_pred = pixel_depth_pred.permute(0, 3, 1, 2)
    if pixel_depth_gt.shape[1] != 1 and pixel_depth_gt.shape[-1] == 1:
        pixel_depth_gt = pixel_depth_gt.permute(0, 3, 1, 2)
    if pixel_normal_pred.shape[1] != 3 and pixel_normal_pred.shape[-1] == 3:
        pixel_normal_pred = pixel_normal_pred.permute(0, 3, 1, 2)

    if tri_id_map.dim() == 3:
        tri_id_map = tri_id_map.unsqueeze(1)
    elif tri_id_map.shape[-1] == 1 and tri_id_map.dim() == 4:
        tri_id_map = tri_id_map.permute(0, 3, 1, 2)

    # =====================================================================
    # 2. 复用刚性深度范围并注入课程学习进度
    # =====================================================================
    if depth_range is not None:
        min_d, max_d = depth_range
        val_min = min_d.mean().item() if isinstance(min_d, torch.Tensor) else float(min_d)
        val_max = max_d.mean().item() if isinstance(max_d, torch.Tensor) else float(max_d)
        span = max(val_max - val_min, 1e-3)
    else:
        span = 400.0

    # 课程学习温标 Schedules
    sigma_max = span * 0.015
    sigma_min = span * 0.0065
    prog_val = max(0.0, min(float(progress), 1.0))
    adaptive_sigma = sigma_max * (1.0 - prog_val) + sigma_min * prog_val
    adaptive_sigma = max(adaptive_sigma, 0.1)

    # =====================================================================
    # 3. 🚀【3D 解析空间正交清洗：利用外层传入解析法线直接通流】
    # =====================================================================
    # 3.a 构造纯净相机晶格位置矩阵
    y, x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij',
    )
    coords = torch.stack([x, y, torch.ones_like(x)], dim=0).float().to(device)  # [3, H, W]
    coords = coords.unsqueeze(0).expand(B, -1, -1, -1)  # [B, 3, H, W]

    # 3.b 提取物理视锥射线 field 形态
    if intrinsics is not None:
        inv_k = torch.linalg.inv(intrinsics.float())
        flat_coords = coords.view(B, 3, -1)
        rays_cam = torch.bmm(inv_k, flat_coords).view(B, 3, H, W)
    else:
        focal = float(max(H, W))
        cx, cy = (W - 1) * 0.5, (H - 1) * 0.5
        rays_cam = torch.stack([(coords[:, 0] - cx) / focal, (coords[:, 1] - cy) / focal, torch.ones_like(coords[:, 0])], dim=1)

    # 归一化提取单位视线射线总线 rays_unit
    rays_unit = F.normalize(rays_cam, p=2, dim=1)  # [B, 3, H, W]

    # 3.c 刚性对齐预测单位法向量
    N_pred = F.normalize(pixel_normal_pred, p=2, dim=1)  # [B, 3, H, W]

    # 3.d 计算流形正交投影余弦因子 cos_theta = |n_pred · r_unit|
    cos_theta = torch.abs(torch.sum(N_pred * rays_unit, dim=1, keepdim=True))  # [B, 1, H, W]
    cos_theta = torch.clamp(cos_theta, min=0.1, max=1.0)

    # 将射线深度绝对差值点乘余弦，还原为表面正交距离
    pixel_err_map = torch.abs(pixel_depth_pred - pixel_depth_gt) * cos_theta

    # =====================================================================
    # 4. 🔥【拓扑双重边界隔离锁机制维系】
    # =====================================================================
    edge_mask = torch.ones_like(pixel_depth_pred, dtype=torch.bool)
    edge_mask[:, :, :, -1] = False
    edge_mask[:, :, -1, :] = False

    tri_id_float = tri_id_map.float()
    tri_id_shift_x = F.pad(tri_id_float[:, :, :, 1:], (0, 1, 0, 0), mode='replicate')
    tri_id_shift_y = F.pad(tri_id_float[:, :, 1:, :], (0, 0, 0, 1), mode='replicate')
    is_mesh_boundary = (tri_id_float != tri_id_shift_x) | (tri_id_float != tri_id_shift_y)

    pixel_valid_mask = (pixel_depth_gt > 0.0) & (~torch.isnan(pixel_depth_gt)) \
                       & (~torch.isnan(pixel_err_map)) & edge_mask & (~is_mesh_boundary)
                       
    if valid_mask is not None:
        if valid_mask.dim() == 3: 
            valid_mask = valid_mask.unsqueeze(1)
        if valid_mask.shape[-2] == H and valid_mask.shape[-1] == W:
            pixel_valid_mask = pixel_valid_mask & (valid_mask > 0.5)

    # =====================================================================
    # 5. 🎯【拓扑内切锁】高性能并行 scatter_add_ 归约
    # =====================================================================
    flat_tri_ids = tri_id_map.view(B, -1).long()
    flat_errors = pixel_err_map.view(B, -1)
    flat_mask = pixel_valid_mask.view(B, -1).float()

    is_bg_pixel = (flat_tri_ids < 0)
    flat_tri_ids = torch.where(is_bg_pixel, torch.zeros_like(flat_tri_ids), flat_tri_ids)
    flat_mask = flat_mask * (~is_bg_pixel).float()

    tri_error_sum = torch.zeros(B, num_triangles, device=device)
    tri_pixel_count = torch.zeros(B, num_triangles, device=device)

    tri_error_sum.scatter_add_(dim=1, index=flat_tri_ids, src=flat_errors * flat_mask)
    tri_pixel_count.scatter_add_(dim=1, index=flat_tri_ids, src=flat_mask)

    E_tri = tri_error_sum / (tri_pixel_count + eps)

    # =====================================================================
    # 6. 相对误差 Cauchy 映射与单向凸空间回归 (动态真值主轨)
    # =====================================================================
    W_GT_dynamic = 1.0 / (1.0 + (E_tri / adaptive_sigma) ** 2)
    W_GT_dynamic = torch.where(tri_pixel_count > 0, W_GT_dynamic, torch.zeros_like(W_GT_dynamic))
    W_GT_dynamic = W_GT_dynamic.unsqueeze(-1)  # [B, N, 1]
    
    # 动态 L1 主损失
    loss_conf_raw = F.l1_loss(W_pred, W_GT_dynamic.detach(), reduction='none')  
    
    # 🎯 物理学防伪修正：剥夺微观奇异面的监督资格
    # 面积 < 3 的微型三角形可能是好面，也极有可能是狗牙错分悬崖。
    # 绝不能强制赋 1.0 (会导致 GNN 学会“小=好”的致命偏见)，必须 Mask 掉不产生任何 loss，让其被 GNN 邻域平滑。
    loss_mask = (tri_pixel_count >= 3).float().unsqueeze(-1)
    
    loss_conf = (loss_conf_raw * loss_mask).sum() / (loss_mask.sum() + 1e-6)

    # =====================================================================
    # 7. 👑【静态几何特定区间辅助监督】(Hinge 极值锚定)
    # =====================================================================
    loss_aux = torch.tensor(0.0, device=device)
    
    # 解析离线预计算的 Planar Mask
    GT_planar_mask = None
    if is_gt_planar is not None:
        if isinstance(is_gt_planar, list):
            padded_list = []
            for b in range(B):
                mask_tensor = is_gt_planar[b].bool().to(device)
                N_b = mask_tensor.shape[0]
                if N_b < num_triangles:
                    padded = F.pad(mask_tensor, (0, num_triangles - N_b), mode='constant', value=False)
                elif N_b > num_triangles:
                    padded = mask_tensor[:num_triangles]
                else:
                    padded = mask_tensor
                padded_list.append(padded)
            GT_planar_mask = torch.stack(padded_list, dim=0).unsqueeze(-1)  # [B, num_triangles, 1]
        else:
            GT_planar_mask = is_gt_planar.bool().to(device)
            if GT_planar_mask.dim() == 2:
                GT_planar_mask = GT_planar_mask.unsqueeze(-1)
    
    # 兼容原有的 W_GT_static 解析逻辑 (保留给 curved 掩码用)
    W_GT_static = None
    if tri_conf_gt is not None:
        if isinstance(tri_conf_gt, list):
            padded_list = []
            for b in range(B):
                conf_tensor = tri_conf_gt[b].float().to(device)
                N_b = conf_tensor.shape[0]
                if N_b < num_triangles:
                    padded = F.pad(conf_tensor, (0, num_triangles - N_b), mode='constant', value=0.0)
                elif N_b > num_triangles:
                    padded = conf_tensor[:num_triangles]
                else:
                    padded = conf_tensor
                padded_list.append(padded)
            W_GT_static = torch.stack(padded_list, dim=0).unsqueeze(-1)  # [B, num_triangles, 1]
        else:
            W_GT_static = tri_conf_gt.float().to(device)
            if W_GT_static.dim() == 2:
                W_GT_static = W_GT_static.unsqueeze(-1)
                
    if GT_planar_mask is not None:
        # 轨 1：几何真平坦区 (利用离线计算的精确掩码: tri_conf > 0.60 & tri_err95 < 0.08) 监督 W_pred 至少为 0.8
        mask_plane = GT_planar_mask & (tri_pixel_count.unsqueeze(-1) >= 3)
        if mask_plane.any():
            loss_plane_aux = torch.clamp(0.8 - W_pred[mask_plane], min=0.0).mean()
            loss_aux = loss_aux + loss_plane_aux
            
    if W_GT_static is not None:
        # 轨 2：几何真曲折区 (W_GT_static <= 0.2) 监督 W_pred 至多为 0.2
        mask_curved = (W_GT_static <= 0.3) & (tri_pixel_count.unsqueeze(-1) >= 3)
        if mask_curved.any():
            loss_curved_aux = torch.clamp(W_pred[mask_curved] - 0.2, min=0.0).mean()
            loss_aux = loss_aux + loss_curved_aux
            
    
    # 早期 Hinge(aux) 主导给稳定锚点，晚期动态真值(dyn)主导做精细拟合
    weight_dyn = 0.3 + 0.7 * progress
    weight_aux = 1.5 - 1.0 * progress
    loss_final = loss_conf * weight_dyn + loss_aux * weight_aux

    # 还原像素空间置信度真值用于 TensorBoard
    tri_id_flat_dense = tri_id_map.squeeze(1).view(B, -1, 1)
    valid_mask_tri = tri_id_flat_dense >= 0
    safe_index_dense = torch.where(valid_mask_tri, tri_id_flat_dense, torch.zeros_like(tri_id_flat_dense))
    dense_flat = torch.gather(W_GT_dynamic, dim=1, index=safe_index_dense.long())
    dense_flat = torch.where(valid_mask_tri, dense_flat, torch.zeros_like(dense_flat))
    W_GT_pixel = dense_flat.permute(0, 2, 1).view(B, 1, H, W)

    return loss_final, W_GT_pixel


def compute_plane_depth_loss(depth_plane, depth_gt, mask, W_plane_pixel=None, gamma=1.5, threshold_low=0.2, threshold_high=0.8):
    """
    【特异性异方差加权】Stage 1 平面级深度重构损失函数
    仅针对 Stage 1 平面优化/传播后的深度预测进行几何置信度动态加权监督。
    
    :param depth_plane: [B, 1, H, W] 或 [B, H, W] 平面深度图 (例如 depth_patchmatch['stage_1'][-1])
    :param depth_gt: [B, 1, H, W] 或 [B, H, W] Stage 1 真实深度
    :param mask: [B, 1, H, W] 或 [B, H, W] Stage 1 有效深度掩码
    :param W_plane_pixel: [B, 1, H, W] 像素级平面置信度权重
    :param gamma: 最大增强权重系数 (默认 1.5)
    :param threshold_low: 置信度截断下界 (默认 0.2)
    :param threshold_high: 置信度完全激活上界 (默认 0.8)
    :return: 加权后的平面深度损失标量
    """
    if depth_plane is None or depth_gt is None or mask is None:
        return torch.tensor(0.0)

    if depth_plane.dim() == 3:
        depth_plane = depth_plane.unsqueeze(1)
    if depth_gt.dim() == 3:
        depth_gt = depth_gt.unsqueeze(1)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)

    mask_s1 = mask > 0.5
    pixel_loss = F.smooth_l1_loss(depth_plane, depth_gt, reduction='none')  # [B, 1, H_s1, W_s1]

    if W_plane_pixel is not None:
        W_val = W_plane_pixel.detach()
        if W_val.dim() == 3:
            W_val = W_val.unsqueeze(1)
            
        # 对齐置信度图分辨率并强制进行梯度阻断双保险
        if W_val.shape[2:] != pixel_loss.shape[2:]:
            W_aligned = F.interpolate(W_val, size=pixel_loss.shape[2:], mode='bilinear', align_corners=True).detach()
        else:
            W_aligned = W_val.detach()

        # SmoothStep 极化映射：保底彻底降为 0.0（非平面完全免除平面监督），最大增强为 gamma (1.5)
        x = torch.clamp((W_aligned - threshold_low) / (threshold_high - threshold_low + 1e-8), 0.0, 1.0)
        smooth_weight = 3.0 * (x ** 2) - 2.0 * (x ** 3)
        pixel_weight = gamma * smooth_weight

        weighted_loss = pixel_loss * pixel_weight
        return weighted_loss[mask_s1].mean()
    else:
        return pixel_loss[mask_s1].mean()


def compute_heteroscedastic_depth_loss(depth_patchmatch, refined_depth, depth_gt, mask, 
                                       W_plane_pixel=None, is_planar_s0=None, gamma=1.5, threshold_low=0.2, threshold_high=0.8):
    """
    【特异性异方差加权】深度重构损失函数 (修正版方案 B)
    仅针对 Stage 1 的最终平面级深度预测（depth_patchmatch['stage_1'][1]）进行置信度加权，
    其余所有阶段、所有像素级深度图均保持原版 Smooth L1 损失计算。
    """
    loss = 0.0
    stage = 4
    device = depth_gt['stage_1'].device

    for l in range(1, stage):
        depth_gt_l = depth_gt[f'stage_{l}']
        mask_l = mask[f'stage_{l}'] > 0.5  # [B, 1, H_l, W_l]
        
        depth_patchmatch_l = depth_patchmatch[f'stage_{l}']
        for i in range(len(depth_patchmatch_l)):
            depth1 = depth_patchmatch_l[i]
            depth2 = depth_gt_l
            
            # 🎯 仅在 Stage 1 的最终平面重建结果 (i == 1) 实施异方差加权
            if l == 1 and i == 1 and W_plane_pixel is not None:
                pixel_loss = F.smooth_l1_loss(depth1, depth2, reduction='none') # [B, 1, H_s1, W_s1]
                
                # 对齐置信度图分辨率并强制进行梯度阻断双保险
                W_val = W_plane_pixel.detach()
                if W_val.shape[2:] != pixel_loss.shape[2:]:
                    W_aligned = F.interpolate(W_val, size=pixel_loss.shape[2:], mode='bilinear', align_corners=True).detach()
                else:
                    W_aligned = W_val.detach()
                
                # SmoothStep 极化映射：保底彻底降为 0.0（非平面完全免除平面监督），最大增强为 gamma (1.5)
                x = torch.clamp((W_aligned - threshold_low) / (threshold_high - threshold_low + 1e-8), 0.0, 1.0)
                smooth_weight = 3.0 * (x ** 2) - 2.0 * (x ** 3)
                pixel_weight = gamma * smooth_weight
                
                weighted_loss = pixel_loss * pixel_weight
                loss = loss + weighted_loss[mask_l].mean()
            else:
                # 其余阶段及 Stage 1 初始像素级深度完全保持原版
                loss = loss + F.smooth_l1_loss(depth1[mask_l], depth2[mask_l], reduction='mean')

    # Stage 0 级 Refine 深度损失保持原版 (结合硬路由隔离)
    l = 0
    depth_refined_l = refined_depth[f'stage_{l}']
    depth_gt_l = depth_gt[f'stage_{l}']
    mask_l = mask[f'stage_{l}'] > 0.5

    # Stage 0 级 Refine 深度损失：全面恢复原版全图有效像素监督（涵盖平面与非平面区）
    active_mask = mask_l

    depth1 = depth_refined_l[active_mask]
    depth2 = depth_gt_l[active_mask]
    
    if depth1.numel() > 0:
        loss = loss + F.smooth_l1_loss(depth1, depth2, reduction='mean')

    return loss


def compute_normal_gt_loss(n_pred, n_gt, is_gt_planar, pixel_counts, min_pixels=3):
    """
    计算基于离线 GT SVD 的绝对法向余弦损失（严格的三角面级别）。
    """
    if is_gt_planar is None or n_gt is None:
        return torch.tensor(0.0, device=n_pred.device, requires_grad=True)

    # 兼容 list 输入
    if isinstance(is_gt_planar, list):
        B = len(is_gt_planar)
        num_tri = n_pred.shape[1]
        
        # 将 list of tensors 转为单个张量 [B, N]
        planar_list = []
        gt_norm_list = []
        for b in range(B):
            planar_b = is_gt_planar[b].bool().to(n_pred.device)
            gt_norm_b = n_gt[b].to(n_pred.device)
            
            # 补齐或截断到 num_tri
            if planar_b.shape[0] < num_tri:
                planar_b = F.pad(planar_b, (0, num_tri - planar_b.shape[0]), value=False)
                gt_norm_b = F.pad(gt_norm_b, (0, 0, 0, num_tri - gt_norm_b.shape[0]), value=0.0)
            elif planar_b.shape[0] > num_tri:
                planar_b = planar_b[:num_tri]
                gt_norm_b = gt_norm_b[:num_tri]
            planar_list.append(planar_b)
            gt_norm_list.append(gt_norm_b)
            
        mask_planar = torch.stack(planar_list, dim=0) # [B, N]
        n_gt_tensor = torch.stack(gt_norm_list, dim=0) # [B, N, 3]
    else:
        mask_planar = is_gt_planar.bool().to(n_pred.device)
        n_gt_tensor = n_gt.to(n_pred.device)
        if mask_planar.dim() == 3:
            mask_planar = mask_planar.squeeze(-1)

    # 1. 真实深度必须极度平坦 (mask_planar)
    # 2. 覆盖像素点不能太少 (抗噪声)
    # 不再强制约束法线朝向 (n_z < 0)，因为 SVD 提取的本征向量可能有随机符号
    valid_mask = mask_planar & (pixel_counts >= min_pixels)
    
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=n_pred.device, requires_grad=True)
    
    # 余弦相似度损失: 取绝对值，完美兼容真值法线正反朝向！
    cos_sim = (n_pred * n_gt_tensor).sum(dim=-1)
    loss = (1.0 - torch.abs(cos_sim))[valid_mask].mean()
    return loss


def compute_pixel_d2n_normal_loss(depth_pixel_pred, depth_gt, intrinsics, mask=None, cliff_threshold=0.8,
                                   W_plane_pixel=None, threshold_low=0.3, threshold_high=0.8):
    """
    P1: 像素级 D2N 几何正则化损失 (Pixel-Level Differentiable Normal Loss with SmoothStep Gating)
    利用可微差分算子从 Stage 1 原生像素级深度求出预测法向量，
    通过余弦相似度损失反向传播，强行熨平弱纹理区域的深度凹凸噪点。
    结合物理平面置信度 SmoothStep 软门控进行极化加权（平面区满载拉直，非平面区彻底关死）。
    同时使用深度梯度断崖掩码过滤掉房檐、物体边界处的飞面法向。

    Args:
        depth_pixel_pred: [B, H, W] 或 [B, 1, H, W] Stage 1 原生像素级 PM 深度 (depth_patchmatch['stage_1'][0])
        depth_gt:         [B, H, W] 或 [B, 1, H, W] Stage 1 GT 深度
        intrinsics:       [B, 3, 3] 参考视角相机内参
        mask:             [B, H, W] 或 [B, 1, H, W] 0-1 有效深度掩码
        cliff_threshold:  断崖过滤阈值 (默认 0.8 米)，过滤楼檐悬崖处的虚假法向
        W_plane_pixel:    [B, 1, H, W] 像素级物理平面置信度（用于软门控极化加权）
        threshold_low:    软门控低阈值 (默认 0.2)，低于此值权重归 0
        threshold_high:   软门控高阈值 (默认 0.8)，高于此值权重饱和至 1.0

    Returns:
        loss_pix_normal:  标量 Loss
    """
    device = depth_pixel_pred.device
    if depth_pixel_pred.dim() == 3:
        depth_pixel_pred = depth_pixel_pred.unsqueeze(1)
    if depth_gt.dim() == 3:
        depth_gt = depth_gt.unsqueeze(1)

    # 1. 前置平滑 + 可微计算预测法向 (smooth=True 抑制微分高频噪声放大)
    N_pred, _ = compute_normal_map_torch(depth_pixel_pred, intrinsics, mask=None, smooth=True)

    # 2. 从 GT 深度计算真值法向 (detach 隔离)
    with torch.no_grad():
        N_gt, _ = compute_normal_map_torch(depth_gt, intrinsics, mask=None, smooth=True)
        N_gt = N_gt.detach()

        # 3. 房檐悬崖断崖掩码 (Occlusion Edge Mask / Cliff Filter)
        # 计算 GT 深度在 x 和 y 方向的相邻差分落差
        diff_x = torch.zeros_like(depth_gt)
        diff_y = torch.zeros_like(depth_gt)
        diff_x[:, :, :, 1:] = torch.abs(depth_gt[:, :, :, 1:] - depth_gt[:, :, :, :-1])
        diff_y[:, :, 1:, :] = torch.abs(depth_gt[:, :, 1:, :] - depth_gt[:, :, :-1, :])
        diff_max = torch.max(diff_x, diff_y)

        valid_cliff = diff_max < cliff_threshold

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            valid_mask = (mask > 0.5) & (depth_gt > 1e-4) & valid_cliff
        else:
            valid_mask = (depth_gt > 1e-4) & valid_cliff

    # 4. 余弦相似度损失: 1.0 - cos_sim (带 clamp 防微小数值越界)
    cos_sim = torch.sum(N_pred * N_gt, dim=1, keepdim=True)  # [B, 1, H, W]
    loss_map = 1.0 - torch.clamp(cos_sim, -1.0, 1.0)

    # 5. 置信度 SmoothStep 软门控极化加权：平面区满额拉直，非平面区彻底归零
    if W_plane_pixel is not None:
        W_val = W_plane_pixel.detach()
        if W_val.shape[2:] != loss_map.shape[2:]:
            W_aligned = F.interpolate(W_val, size=loss_map.shape[2:], mode='bilinear', align_corners=True).detach()
        else:
            W_aligned = W_val
        x = torch.clamp((W_aligned - threshold_low) / (threshold_high - threshold_low + 1e-8), 0.0, 1.0)
        smooth_weight = 3.0 * (x ** 2) - 2.0 * (x ** 3)
        loss_map = loss_map * smooth_weight

    if valid_mask.sum() > 0:
        loss_pix_normal = loss_map[valid_mask].mean()
    else:
        loss_pix_normal = torch.tensor(0.0, device=device, requires_grad=True)

    return loss_pix_normal

