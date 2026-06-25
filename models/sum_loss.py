import torch
import argparse
import os
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F

import math

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
    # 2. 🚀 空间流形消毒与低通钝化 (熨平晶格多视图匹配的微观量化锯齿)
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
    # 【🎯 修正点二：显式转型为 Float32，破除 Int64 张量在低版本 PyTorch F.pad 算子里缺失特化模板的技术债】
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

    # 【🎯 修正点四：对 3D 掩码强行注入 unsqueeze(1) 锁定轴心，封杀隐式右对齐广播炸出的维度断层】
    if valid_mask is not None:
        if valid_mask.dim() == 3:
            valid_mask = valid_mask.unsqueeze(1)
        final_valid = final_valid & (valid_mask > 0.5)

    relevant_loss = gated_loss_map[final_valid]

    # 无效区域自卫保底（当全图被完全裁切放逐时的极值自防卫，防止分母归零炸出空均值 NaN）
    if relevant_loss.numel() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return relevant_loss.mean()


def compute_confidence_supervision_loss(pixel_depth_pred, pixel_depth_gt, pixel_normal_pred, tri_id_map, W_pred, progress,
                                        depth_range=None, valid_mask=None, eps=1e-7, intrinsics=None, pixel_counts=None,
                                        pixel_normal_gt=None, tri_id_map_s1=None):
    """
    【三位一体协同校准·完全体】平面置信度回归损失函数
    
    1. 静态真值在线解构 (W_GT_static):
       - GPU 并行版 OLS 拟合残差估计：基于 GT 深度和 GT 法线计算每个三角形内点到拟合平面的平均正交距离。
       - 悬边噪点 Dropout：通过深度突变差分与三角形边界膨胀求交，剔除边界 1-2 像素的悬空污染。
       - 极小三角形深度标准差门控：在 N < 15 时，若标准差 std(Z) < 0.03m 则直接设为 1.0 置信度，否则为 0.0。
    2. 区间忽略与双峰损失加权:
       - Margin-free Interval Loss: 对于 W_GT 在 [0.3, 0.7] 的区域，残差在 0.2 以内免于惩罚。
       - Double-Peak Gradient Gating: 乘上 V字型权重 4.0 * (W_GT - 0.5)^2，熔断中间过渡带梯度。
    3. 各项异性非对称 Huber 惩罚:
       - 预测高于真值(虚报) 乘以 k_fp = 2.0，预测低于真值(漏报) 乘以 k_fn = 0.5。
    4. 功率降档补偿:
       - 全局乘以 1.8 因子补偿双峰调制带来的数值功率降档。
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

    # 获得 GT 深度的 H, W 尺寸 (在 Stage 0 下计算)
    H_gt, W_gt = pixel_depth_gt.shape[-2], pixel_depth_gt.shape[-1]

    if tri_id_map.dim() == 3:
        tri_id_map = tri_id_map.unsqueeze(1)
    elif tri_id_map.shape[-1] == 1 and tri_id_map.dim() == 4:
        tri_id_map = tri_id_map.permute(0, 3, 1, 2)

    # =====================================================================
    # 2. 🚀【悬边噪点 Dropout 机制】
    # =====================================================================
    # 计算深度突变以识别深度断裂线（突变阈值 0.25米）
    dx = torch.abs(pixel_depth_gt[:, :, :, 1:] - pixel_depth_gt[:, :, :, :-1])
    dy = torch.abs(pixel_depth_gt[:, :, 1:, :] - pixel_depth_gt[:, :, :-1, :])
    dx = F.pad(dx, (0, 1, 0, 0), mode='replicate')
    dy = F.pad(dy, (0, 0, 0, 1), mode='replicate')
    depth_mutant = (dx > 0.25) | (dy > 0.25)

    # 计算三角形 ID 的分界线（边界线）
    tri_id_float = tri_id_map.float()
    dtri_x = tri_id_float[:, :, :, 1:] != tri_id_float[:, :, :, :-1]
    dtri_y = tri_id_float[:, :, 1:, :] != tri_id_float[:, :, :-1, :]
    dtri_x = F.pad(dtri_x, (0, 1, 0, 0), mode='constant', value=False)
    dtri_y = F.pad(dtri_y, (0, 0, 0, 1), mode='constant', value=False)
    tri_boundary = dtri_x | dtri_y
    
    # 膨胀边界，使用 3x3 max_pool2d
    tri_boundary_expanded = F.max_pool2d(tri_boundary.float(), kernel_size=3, stride=1, padding=1) > 0.5
    
    # 悬边噪点掩码
    discard_mask = depth_mutant & tri_boundary_expanded

    # =====================================================================
    # 3. 🛡️ 边界过滤与有效掩码构建
    # =====================================================================
    edge_mask = torch.ones_like(pixel_depth_gt, dtype=torch.bool)
    edge_mask[:, :, :, -1] = False
    edge_mask[:, :, -1, :] = False

    tri_id_shift_x = F.pad(tri_id_float[:, :, :, 1:], (0, 1, 0, 0), mode='replicate')
    tri_id_shift_y = F.pad(tri_id_float[:, :, 1:, :], (0, 0, 0, 1), mode='replicate')
    is_mesh_boundary = (tri_id_float != tri_id_shift_x) | (tri_id_float != tri_id_shift_y)

    pixel_valid_mask = (pixel_depth_gt > 0.0) & (~torch.isnan(pixel_depth_gt)) \
                       & edge_mask & (~is_mesh_boundary) & (~discard_mask)
                       
    if valid_mask is not None:
        if valid_mask.dim() == 3: 
            valid_mask = valid_mask.unsqueeze(1)
        # 仅在分辨率大小一致时应用外部 valid_mask
        if valid_mask.shape[-2] == H_gt and valid_mask.shape[-1] == W_gt:
            pixel_valid_mask = pixel_valid_mask & (valid_mask > 0.5)

    # =====================================================================
    # 4. 🚀【3D 解析空间正交投影】
    # =====================================================================
    y, x = torch.meshgrid(
        torch.arange(H_gt, device=device),
        torch.arange(W_gt, device=device),
        indexing='ij',
    )
    coords = torch.stack([x, y, torch.ones_like(x)], dim=0).float().to(device)  # [3, H_gt, W_gt]
    coords = coords.unsqueeze(0).expand(B, -1, -1, -1)  # [B, 3, H_gt, W_gt]

    if intrinsics is not None:
        inv_k = torch.linalg.inv(intrinsics.float())
        flat_coords = coords.view(B, 3, -1)
        rays_cam = torch.bmm(inv_k, flat_coords).view(B, 3, H_gt, W_gt)
    else:
        focal = float(max(H_gt, W_gt))
        cx, cy = (W_gt - 1) * 0.5, (H_gt - 1) * 0.5
        rays_cam = torch.stack([(coords[:, 0] - cx) / focal, (coords[:, 1] - cy) / focal, torch.ones_like(coords[:, 0])], dim=1)

    # 3D 点云
    pts_3d = rays_cam * pixel_depth_gt  # [B, 3, H_gt, W_gt]

    # 获取物理 GT 法向量进行拟合
    if pixel_normal_gt is None:
        # 在线计算 GT 法向量进行兜底
        P_right = F.pad(pts_3d[:, :, :, 1:], (0, 1, 0, 0), mode='replicate')
        P_down = F.pad(pts_3d[:, :, 1:, :], (0, 0, 0, 1), mode='replicate')
        v1 = P_right - pts_3d
        v2 = P_down - pts_3d
        n_gt_raw = torch.cross(v1, v2, dim=1)
        pixel_normal_gt = F.normalize(n_gt_raw, p=2, dim=1)
    else:
        if pixel_normal_gt.shape[1] != 3 and pixel_normal_gt.shape[-1] == 3:
            pixel_normal_gt = pixel_normal_gt.permute(0, 3, 1, 2)
        pixel_normal_gt = F.normalize(pixel_normal_gt, p=2, dim=1)

    # =====================================================================
    # 5. 🎯【GPU 高性能归约统计】
    # =====================================================================
    flat_tri_ids = tri_id_map.view(B, -1).long()
    flat_mask = pixel_valid_mask.view(B, -1).float()

    is_bg_pixel = (flat_tri_ids < 0)
    flat_tri_ids = torch.where(is_bg_pixel, torch.zeros_like(flat_tri_ids), flat_tri_ids)
    flat_mask = flat_mask * (~is_bg_pixel).float()

    flat_pts = pts_3d.view(B, 3, -1)
    flat_normals = pixel_normal_gt.view(B, 3, -1)

    tri_pts_sum = torch.zeros(B, 3, num_triangles, device=device)
    tri_normal_sum = torch.zeros(B, 3, num_triangles, device=device)
    tri_z_sum = torch.zeros(B, num_triangles, device=device)
    tri_z2_sum = torch.zeros(B, num_triangles, device=device)
    tri_count = torch.zeros(B, num_triangles, device=device)

    expanded_index = flat_tri_ids.unsqueeze(1).expand(-1, 3, -1)
    tri_pts_sum.scatter_add_(dim=2, index=expanded_index, src=flat_pts * flat_mask.unsqueeze(1))
    tri_normal_sum.scatter_add_(dim=2, index=expanded_index, src=flat_normals * flat_mask.unsqueeze(1))
    tri_z_sum.scatter_add_(dim=1, index=flat_tri_ids, src=flat_pts[:, 2, :] * flat_mask)
    tri_z2_sum.scatter_add_(dim=1, index=flat_tri_ids, src=(flat_pts[:, 2, :] ** 2) * flat_mask)
    tri_count.scatter_add_(dim=1, index=flat_tri_ids, src=flat_mask)

    # 计算重心和平均法向
    centroid = tri_pts_sum / (tri_count.unsqueeze(1) + eps)
    n_avg = F.normalize(tri_normal_sum, p=2, dim=1)

    # 映射回像素空间计算正交距离
    expanded_flat_tri_ids = flat_tri_ids.unsqueeze(1).expand(-1, 3, -1)
    flat_centroid = torch.gather(centroid, dim=2, index=expanded_flat_tri_ids)
    flat_n_avg = torch.gather(n_avg, dim=2, index=expanded_flat_tri_ids)
    flat_dist = torch.abs(torch.sum((flat_pts - flat_centroid) * flat_n_avg, dim=1))  # [B, HW]

    # 三角形绝对正交残差
    tri_error_sum = torch.zeros(B, num_triangles, device=device)
    tri_error_sum.scatter_add_(dim=1, index=flat_tri_ids, src=flat_dist * flat_mask)
    E_tri = tri_error_sum / (tri_count + eps)

    # 深度标准差 (Z-std)
    mean_z = tri_z_sum / (tri_count + eps)
    mean_z2 = tri_z2_sum / (tri_count + eps)
    z_var = torch.clamp(mean_z2 - mean_z ** 2, min=0.0)
    z_std = torch.sqrt(z_var + eps)

    # =====================================================================
    # 6. 🎯【静态置信度真值生成机制】
    # =====================================================================
    if depth_range is not None:
        min_d, max_d = depth_range
        val_min = min_d.mean().item() if isinstance(min_d, torch.Tensor) else float(min_d)
        val_max = max_d.mean().item() if isinstance(max_d, torch.Tensor) else float(max_d)
        span = max(val_max - val_min, 1e-3)
    else:
        span = 400.0

    sigma_max = span * 0.015
    sigma_min = span * 0.0065
    prog_val = max(0.0, min(float(progress), 1.0))
    adaptive_sigma = max(sigma_max * (1.0 - prog_val) + sigma_min * prog_val, 0.1)

    scale_compensator = 1.0 + 1.5 * torch.exp(-tri_count / 30.0)
    sigma_adapted = adaptive_sigma * scale_compensator
    conf_large = 1.0 / (1.0 + (E_tri / sigma_adapted) ** 2)

    # 极小面片采用深度标准差门控 (小于 3 厘米视为微观平面)
    conf_small = torch.where(z_std < 0.03, torch.ones_like(E_tri), torch.zeros_like(E_tri))
    
    # 恢复原分辨率 Stage 0 的点数带宽，大中三角形起算点恢复为 15
    W_GT_static = torch.where(tri_count >= 15, conf_large, conf_small)
    # 点数不足 4 个的超碎面不参与损失计算 (设为 0)
    W_GT_static = torch.where(tri_count >= 4, W_GT_static, torch.zeros_like(W_GT_static))
    W_GT_static = W_GT_static.unsqueeze(-1).detach()  # [B, num_triangles, 1]

    # =====================================================================
    # 7. 🔥【三位一体协同损失与非对称惩罚】
    # =====================================================================
    diff = W_pred - W_GT_static
    abs_diff = torch.abs(diff)

    # 7.a 区间忽略 (Margin-free Interval Loss)
    gray_mask = (W_GT_static >= 0.3) & (W_GT_static <= 0.7)
    abs_diff_adjusted = torch.where(gray_mask, torch.clamp(abs_diff - 0.2, min=0.0), abs_diff)
    diff_adjusted = torch.sign(diff) * abs_diff_adjusted

    # 7.b Huber 损失计算
    loss_huber = torch.where(
        torch.abs(diff_adjusted) < 0.1,
        0.5 * (diff_adjusted ** 2) / 0.1,
        torch.abs(diff_adjusted) - 0.05
    )

    # 7.c 双峰梯度调制
    w_loss = 4.0 * ((W_GT_static - 0.5) ** 2)

    # 7.d 各项异性非对称权重 (虚报惩罚 2.0，漏报折算为 0.5)
    asym_weight = torch.where(diff_adjusted > 0, torch.tensor(2.0, device=device), torch.tensor(0.5, device=device))
    loss_conf_per_tri = loss_huber * asym_weight * w_loss

    # =====================================================================
    # 8. 🛡️【有效掩码与量纲功率补偿】
    # =====================================================================
    # 高清分辨率下参与监督的三角形有效像素门控设为 >= 4
    valid_tri_mask = (tri_count >= 4).unsqueeze(-1)
    loss_valid = loss_conf_per_tri[valid_tri_mask]
    
    if loss_valid.numel() == 0:
        loss_final = torch.tensor(0.0, device=device, requires_grad=True)
    else:
        loss_final = loss_valid.mean() * 1.8

    # =====================================================================
    # 9. 🎨【还原像素空间置信度真值用于 TensorBoard】
    # =====================================================================
    # 优先采用传入的 1/2 分辨率的 tri_id_map_s1 进行反投影，以便在 TensorBoard 完美显示
    target_tri_id_map = tri_id_map_s1 if tri_id_map_s1 is not None else tri_id_map
    if target_tri_id_map.dim() == 3:
        target_tri_id_map = target_tri_id_map.unsqueeze(1)
    elif target_tri_id_map.shape[-1] == 1 and target_tri_id_map.dim() == 4:
        target_tri_id_map = target_tri_id_map.permute(0, 3, 1, 2)
        
    B_out, _, H_out, W_out = target_tri_id_map.shape
    tri_id_map_sq = target_tri_id_map.squeeze(1)
    tri_id_flat_dense = tri_id_map_sq.view(B_out, -1, 1)
    valid_mask_tri = tri_id_flat_dense >= 0
    safe_index_dense = torch.where(valid_mask_tri, tri_id_flat_dense, torch.zeros_like(tri_id_flat_dense))
    dense_flat = torch.gather(W_GT_static, dim=1, index=safe_index_dense.long())
    dense_flat = torch.where(valid_mask_tri, dense_flat, torch.zeros_like(dense_flat))
    W_GT_pixel = dense_flat.permute(0, 2, 1).view(B_out, 1, H_out, W_out)

    return loss_final, W_GT_pixel
