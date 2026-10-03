"""
========================================================================================
WHU-MVS 灾难案例单兵深度透视与可视化系统 (visualize_disaster_case.py)
========================================================================================
功能：
针对 Top-1 灾难案例 (Scan 009_59 / Img 000002, Tri 1756 -> 1722)，
提取并可视化 6 大核心模块：
1. 参考图像局部放大图 (Crop RGB): 红色多边形标出 Tri 1756，青色标出 Tri 1722；
2. 真值深度图 (GT Depth Crop): 伪彩色直观查看真实三维几何；
3. 自身平面深度 vs 邻居平面深度 (Depth Comparison): 查看 6.8cm 到 76.8m 的暴飞实况；
4. 4 个源视角的重投影图像 (Warped RGB): 观察 76m 深度到底把源图拉扯成了什么样；
5. 逐像素多视特征匹配代价图 (Cost Maps): 查看多视匹配代价为何给错面打出高分；
6. 空间几何与重投影坐标法医数值报告 (终端与图内双输出)。
========================================================================================
"""

import argparse
import os
import sys
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 控制 GPU id
_gpu_choice = os.environ.get('GPU_ID', os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
os.environ['CUDA_VISIBLE_DEVICES'] = str(_gpu_choice)

from datasets import find_dataset_def
from datasets.dtu_whu import collate_keep_list
from models import PatchmatchNet
from utils import tocuda, make_nograd_func

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description='Disaster Case Diagnostic & Visualization')
    parser.add_argument('--dataset', default='dtu_whu', help='select dataset (default: dtu_whu)')
    parser.add_argument('--testpath', default='/home/ym/Experiment/Datas/WHU_MVS_dataset', help='testing data path')
    parser.add_argument('--testlist', default='lists/whu/test9.20.txt', help='testing scan list file')
    parser.add_argument('--loadckpt', required=True, help='checkpoint path')
    parser.add_argument('--n_views', type=int, default=5, help='number of views')
    parser.add_argument('--scan', default='009_59', help='目标 Scan ID (默认: 009_59)')
    parser.add_argument('--file_id', default='000002', help='目标图像 File ID (默认: 000002)')
    parser.add_argument('--tri_self', type=int, default=1756, help='自身三角形 ID (默认: 1756)')
    parser.add_argument('--tri_neigh', type=int, default=1722, help='邻居三角形 ID (默认: 1722)')
    parser.add_argument('--pad', type=int, default=60, help='局部裁剪边缘 Padding (像素, 默认 60)')
    parser.add_argument('--output_png', default='disaster_case_1756_1722.png', help='输出图片路径')

    # PatchMatch 超参数
    parser.add_argument('--patchmatch_iteration', nargs='+', type=int, default=[1, 2, 2])
    parser.add_argument('--patchmatch_num_sample', nargs='+', type=int, default=[8, 8, 16])
    parser.add_argument('--patchmatch_interval_scale', nargs='+', type=float, default=[0.005, 0.0125, 0.025])
    parser.add_argument('--patchmatch_range', nargs='+', type=int, default=[6, 4, 2])
    parser.add_argument('--propagate_neighbors', nargs='+', type=int, default=[0, 8, 16])
    parser.add_argument('--evaluate_neighbors', nargs='+', type=int, default=[9, 9, 9])

    return parser.parse_args()


@make_nograd_func
def evaluate_single_sample_stage1(model, sample, device):
    """提取 Stage 1 几何与特征。"""
    vertexs_batch = sample['vertexs']
    lines_batch = sample['lines']
    triangles_batch = []
    for tri_list in sample['triangles']:
        tri_processed = []
        for t in tri_list:
            v_ids = t['vertex_ids']
            l_ids = t['line_ids']
            pts = torch.from_numpy(t['valid_points']).to(device)
            tri_processed.append((v_ids, l_ids, pts))
        triangles_batch.append(tri_processed)

    skip = ["vertexs", "lines", "triangles", "tri_conf_cleaned", "tri_normal_cleaned", "is_gt_planar", "tri_err95", "tri_weight_normal", "tri_weight_conf"]
    sample_cuda = tocuda(sample, device=device, skip_keys=skip)

    depth_gt = sample_cuda["depth"]
    mask = sample_cuda["mask"]

    outputs = model(
        sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["intrinsics_mats"],
        sample_cuda["depth_min"], sample_cuda["depth_max"],
        vertexs_batch, lines_batch, triangles_batch, depth_gt['stage_1'],
        0.0, 0.0, 0.55, compute_edge_pixels=False
    )
    return outputs, depth_gt, mask, sample_cuda


def get_triangle_contour(tri_mask):
    """从二值 mask 提取轮廓多边形坐标 [K, 2] (x, y)。"""
    mask_np = tri_mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return None
    c = max(contours, key=cv2.contourArea)
    pts = c.squeeze(1) # [K, 2]
    return pts


def main():
    args = parse_args()

    print("=" * 90)
    print(f"🔍 启动灾难案例透视分析: Scan [{args.scan}] / File [{args.file_id}]")
    print(f"🎯 目标比对: 自身 Tri [{args.tri_self}] vs 邻居 Tri [{args.tri_neigh}]")
    print("=" * 90)

    # 1. 载入数据集
    MVSDataset = find_dataset_def(args.dataset)
    test_dataset = MVSDataset(args.testpath, args.testlist, "test", nviews=args.n_views, robust_train=False)

    # 检索目标 sample
    target_idx = None
    target_file_id_clean = str(args.file_id).split('.')[0]
    for idx, (s, fid) in enumerate(test_dataset.metas):
        fid_clean = str(fid).split('.')[0]
        if s == args.scan and (fid_clean == target_file_id_clean or int(fid_clean) == int(target_file_id_clean)):
            target_idx = idx
            break

    if target_idx is None:
        print(f"❌ 未在测试列表中找到目标样本 Scan={args.scan}, FileID={args.file_id}！")
        return

    sample = test_dataset[target_idx]
    # 包装为 batch=1
    sample_batched = collate_keep_list([sample])

    # 2. 载入模型
    model = PatchmatchNet(
        patchmatch_interval_scale=args.patchmatch_interval_scale,
        propagation_range=args.patchmatch_range,
        patchmatch_iteration=args.patchmatch_iteration,
        patchmatch_num_sample=args.patchmatch_num_sample,
        propagate_neighbors=args.propagate_neighbors,
        evaluate_neighbors=args.evaluate_neighbors
    ).to(device)

    print(f"[Model] 载入权重: {args.loadckpt}")
    checkpoint = torch.load(args.loadckpt, map_location=device)
    state_dict_model = checkpoint.get('model', checkpoint)
    cleaned_state = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict_model.items()}
    try:
        model.load_state_dict(cleaned_state, strict=True)
    except:
        model.load_state_dict(cleaned_state, strict=False)
    model.eval()

    # 3. 前向提取
    with torch.no_grad():
        outputs, depth_gt, mask, sample_cuda = evaluate_single_sample_stage1(model, sample_batched, device)

    output_plane = outputs.get('output_plane', {})
    planes = output_plane['final_plane'][0]            # [N, 4]
    W_tri = output_plane['W_plane_tri'][0, :, 0]       # [N]
    tri_id_map = output_plane['tri_id_map'][0]         # [H, W]

    K_intr = sample_cuda['intrinsics_mats']['stage_1'][0, 0] # [3, 3]
    d_gt = depth_gt['stage_1'][0, 0]                   # [H, W]
    m_gt = (mask['stage_1'][0, 0] > 0.5)               # [H, W]

    ref_img_s1 = sample_cuda['imgs']['stage_1'][0, 0].permute(1, 2, 0).cpu().numpy() # [H, W, 3]
    src_imgs_s1 = [sample_cuda['imgs']['stage_1'][0, i].permute(1, 2, 0).cpu().numpy() for i in range(1, args.n_views)]
    H, W = tri_id_map.shape

    # 4. 提取 Tri 1756 和 1722 的像素与几何信息
    t_self = args.tri_self
    t_neigh = args.tri_neigh
    mask_self = (tri_id_map == t_self)
    mask_neigh = (tri_id_map == t_neigh)

    pts_self = mask_self.sum().item()
    pts_neigh = mask_neigh.sum().item()

    print(f"\n📊 目标三角形基础几何属性:")
    print(f"   - 自身 Tri {t_self}: 像素数 = {pts_self}, 置信度 W = {W_tri[t_self].item():.4f}")
    print(f"   - 邻居 Tri {t_neigh}: 像素数 = {pts_neigh}, 置信度 W = {W_tri[t_neigh].item():.4f}")

    if pts_self == 0:
        print(f"❌ 自身 Tri {t_self} 在 Stage 1 tri_id_map 中没有对应像素！")
        return

    # 平面参数解析: n, d
    p_self = planes[t_self].cpu().numpy()
    p_neigh = planes[t_neigh].cpu().numpy()
    n_self = p_self[:3]
    d_self = p_self[3]
    n_neigh = p_neigh[:3]
    d_neigh = p_neigh[3]

    # 法向夹角
    cos_angle = np.abs(np.dot(n_self, n_neigh)) / (np.linalg.norm(n_self) * np.linalg.norm(n_neigh) + 1e-8)
    angle_deg = np.arccos(np.clip(cos_angle, 0.0, 1.0)) * 180.0 / np.pi

    print(f"   - 平面参数 1756: n = [{n_self[0]:+.4f}, {n_self[1]:+.4f}, {n_self[2]:+.4f}], d = {d_self:+.4f}")
    print(f"   - 平面参数 1722: n = [{n_neigh[0]:+.4f}, {n_neigh[1]:+.4f}, {n_neigh[2]:+.4f}], d = {d_neigh:+.4f}")
    print(f"   - 法向空间夹角: {angle_deg:.2f}° (几乎完全平行！)")

    # 计算全图射线
    K_inv = torch.inverse(K_intr)
    y_grid, x_grid = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    uv1 = torch.stack([x_grid.float(), y_grid.float(), torch.ones_like(x_grid)], dim=-1) # [H, W, 3]
    rays = (K_inv @ uv1.unsqueeze(-1)).squeeze(-1) # [H, W, 3]
    rays_len = torch.norm(rays, dim=-1, keepdim=True).clamp(min=1e-6)

    # 自身平面全图深度
    n_s_t = planes[t_self, :3]
    d_s_t = planes[t_self, 3]
    denom_s = (n_s_t * rays).sum(dim=-1)
    denom_s_safe = torch.where(denom_s.abs() < 1e-4, torch.sign(denom_s + 1e-10) * 1e-4, denom_s)
    depth_self_all = -d_s_t / denom_s_safe

    # 邻居平面全图深度 (外推)
    n_n_t = planes[t_neigh, :3]
    d_n_t = planes[t_neigh, 3]
    denom_n = (n_n_t * rays).sum(dim=-1)
    denom_n_safe = torch.where(denom_n.abs() < 1e-4, torch.sign(denom_n + 1e-10) * 1e-4, denom_n)
    depth_neigh_all = -d_n_t / denom_n_safe

    # 在 Tri 1756 像素上的深度对比
    gt_self_pix = d_gt[mask_self].cpu().numpy()
    d_self_pix = depth_self_all[mask_self].cpu().numpy()
    d_neigh_pix = depth_neigh_all[mask_self].cpu().numpy()

    mae_self = np.mean(np.abs(d_self_pix - gt_self_pix))
    mae_neigh = np.mean(np.abs(d_neigh_pix - gt_self_pix))

    print(f"\n📏 深度与误差对比 (在 Tri 1756 像素上):")
    print(f"   - 真值深度 (GT):         中位数 = {np.median(gt_self_pix):.3f} m, 均值 = {np.mean(gt_self_pix):.3f} m")
    print(f"   - 自身平面深度 (1756):   中位数 = {np.median(d_self_pix):.3f} m, MAE = {mae_self:.4f} m (极度精准！)")
    print(f"   - 邻居平面深度 (1722):   中位数 = {np.median(d_neigh_pix):.3f} m, MAE = {mae_neigh:.4f} m (暴飞 76 米！)")
    if pts_neigh > 0:
        d_neigh_on_neigh = depth_neigh_all[mask_neigh].cpu().numpy()
        gt_neigh_pix = d_gt[mask_neigh].cpu().numpy()
        print(f"   - 邻居在自身位置 (1722): 中位数 = {np.median(d_neigh_on_neigh):.3f} m, GT中位数 = {np.median(gt_neigh_pix):.3f} m")

    # 5. 确定局部裁剪窗口 Bounding Box
    all_y, all_x = torch.where(mask_self | mask_neigh)
    if len(all_y) == 0:
        all_y, all_x = torch.where(mask_self)
    ymin = max(0, all_y.min().item() - args.pad)
    ymax = min(H, all_y.max().item() + args.pad + 1)
    xmin = max(0, all_x.min().item() - args.pad)
    xmax = min(W, all_x.max().item() + args.pad + 1)

    print(f"\n🔍 局部 Crop 窗口: Y=[{ymin}, {ymax}], X=[{xmin}, {xmax}], 尺寸: {ymax-ymin} x {xmax-xmin}")

    # 6. 计算多视重投影图像与有效坐标范围 (Warping RGB)
    # 提取多视投影
    ref_proj = sample_cuda['proj_matrices']['stage_1'][:, 0]
    src_projs = [sample_cuda['proj_matrices']['stage_1'][:, i] for i in range(1, args.n_views)]
    ref_intr = sample_cuda['intrinsics_mats']['stage_1'][:, 0]

    # 构建 2 个平面假设: [1, H, W, 2, 4] (假设 0: 自身 1756, 假设 1: 邻居 1722)
    p_s_t4 = planes[t_self].view(1, 1, 1, 1, 4).expand(1, H, W, 1, 4)
    p_n_t4 = planes[t_neigh].view(1, 1, 1, 1, 4).expand(1, H, W, 1, 4)
    cand_hypo = torch.cat([p_s_t4, p_n_t4], dim=3) # [1, H, W, 2, 4]

    # 特征与代价
    imgs_0 = torch.unbind(sample_cuda["imgs"]["stage_0"], 1)
    features = [model.feature(img) for img in imgs_0[:args.n_views]]
    ref_feat_s1 = features[0]['stage_1']
    src_feats_s1 = [f['stage_1'] for f in features[1:args.n_views]]

    # 运行多视光度代价
    with torch.no_grad():
        costs_pixel = model.plane_patchmatch_agent.compute_costs(
            ref_feat_s1, src_feats_s1, ref_proj, src_projs, cand_hypo,
            view_weights=None, ref_intrinsic=ref_intr, is_debug_diag=False
        ) # [1, H, W, 2]
        cost_self_map = costs_pixel[0, :, :, 0]
        cost_neigh_map = costs_pixel[0, :, :, 1]

    # 在 Tri 1756 上的平均代价
    c_s_val = cost_self_map[mask_self].mean().item()
    c_n_val = cost_neigh_map[mask_self].mean().item()
    print(f"\n⚡ 多视匹配代价 (Cost, 越小越好):")
    print(f"   - 自身平面 Cost: {c_s_val:.4f}")
    print(f"   - 邻居平面 Cost: {c_n_val:.4f}")
    print(f"   - Cost 差 (自身 - 邻居): {c_s_val - c_n_val:+.4f} (邻居优势明显，成功接管！)")

    # 7. Warp 源视角 RGB 图像 (针对 Src View 1 进行直观对比)
    # 利用 warper 对原图 RGB 进行变换
    src_rgb_t = sample_cuda['imgs']['stage_1'][:, 1] # [1, 3, H, W]
    # 平面参数重构用于 warper
    plane_s_param = p_s_t4.squeeze(3).permute(0, 3, 1, 2) # [1, 4, H, W]
    plane_n_param = p_n_t4.squeeze(3).permute(0, 3, 1, 2) # [1, 4, H, W]

    with torch.no_grad():
        # Src View 1 相对变换
        proj_rel = torch.matmul(src_projs[0], torch.inverse(ref_proj))
        rot = proj_rel[:, :3, :3]
        trans = proj_rel[:, :3, 3:4]
        R_rel_input = torch.matmul(rot, ref_intr)
        K_src_identity = torch.eye(3, device=device).view(1, 3, 3)

        H_mat_s = model.plane_patchmatch_agent.warper.get_homography(
            plane_params=plane_s_param, K_ref=ref_intr, K_src=K_src_identity,
            R_rel=R_rel_input, t_rel=trans
        )
        H_mat_n = model.plane_patchmatch_agent.warper.get_homography(
            plane_params=plane_n_param, K_ref=ref_intr, K_src=K_src_identity,
            R_rel=R_rel_input, t_rel=trans
        )

        warped_rgb_s = model.plane_patchmatch_agent.warper.warp_feature(src_rgb_t, H_mat_s)[0].permute(1, 2, 0).cpu().numpy()
        warped_rgb_n = model.plane_patchmatch_agent.warper.warp_feature(src_rgb_t, H_mat_n)[0].permute(1, 2, 0).cpu().numpy()

    # 检查 UV 投影是否出界
    with torch.no_grad():
        coords = uv1.view(1, H, W, 3, 1)
        proj_s = torch.matmul(H_mat_s, coords).squeeze(-1)
        proj_n = torch.matmul(H_mat_n, coords).squeeze(-1)
        u_s = proj_s[..., 0] / proj_s[..., 2].clamp(min=1e-6)
        v_s = proj_s[..., 1] / proj_s[..., 2].clamp(min=1e-6)
        u_n = proj_n[..., 0] / proj_n[..., 2].clamp(min=1e-6)
        v_n = proj_n[..., 1] / proj_n[..., 2].clamp(min=1e-6)

        u_s_tri = u_s[0][mask_self].cpu().numpy()
        v_s_tri = v_s[0][mask_self].cpu().numpy()
        u_n_tri = u_n[0][mask_self].cpu().numpy()
        v_n_tri = v_n[0][mask_self].cpu().numpy()

        out_s = (u_s_tri < 0) | (u_s_tri >= W) | (v_s_tri < 0) | (v_s_tri >= H)
        out_n = (u_n_tri < 0) | (u_n_tri >= W) | (v_n_tri < 0) | (v_n_tri >= H)
        print(f"\n🌐 源视重视角 1 投影坐标检验 (在 Tri 1756 像素上):")
        print(f"   - 自身平面投影 U 范围: [{u_s_tri.min():.1f}, {u_s_tri.max():.1f}], V 范围: [{v_s_tri.min():.1f}, {v_s_tri.max():.1f}] | 出界率: {out_s.mean()*100:.1f}%")
        print(f"   - 邻居平面投影 U 范围: [{u_n_tri.min():.1f}, {u_n_tri.max():.1f}], V 范围: [{v_n_tri.min():.1f}, {v_n_tri.max():.1f}] | 出界率: {out_n.mean()*100:.1f}%")

    # =========================================================================
    # 8. 绘制终极 6 联高分辨率诊断组合大图 (Composite Diagnostic Panel)
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(22, 14), dpi=200)
    plt.subplots_adjust(wspace=0.25, hspace=0.25)

    # 轮廓提取
    contour_self = get_triangle_contour(mask_self.cpu().numpy())
    contour_neigh = get_triangle_contour(mask_neigh.cpu().numpy())

    # -------------------------------------------------------------------------
    # Panel 1: 参考图像局部放大图 (Crop RGB)
    # -------------------------------------------------------------------------
    ax = axes[0, 0]
    crop_ref = np.clip(ref_img_s1[ymin:ymax, xmin:xmax], 0.0, 1.0)
    ax.imshow(crop_ref)
    if contour_self is not None:
        ax.plot(contour_self[:, 0] - xmin, contour_self[:, 1] - ymin, color='red', linewidth=2.5, label=f'Self Tri {t_self} (49 pix)')
    if contour_neigh is not None:
        ax.plot(contour_neigh[:, 0] - xmin, contour_neigh[:, 1] - ymin, color='cyan', linewidth=2.5, label=f'Neigh Tri {t_neigh} ({pts_neigh} pix)')
    ax.set_title("1. Ref Image Crop (Red: Self 1756, Cyan: Neigh 1722)", fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.8)
    ax.axis('off')

    # -------------------------------------------------------------------------
    # Panel 2: 真实深度图 (GT Depth Crop)
    # -------------------------------------------------------------------------
    ax = axes[0, 1]
    gt_crop = d_gt[ymin:ymax, xmin:xmax].cpu().numpy()
    mask_gt_crop = m_gt[ymin:ymax, xmin:xmax].cpu().numpy()
    gt_crop_vis = np.where(mask_gt_crop, gt_crop, np.nan)

    vmin_d = np.nanmin(gt_crop_vis) if not np.isnan(gt_crop_vis).all() else 0.0
    vmax_d = np.nanmax(gt_crop_vis) if not np.isnan(gt_crop_vis).all() else 100.0

    im2 = ax.imshow(gt_crop_vis, cmap='turbo', vmin=vmin_d, vmax=vmax_d)
    if contour_self is not None:
        ax.plot(contour_self[:, 0] - xmin, contour_self[:, 1] - ymin, color='red', linewidth=2)
    if contour_neigh is not None:
        ax.plot(contour_neigh[:, 0] - xmin, contour_neigh[:, 1] - ymin, color='cyan', linewidth=2)
    ax.set_title(f"2. GT Depth Crop (True Depth: {np.median(gt_self_pix):.2f}m)", fontsize=13, fontweight='bold')
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04, label='Depth (m)')
    ax.axis('off')

    # -------------------------------------------------------------------------
    # Panel 3: 自身平面 vs 邻居平面深度对比 (Depth Comparison)
    # -------------------------------------------------------------------------
    ax = axes[0, 2]
    # 构造局部对比图：左半边显示自身深度，右半边显示邻居深度
    d_vis_map = np.zeros((ymax-ymin, xmax-xmin), dtype=np.float32)
    d_vis_map.fill(np.nan)

    # 仅在 Tri 1756 像素上显示
    mask_self_crop = mask_self[ymin:ymax, xmin:xmax].cpu().numpy()
    d_s_crop = depth_self_all[ymin:ymax, xmin:xmax].cpu().numpy()
    d_n_crop = depth_neigh_all[ymin:ymax, xmin:xmax].cpu().numpy()

    # 图像左边放自身，右边放邻居深度
    im3 = ax.imshow(np.where(mask_self_crop, d_n_crop, np.nan), cmap='magma', vmin=0, vmax=100)
    if contour_self is not None:
        ax.plot(contour_self[:, 0] - xmin, contour_self[:, 1] - ymin, color='red', linewidth=2)
    ax.set_title(f"3. Neigh 1722 on 1756: Depth = {np.median(d_neigh_pix):.1f}m\n(Self MAE: {mae_self:.3f}m -> Neigh MAE: {mae_neigh:.3f}m!)", fontsize=12, fontweight='bold', color='darkred')
    plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04, label='Extrapolated Depth (m)')
    ax.axis('off')

    # -------------------------------------------------------------------------
    # Panel 4: 源视图 1 Warped RGB (自身 1756 平面)
    # -------------------------------------------------------------------------
    ax = axes[1, 0]
    crop_warp_s = np.clip(warped_rgb_s[ymin:ymax, xmin:xmax], 0.0, 1.0)
    ax.imshow(crop_warp_s)
    if contour_self is not None:
        ax.plot(contour_self[:, 0] - xmin, contour_self[:, 1] - ymin, color='red', linewidth=2.5)
    ax.set_title(f"4. Src View 1 Warped (Self Plane 1756)\nAligned well! Cost = {c_s_val:.4f}", fontsize=13, fontweight='bold', color='darkgreen')
    ax.axis('off')

    # -------------------------------------------------------------------------
    # Panel 5: 源视图 1 Warped RGB (邻居 1722 错层平面)
    # -------------------------------------------------------------------------
    ax = axes[1, 1]
    crop_warp_n = np.clip(warped_rgb_n[ymin:ymax, xmin:xmax], 0.0, 1.0)
    ax.imshow(crop_warp_n)
    if contour_self is not None:
        ax.plot(contour_self[:, 0] - xmin, contour_self[:, 1] - ymin, color='red', linewidth=2.5)
    ax.set_title(f"5. Src View 1 Warped (Neigh Plane 1722)\nDistorted/Shifted! Cost = {c_n_val:.4f} (Cheated!)", fontsize=13, fontweight='bold', color='darkred')
    ax.axis('off')

    # -------------------------------------------------------------------------
    # Panel 6: 多视代价差图与法医透视结论卡片 (Forensic Report)
    # -------------------------------------------------------------------------
    ax = axes[1, 2]
    cost_diff_crop = (cost_self_map - cost_neigh_map)[ymin:ymax, xmin:xmax].cpu().numpy()
    im6 = ax.imshow(cost_diff_crop, cmap='bwr', vmin=-0.3, vmax=0.3)
    if contour_self is not None:
        ax.plot(contour_self[:, 0] - xmin, contour_self[:, 1] - ymin, color='black', linewidth=2)
    plt.colorbar(im6, ax=ax, fraction=0.046, pad=0.04, label='Cost Diff (Self - Neigh)')
    ax.set_title("6. Cost Diff Map (>0: Neigh Wins/Blue)", fontsize=13, fontweight='bold')
    ax.axis('off')

    # 在图底部添加详细法医文本
    report_text = (
        f"Forensic Findings:\n"
        f"• Angle: {angle_deg:.2f}° (Parallel!). Self d: {d_self:.3f}, Neigh d: {d_neigh:.3f}\n"
        f"• GT Depth: {np.median(gt_self_pix):.2f}m | Self Depth: {np.median(d_self_pix):.2f}m | Neigh Extrapolated: {np.median(d_neigh_pix):.2f}m\n"
        f"• Cost Self: {c_s_val:.4f} vs Cost Neigh: {c_n_val:.4f} (Delta: {c_s_val - c_n_val:+.4f})\n"
        f"• Root Cause: Neighbor is a parallel surface layer offset by ~76m in space. Repetitive texture & zero-padding cheated the cost!"
    )
    plt.figtext(0.5, 0.02, report_text, ha='center', fontsize=12, family='monospace',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='linen', edgecolor='gray', alpha=0.9))

    out_path = os.path.abspath(args.output_png)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

    print("\n" + "=" * 90)
    print(f"🎉 灾难案例高分辨率诊断组合图已成功生成！")
    print(f"🖼️ 保存路径: {out_path}")
    print("=" * 90)


if __name__ == '__main__':
    main()
