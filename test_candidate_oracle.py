"""
========================================================================================
WHU-MVS 独立离线诊断系统：视场有效性 (FOV Validity) 极速 A/B 对照评测 (test_candidate_oracle.py)
========================================================================================
核心升级：
1. 极速一次性前向同时计算：
   - Mode A (原版多视代价)：保留出界 0 特征参与平均；
   - Mode B (加视场有效性 FOV Mask)：严格剔除投影出界视角，仅对在视野内的源视角加权；
   - 全出界或不可见候选直接赋予惩罚代价 (1e9)，禁止接管。
2. 零训练成本，全量 383,600 三角面、1,150 万固定像素严格代数闭合。
3. 彻底删除多余的缓慢个例深挖循环，纯 GPU 矢量化，极速完成评测并保存至 TXT。
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

# 控制 GPU id
_gpu_choice = os.environ.get('GPU_ID', os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
os.environ['CUDA_VISIBLE_DEVICES'] = str(_gpu_choice)

from datasets import find_dataset_def
from datasets.dtu_whu import collate_keep_list
from models import PatchmatchNet
from models.net import build_neighbor_indices
from utils import tocuda, make_nograd_func

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description='WHU-MVS Candidate Evaluation & FOV-Cost Selection')
    parser.add_argument('--mode', default='cost_select', choices=['cost_select', 'oracle'],
                        help='评测模式: cost_select (默认, A/B 对照), oracle (传统远距离上界)')
    parser.add_argument('--cost_mode', default='both', choices=['both', 'orig', 'fov'],
                        help='代价评估模式: both (同时跑 A 和 B 对照), orig (仅 Mode A), fov (仅 Mode B)')
    parser.add_argument('--dataset', default='dtu_whu', help='select dataset (default: dtu_whu)')
    parser.add_argument('--testpath', default='/home/ym/Experiment/Datas/WHU_MVS_dataset', help='testing data path')
    parser.add_argument('--testlist', default='lists/whu/test9.20.txt', help='testing scan list file')
    parser.add_argument('--loadckpt', required=True, help='checkpoint path')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size (fixed to 1)')
    parser.add_argument('--n_views', type=int, default=5, help='number of views')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers for DataLoader')
    parser.add_argument('--err_thresh', type=float, default=0.10, help='错误面判定门限 (米, 默认 0.10m=10cm)')
    parser.add_argument('--conf_thresh', type=float, default=0.80, help='可信候选池置信度门限 (默认 0.80)')
    parser.add_argument('--margins', nargs='+', type=float, default=[0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15],
                        help='优势门限 Margin 扫描列表 (默认: 0.0 0.005 0.01 0.02 0.03 0.05 0.08 0.10 0.15)')
    parser.add_argument('--output_txt', default='cost_select_fov_ab_report.txt',
                        help='诊断详细报告输出 TXT 文件路径 (默认: cost_select_fov_ab_report.txt)')

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
    """极速前向推断：提取 Stage 1 几何与特征。"""
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


def compute_costs_dual(agent, ref_feature, src_features, ref_proj, src_projs, current_hypotheses, ref_intrinsic):
    """
    一次性极速计算 Mode A (原版代价) 与 Mode B (加视场有效性 FOV Mask) 两种代价体积。
    Returns:
        cost_A: [B, H, W, K] (原版)
        cost_B: [B, H, W, K] (加视场掩码，全出界像素置为 1e9)
        valid_views_count: [B, H, W, K] (有效源视角数量 0~Nviews-1)
    """
    B, H, W, K, _ = current_hypotheses.shape
    C = ref_feature.shape[1]
    device = ref_feature.device
    num_src_views = len(src_features)

    # 1. 预处理平面参数 [B*K, 4, H, W]
    plane_params = current_hypotheses.permute(0, 3, 4, 1, 2).reshape(B * K, 4, H, W)

    # 2. Ref 特征分组并归一化
    ref_feat_grouped = ref_feature.view(B, agent.G, C // agent.G, H, W)
    ref_feat_expanded = ref_feat_grouped.unsqueeze(1).repeat(1, K, 1, 1, 1, 1).view(B * K, agent.G, C // agent.G, H, W)
    ref_feat_norm = F.normalize(ref_feat_expanded, p=2, dim=2)

    # 3. 准备变换参数
    K_ref_expand = ref_intrinsic.unsqueeze(1).repeat(1, K, 1, 1).view(B * K, 3, 3)
    K_src_identity = torch.eye(3, device=device).view(1, 3, 3).expand(B * K, -1, -1)

    # 生成网格坐标 (u, v, 1)
    y_g, x_g = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    coords = torch.stack([x_g, y_g, torch.ones_like(x_g)], dim=-1).float() # [H, W, 3]
    coords_4d = coords.view(1, H, W, 3, 1).expand(B * K, -1, -1, -1, -1)

    # 累加容器
    sim_sum_A = 0.0 # Mode A: 原版全累加
    sim_sum_B = 0.0 # Mode B: 仅有效视场内累加
    valid_weight_B = 0.0 # Mode B: 视场掩码累加

    for i, (src_feat, src_proj) in enumerate(zip(src_features, src_projs)):
        with torch.no_grad():
            proj_rel = torch.matmul(src_proj, torch.inverse(ref_proj))
            rot = proj_rel[:, :3, :3]
            trans = proj_rel[:, :3, 3:4]
            rot_expand = rot.unsqueeze(1).repeat(1, K, 1, 1).view(B * K, 3, 3)
            trans_expand = trans.unsqueeze(1).repeat(1, K, 1, 1).view(B * K, 3, 1)
            R_rel_input = torch.matmul(rot_expand, K_ref_expand)

        H_mats = agent.warper.get_homography(
            plane_params=plane_params, K_ref=K_ref_expand, K_src=K_src_identity,
            R_rel=R_rel_input, t_rel=trans_expand
        )

        # 投影坐标与透视除法
        projected_coords = torch.matmul(H_mats, coords_4d).squeeze(-1) # [B*K, H, W, 3]
        u_prime = projected_coords[..., 0]
        v_prime = projected_coords[..., 1]
        w_prime = projected_coords[..., 2]

        valid_z = w_prime > 1e-6
        w_prime_safe = torch.where(valid_z, w_prime, torch.ones_like(w_prime) * 1e-12)
        u_norm = u_prime / w_prime_safe
        v_norm = v_prime / w_prime_safe

        # 视场有效性掩码 (FOV Valid Mask): 位于源图像范围内且在相机前方
        fov_mask = valid_z & (u_norm >= 0.0) & (u_norm <= (W - 1.0)) & (v_norm >= 0.0) & (v_norm <= (H - 1.0))
        fov_mask_4d = fov_mask.unsqueeze(1).float() # [B*K, 1, H, W]

        # 归一化采样
        uv_grid = torch.stack([2.0 * u_norm / (W - 1.0) - 1.0, 2.0 * v_norm / (H - 1.0) - 1.0], dim=-1)
        src_feat_expand = src_feat.unsqueeze(1).repeat(1, K, 1, 1, 1).view(B * K, C, H, W)
        warped_src = F.grid_sample(src_feat_expand, uv_grid, align_corners=True, padding_mode='zeros')

        # 余弦相似度
        warped_src_grouped = warped_src.view(B * K, agent.G, C // agent.G, H, W)
        warped_src_norm = F.normalize(warped_src_grouped, p=2, dim=2)
        sim_view = (warped_src_norm * ref_feat_norm).mean(dim=2).mean(dim=1, keepdim=True) # [B*K, 1, H, W]

        # Mode A 累加
        sim_sum_A = sim_sum_A + sim_view
        # Mode B 累加 (出界视角赋 0 权重)
        sim_sum_B = sim_sum_B + (sim_view * fov_mask_4d)
        valid_weight_B = valid_weight_B + fov_mask_4d

    # 还原代价
    # Mode A (原版平均)
    sim_fused_A = sim_sum_A / max(num_src_views, 1)
    cost_A = -sim_fused_A.view(B, K, H, W).permute(0, 2, 3, 1)

    # Mode B (加权平均，全出界像素惩罚为 1e9)
    has_valid_view = valid_weight_B > 0.5 # 至少有 1 个视角在视场内
    sim_fused_B = torch.where(has_valid_view, sim_sum_B / (valid_weight_B + 1e-6), torch.tensor(-1e9, device=device))
    cost_B = -sim_fused_B.view(B, K, H, W).permute(0, 2, 3, 1)

    valid_views_count = valid_weight_B.view(B, K, H, W).permute(0, 2, 3, 1)

    return cost_A, cost_B, valid_views_count


def diagnose_scan_fov_ab(model, outputs, depth_gt, mask, sample_cuda, args):
    """
    单个 Scan 的 A/B 极速对照评测算子：
    计算原版代价 A 与视场掩码代价 B 下的整面决策指标。
    """
    output_plane = outputs.get('output_plane', {})
    if 'final_plane' not in output_plane or 'tri_id_map' not in output_plane:
        return None

    planes = output_plane['final_plane'][0]            # [N, 4]
    W_tri = output_plane['W_plane_tri'][0, :, 0]       # [N]
    tri_id_map = output_plane['tri_id_map'][0]         # [H, W]

    K_intr = sample_cuda['intrinsics_mats']['stage_1'][0, 0] # [3, 3]
    d_gt = depth_gt['stage_1'][0, 0]                   # [H, W]
    m_gt = (mask['stage_1'][0, 0] > 0.5)               # [H, W]
    d_min = sample_cuda['depth_min'][0].item()
    d_max = sample_cuda['depth_max'][0].item()

    N = planes.shape[0]
    H, W = tri_id_map.shape

    # 1. 局部 3 邻居索引
    tri_infos_obj = outputs.get('tri_infos', None)
    neigh_idx = None
    if tri_infos_obj is not None:
        try:
            if isinstance(tri_infos_obj, list) and len(tri_infos_obj) > 0 and isinstance(tri_infos_obj[0], dict):
                first_item = tri_infos_obj[0]
                if 'batch_num_tri' in first_item:
                    max_tri = first_item['batch_num_tri']
                    if hasattr(max_tri, '__iter__'): max_tri = max(max_tri)
                    neigh_idx = build_neighbor_indices(tri_infos_obj, max_tri, planes.device)[0]
            elif isinstance(tri_infos_obj, dict):
                if 'batch_num_tri' in tri_infos_obj:
                    max_tri = tri_infos_obj['batch_num_tri']
                    if hasattr(max_tri, '__iter__'): max_tri = max(max_tri)
                    neigh_idx = build_neighbor_indices([tri_infos_obj], max_tri, planes.device)[0]
        except Exception:
            neigh_idx = None

    if neigh_idx is None or neigh_idx.shape[0] != N:
        neigh_idx = torch.arange(N, device=planes.device).unsqueeze(1).expand(-1, 3)

    cand_idx_matrix = torch.cat([torch.arange(N, device=planes.device).unsqueeze(1), neigh_idx], dim=-1) # [N, 4]
    cand_planes = planes[cand_idx_matrix].unsqueeze(0) # [1, N, 4, 4]

    # 2. 一次性提取特征与双模式代价
    with torch.no_grad():
        imgs_0 = torch.unbind(sample_cuda["imgs"]["stage_0"], 1)
        features = [model.feature(img) for img in imgs_0[:args.n_views]]
        ref_feat_s1 = features[0]['stage_1']
        src_feats_s1 = [f['stage_1'] for f in features[1:args.n_views]]
        ref_proj = sample_cuda['proj_matrices']['stage_1'][:, 0]
        src_projs = [sample_cuda['proj_matrices']['stage_1'][:, i] for i in range(1, args.n_views)]
        ref_intr = sample_cuda['intrinsics_mats']['stage_1'][:, 0]

        pixel_hypotheses = model.plane_patchmatch_agent.map_tri_to_pixel(cand_planes, tri_id_map.unsqueeze(0), H, W)

        # 极速双模式代价
        cost_A_pix, cost_B_pix, valid_views_pix = compute_costs_dual(
            model.plane_patchmatch_agent, ref_feat_s1, src_feats_s1, ref_proj, src_projs,
            pixel_hypotheses, ref_intr
        )

        cand_costs_A = model.plane_patchmatch_agent.aggregate_costs_per_triangle(cost_A_pix, tri_id_map.unsqueeze(0), N)[0]
        cand_costs_B = model.plane_patchmatch_agent.aggregate_costs_per_triangle(cost_B_pix, tri_id_map.unsqueeze(0), N)[0]
        cand_valid_views = model.plane_patchmatch_agent.aggregate_costs_per_triangle(valid_views_pix, tri_id_map.unsqueeze(0), N)[0]

    # 3. 纯几何合法性与像素级真实误差提取
    valid_pix = (tri_id_map >= 0) & m_gt & (d_gt > 0)
    if not valid_pix.any():
        return None

    y_idx, x_idx = torch.where(valid_pix)
    tri_ids_pix = tri_id_map[valid_pix].long()        # [M]
    gt_pix = d_gt[valid_pix]                          # [M]

    K_inv = torch.inverse(K_intr)
    ones = torch.ones_like(x_idx, dtype=torch.float32)
    uv1 = torch.stack([x_idx.float(), y_idx.float(), ones], dim=-1) # [M, 3]
    rays = (K_inv @ uv1.unsqueeze(-1)).squeeze(-1)    # [M, 3]
    rays_len = torch.norm(rays, dim=-1, keepdim=True).clamp(min=1e-6)

    pix_counts = torch.zeros(N, device=planes.device, dtype=torch.float32)
    pix_counts.scatter_add_(0, tri_ids_pix, torch.ones_like(tri_ids_pix, dtype=torch.float32))

    d_est_cands, is_geo_valid_cands, mae_cands = [], [], []

    for k in range(4):
        cand_k = cand_idx_matrix[tri_ids_pix, k]
        n_k = planes[cand_k, :3]
        d_k = planes[cand_k, 3]
        denom_k = (n_k * rays).sum(dim=-1)
        denom_k_safe = torch.where(denom_k.abs() < 1e-4, torch.sign(denom_k + 1e-10) * 1e-4, denom_k)
        d_est_k = -d_k / denom_k_safe
        cos_k = denom_k.abs() / (torch.norm(n_k, dim=-1) * rays_len.squeeze(-1)).clamp(min=1e-6)

        valid_k_pix = (d_est_k > 0) & torch.isfinite(d_est_k) & (cos_k >= 0.05) & \
                      (d_est_k >= 0.05 * d_min) & (d_est_k <= 5.0 * d_max)

        valid_cnt_k = torch.zeros(N, device=planes.device, dtype=torch.float32)
        valid_cnt_k.scatter_add_(0, tri_ids_pix, valid_k_pix.float())
        is_geo_valid_k = (valid_cnt_k == pix_counts) & (pix_counts >= 5)

        err_k_pix = (d_est_k - gt_pix).abs()
        err_sum_k = torch.zeros(N, device=planes.device, dtype=torch.float32)
        err_sum_k.scatter_add_(0, tri_ids_pix, err_k_pix)
        mae_k_tri = err_sum_k / pix_counts.clamp(min=1.0)
        mae_k_tri = torch.where(is_geo_valid_k, mae_k_tri, torch.tensor(999.0, device=planes.device))

        d_est_cands.append(d_est_k)
        is_geo_valid_cands.append(is_geo_valid_k)
        mae_cands.append(mae_k_tri)

    d_est_stack = torch.stack(d_est_cands, dim=-1)          # [M, 4]
    is_geo_valid_stack = torch.stack(is_geo_valid_cands, dim=-1) # [N, 4]
    mae_stack = torch.stack(mae_cands, dim=-1)              # [N, 4]

    is_planar_tri = (W_tri >= args.conf_thresh) & (pix_counts >= 5)
    is_geo_valid_0 = is_geo_valid_stack[:, 0]
    is_eval_planar = is_planar_tri & is_geo_valid_0
    cnt_eval = is_eval_planar.sum().item()
    if cnt_eval == 0:
        return None

    eval_pix_mask = is_eval_planar[tri_ids_pix] # [M]
    n_eval_pix = eval_pix_mask.sum().item()
    if n_eval_pix == 0:
        return None

    d_base_pix = d_est_stack[eval_pix_mask, 0]
    gt_eval_pix = gt_pix[eval_pix_mask]
    e_old = (d_base_pix - gt_eval_pix).abs()
    base_err_sum = e_old.sum().item()
    glob_curr_m = base_err_sum / n_eval_pix

    # Oracle 计算
    oracle_valid_mae = torch.where(is_geo_valid_stack, mae_stack, torch.tensor(999.0, device=planes.device))
    best_oracle_cand = torch.argmin(oracle_valid_mae, dim=-1)
    best_oracle_pix_cand = best_oracle_cand[tri_ids_pix[eval_pix_mask]]
    d_oracle_pix = d_est_stack[eval_pix_mask].gather(1, best_oracle_pix_cand.unsqueeze(1)).squeeze(1)
    e_oracle = (d_oracle_pix - gt_eval_pix).abs()
    glob_oracle_m = e_oracle.sum().item() / n_eval_pix
    delta_oracle_mm = (glob_curr_m - glob_oracle_m) * 1000.0

    mae_curr = mae_stack[:, 0]
    is_err = is_eval_planar & (mae_curr > args.err_thresh)
    is_correct = is_eval_planar & (mae_curr <= args.err_thresh)
    cnt_err = is_err.sum().item()
    cnt_correct = is_correct.sum().item()

    mae_oracle_tri = mae_stack[torch.arange(N, device=planes.device), best_oracle_cand]
    is_oracle_improved = is_err & (mae_oracle_tri < mae_curr - 0.005)
    is_oracle_cured = is_err & (mae_oracle_tri <= args.err_thresh)
    oracle_rescue_tri_cnt = is_oracle_improved.sum().item()
    oracle_cure_tri_cnt = is_oracle_cured.sum().item()

    # 4. 分别针对 Mode A 与 Mode B 执行门槛扫描
    def evaluate_cost_mode(cand_costs_tensor):
        cand_costs_valid = torch.where(is_geo_valid_stack, cand_costs_tensor, torch.tensor(1e9, device=cand_costs_tensor.device))
        cost_self = cand_costs_valid[:, 0]
        best_neigh_offset = torch.argmin(cand_costs_valid[:, 1:], dim=-1)
        best_neigh_idx = 1 + best_neigh_offset
        best_neigh_cost = cand_costs_valid.gather(1, best_neigh_idx.unsqueeze(1)).squeeze(1)

        mode_stats = {}
        for tau in args.margins:
            should_switch = (best_neigh_cost < cost_self - tau) & (best_neigh_cost < 900.0) & (cost_self < 900.0)
            best_cand_tau = torch.where(should_switch, best_neigh_idx, torch.zeros_like(best_neigh_idx))

            switched_mask = is_eval_planar & (best_cand_tau > 0)
            switched_tri_cnt = switched_mask.sum().item()
            switched_pix_cnt = pix_counts[switched_mask].sum().item()

            chosen_cand_pix = best_cand_tau[tri_ids_pix[eval_pix_mask]]
            d_chosen_pix = d_est_stack[eval_pix_mask].gather(1, chosen_cand_pix.unsqueeze(1)).squeeze(1)
            e_new = (d_chosen_pix - gt_eval_pix).abs()

            diff = e_old - e_new
            earned_pix_sum = torch.clamp(diff, min=0.0).sum().item()
            damaged_pix_sum = torch.clamp(-diff, min=0.0).sum().item()
            net_err_sum = e_new.sum().item()

            if switched_tri_cnt == 0:
                earned_pix_sum = 0.0
                damaged_pix_sum = 0.0

            mae_cost_tau = mae_stack[torch.arange(N, device=planes.device), best_cand_tau]
            is_cost_improved = is_err & (mae_cost_tau < mae_curr - 0.005)
            is_cost_cured = is_err & (mae_cost_tau <= args.err_thresh)
            is_cost_harmed = is_correct & (mae_cost_tau > mae_curr + 0.005)

            mode_stats[tau] = {
                "switched_tri_cnt": switched_tri_cnt,
                "switched_pix_cnt": switched_pix_cnt,
                "cnt_rescued": is_cost_improved.sum().item(),
                "cnt_cured": is_cost_cured.sum().item(),
                "cnt_harmed": is_cost_harmed.sum().item(),
                "earned_pix_sum": earned_pix_sum,
                "damaged_pix_sum": damaged_pix_sum,
                "cost_err_sum": net_err_sum
            }
        return mode_stats

    stats_A = evaluate_cost_mode(cand_costs_A)
    stats_B = evaluate_cost_mode(cand_costs_B)

    # 视场出界统计 (在有效评测三角形上，统计邻居候选平面的有效视角数量分布)
    neigh_valid_views_eval = cand_valid_views[is_eval_planar, 1:].cpu().numpy() # [cnt_eval, 3]
    # 每个邻居候选有效视角数四舍五入为整数 0, 1, 2, 3, 4
    v_counts = np.round(neigh_valid_views_eval).astype(int).clip(0, 4)
    out_views_stats = {
        "v4_cnt": int((v_counts == 4).sum()),
        "v3_cnt": int((v_counts == 3).sum()),
        "v2_cnt": int((v_counts == 2).sum()),
        "v1_cnt": int((v_counts == 1).sum()),
        "v0_cnt": int((v_counts == 0).sum()),
        "total_neigh_eval": int(cnt_eval * 3)
    }

    return {
        "cnt_eval": cnt_eval,
        "n_eval_pix": n_eval_pix,
        "cnt_err": cnt_err,
        "cnt_correct": cnt_correct,
        "base_err_sum": base_err_sum,
        "glob_curr_m": glob_curr_m,
        "glob_oracle_m": glob_oracle_m,
        "delta_oracle_mm": delta_oracle_mm,
        "oracle_rescue_tri_cnt": oracle_rescue_tri_cnt,
        "oracle_cure_tri_cnt": oracle_cure_tri_cnt,
        "stats_A": stats_A,
        "stats_B": stats_B,
        "out_views_stats": out_views_stats
    }


def main_cost_select(args):
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    # 双通道输出
    out_dir = os.path.dirname(os.path.abspath(args.output_txt))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    report_file = open(args.output_txt, 'w', encoding='utf-8')

    def log_print(msg=""):
        print(msg)
        report_file.write(str(msg) + "\n")
        report_file.flush()

    log_print("=" * 145)
    log_print("🚀 WHU-MVS 视场有效性 (FOV Validity) 极速 A/B 对照评测大表")
    log_print("=" * 145)
    log_print(f"元数据配置:")
    log_print(f"   - 检查点模型: {args.loadckpt}")
    log_print(f"   - 评估模式: Mode A (原版代价) vs Mode B (加视场掩码 FOV-Mask)")
    log_print(f"   - 评测列表: {args.testlist} | 门限: 错误面 > {args.err_thresh}m ({int(args.err_thresh*1000)}mm), 可信候选池 >= {args.conf_thresh}")
    log_print("-" * 145)

    # 1. 数据集与模型
    MVSDataset = find_dataset_def(args.dataset)
    test_dataset = MVSDataset(args.testpath, args.testlist, "test", nviews=args.n_views, robust_train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_keep_list, num_workers=args.num_workers)

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

    per_scan_results = {}
    log_print("\n" + "-" * 145)
    log_print(f"{'Scan':<10} | {'有效面(错/对)':<16} | {'有效像素':<10} | {'基准 MAE -> Oracle (m)':<28} | {'Mode A (τ=0 改善)':<20} | {'Mode B (τ=0 改善)':<20}")
    log_print("-" * 145)

    t0 = time.time()
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            scan, file_id = test_dataset.metas[batch_idx]
            outputs, depth_gt, mask, sample_cuda = evaluate_single_sample_stage1(model, sample, device)

            res = diagnose_scan_fov_ab(model, outputs, depth_gt, mask, sample_cuda, args)
            if res is None:
                continue

            if scan not in per_scan_results:
                per_scan_results[scan] = []
            per_scan_results[scan].append(res)

            s_tri = f"{res['cnt_eval']} ({res['cnt_err']}/{res['cnt_correct']})"
            s_pix = f"{res['n_eval_pix']}"
            s_base_ora = f"{res['glob_curr_m']:.4f} -> {res['glob_oracle_m']:.4f}m"

            cost_A_0_m = res['stats_A'][0.0]['cost_err_sum'] / res['n_eval_pix']
            cost_B_0_m = res['stats_B'][0.0]['cost_err_sum'] / res['n_eval_pix']
            delta_A_0 = (res['glob_curr_m'] - cost_A_0_m) * 1000.0
            delta_B_0 = (res['glob_curr_m'] - cost_B_0_m) * 1000.0

            log_print(f"{scan:<10} | {s_tri:<16} | {s_pix:<10} | {s_base_ora:<28} | {delta_A_0:>+12.2f} mm         | {delta_B_0:>+12.2f} mm")

    all_res = [r for records in per_scan_results.values() for r in records]
    if len(all_res) == 0:
        log_print("未采集到有效评测数据。")
        report_file.close()
        return

    # 全局代数闭合汇总
    total_pixels = sum(r['n_eval_pix'] for r in all_res)
    total_base_err = sum(r['base_err_sum'] for r in all_res)
    total_eval_tri = sum(r['cnt_eval'] for r in all_res)
    total_err_cnt = sum(r['cnt_err'] for r in all_res)
    total_correct_cnt = sum(r['cnt_correct'] for r in all_res)

    glob_curr_m = total_base_err / max(total_pixels, 1)
    total_oracle_err = sum(r['glob_oracle_m'] * r['n_eval_pix'] for r in all_res)
    glob_oracle_m = total_oracle_err / max(total_pixels, 1)
    delta_oracle_mm = (glob_curr_m - glob_oracle_m) * 1000.0

    oracle_rescue_tri_cnt = sum(r['oracle_rescue_tri_cnt'] for r in all_res)
    oracle_cure_tri_cnt = sum(r['oracle_cure_tri_cnt'] for r in all_res)
    oracle_rescue_tri_pct = (oracle_rescue_tri_cnt / total_err_cnt * 100.0) if total_err_cnt > 0 else 0.0
    oracle_cure_tri_pct = (oracle_cure_tri_cnt / total_err_cnt * 100.0) if total_err_cnt > 0 else 0.0

    # 视场出界全景统计
    total_v4 = sum(r['out_views_stats']['v4_cnt'] for r in all_res)
    total_v3 = sum(r['out_views_stats']['v3_cnt'] for r in all_res)
    total_v2 = sum(r['out_views_stats']['v2_cnt'] for r in all_res)
    total_v1 = sum(r['out_views_stats']['v1_cnt'] for r in all_res)
    total_v0 = sum(r['out_views_stats']['v0_cnt'] for r in all_res)
    total_neigh_eval = sum(r['out_views_stats']['total_neigh_eval'] for r in all_res)

    log_print("\n" + "=" * 145)
    log_print("🌐 视场有效性全景分布报告 (Candidate View Visibility Distribution):")
    log_print(f"   - 评估邻居候选总配对数: {total_neigh_eval} 个")
    log_print(f"   - 4 视完全有效 (4/4 Views In-FOV): {total_v4:>8} ({total_v4/total_neigh_eval*100:.1f}%)")
    log_print(f"   - 3 视有效 (1 视出界):             {total_v3:>8} ({total_v3/total_neigh_eval*100:.1f}%)")
    log_print(f"   - 2 视有效 (2 视出界):             {total_v2:>8} ({total_v2/total_neigh_eval*100:.1f}%)")
    log_print(f"   - 1 视有效 (3 视出界):             {total_v1:>8} ({total_v1/total_neigh_eval*100:.1f}%)")
    log_print(f"   - 0 视完全出界 (All Out-of-FOV):   {total_v0:>8} ({total_v0/total_neigh_eval*100:.1f}%)  <-- 致命伪候选！")
    log_print("=" * 145)

    def print_mode_table(mode_name, stats_key):
        log_print(f"\n📊 模式 {mode_name} 扫描结论:")
        log_print(f"基准平面精度 (Baseline): {glob_curr_m:.4f} m | 局部 Oracle 极限: {glob_oracle_m:.4f} m (理论最大红利: +{delta_oracle_mm:.2f} mm)")
        log_print("-" * 145)
        log_print(f"{'Margin (τ)':<12} | {'接管面数':<10} | {'接管像素比':<10} | {'面改善率':<8} | {'面误伤率':<8} | {'彻底修复率':<10} | {'赚取 (Δ+)':<12} | {'损失 (Δ-)':<12} | {'全图平面 MAE':<15} | {'实际净改善':<12} | {'全局兑现率':<10}")
        log_print("-" * 145)

        for tau in args.margins:
            sw_tri = sum(r[stats_key][tau]['switched_tri_cnt'] for r in all_res)
            sw_pix = sum(r[stats_key][tau]['switched_pix_cnt'] for r in all_res)
            sw_pix_pct = (sw_pix / max(total_pixels, 1) * 100.0)

            earned = sum(r[stats_key][tau]['earned_pix_sum'] for r in all_res)
            damaged = sum(r[stats_key][tau]['damaged_pix_sum'] for r in all_res)
            cost_err = sum(r[stats_key][tau]['cost_err_sum'] for r in all_res)

            d_plus_mm = (earned / max(total_pixels, 1)) * 1000.0
            d_minus_mm = (damaged / max(total_pixels, 1)) * 1000.0
            net_delta_mm = d_plus_mm - d_minus_mm
            mae_m = cost_err / max(total_pixels, 1)

            if sw_tri == 0:
                d_plus_mm, d_minus_mm, net_delta_mm = 0.0, 0.0, 0.0

            real_pct = (net_delta_mm / delta_oracle_mm * 100.0) if delta_oracle_mm > 0.001 else 0.0

            rescued = sum(r[stats_key][tau]['cnt_rescued'] for r in all_res)
            cured = sum(r[stats_key][tau]['cnt_cured'] for r in all_res)
            harmed = sum(r[stats_key][tau]['cnt_harmed'] for r in all_res)

            res_pct = (rescued / total_err_cnt * 100.0) if total_err_cnt > 0 else 0.0
            cure_pct = (cured / total_err_cnt * 100.0) if total_err_cnt > 0 else 0.0
            harm_pct = (harmed / total_correct_cnt * 100.0) if total_correct_cnt > 0 else 0.0

            s_tau = f"{tau:.3f}" if tau > 0 else "0.000 (原版)"
            log_print(f"{s_tau:<12} | {sw_tri:>10} | {sw_pix_pct:>9.2f}% | {res_pct:>7.1f}% | {harm_pct:>7.1f}% | {cure_pct:>9.1f}% | {f'+{d_plus_mm:.2f}mm':>12} | {f'-{d_minus_mm:.2f}mm':>12} | {mae_m:>13.4f} m | {f'{net_delta_mm:+.2f} mm':>12} | {f'{real_pct:+.1f}%':>10}")

        log_print("-" * 145)
        log_print(f"{'[Oracle 极限]':<12} | {'--':>10} | {'--':>10} | {f'{oracle_rescue_tri_pct:.1f}%':>8} | {'0.0%':>8} | {f'{oracle_cure_tri_pct:.1f}%':>10} | {f'+{delta_oracle_mm:.2f}mm':>12} | {'-0.00mm':>12} | {glob_oracle_m:>13.4f} m | {f'+{delta_oracle_mm:.2f} mm':>12} | {'+100.0%':>10}")

    # 打印 Mode A 表格
    print_mode_table("Mode A (原版多视代价 - 允许出界零填充)", "stats_A")
    # 打印 Mode B 表格
    print_mode_table("Mode B (加视场有效性 FOV-Masked - 剔除出界视角)", "stats_B")

    log_print(f"\n⏱️ 诊断耗时: {time.time() - t0:.1f}s (纯 GPU 一次性双模式极速评测)")
    log_print(f"📄 完整 A/B 对照报告已保存至: {os.path.abspath(args.output_txt)}")
    log_print("=" * 145)
    report_file.close()


def main():
    args = parse_args()
    main_cost_select(args)


if __name__ == '__main__':
    main()
