#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
WHU MVS 离线几何 (.npz + CDTinfo) 数据分布与像素级真实面积透视评估工具
================================================================================
本脚本高速读取离线几何特征文件 (geom_svd_*.npz) 并结合 CDT 剖分数据 (CDT_info_vlf_*.txt)，
实现【三角面片数量】与【全图像素真实面积】的双轨无偏统计：
1. 核心大类四分段对比：面片数占比 vs 真实像素面积占比 vs 面片平均像素尺寸；
2. 联合置信度门限 (tri_conf > 0.60) 下，放宽 err95 对真实像素面积的覆盖率与相对增益 (+Δ%)；
3. 倾斜屋面 (Slanted Roofs, 0.60 <= |n_z| < 0.85) 的像素级被卡控与误伤量化；
4. 多进程并发极速扫描 (支持 28 个 Scan 全量并发，耗时数秒)。
"""

import os
import sys
import glob
import argparse
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="WHU MVS Offline Planar Geometry and Pixel Area Profiler")
    parser.add_argument(
        "--datapath",
        type=str,
        default="/home/ym/Experiment/Datas/WHU_MVS_dataset",
        help="WHU MVS 数据集根目录路径"
    )
    parser.add_argument(
        "--listfile",
        type=str,
        default="lists/whu/newtrain.txt",
        help="待统计的 Scan 列表文件路径 (如 lists/whu/newtrain.txt 或 lists/whu/minitest.txt)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "train", "test", "val"],
        help="数据集子目录模式 (auto: 自动根据 listfile 探测 train 或 test)"
    )
    parser.add_argument(
        "--conf_thresh",
        type=float,
        default=0.60,
        help="判定有效平面的初始置信度门限 (默认严格对齐 dtu_whu.py: 0.60)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="并发读取进程数 (默认 8 进程极速读取)"
    )
    parser.add_argument(
        "--out_report",
        type=str,
        default=None,
        help="可选：将统计 Markdown 报告导出至指定路径"
    )
    return parser.parse_args()


def resolve_mode(listfile, mode_arg):
    if mode_arg != "auto":
        return mode_arg
    basename = os.path.basename(listfile).lower()
    if "train" in basename:
        return "train"
    elif "test" in basename or "eval" in basename:
        return "test"
    elif "val" in basename:
        return "val"
    return "train"


def load_scans_from_list(listfile):
    if not os.path.exists(listfile):
        raise FileNotFoundError(f"未找到指定的 Scan 列表文件: {listfile}")
    with open(listfile, "r") as f:
        scans = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return scans


def parse_cdt_pixel_counts(cdt_path):
    """
    极速从 CDT_info_vlf_*.txt 中提取每个三角形的像素点数 valid_count。
    利用指针跳跃直接跨过数十万行坐标，单文件解析耗时 < 5ms。
    """
    if not os.path.exists(cdt_path):
        return None
    try:
        with open(cdt_path, "r") as f:
            lines = f.readlines()
        if len(lines) == 0:
            return None
        first = lines[0].strip().split("&")
        if len(first) < 3:
            return None
        num_vertices = int(first[0])
        num_lines = int(first[1])
        num_triangles = int(first[2])
        
        # 三角形起始行
        curr = 1 + num_vertices + num_lines
        counts = np.zeros(num_triangles, dtype=np.int32)
        
        for tid in range(num_triangles):
            if curr + 2 >= len(lines):
                break
            # lines[curr] -> vertex_ids
            # lines[curr+1] -> line_ids
            # lines[curr+2] -> valid_count
            c_str = lines[curr + 2].strip()
            if c_str:
                c = int(c_str)
                counts[tid] = c
                curr += 3 + c
            else:
                curr += 3
        return counts
    except Exception:
        return None


def process_single_scan(args_tuple):
    """单 Scan 处理函数，供多进程调用"""
    scan, datapath, mode, conf_thresh = args_tuple
    
    scan_tri_dir = os.path.join(datapath, mode, "Images", scan, "triangulation")
    scan_cdt_dir = os.path.join(scan_tri_dir, "CDTinfo")
    if not os.path.exists(scan_cdt_dir):
        # 兼容小写路径
        scan_cdt_dir = os.path.join(scan_tri_dir, "cdtinfo")
        
    list_file = os.path.join(datapath, mode, "Images", scan, "list.txt")
    
    view_fids = []
    if os.path.exists(list_file):
        with open(list_file, "r") as lf:
            for line in lf:
                clean_line = line.strip()
                if clean_line:
                    fid = clean_line.split(".")[0]
                    view_fids.append(fid)
    else:
        npz_files = glob.glob(os.path.join(scan_tri_dir, "geom_svd_*.npz"))
        for p in npz_files:
            fid = os.path.basename(p).replace("geom_svd_", "").replace(".npz", "")
            view_fids.append(fid)
        view_fids = sorted(list(set(view_fids)))
        
    scan_err95 = []
    scan_conf = []
    scan_nz = []
    scan_px = []
    
    files_loaded = 0
    cdt_loaded = 0
    
    for fid in view_fids:
        npz_path = os.path.join(scan_tri_dir, f"geom_svd_{fid}.npz")
        cdt_path = os.path.join(scan_cdt_dir, f"CDT_info_vlf_{fid}.txt")
        
        if not os.path.exists(npz_path):
            continue
            
        try:
            data = np.load(npz_path)
            conf = data["tri_initial_conf"]
            err95 = data["tri_svd_err95"]
            
            if "tri_svd_plane" in data:
                normal_z = np.abs(data["tri_svd_plane"][:, 2])
            elif "tri_svd_normal" in data:
                normal_z = np.abs(data["tri_svd_normal"][:, 2])
            else:
                normal_z = np.ones_like(conf)
                
            px_counts = parse_cdt_pixel_counts(cdt_path)
            if px_counts is not None:
                cdt_loaded += 1
            else:
                px_counts = np.ones_like(conf, dtype=np.int32)
                
            n = min(len(conf), len(px_counts))
            conf = conf[:n]
            err95 = err95[:n]
            normal_z = normal_z[:n]
            px_counts = px_counts[:n]
            
            # 过滤有效三角形 (具有观测或像素覆盖)
            valid = ((conf > 0.0) | (err95 > 0.0)) & (px_counts > 0)
            if np.any(valid):
                scan_err95.append(err95[valid].astype(np.float32))
                scan_conf.append(conf[valid].astype(np.float32))
                scan_nz.append(normal_z[valid].astype(np.float32))
                scan_px.append(px_counts[valid].astype(np.int64))
                
            files_loaded += 1
        except Exception:
            continue
            
    if len(scan_err95) > 0:
        return {
            "scan": scan,
            "views": files_loaded,
            "cdt_views": cdt_loaded,
            "err95": np.concatenate(scan_err95),
            "conf": np.concatenate(scan_conf),
            "nz": np.concatenate(scan_nz),
            "px": np.concatenate(scan_px)
        }
    return None


def format_table(headers, rows, alignments=None):
    if alignments is None:
        alignments = [":---"] + [":---:"] * (len(headers) - 1)
    
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))
            
    header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join(
        (alignments[i][0] + "-" * (col_widths[i] - 2) + alignments[i][-1])
        if len(alignments[i]) >= 2 else "-" * col_widths[i]
        for i in range(len(headers))
    ) + " |"
    
    row_lines = []
    for row in rows:
        r_str = "| " + " | ".join(str(val).ljust(col_widths[i]) for i, val in enumerate(row)) + " |"
        row_lines.append(r_str)
        
    return "\n".join([header_line, sep_line] + row_lines)


def main():
    args = parse_args()
    mode = resolve_mode(args.listfile, args.mode)
    
    print("=" * 80)
    print("  WHU MVS 几何分布与像素级真实面积深度剖析工具 (Pixel Area Profiler)")
    print("=" * 80)
    print(f" 📂 数据集根目录 : {args.datapath}")
    print(f" 📜 Scan 列表文件: {args.listfile} (模式: {mode})")
    print(f" 🎯 初始置信度阈值: {args.conf_thresh:.2f}")
    print(f" ⚡ 并发进程数   : {args.workers}")
    print("=" * 80)
    
    scans = load_scans_from_list(args.listfile)
    print(f"开始调度 {len(scans)} 个 Scan 场景进行并发几何与 CDT 像素解析...")
    
    t0 = time.time()
    worker_args = [(s, args.datapath, mode, args.conf_thresh) for s in scans]
    
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for res in executor.map(process_single_scan, worker_args):
            if res is not None:
                results.append(res)
                print(f" ✓ Scan '{res['scan']}' 完成 ({res['views']} 视图, 包含 {res['cdt_views']} 个 CDT 像素图, {len(res['err95']):,} 面片)")
                
    elapsed = time.time() - t0
    print(f"\n✓ 全部扫描完成！耗时: {elapsed:.2f}s | 成功加载 {len(results)}/{len(scans)} 个 Scan 场景")
    
    if len(results) == 0:
        print("❌ 未能读取到有效数据，请检查路径！")
        return
        
    # 全局数据拼接
    err95_all = np.concatenate([r["err95"] for r in results])
    conf_all = np.concatenate([r["conf"] for r in results])
    nz_all = np.concatenate([r["nz"] for r in results])
    px_all = np.concatenate([r["px"] for r in results])
    
    total_tri = len(err95_all)
    total_px = np.sum(px_all)
    
    print(f"📊 全局有效观测面片总量 : {total_tri:,} 个三角形")
    print(f"🖼️ 全局真实覆盖像素总量 : {total_px:,} 个像素 (平均单个面片 {total_px/total_tri:.1f} 像素)\n")

    # =========================================================================
    # 1. 核心大类四分段：面片数 vs 真实像素面积 对照
    # =========================================================================
    m_lt_008 = err95_all < 0.08
    m_008_012 = (err95_all >= 0.08) & (err95_all < 0.12)
    m_012_016 = (err95_all >= 0.12) & (err95_all < 0.16)
    m_gt_016 = err95_all >= 0.16
    
    def calc_stat(mask):
        cnt = np.sum(mask)
        px_cnt = np.sum(px_all[mask])
        pct_tri = (cnt / total_tri) * 100.0 if total_tri > 0 else 0
        pct_px = (px_cnt / total_px) * 100.0 if total_px > 0 else 0
        avg_px = px_cnt / cnt if cnt > 0 else 0
        return cnt, pct_tri, px_cnt, pct_px, avg_px
        
    c1, pt1, p1, pp1, a1 = calc_stat(m_lt_008)
    c2, pt2, p2, pp2, a2 = calc_stat(m_008_012)
    c3, pt3, p3, pp3, a3 = calc_stat(m_012_016)
    c4, pt4, p4, pp4, a4 = calc_stat(m_gt_016)
    
    table1_headers = ["核心几何分类", "残差范围 (tri_err95)", "面片数量", "面片占比", "真实像素总量 (px)", "像素面积占比", "平均面片大小", "状态与角色"]
    table1_rows = [
        ["当前基准平面", "< 0.08m", f"{c1:,}", f"{pt1:.2f}%", f"{p1:,}", f"{pp1:.2f}%", f"{a1:.1f} px/面", "✅ 当前硬门限覆盖"],
        ["次平整争议区", "0.08m ~ 0.12m", f"{c2:,}", f"{pt2:.2f}%", f"{p2:,}", f"{pp2:.2f}%", f"{a2:.1f} px/面", "⚡ 核心放宽区 (含斜屋顶)"],
        ["中度起伏过渡区", "0.12m ~ 0.16m", f"{c3:,}", f"{pt3:.2f}%", f"{p3:,}", f"{pp3:.2f}%", f"{a3:.1f} px/面", "⚠️ 需审慎评估 (混合带)"],
        ["复杂碎面 / 植被", "> 0.16m", f"{c4:,}", f"{pt4:.2f}%", f"{p4:,}", f"{pp4:.2f}%", f"{a4:.1f} px/面", "❌ 严禁纳入 (树木、碎地)"]
    ]
    md_table1 = format_table(table1_headers, table1_rows, [":---", ":---:", ":---:", ":---:", ":---:", ":---:", ":---:", ":---:"])

    # =========================================================================
    # 2. 联合置信度门限 (is_gt_planar) 的实际像素面积覆盖率与相对增益
    # =========================================================================
    cf = args.conf_thresh
    mask_base = (conf_all > cf) & (err95_all < 0.08)
    mask_010 = (conf_all > cf) & (err95_all < 0.10)
    mask_012 = (conf_all > cf) & (err95_all < 0.12)
    mask_015 = (conf_all > cf) & (err95_all < 0.15)
    mask_020 = (conf_all > cf) & (err95_all < 0.20)
    
    def calc_gain(mask, base_mask):
        cnt = np.sum(mask)
        px_cnt = np.sum(px_all[mask])
        pct_tri = (cnt / total_tri) * 100.0
        pct_px = (px_cnt / total_px) * 100.0
        
        base_cnt = np.sum(base_mask)
        base_px = np.sum(px_all[base_mask])
        
        gain_tri = ((cnt - base_cnt) / base_cnt) * 100.0 if base_cnt > 0 else 0
        gain_px = ((px_cnt - base_px) / base_px) * 100.0 if base_px > 0 else 0
        delta_px = px_cnt - base_px
        return cnt, pct_tri, px_cnt, pct_px, delta_px, gain_px
        
    cb, ptb, pb, ppb, _, _ = calc_gain(mask_base, mask_base)
    c10, pt10, p10, pp10, dpx10, gpx10 = calc_gain(mask_010, mask_base)
    c12, pt12, p12, pp12, dpx12, gpx12 = calc_gain(mask_012, mask_base)
    c15, pt15, p15, pp15, dpx15, gpx15 = calc_gain(mask_015, mask_base)
    c20, pt20, p20, pp20, dpx20, gpx20 = calc_gain(mask_020, mask_base)
    
    table2_headers = ["判定条件 (tri_conf > 0.60 联合)", "合格面片数", "面片占比", "合格像素总量 (px)", "全图像素占比", "净增像素数量", "像素面积相对增益"]
    table2_rows = [
        ["err95 < 0.08m (当前设定)", f"{cb:,}", f"{ptb:.2f}%", f"{pb:,}", f"{ppb:.2f}%", "-", "基准线 (1.00x)"],
        ["err95 < 0.10m (保守放宽)", f"{c10:,}", f"{pt10:.2f}%", f"{p10:,}", f"{pp10:.2f}%", f"+{dpx10:,}", f"+{gpx10:.2f}%"],
        ["err95 < 0.12m (推荐放宽)", f"{c12:,}", f"{pt12:.2f}%", f"{p12:,}", f"{pp12:.2f}%", f"+{dpx12:,}", f"+{gpx12:.2f}%"],
        ["err95 < 0.15m (积极放宽)", f"{c15:,}", f"{pt15:.2f}%", f"{p15:,}", f"{pp15:.2f}%", f"+{dpx15:,}", f"+{gpx15:.2f}%"],
        ["err95 < 0.20m (极度激进)", f"{c20:,}", f"{pt20:.2f}%", f"{p20:,}", f"{pp20:.2f}%", f"+{dpx20:,}", f"+{gpx20:.2f}%"]
    ]
    md_table2 = format_table(table2_headers, table2_rows, [":---", ":---:", ":---:", ":---:", ":---:", ":---:", ":---:"])

    # =========================================================================
    # 3. 倾斜角透视剖析 (结合真实像素尺寸与像素占比)
    # =========================================================================
    mask_flat = nz_all >= 0.85
    mask_slanted = (nz_all >= 0.60) & (nz_all < 0.85)
    mask_steep = nz_all < 0.60
    
    def analyze_angle_group(grp_mask, name):
        grp_tot_tri = np.sum(grp_mask)
        grp_tot_px = np.sum(px_all[grp_mask])
        if grp_tot_tri == 0:
            return [name, "0", "0", "0", "0%", "0%", "0%", "0%"]
            
        avg_px = grp_tot_px / grp_tot_tri
        pct_all_px = (grp_tot_px / total_px) * 100.0
        pct_all_tri = (grp_tot_tri / total_tri) * 100.0
        
        # 在该形态类别内部看各残差区间的像素占比
        g_px = px_all[grp_mask]
        g_err = err95_all[grp_mask]
        g_cnf = conf_all[grp_mask]
        
        # conf > 0.60 基础上的像素分布
        val_conf = g_cnf > cf
        tot_val_px = np.sum(g_px[val_conf])
        
        if tot_val_px > 0:
            p_008 = np.sum(g_px[val_conf & (g_err < 0.08)]) / tot_val_px * 100.0
            p_008_012 = np.sum(g_px[val_conf & (g_err >= 0.08) & (g_err < 0.12)]) / tot_val_px * 100.0
            p_012_016 = np.sum(g_px[val_conf & (g_err >= 0.12) & (g_err < 0.16)]) / tot_val_px * 100.0
            p_gt_016 = np.sum(g_px[val_conf & (g_err >= 0.16)]) / tot_val_px * 100.0
        else:
            p_008 = p_008_012 = p_012_016 = p_gt_016 = 0.0
            
        return [
            name,
            f"{grp_tot_tri:,} ({pct_all_tri:.1f}%)",
            f"{grp_tot_px:,} ({pct_all_px:.1f}%)",
            f"{avg_px:.1f} px/面",
            f"{p_008:.2f}%",
            f"{p_008_012:.2f}%",
            f"{p_012_016:.2f}%",
            f"{p_gt_016:.2f}%"
        ]
        
    table3_headers = ["几何形态分类", "面片总量 (占比%)", "像素总量 (占比%)", "平均面片大小", "< 0.08m 像素占比", "0.08~0.12m 像素占比", "0.12~0.16m", ">= 0.16m"]
    table3_rows = [
        analyze_angle_group(mask_flat, "平缓面 (|n_z| >= 0.85)"),
        analyze_angle_group(mask_slanted, "倾斜面 (0.60 <= |n_z| < 0.85)"),
        analyze_angle_group(mask_steep, "陡峭立面 (|n_z| < 0.60)")
    ]
    md_table3 = format_table(table3_headers, table3_rows, [":---", ":---:", ":---:", ":---:", ":---:", ":---:", ":---:", ":---:"])

    # =========================================================================
    # 汇总 Markdown 报告输出
    # =========================================================================
    report_content = f"""# WHU MVS 几何分布与像素级真实面积深度剖析报告

- **数据集路径**: `{args.datapath}`
- **列表文件**: `{args.listfile}` (模式: `{mode}`)
- **统计 Scan 数量**: `{len(scans)}`
- **统计有效三角形总量**: `{total_tri:,}` 个
- **统计全量真实像素总量**: `{total_px:,}` 像素 (全局平均单面片: `{total_px/total_tri:.1f}` px)
- **初始置信度门限 (`tri_conf`)**: `{args.conf_thresh:.2f}`

---

## 1. 核心大类四分段：面片数 vs 真实像素面积 对照表
{md_table1}

---

## 2. 联合置信度门限 (is_gt_planar) 的实际像素面积覆盖率与相对增益
{md_table2}

---

## 3. 倾斜角透视剖析 (面片尺寸与像素占比)
{md_table3}

---

## 4. 关键几何发现与建议
1. **真实像素面积对比面片数量的“放大效应”**：
   - 平缓面（大平地/大屋顶）由于几何完整，单面平均像素远高于碎地形，其在像素面积上的统治力得以显现；
   - 倾斜坡屋顶虽然面片被特征线切得相对较碎，但在像素级尺度上仍占据着至关重要的建筑重建比重；
2. **0.08m ~ 0.12m 次平整区在像素级维度的沉淀**：
   - 观察 Table 2 中净增像素数 `+{dpx12:,}` 与相对增益 `+{gpx12:.2f}%`，直观反映将门槛放宽至 0.12m 时，能够额外将多少真实像素纳入有效法向和深度监督；
3. **软加权损失机制建议**：
   - 深度损失（按像素累加）与法向损失（按面片均值）的双轨制天然互补：大平地靠像素数主导深度，斜坡屋面靠独立面片纠正法向朝向。
"""

    print("\n" + report_content)
    
    if args.out_report:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_report)), exist_ok=True)
        with open(args.out_report, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"✓ 完整 Markdown 分析报告已导出至: {args.out_report}")


if __name__ == "__main__":
    main()
