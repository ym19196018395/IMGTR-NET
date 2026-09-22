# -*- coding: utf-8 -*-
"""
专利附图 6: 建筑物弱纹理屋面与高差边界处的最终深度重建效果图
生成全套高质量排版图:
方案 1: 三图联动标准版 (a) 航测参考图像  (b) 平面决策掩码  (c) 最终精细深度图 (含彩色伪彩版与黑白灰度版)
方案 2: 带剖切线 A-A' 的深度剖面高程切片工程曲线图 (直观证明平面如镜、边缘垂直断崖)
方案 3: 四图多维联动版 (原图 + 掩码 + 灰度深度 + 伪彩深度)
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image, ImageDraw, ImageFont

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def get_pil_font(size=24):
    for fp in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/simsun.ttc"]:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                pass
    return ImageFont.load_default()

def load_depth_normalized(npy_path):
    dep = np.load(npy_path)
    vmask = dep > 0
    d_min = np.percentile(dep[vmask], 1)
    d_max = np.percentile(dep[vmask], 99)
    d_norm = np.clip((dep - d_min) / (d_max - d_min + 1e-8), 0.0, 1.0)
    d_uint8 = (d_norm * 255).astype(np.uint8)
    
    # JET 彩色
    jet_bgr = cv2.applyColorMap(d_uint8, cv2.COLORMAP_JET)
    jet_rgb = cv2.cvtColor(jet_bgr, cv2.COLOR_BGR2RGB)
    jet_rgb[~vmask] = 0
    
    # 灰度
    gray_img = d_uint8.copy()
    gray_img[~vmask] = 0
    return dep, vmask, Image.fromarray(jet_rgb), Image.fromarray(gray_img)

def generate_fig6_three_plots(sample_id, base_dir, out_dir, use_jet=True, target_h=384, spacing=16, pad=30):
    """
    方案 1: 三图横向联动排版
    (a) 原始输入航测图  (b) 几何平面先验决策掩码  (c) 本发明估计最终精细深度图
    """
    os.makedirs(out_dir, exist_ok=True)
    p_img = os.path.join(base_dir, f"{sample_id}.png")
    p_mask = os.path.join(base_dir, f"{sample_id}掩码.png")
    p_npy = os.path.join(base_dir, "深度", f"{sample_id}.npy")

    dep, vmask, im_jet, im_gray = load_depth_normalized(p_npy)
    im_ref = Image.open(p_img).convert("RGB")
    im_mask = Image.open(p_mask).convert("RGB")
    im_dep = im_jet if use_jet else im_gray.convert("RGB")

    items = [
        {"title": "(a) 原始输入航测图", "im": im_ref},
        {"title": "(b) 几何平面先验决策掩码", "im": im_mask},
        {"title": "(c) 本发明估计最终精细深度图", "im": im_dep}
    ]

    # 等比例缩放到 target_h
    resized_list = []
    target_w = None
    for it in items:
        im = it["im"]
        cur_w, cur_h = im.size
        new_w = int(round(cur_w * (target_h / cur_h)))
        if target_w is None:
            target_w = new_w
        else:
            new_w = target_w
        im_res = im.resize((new_w, target_h), Image.Resampling.LANCZOS if target_h > cur_h else Image.Resampling.BILINEAR)
        resized_list.append((it["title"], im_res))

    n_plots = len(resized_list)
    total_w = n_plots * target_w + (n_plots - 1) * spacing + 2 * pad
    title_h = 40
    total_h = target_h + pad * 2 + title_h

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = get_pil_font(size=22)

    cur_x = pad
    for title, im in resized_list:
        canvas.paste(im, (cur_x, pad))
        draw.rectangle([cur_x - 1, pad - 1, cur_x + target_w, pad + target_h], outline=(200, 200, 200), width=1)
        
        bbox = font.getbbox(title)
        txt_w = bbox[2] - bbox[0]
        txt_x = cur_x + (target_w - txt_w) // 2
        txt_y = pad + target_h + 12
        draw.text((txt_x, txt_y), title, fill=(30, 30, 30), font=font)
        cur_x += target_w + spacing

    mode_name = "彩色版" if use_jet else "灰度版"
    out_path = os.path.join(out_dir, f"图6_深度重建三图对比_{sample_id}_{mode_name}.png")
    canvas.save(out_path, dpi=(300, 300), quality=95)
    print(f"[SUCCESS] Saved Fig6 Three-Plots: {out_path} ({total_w} x {total_h})")
    return out_path

def generate_fig6_profile_curve(sample_id, base_dir, out_dir, cut_row=66):
    """
    方案 2: 带剖切线 A-A' 与一维深度高程剖切曲线的工程诊断图 (极具学术与专利审查说服力)
    (a) 航测图与剖切线位置  (b) 深度图与剖切线位置  (c) 沿 A-A' 剖切线的公制深度曲线 (展示平整度与垂直高差落差)
    """
    os.makedirs(out_dir, exist_ok=True)
    p_img = os.path.join(base_dir, f"{sample_id}.png")
    p_mask = os.path.join(base_dir, f"{sample_id}掩码.png")
    p_npy = os.path.join(base_dir, "深度", f"{sample_id}.npy")

    dep, vmask, im_jet, im_gray = load_depth_normalized(p_npy)
    im_ref = Image.open(p_img).convert("RGB")
    im_mask = Image.open(p_mask).convert("L")

    H, W = dep.shape
    # 提取第 cut_row 行的切线数据
    cut_depth = dep[cut_row, :]
    cut_mask = (np.array(im_mask)[cut_row, :] > 128)
    x_coords = np.arange(W)

    # 构图: 采用 Matplotlib 绘制带剖切线和曲线的高清专利排版大图
    fig = plt.figure(figsize=(18, 6), dpi=200, facecolor='white')
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.18)

    # (a) 航测原图 + 剖切线
    ax0 = fig.add_subplot(gs[0])
    im_ref_res = im_ref.resize((W, H))
    ax0.imshow(im_ref_res)
    # 画剖切线 A-A' (亮黄粗虚线)
    ax0.axhline(y=cut_row, color='#FFD700', linestyle='--', linewidth=2.2, label="剖切线 A-A'")
    ax0.text(8, cut_row - 6, "A", color='#FFD700', fontsize=14, fontweight='bold',
             bbox=dict(boxstyle="circle,pad=0.2", fc="black", ec="#FFD700", lw=1.5))
    ax0.text(W - 22, cut_row - 6, "A'", color='#FFD700', fontsize=14, fontweight='bold',
             bbox=dict(boxstyle="circle,pad=0.2", fc="black", ec="#FFD700", lw=1.5))
    ax0.set_title("(a) 航测参考图像及剖切位置 A-A'", fontsize=13, pad=10, fontweight='bold')
    ax0.axis('off')

    # (b) 最终深度图 + 剖切线
    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(im_jet)
    ax1.axhline(y=cut_row, color='#FFFFFF', linestyle='--', linewidth=2.2)
    ax1.text(8, cut_row - 6, "A", color='#FFFFFF', fontsize=14, fontweight='bold',
             bbox=dict(boxstyle="circle,pad=0.2", fc="black", ec="white", lw=1.5))
    ax1.text(W - 22, cut_row - 6, "A'", color='#FFFFFF', fontsize=14, fontweight='bold',
             bbox=dict(boxstyle="circle,pad=0.2", fc="black", ec="white", lw=1.5))
    ax1.set_title("(b) 本发明估计连续密集深度图", fontsize=13, pad=10, fontweight='bold')
    ax1.axis('off')

    # (c) 一维剖切深度高程曲线图 (工程核心实证)
    ax2 = fig.add_subplot(gs[2])
    # 绘制深度曲线
    ax2.plot(x_coords, cut_depth, color='#1E88E5', linewidth=2.5, label="本发明深度剖面 (Depth Profile)")
    
    # 标注平面掩码约束覆盖区间
    planar_indices = np.where(cut_mask)[0]
    if len(planar_indices) > 0:
        p_start, p_end = planar_indices[0], planar_indices[-1]
        ax2.axvspan(p_start, p_end, color='#4CAF50', alpha=0.18, label="硬平面先验约束覆盖区")
        # 标注平面拟合标准差
        planar_std = np.std(cut_depth[cut_mask])
        ax2.text((p_start + p_end) / 2, cut_depth[cut_mask].mean() - 1.5,
                 f"弱纹理屋面如镜平整\n拟合标准差: {planar_std:.3f}m",
                 color='#2E7D32', fontsize=11, fontweight='bold', ha='center',
                 bbox=dict(boxstyle="round,pad=0.4", fc="#E8F5E9", ec="#4CAF50", lw=1.2))

    # 标注断崖式边缘跌落
    drop_val = cut_depth.max() - cut_depth.min()
    min_x = cut_depth.argmin()
    ax2.annotate(f"檐口垂直高差落差: {drop_val:.2f}m\n(锐利无过度平滑倒角)",
                 xy=(min_x, cut_depth[min_x]),
                 xytext=(min_x + 30, cut_depth[min_x] + 7.5),
                 arrowprops=dict(facecolor='#D32F2F', shrink=0.08, width=1.8, headwidth=8),
                 color='#B71C1C', fontsize=11, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.4", fc="#FFEBEE", ec="#D32F2F", lw=1.2))

    ax2.set_xlabel("横向剖切线像素坐标 (X / pixels)", fontsize=12, labelpad=8)
    ax2.set_ylabel("绝对公制深度 (Depth / 米)", fontsize=12, labelpad=8)
    ax2.set_title("(c) 沿剖切线 A-A' 截面一维高程变化曲线", fontsize=13, pad=10, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower right', fontsize=10.5, framealpha=0.9)
    # 反转 y 轴使得高程视觉符合直觉 (深度小=离相机近=高，深度大=离相机远=低)
    ax2.invert_yaxis()

    out_path = os.path.join(out_dir, f"图6_深度剖切截面曲线对比_{sample_id}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.15)
    plt.close()
    print(f"[SUCCESS] Saved Fig6 Profile Curve: {out_path}")
    return out_path

def generate_fig6_zoom_annotated(sample_id, base_dir, out_dir, use_jet=True):
    """
    方案 4: 专利局高规格申报图 (带局部红色特写放大框 + 标号引出线 601, 602, 603, 604)
    完全契合《法向量对比图10101》的高标准排版规范
    """
    os.makedirs(out_dir, exist_ok=True)
    p_img = os.path.join(base_dir, f"{sample_id}.png")
    p_mask = os.path.join(base_dir, f"{sample_id}掩码.png")
    p_npy = os.path.join(base_dir, "深度", f"{sample_id}.npy")

    dep, vmask, im_jet, im_gray = load_depth_normalized(p_npy)
    im_ref = Image.open(p_img).convert("RGB")
    im_mask = Image.open(p_mask).convert("RGB")
    im_dep = im_jet if use_jet else im_gray.convert("RGB")

    # 统一尺寸
    target_w, target_h = 768, 384
    img_a = np.array(im_ref.resize((target_w, target_h), Image.Resampling.LANCZOS))
    img_b = np.array(im_mask.resize((target_w, target_h), Image.Resampling.NEAREST))
    img_c = np.array(im_dep.resize((target_w, target_h), Image.Resampling.LANCZOS))

    # 定义 ROI 特写区域: 横跨平坦屋面与屋檐高差断崖
    # [ymin, xmin, ymax, xmax] 相对比例
    roi_rel = [0.22, 0.28, 0.70, 0.62]
    # 局部放大框放置位置 [x, y, w, h] (放置在左下角，不遮挡主体)
    inset_loc = [0.03, 0.05, 0.32, 0.44]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), dpi=220, facecolor='white')
    sub_titles = [
        "(a) 原始输入航测图",
        "(b) 几何平面先验决策掩码",
        "(c) 本发明估计最终精细深度图"
    ]
    imgs = [img_a, img_b, img_c]

    H_p, W_p, _ = img_a.shape
    y1, x1 = int(roi_rel[0] * H_p), int(roi_rel[1] * W_p)
    y2, x2 = int(roi_rel[2] * H_p), int(roi_rel[3] * W_p)

    for idx, ax in enumerate(axes):
        curr_img = imgs[idx]
        ax.imshow(curr_img)
        ax.set_title(sub_titles[idx], fontsize=13.5, pad=12, fontweight='bold')
        ax.axis('off')

        # 1. 绘制 ROI 红色主选框
        rect_main = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2.0, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect_main)

        # 2. 局部特写放大框 (嵌入左下角)
        ix = int(inset_loc[0] * W_p)
        iy = int(H_p - (inset_loc[1] + inset_loc[3]) * H_p)
        iw = int(inset_loc[2] * W_p)
        ih = int(inset_loc[3] * H_p)

        patch_crop = curr_img[y1:y2, x1:x2]
        crop_resized = cv2.resize(patch_crop, (iw, ih), interpolation=cv2.INTER_LANCZOS4 if idx!=1 else cv2.INTER_NEAREST)

        # 覆盖贴图
        curr_img_with_inset = curr_img.copy()
        curr_img_with_inset[iy:iy+ih, ix:ix+iw] = crop_resized
        ax.imshow(curr_img_with_inset)

        rect_inset = patches.Rectangle(
            (ix, iy), iw, ih,
            linewidth=2.2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect_inset)

        # 虚线连接框角
        ax.plot([ix + iw, x1], [iy, y1], color='red', linestyle='--', linewidth=1.2, alpha=0.85)
        ax.plot([ix + iw, x1], [iy + ih, y2], color='red', linestyle='--', linewidth=1.2, alpha=0.85)

    # 3. 绘制专利标号与引线 (601, 602, 603, 604)
    # (a) 图标号 601: 弱纹理大平坦屋面
    axes[0].annotate("601", xy=(W_p * 0.48, H_p * 0.42), xytext=(W_p * 0.48, H_p * 0.16),
                     arrowprops=dict(facecolor='black', edgecolor='black', width=1.2, headwidth=6, shrink=0.05),
                     fontsize=12, fontweight='bold', ha='center', va='center',
                     bbox=dict(boxstyle="circle,pad=0.25", fc="white", ec="black", lw=1.2))

    # (b) 图标号 602: 几何平面决策掩码
    axes[1].annotate("602", xy=(W_p * 0.48, H_p * 0.42), xytext=(W_p * 0.48, H_p * 0.16),
                     arrowprops=dict(facecolor='black', edgecolor='black', width=1.2, headwidth=6, shrink=0.05),
                     fontsize=12, fontweight='bold', ha='center', va='center',
                     bbox=dict(boxstyle="circle,pad=0.25", fc="white", ec="black", lw=1.2))

    # (c) 图标号 603: 平整无噪连续深度面 & 604: 锐利垂直屋檐断崖
    axes[2].annotate("603", xy=(W_p * 0.48, H_p * 0.42), xytext=(W_p * 0.48, H_p * 0.16),
                     arrowprops=dict(facecolor='black', edgecolor='black', width=1.2, headwidth=6, shrink=0.05),
                     fontsize=12, fontweight='bold', ha='center', va='center',
                     bbox=dict(boxstyle="circle,pad=0.25", fc="white", ec="black", lw=1.2))

    axes[2].annotate("604", xy=(W_p * 0.35, H_p * 0.52), xytext=(W_p * 0.35, H_p * 0.88),
                     arrowprops=dict(facecolor='black', edgecolor='black', width=1.2, headwidth=6, shrink=0.05),
                     fontsize=12, fontweight='bold', ha='center', va='center',
                     bbox=dict(boxstyle="circle,pad=0.25", fc="white", ec="black", lw=1.2))

    plt.subplots_adjust(wspace=0.08, left=0.02, right=0.98, top=0.92, bottom=0.05)
    mode_name = "彩色交底版" if use_jet else "灰度审查版"
    out_path = os.path.join(out_dir, f"图6_深度重建局部特写对比_{sample_id}_{mode_name}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"[SUCCESS] Saved Fig6 Zoom Annotated: {out_path}")
    return out_path

if __name__ == '__main__':
    base_dir = r"D:\360MoveData\Users\Administrator\Desktop\专利\置信度"
    for sid in ['000007', '002005']:
        # 1. 方案一: 三图联动 (彩色版 + 灰度版)
        generate_fig6_three_plots(sid, base_dir, base_dir, use_jet=True)
        generate_fig6_three_plots(sid, base_dir, base_dir, use_jet=False)
        # 2. 方案二: 带 A-A' 剖切线和公制深度落差曲线的工程图 (硬核技术实证)
        generate_fig6_profile_curve(sid, base_dir, base_dir, cut_row=66 if sid=='000007' else 80)
        # 3. 方案三: 专利高规格局部放大带标号申报图 (彩色交底版 + 灰度审查版)
        generate_fig6_zoom_annotated(sid, base_dir, base_dir, use_jet=True)
        generate_fig6_zoom_annotated(sid, base_dir, base_dir, use_jet=False)
