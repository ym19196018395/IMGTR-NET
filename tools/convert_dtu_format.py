#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
organize_scans.py

功能：
    根据一个包含 scan 名称的 txt 文件，对 DATA_ROOT 中对应的目录进行结构整理。
    例如 txt 中有：
        scan2
        scan6
        scan7
    该脚本将在 DATA_ROOT 中找到：
        scan2_train, scan6_train, scan7_train ...
    并在每个目录中执行：
        1. 创建 urd/ 文件夹，并将图片移动进去
        2. 创建 triangulation/ 文件夹（空）
        3. 创建 list 文件，记录所有图片文件名（一行一个）

用法：
    python organize_scans.py /path/to/DATA_ROOT /path/to/scans.txt
"""

import sys
import os
import shutil
from pathlib import Path

# 支持的图片后缀（不区分大小写）
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

def is_image_file(p: Path):
    """判断文件是否为图片文件"""
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS

def process_scan(data_root: Path, scan_name: str):
    """
    处理单个 scan 目录
    scan_name: 例如 "scan2"，对应目录为 scan2_train
    """
    scan_dir_name = f"{scan_name}_train"
    scan_dir = data_root / scan_dir_name

    # 判断 scan 目录是否存在
    if not scan_dir.exists() or not scan_dir.is_dir():
        print(f"[跳过] 未找到目录: {scan_dir_name}")
        return False

    # 创建所需目录
    urd_dir = scan_dir / "urd"
    tri_dir = scan_dir / "triangulation"
    list_file = scan_dir / "list.txt"   # 这里改成 list.txt !!!

    urd_dir.mkdir(exist_ok=True)
    tri_dir.mkdir(exist_ok=True)

    # 找到 scan_train 根目录下的所有图片（不递归）
    image_files = sorted([p for p in scan_dir.iterdir() if is_image_file(p)])

    moved_count = 0
    for img in image_files:
        target = urd_dir / img.name
        # 如果目标中已经存在该图片，则跳过（避免重复移动）
        if target.exists():
            continue
        try:
            shutil.move(str(img), str(target))
            moved_count += 1
        except Exception as e:
            print(f"[错误] 移动文件失败 {img} -> {target}: {e}")

    # urd 中所有图片（包括可能之前已存在的）
    urd_images = sorted([p for p in urd_dir.iterdir() if is_image_file(p)])

    # 写 list 文件
    try:
        with list_file.open('w', encoding='utf-8') as f:
            for p in urd_images:
                f.write(p.name + "\n")
    except Exception as e:
        print(f"[错误] 无法写入 list 文件: {list_file}, {e}")
        return False

    print(f"[完成] {scan_dir_name}: 移动 {moved_count} 个文件到 urd/，生成 list 和 triangulation/。")
    return True

def read_scan_list(scan_list_path: Path):
    """
        读取 txt 文件，支持：
        scan2
        scan6 scan7 scan8
    :param scan_list_path:
    :return:
    """
    scans = []
    with scan_list_path.open('r', encoding='utf-8') as f:
        for line in f:
            names = line.strip().split()
            scans.extend(names)
    return scans

def main():
    if len(sys.argv) != 3:
        print("使用方式：python convert_dtu_format.py DATA_ROOT_PATH SCAN_LIST_TXT")
        print("例如：python convert_dtu_format.py /home/ym/DATA_ROOT ./scans.txt")
        sys.exit(1)

    data_root = Path(sys.argv[1]).expanduser().resolve()
    scan_list_path = Path(sys.argv[2]).expanduser().resolve()

    if not data_root.exists():
        print(f"[错误] DATA_ROOT 不存在: {data_root}")
        sys.exit(1)
    if not scan_list_path.exists():
        print(f"[错误] scans.txt 不存在: {scan_list_path}")
        sys.exit(1)

    scans = read_scan_list(scan_list_path)
    print(f"[信息] 共读取到 {len(scans)} 个 scan 条目: {scans}")

    success = 0
    for scan in scans:
        if process_scan(data_root, scan):
            success += 1

    print(f"[完成] 成功处理 {success}/{len(scans)} 个 scan。")

if __name__ == "__main__":
    main()
