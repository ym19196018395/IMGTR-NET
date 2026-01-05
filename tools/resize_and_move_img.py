# -*- coding: utf-8 -*-
"""
blended_cleanup.py

功能：
    批量处理 DATA_ROOT 下由 BlendedMVG_list 指定的 scan 文件夹：
      - 将每个 scan_xxx/blended_images 中的图像（非_masked）缩小为原来宽高的 1/2（面积 1/4）
      - 将缩小后的图像保存到 scan_xxx/urd/（自动创建）
      - 在 scan_xxx/ 下写入 list.txt，包含缩小后图像的文件名（带后缀），一行一个

使用方式（示例）:
    from blended_cleanup import batch_resize_blended_images
    batch_resize_blended_images(data_root="E:/DATA_ROOT",
                                blended_list_file="E:/DATA_ROOT/BlendedMVG_list",
                                preview_limit=10,
                                verbose=True)

说明：
    - blended_list_file: 是 BlendedMVG_list 的路径（文件中每行是一个 scan 文件夹名）
    - 脚本会智能识别带有 "_masked" 的掩码文件并优先选择非 masked 图像作为原图
    - 若某个 base 仅存在 masked 文件，则仍会处理（并在日志中标记）
    - 默认只处理常见图片后缀： .jpg .jpeg .png .bmp .tif .tiff
"""
import math
import shutil
from pathlib import Path
from PIL import Image, ImageFilter
import os
from typing import Optional, List, Dict, Tuple

# 支持的图片扩展名（小写）
_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

def _is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in _IMAGE_EXTS

def resize_and_save_png_high_quality(
    input_path,
    out_path,
    scale: float = 0.5,
    resample=Image.LANCZOS,
    apply_unsharp: bool = True,
    unsharp_radius: float = 1.0,
    unsharp_percent: int = 150,
    unsharp_threshold: int = 3,
    png_compress_level: int = 6,
    optimize_png: bool = True,
    convert_mode: str = "RGBA"
) -> Path:
    """
    高质量下采样并**强制保存为 PNG**（即使 out_path 后缀不是 .png 也会改为 .png）。

    参数:
      input_path : 输入图片路径（str 或 Path）
      out_path   : 目标路径（str 或 Path），但会被强制改为 .png
      scale      : 缩放比例（0.5 表示宽高各 /2）
      resample   : PIL 重采样方法（默认 LANCZOS）
      apply_unsharp : 是否应用 UnsharpMask（默认 True）
      unsharp_radius, unsharp_percent, unsharp_threshold : UnsharpMask 参数
      png_compress_level : PNG 压缩等级 0-9（越大文件越小但越慢），默认 6
      optimize_png : 是否启用 optimize=True（通常能减小文件但更慢）
      convert_mode: 保存前转换的模式，默认 "RGBA"（保留透明通道，如需去掉 alpha 用 "RGB"）

    返回:
      保存后的 Path（后缀为 .png）
    """
    input_path = Path(input_path)
    out_path = Path(out_path)

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    # 强制输出为 .png（保留原名但换后缀）
    out_path_png = out_path.with_suffix('.png')
    out_path_png.parent.mkdir(parents=True, exist_ok=True)

    # 打开并处理图片
    with Image.open(input_path) as img:
        # 可选转换模式（RGBA 可保留 alpha）
        try:
            img = img.convert(convert_mode)
        except Exception:
            # 如果转换失败，则继续用打开的模式
            pass

        orig_w, orig_h = img.size
        new_w = max(1, int(math.floor(orig_w * scale)))
        new_h = max(1, int(math.floor(orig_h * scale)))

        # 高质量重采样
        resized = img.resize((new_w, new_h), resample=resample)

        # 可选锐化（UnsharpMask）
        if apply_unsharp:
            try:
                resized = resized.filter(ImageFilter.UnsharpMask(
                    radius=unsharp_radius,
                    percent=unsharp_percent,
                    threshold=unsharp_threshold
                ))
            except Exception:
                # 在极少数 PIL 版本中可能失败，忽略错误继续
                pass

        # 保存为 PNG，使用高压缩但保真设置
        save_kwargs = {
            'compress_level': int(max(0, min(9, png_compress_level))),
            'optimize': bool(optimize_png)
        }

        # 如果图像是 RGBA（有 alpha），直接保存；若不需要 alpha 可改为 RGB
        try:
            resized.save(out_path_png, format='PNG', **save_kwargs)
        except TypeError:
            # 某些 Pillow 版本对 optimize/compress_level 的支持不同：降级为只传 compress_level
            resized.save(out_path_png, format='PNG', compress_level=save_kwargs['compress_level'])
        except Exception as e:
            # 最后兜底：直接保存（不带参数）
            resized.save(out_path_png, format='PNG')

    return out_path_png

def _collect_base_map(blended_dir: Path) -> Dict[str, Dict[str, Path]]:
    """
    读取 blended_dir 下所有文件，按 base name（不含扩展，去掉尾部 '_masked'）分组。
    返回 dict:
      base -> {'orig': Path, 'mask': Path, 'candidates': [Path,...]}
    逻辑：
      - 如果文件名中包含 '_masked' (不区分大小写)，则视为 mask，并记录 base = stem.replace('_masked','')
      - 其它文件视为候选原图，记录在 candidates
      - 最终为每个 base 尝试确定 'orig'（优先 candidates 中的文件）
    """
    files = [p for p in blended_dir.iterdir() if p.is_file()]
    base_map = {}
    for p in files:
        if not _is_image_file(p):
            continue
        stem = p.stem  # 文件名无扩展
        lower = stem.lower()
        if lower.endswith('_masked'):
            base = stem[:-7]  # 去掉 '_masked'
            rec = base_map.setdefault(base, {'orig': None, 'mask': None, 'candidates': []})
            rec['mask'] = p
        else:
            base = stem
            rec = base_map.setdefault(base, {'orig': None, 'mask': None, 'candidates': []})
            rec['candidates'].append(p)
    # 选择 orig：优先 candidates（若有多个选择一个 jpg 或第一个）
    for base, rec in base_map.items():
        if rec['candidates']:
            # 尝试优先选择 jpg -> png -> others
            choose = None
            for ext_pref in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                for cand in rec['candidates']:
                    if cand.suffix.lower() == ext_pref:
                        choose = cand
                        break
                if choose:
                    break
            if not choose:
                choose = rec['candidates'][0]
            rec['orig'] = choose
        else:
            # 没有候选原图，仅有 mask（极端情况）
            if rec['mask'] is not None:
                rec['orig'] = rec['mask']  # 退而求其次：用 mask 文件处理
    return base_map

def batch_resize_blended_images(
    data_root: str,
    blended_list_file: str,
    output_subdir: str = 'urd',
    list_filename: str = 'list.txt',
    overwrite: bool = False,
    preview_limit: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    批量处理 DATA_ROOT 下由 blended_list_file 指定的 scan 文件夹。
    参数:
      - data_root: DATA_ROOT 路径（str）
      - blended_list_file: BlendedMVG_list 文件路径（每行一个 scan 文件夹名）
      - output_subdir: 要把缩小图片放入的子文件夹名（默认 'urd'）
      - list_filename: 要写入的列表文件名（默认 'list.txt'），位于每个 scan 文件夹下
      - overwrite: 若为 True，则若目标图片已存在则重新生成；否则跳过已存在文件
      - preview_limit: 若不为 None，仅对每个 scan 处理前 preview_limit 张图片（用于测试）
      - verbose: 是否打印处理日志
    返回:
      一个字典，键为 scan 名，值为 dictionary 包含处理结果统计：
         { 'processed': n, 'skipped': m, 'saved_files': [ ... ], 'errors': [ ... ] }
    """
    data_root_p = Path(data_root)
    blended_list_p = Path(blended_list_file)
    if not data_root_p.exists():
        raise FileNotFoundError(f"DATA_ROOT 不存在: {data_root}")
    if not blended_list_p.exists():
        raise FileNotFoundError(f"Blended list 文件不存在: {blended_list_file}")

    # 读取扫描列表（逐行可能包含多个名字，按空白拆分）
    scans = []
    with blended_list_p.open('r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            for p in parts:
                if p:
                    scans.append(p)

    result_summary = {}

    for scan in scans:
        try:
            scan_dir = data_root_p / scan
            if not scan_dir.exists() or not scan_dir.is_dir():
                if verbose:
                    print(f"[跳过] 未找到 scan 目录: {scan_dir}")
                result_summary[scan] = {'processed': 0, 'skipped': 0, 'saved_files': [], 'errors': [f"scan dir not found: {scan_dir}"]}
                continue

            blended_dir = scan_dir / "blended_images"
            if not blended_dir.exists() or not blended_dir.is_dir():
                if verbose:
                    print(f"[跳过] {scan}: blended_images 文件夹不存在")
                result_summary[scan] = {'processed': 0, 'skipped': 0, 'saved_files': [], 'errors': [f"blended_images not found: {blended_dir}"]}
                continue

            # 创建 urd 目录
            urd_dir = scan_dir / output_subdir
            urd_dir.mkdir(parents=True, exist_ok=True)

            # 收集 base map（排除非图片文件）
            base_map = _collect_base_map(blended_dir)
            base_keys = sorted(base_map.keys())

            # 如果没有检测到任何图片，可以尝试用文件总数/2估算数量（退化处理）
            if not base_keys:
                # fallback: 以 blended_dir 所有图片文件计数 //2
                files = [p for p in blended_dir.iterdir() if p.is_file() and _is_image_file(p)]
                est_n = max(0, len(files) // 2)
                if verbose:
                    print(f"[警告] {scan}: 未按名称匹配到 base，估算图片数量 = {est_n}")
                # 选择前 est_n 个（按名称排序）作为基准
                files_sorted = sorted(files)
                chosen = files_sorted[:est_n]
                # 将 chosen 每个直接作为单个 base
                base_keys = [p.stem for p in chosen]
                for idx, p in enumerate(chosen):
                    base_map[base_keys[idx]] = {'orig': p, 'mask': None, 'candidates': [p]}

            # 如果 preview_limit 设置，截断 base_keys
            if preview_limit is not None:
                base_keys = base_keys[:preview_limit]

            processed = 0
            skipped = 0
            saved_files = []
            errors = []

            for base in base_keys:
                rec = base_map[base]
                src = rec.get('orig', None)
                if src is None:
                    # 没有可处理的文件
                    errors.append(f"{base}: no source image found")
                    if verbose:
                        print(f"[跳过] {scan}/{base}: 没有找到可处理的原图（甚至没有 masked）")
                    skipped += 1
                    continue

                # 确定保存文件名：用原始文件名（包含扩展）
                out_name = src.name
                out_name = os.path.splitext(out_name)[0] + ".png"
                out_path = urd_dir / out_name

                if out_path.exists() and not overwrite:
                    if verbose:
                        print(f"[跳过] 已存在: {out_path}")
                    skipped += 1
                    saved_files.append(str(out_path))
                    continue

                # 执行缩放并保存
                try:
                    resize_and_save_png_high_quality(
                            input_path=src,out_path=out_path,  # 注意后缀会被改成 .png
                            scale=0.5,apply_unsharp=True,
                            png_compress_level=6,optimize_png=True,
                            convert_mode="RGBA"
                        )
                    processed += 1
                    saved_files.append(str(out_path))
                    if verbose:
                        print(f"[保存] {scan}: {src.name} -> {out_path.relative_to(data_root_p)}")
                except Exception as e:
                    err = f"{scan}/{base} 处理失败: {e}"
                    errors.append(err)
                    if verbose:
                        print(f"[错误] {err}")

            # 写 list.txt：把 urd 目录中所有文件名（仅图片）写入 list 文件（按字母排序）
            list_path = scan_dir / list_filename
            urd_images = sorted([p.name for p in urd_dir.iterdir() if p.is_file() and _is_image_file(p)])
            try:
                with list_path.open('w', encoding='utf-8') as f:
                    for nm in urd_images:
                        f.write(nm + "\n")
                if verbose:
                    print(f"[写入] {list_path} (共 {len(urd_images)} 行)")
            except Exception as e:
                err = f"{scan} 写 list.txt 失败: {e}"
                errors.append(err)
                if verbose:
                    print(f"[错误] {err}")

            result_summary[scan] = {
                'processed': processed,
                'skipped': skipped,
                'saved_files': saved_files,
                'errors': errors
            }

        except Exception as e:
            # 捕获单个 scan 处理的异常，不中断批量任务
            import traceback as tb
            tb_str = tb.format_exc()
            result_summary[scan] = {'processed': 0, 'skipped': 0, 'saved_files': [], 'errors': [str(e), tb_str]}
            if verbose:
                print(f"[致命] 处理 {scan} 出错: {e}")
                print(tb_str)

    return result_summary


def batch_create_trigulation(data_root: str,
    blended_list_file: str,
    output_subdir: str = 'urd'):
    data_root_p = Path(data_root)
    blended_list_p = Path(blended_list_file)
    if not data_root_p.exists():
        raise FileNotFoundError(f"DATA_ROOT 不存在: {data_root}")
    if not blended_list_p.exists():
        raise FileNotFoundError(f"Blended list 文件不存在: {blended_list_file}")

    # 读取扫描列表（逐行可能包含多个名字，按空白拆分）
    scans = []
    with blended_list_p.open('r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            for p in parts:
                if p:
                    scans.append(p)

    result_summary = {}

    for scan in scans:
        scan_dir = data_root_p / scan

        # 创建 triangulation 目录
        urd_dir = scan_dir / output_subdir
        urd_dir.mkdir(parents=True, exist_ok=True)


def batch_delete_trigulation(data_root: str,
    blended_list_file: str,
    output_subdir: str = 'urd'):
    data_root_p = Path(data_root)
    blended_list_p = Path(blended_list_file)
    if not data_root_p.exists():
        raise FileNotFoundError(f"DATA_ROOT 不存在: {data_root}")
    if not blended_list_p.exists():
        raise FileNotFoundError(f"Blended list 文件不存在: {blended_list_file}")

    # 读取扫描列表（逐行可能包含多个名字，按空白拆分）
    scans = []
    with blended_list_p.open('r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            for p in parts:
                if p:
                    scans.append(p)

    result_summary = {}

    for scan in scans:
        scan_dir = data_root_p / scan

        # 删除 triangulation 目录中，除开CDTinfo的文件夹
        urd_dir = scan_dir / output_subdir
        clean_directory(urd_dir)

        # ---------------------------------------------------------
        # 第二步：清理 CDTinfo 文件夹内部的文件
        # ---------------------------------------------------------
        cdtinfo_folder_name = "CDTinfo"
        # 进入文件夹，删除不需要文件
        cdtinfo_full_path=os.path.join(urd_dir, cdtinfo_folder_name)
        if os.path.exists(cdtinfo_full_path):
            print(f"📂 进入 {cdtinfo_folder_name} 进行文件过滤...")

            files = os.listdir(cdtinfo_full_path)
            kept_count = 0
            deleted_count = 0

            for filename in files:
                file_path = os.path.join(cdtinfo_full_path, filename)

                # 只处理文件，不处理CDTinfo里面可能存在的子文件夹（如果需要处理子文件夹请告诉我）
                if os.path.isfile(file_path):
                    # 核心判断逻辑：前缀是否为 CDT_info
                    if filename.startswith("CDT_info"):
                        # 保留
                        # print(f"  🛡️  保留文件: {filename}") # 如果文件太多，可以注释掉这行
                        kept_count += 1
                    else:
                        # 删除
                        try:
                            os.remove(file_path)
                            print(f"  🗑️  删除文件: {filename}")
                            deleted_count += 1
                        except Exception as e:
                            print(f"  ❌ 删除文件失败 {filename}: {e}")

            print(f"\n📊 CDTinfo 清理结果: 保留了 {kept_count} 个文件，删除了 {deleted_count} 个文件。")
        else:
            print(f"⚠️ 警告: 在 下没有找到 {cdtinfo_folder_name} 文件夹，跳过第二步。")

def batch_delete_urd(data_root: str,
    blended_list_file: str,
    output_subdir: str = 'urd'):
    data_root_p = Path(data_root)
    blended_list_p = Path(blended_list_file)
    if not data_root_p.exists():
        raise FileNotFoundError(f"DATA_ROOT 不存在: {data_root}")
    if not blended_list_p.exists():
        raise FileNotFoundError(f"Blended list 文件不存在: {blended_list_file}")

    # 读取扫描列表（逐行可能包含多个名字，按空白拆分）
    scans = []
    with blended_list_p.open('r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            for p in parts:
                if p:
                    scans.append(p)

    for scan in scans:
        scan_dir = data_root_p / scan

        delete_specific_folders(scan_dir)


def clean_directory(target_path):
    """
    删除 target_path 下除了 'CDTinfo' 以外的所有文件夹。
    """
    # 需要保留的文件夹名称
    FOLDER_TO_KEEP = "CDTinfo"

    # 检查路径是否存在
    if not os.path.exists(target_path):
        print(f"❌ 错误：找不到路径 -> {target_path}")
        return

    print(f"📂 正在处理目录：{target_path}")
    print("-" * 40)
    # 获取目录下所有内容
    items = os.listdir(target_path)
    deleted_count = 0
    for item in items:
        full_path = os.path.join(target_path, item)
        # 判断是否为文件夹
        if os.path.isdir(full_path):
        # 如果文件夹名称不是 CDTinfo，则删除
            if item != FOLDER_TO_KEEP:
                try:
                    shutil.rmtree(full_path)
                    print(f"✅ 已删除文件夹：{item}")
                    deleted_count += 1
                except Exception as e:
                    print(f"❌ 删除失败 {item}: {e}")
            else:
                print(f"🛡️ 已保留关键文件夹：{item}")


def delete_specific_folders(target_path):
    """
    在 target_path 下，专门删除 'urd' 和 'triangulation' 这两个文件夹。
    保留其他所有内容。
    """
    # 🎯 定义需要删除的文件夹名称列表
    FOLDERS_TO_DELETE = ["urd", "triangulation"]

    # 检查主路径是否存在
    if not os.path.exists(target_path):
        print(f"❌ 错误：找不到路径 -> {target_path}")
        return

    print(f"📂 正在检查目录：{target_path}")
    print("-" * 40)

    deleted_count = 0

    # 遍历你要删除的目标列表
    for folder_name in FOLDERS_TO_DELETE:
        # 拼接完整的绝对路径
        full_path = os.path.join(target_path, folder_name)

        # 检查这个文件夹是否存在
        if os.path.exists(full_path) and os.path.isdir(full_path):
            try:
                shutil.rmtree(full_path)
                print(f"✅ 已成功删除：{folder_name}")
                deleted_count += 1
            except OSError as e:
                # Linux 上常见的错误是权限不足 (Permission denied)
                print(f"❌ 删除失败 {folder_name}: {e}")
        else:
            print(f"⚠️  未找到文件夹或非文件夹：{folder_name} (可能已被删除)")

    print("-" * 40)
    print(f"🎉 操作结束。本次共删除了 {deleted_count} 个文件夹。")

if __name__ == "__main__":
    # 批量将图片缩小并导入urd文件夹 之前是1/2
    # res = batch_resize_blended_images(
    #     data_root=r"E:\RemoteCodeEx\Datas\blendmvs-data",
    #     blended_list_file=r"E:\RemoteCodeEx\Datas\blendmvs-data\BlendedMVG_list.txt",
    #     output_subdir="urd",
    #     list_filename="list.txt",
    #     overwrite=False,
    #     preview_limit=None,
    #     verbose=True
    # )
    #
    # # 打印处理摘要
    # for scan, info in res.items():
    #     print(scan, info)

    # 批量创建一个文件夹
    # batch_create_trigulation(data_root=r"E:\RemoteCodeEx\Datas\blendmvs-data",
    #                          blended_list_file=r"E:\RemoteCodeEx\Datas\blendmvs-data\BlendedMVG_list.txt",
    #                          output_subdir="triangulation")

    # 批量删除 triangulation 目录中，除开CDTinfo的文件夹
    # batch_delete_trigulation(data_root=r"E:\RemoteCodeEx\Datas\blendmvs-data",
    #                          blended_list_file=r"E:\RemoteCodeEx\Datas\blendmvs-data\BlendedMVG_list.txt",
    #                          output_subdir="triangulation")

    # 批量删除 urd和triangulation
    batch_delete_urd(data_root=r"/home/ym/Experiment/Datas/blendmvs-data",
                             blended_list_file=r"/home/ym/Experiment/Datas/blendmvs-data/BlendedMVG_list.txt",
                             output_subdir="triangulation")