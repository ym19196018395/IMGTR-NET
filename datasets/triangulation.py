#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import traceback
import os
from pathlib import Path
from typing import List, Tuple, Optional
import math
from PIL import Image, ImageDraw, ImageFont
import random
from collections import namedtuple

from pandas.core.dtypes.inference import is_float

# 类型别名
FloatVertex = Tuple[float, float]
PixelVertex = Tuple[int, int]

# 定义线的数据结构：两个端点ID + 两个关联面ID
Line = namedtuple('Line', ['p1', 'p2', 'face1', 'face2'])
# 定义三角形的数据结构：顶点ID + 关联线ID + 有效像素点容器
Triangle = namedtuple('Triangle', ['vertex_ids', 'line_ids', 'valid_points'])
Cdt_data=namedtuple(
    'Cdt_data',['vertexs', 'lines', 'triangles'] )

def quantize_vertices(vertices: List[FloatVertex],
                      image_size: Optional[Tuple[int, int]],
                      method: str = 'round') -> List[PixelVertex]:
    """
    将浮点顶点转换为像素顶点（整数）并裁剪到图像边界。
    参数:
      - vertices: [(x,y), ...] 浮点坐标（亚像素）
      - image_size: (width, height) 或 None（如果为 None，不做裁剪）
      - method: 'round'|'floor'|'ceil'
    返回:
      - pixel_vertices: [(px,py), ...] 整数像素坐标
    """
    def quant(x, m):
        if method == 'round':
            return int(round(x))
        elif method == 'floor':
            return int(math.floor(x))
        elif method == 'ceil':
            return int(math.ceil(x))
        else:
            raise ValueError("method must be 'round'|'floor'|'ceil'")

    W = H = None
    if image_size is not None:
        W, H = image_size

    pixel_vertices: List[PixelVertex] = []
    for (x, y) in vertices:
        px = quant(x, method)
        py = quant(y, method)
        if W is not None and H is not None:
            px = max(0, min(W - 1, px))
            py = max(0, min(H - 1, py))
        pixel_vertices.append((px, py))
    return pixel_vertices


def draw_mesh_on_image(image_path: str,
                       pixel_vertices: List[PixelVertex],
                       triangles: List[Triangle],
                       out_path: Optional[str] = None,
                       draw_vertices: bool = True,
                       vertex_radius: int = 1,
                       line_width: int = 1,
                       fill_alpha: int = 0,
                       vertex_color: Tuple[int,int,int]=(255,0,0),
                       edge_color: Tuple[int,int,int]=(0,254,0),
                       fill_color: Tuple[int,int,int]=(0,0,0),
                       draw_vertex_index: bool = False,
                       font_path: Optional[str] = None,
                       draw_edge_index: bool = False,
                       ) -> Image.Image:
    """
    在图片上绘制三角形网格并返回 PIL.Image。
    :param image_path:原图路径
    :param pixel_vertices:像素顶点列表 [(px,py), ...]
    :param triangles:三角形索引列表 [(i,j,k), ...]，索引对应 pixel_vertices 的位置
    :param out_path:若提供则保存结果图像到该路径
    :param draw_vertices:是否绘制顶点圆点
    :param vertex_radius:顶点点半径
    :param line_width:三角边宽度
    :param fill_alpha:填充透明度 (0-255)，若 0 则不填充
    :param vertex_color:RGB 三元组
    :param edge_color:
    :param fill_color:
    :param draw_vertex_index:是否在顶点旁画索引编号（可能影响密集点的可读性）
    :param font_path:可选 ttf 字体文件
    :return:overlay PIL.Image 对象 (RGB)
    """
    img = Image.open(image_path).convert("RGBA")
    W, H = img.size

    # 新建一个透明的 overlay 用于绘制，最后与原图合并
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # 预载字体
    font = None
    if draw_vertex_index:
        try:
            if font_path:
                font = ImageFont.truetype(font_path, 12)
            else:
                font = ImageFont.load_default()
        except:
            font = None

    # 先填充三角形 (如果 fill_alpha > 0)
    if fill_alpha > 0:
        for tri in triangles:
            i, j, k = tri.vertex_ids
            p1 = pixel_vertices[i]
            p2 = pixel_vertices[j]
            p3 = pixel_vertices[k]
            polygon = [p1, p2, p3]
            # RGBA color, alpha = fill_alpha
            fill_color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
            rgba = (fill_color[0], fill_color[1], fill_color[2], fill_alpha)
            draw.polygon(polygon, fill=rgba)

    # 然后画边界线
    if draw_edge_index == True:
        for tri in triangles:
            # 随机颜色
            # edge_color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
            i, j, k = tri.vertex_ids
            p1 = pixel_vertices[i]
            p2 = pixel_vertices[j]
            p3 = pixel_vertices[k]
            # draw line segments (p1-p2, p2-p3, p3-p1)
            rgba = (edge_color[0], edge_color[1], edge_color[2], 255)
            draw.line([p1, p2], fill=rgba, width=line_width)
            draw.line([p2, p3], fill=rgba, width=line_width)
            draw.line([p3, p1], fill=rgba, width=line_width)

    # 绘制顶点点
    if draw_vertices:
        for idx, (px, py) in enumerate(pixel_vertices):
            bbox = (px - vertex_radius, py - vertex_radius, px + vertex_radius, py + vertex_radius)
            draw.ellipse(bbox, outline=vertex_color + (255,), width=1)
            # 填充小圆（白色）以增强可见性
            draw.ellipse(
                (px - vertex_radius + 1, py - vertex_radius + 1, px + vertex_radius - 1, py + vertex_radius - 1),
                fill=(255, 255, 255, 255))
            if draw_vertex_index and font:
                draw.text((px + vertex_radius + 2, py - vertex_radius - 2), str(idx), fill=(255, 0, 0, 255), font=font)

    # 合并 overlay 到原图
    result = Image.alpha_composite(img, overlay).convert("RGB")

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        result.save(out_path)

    return result


def get_cdt_datas(cdt_file_path: str,H:int,W:int) -> Cdt_data:
    '''
    获取三角剖完的数据
    :param cdt_file_path:
    :return: 返回顶点数据和三角关系,vertices,triangles
    '''

    # 开始处理CDT
    # 加载三角网数据
    vertices_float = []  # 顶点坐标：[(x1, y1), (x2, y2), ...]
    vertices_int=[]
    lines = []  # 线：[Line(p1, p2, face1, face2), ...]
    triangles = []  # 三角形：[Triangle(vertex_ids, line_ids, valid_points), ...]
    errors = []  # 错误信息记录
    cdt_data=[]
    try:
        with open(cdt_file_path, 'r') as f:
            lines_raw = f.readlines()  # txt文本的行数
            # 1.读取第一行， "8&17&10&" 获取点 线 面的数据
            first_line = lines_raw[0].strip()
            # 如果包含 "&" 则按 & 分割并取第一个非空部分
            if "&" in first_line:
                num_vertices = int(first_line.split("&")[0])
                num_lines = int(first_line.split("&")[1])
                num_triangles = int(first_line.split("&")[2])
            else:
                print(f"cdt数据格式出现错误")
                traceback.print_exc()

            # 2.收集顶点坐标
            for i in range(1, num_vertices + 1):
                if i < len(lines_raw):
                    parts = lines_raw[i].strip().split()
                    if len(parts) >= 2:
                        vertices_float.append((float(parts[0]), float(parts[1])))

            # 3.收集线数据，分别是两个端点+两个邻界面
            line_start_idx = num_vertices + 1
            line_end_idx = line_start_idx + num_lines  # 线结束索引（不包含）
            if num_vertices + 1 < len(lines_raw):
                # 进行了一个修改，+1
                for i in range(line_start_idx, line_end_idx):
                    parts = lines_raw[i].split()
                    if len(parts) < 4:
                        errors.append(
                            f"线第{i - line_start_idx + 1}行数据不足，预期4个参数，实际{len(parts)}个：{lines_raw[i]}")
                        continue
                    try:
                        p1 = int(parts[0])  # 端点1 ID
                        p2 = int(parts[1])  # 端点2 ID
                        face1 = int(parts[2])  # 关联面1 ID
                        face2 = int(parts[3])  # 关联面2 ID
                        lines.append(Line(p1, p2, face1, face2))
                    except ValueError as e:
                        errors.append(f"线第{i - line_start_idx + 1}行参数转换失败：{e}，内容：{lines_raw[i]}")

            # 4. 读取三角形（从line_end_idx开始，每个三角形包含多行）
            tri_start_idx = line_end_idx
            current_idx = tri_start_idx  # 当前读取索引
            for tri_idx in range(num_triangles):

                # 4.1 三角形顶点ID（一行，3个整数）
                vertex_ids_line = lines_raw[current_idx]
                vertex_parts = vertex_ids_line.split()
                if len(vertex_parts) < 3:
                    errors.append(
                        f"三角形{tri_idx + 1}顶点ID不足，预期3个，实际{len(vertex_parts)}个：{vertex_ids_line}")
                    current_idx += 1  # 尝试跳过该行
                    continue
                try:
                    v1, v2, v3 = map(int, vertex_parts[:3])
                    vertex_ids = (v1, v2, v3)
                except ValueError as e:
                    errors.append(f"三角形{tri_idx + 1}顶点ID转换失败：{e}，内容：{vertex_ids_line}")
                    current_idx += 1
                    continue
                current_idx += 1  # 移动到下一行
                # 4.2 三角形三条线段
                line_ids_line = lines_raw[current_idx]
                line_parts = line_ids_line.split()
                l1, l2, l3 = map(int, line_parts[:3])
                line_ids = (l1, l2, l3)
                current_idx += 1  # 移动到下一行

                # 4.3 有效像素点数量（一行，1个整数）
                valid_count_line = lines_raw[current_idx]
                try:
                    valid_count = int(valid_count_line)
                except ValueError as e:
                    errors.append(f"三角形{tri_idx + 1}有效点数量转换失败：{e}，内容：{valid_count_line}")
                    current_idx += 1
                    continue
                current_idx += 1  # 移动到下一行

                # 4.4 有效像素点坐标（valid_count行，每行2个浮点数）
                valid_points = []
                for pt_idx in range(valid_count):
                    if current_idx >= len(lines_raw):
                        errors.append(f"三角形{tri_idx + 1}有效点不完整，预期{valid_count}个，实际{pt_idx}个")
                        break
                    pt_line = lines_raw[current_idx]
                    pt_parts = pt_line.split()
                    px = int(pt_parts[0])
                    py = int(pt_parts[1])
                    valid_points.append((px, py))
                    current_idx += 1  # 移动到下一行

                # 存储当前三角形
                tri=Triangle(vertex_ids=vertex_ids,line_ids=line_ids,valid_points=valid_points)
                triangles.append(tri)

            # 校验读取数量
            if len(vertices_float) != num_vertices:
                errors.append(f"顶点数量不匹配，预期{num_vertices}个，实际{len(vertices_float)}个")
            if len(lines) != num_lines:
                errors.append(f"线数量不匹配，预期{num_lines}条，实际{len(lines)}条")
            if len(triangles) != num_triangles:
                errors.append(f"三角形数量不匹配，预期{num_triangles}个，实际{len(triangles)}个")
            # 量化处理将亚像素级像素转化为图像像素 ym-issue-11.29 因为后面是双线性插值提取不需要化为整数
            # vertices_int = quantize_vertices(vertices_float, (W, H), method='round')

    except Exception as e:
        errors.append(f"读取文件时发生错误：{str(e)}\n{traceback.format_exc()}")

    cdt_data=(Cdt_data(vertices_float,lines, triangles))
    return cdt_data


def test():

    root_path="E:\RemoteCodeEx\Datas\mvs_training\dtu\Rectified\scan16_train\\triangulation"
    # root_path = "E:\RemoteCodeEx\Triangulation\\test-save"
    cdt_path=os.path.join(root_path, "CDTinfo")
    img_id="rect_001_1_r5000"
    # img_id ="2"
    image_path = os.path.join(root_path, f"grayImgs\gray_{img_id}.png")
    img = Image.open(image_path)
    W,H = img.size
    cdt_data=get_cdt_datas(img_id,cdt_path,W=W,H=H)
    vertices, lines, triangles = cdt_data.vertexs, cdt_data.lines, cdt_data.triangles

    # 量化处理将亚像素级像素转化为图像像素
    pixel_vertices = quantize_vertices(vertices, (W, H), method='round')

    # 把三角网绘到图像并保存：
    out_img = draw_mesh_on_image(image_path, pixel_vertices, triangles,
                                    out_path=os.path.join(root_path, f"save_overlay{img_id}_edge_color.png"),
                                    draw_vertices=True, draw_vertex_index=False,draw_edge_index=True)




