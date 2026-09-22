import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from datasets.data_io import read_pfm
from datasets.triangulation import get_cdt_datas


def read_pair_file(filename):
    """
    解析标准 pair.txt 文件
    格式:
      第一行: 参考图总数
      随后每两行为一组:
        行 1: 参考图 ID (例如 1)
        行 2: 源图数量及各源图 ID 与得分 (例如 2  0 2000.0  2 2000.0)
    返回: List[(ref_view_id, [src_view_id_1, src_view_id_2, ...])]
    """
    with open(filename, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    if not lines:
        return []

    num_viewpoint = int(lines[0])
    pairs = []
    line_idx = 1
    for _ in range(num_viewpoint):
        if line_idx >= len(lines):
            break
        ref_view = int(lines[line_idx])
        src_tokens = lines[line_idx + 1].split()
        num_src = int(src_tokens[0])
        src_views = [int(src_tokens[2 * i + 1]) for i in range(num_src)]
        pairs.append((ref_view, src_views))
        line_idx += 2
    return pairs


class MVSDataset(Dataset):
    """
    ENRICH-Aerial_Data 航空遥感数据集专用轻量 DataLoader
    适配 2112x1408 超大分辨率与 1 参 2 源 (严格 3 视角)
    专用于纯推理/评测模式：彻底剔除离线物理法向与连续软加权等训练专用损失监督
    """
    def __init__(self, datapath, listfile, mode="test", nviews=3, stages=4, **kwargs):
        super().__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.stages = stages

        self.metas = self.build_metas()

    def build_metas(self):
        """
        读取 scan_list.txt 并结合各场景下的 pair.txt 构建样本元数据
        元数据格式: (scan_name, ref_view_id, src_view_ids)
        """
        metas = []
        if os.path.isfile(self.listfile):
            with open(self.listfile, 'r', encoding='utf-8') as f:
                scans = [line.strip() for line in f.readlines() if line.strip()]
        else:
            # 若未直接指定列表文件，自动扫描根目录下所有 scan_ 开头的子目录
            scans = sorted([d for d in os.listdir(self.datapath) if os.path.isdir(os.path.join(self.datapath, d)) and d.startswith("scan_")])

        for scan in scans:
            scan_dir = os.path.join(self.datapath, scan)
            pair_file = os.path.join(scan_dir, "pair.txt")
            if not os.path.exists(pair_file):
                # 默认 fallback: 1 参 2 源 (ref: 1, src: [0, 2])
                metas.append((scan, 1, [0, 2]))
                continue

            pairs = read_pair_file(pair_file)
            for ref_view, src_views in pairs:
                # 截取前 (nviews - 1) 个源视角
                chosen_src = src_views[:self.nviews - 1]
                metas.append((scan, ref_view, chosen_src))

        print(f"[Dataset] 成功挂载 ENRICH-Aerial_Data 数据集: 共解析 {len(metas)} 个测试样本 (视角数: {self.nviews})")
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename):
        """
        读取标准 MVS / DTU 格式相机参数文件 (*_cam.txt)
        包含 4x4 外参矩阵、3x3 内参矩阵、深度范围 [depth_min, depth_interval, depth_num, depth_max]
        """
        with open(filename, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        # 1. 读取外参 (行 1~4)
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))

        # 2. 读取内参 (行 6~8)
        intrinsics = np.fromstring(' '.join(lines[6:9]), dtype=np.float32, sep=' ').reshape((3, 3))

        # 3. 读取深度范围 (行 9)
        tokens = list(map(float, lines[9].split()))
        depth_min = tokens[0]
        if len(tokens) >= 4:
            depth_interval = tokens[1]
            depth_num = int(tokens[2])
            depth_max = tokens[3]
        elif len(tokens) == 2:
            depth_max = tokens[1]
            depth_interval = (depth_max - depth_min) / 192.0
            depth_num = 192
        else:
            depth_interval = tokens[1] if len(tokens) > 1 else 1.0
            depth_max = tokens[-1]
            depth_num = 192

        return intrinsics, extrinsics, depth_min, depth_max, depth_interval

    def read_img(self, filename):
        """
        读取 RGB 图像并构建 4 级下采样金字塔 (1/8, 1/4, 1/2, 1/1)
        """
        img = Image.open(filename).convert('RGB')
        np_img = np.array(img, dtype=np.float32) / 255.0  # 归一化到 [0, 1]
        h, w, _ = np_img.shape

        np_img_ms = {
            "stage_3": cv2.resize(np_img, (w // 8, h // 8), interpolation=cv2.INTER_LINEAR),
            "stage_2": cv2.resize(np_img, (w // 4, h // 4), interpolation=cv2.INTER_LINEAR),
            "stage_1": cv2.resize(np_img, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR),
            "stage_0": np_img
        }
        return np_img_ms

    def read_depth_ms(self, filename, depth_min=0.0):
        """
        读取 PFM 格式的高精度物理真值深度图 (米) 并构建 4 级金字塔与有效掩码
        """
        depth_raw, _ = read_pfm(filename)
        if depth_raw.ndim == 3:
            depth_raw = depth_raw.squeeze(2)

        depth_raw = np.nan_to_num(depth_raw, nan=0.0, posinf=0.0, neginf=0.0)
        mask_hr = (depth_raw > depth_min).astype(np.float32)
        depth_max = float(depth_raw.max()) if depth_raw.max() > 0 else 1000.0

        h, w = depth_raw.shape
        depth_lr_ms = {
            "stage_3": cv2.resize(depth_raw, (w // 8, h // 8), interpolation=cv2.INTER_NEAREST),
            "stage_2": cv2.resize(depth_raw, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage_1": cv2.resize(depth_raw, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage_0": depth_raw
        }
        mask_lr_ms = {
            "stage_3": cv2.resize(mask_hr, (w // 8, h // 8), interpolation=cv2.INTER_NEAREST),
            "stage_2": cv2.resize(mask_hr, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage_1": cv2.resize(mask_hr, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage_0": mask_hr
        }
        return depth_lr_ms, mask_lr_ms, depth_max

    def __getitem__(self, idx):
        scan, ref_view, src_views = self.metas[idx]
        view_ids = [ref_view] + src_views

        imgs_0, imgs_1, imgs_2, imgs_3 = [], [], [], []
        proj_matrices_0, proj_matrices_1, proj_matrices_2, proj_matrices_3 = [], [], [], []
        intrinsics_matrices_0, intrinsics_matrices_1, intrinsics_matrices_2, intrinsics_matrices_3 = [], [], [], []

        ref_depth_dict = None
        ref_mask_dict = None
        ref_depth_min = 0.0
        ref_depth_max = 1000.0

        scan_dir = os.path.join(self.datapath, scan)

        for i, vid in enumerate(view_ids):
            img_path = os.path.join(scan_dir, "images", f"{vid:08d}.jpg")
            cam_path = os.path.join(scan_dir, "cams", f"{vid:08d}_cam.txt")

            # 1. 读取多尺度图像
            imgs = self.read_img(img_path)
            imgs_0.append(imgs['stage_0'])
            imgs_1.append(imgs['stage_1'])
            imgs_2.append(imgs['stage_2'])
            imgs_3.append(imgs['stage_3'])

            # 2. 读取相机矩阵
            intrinsics_base, extrinsics, d_min, d_max, d_interval = self.read_cam_file(cam_path)

            if i == 0:  # 参考视角
                ref_depth_min = d_min
                ref_depth_max = d_max
                depth_path = os.path.join(scan_dir, "depths", f"{vid:08d}.pfm")
                if os.path.exists(depth_path):
                    ref_depth_dict, ref_mask_dict, _ = self.read_depth_ms(depth_path, depth_min=ref_depth_min)
                else:
                    # 若无深度真值，生成虚拟全 0 深度和掩码
                    H_raw, W_raw = imgs['stage_0'].shape[:2]
                    ref_depth_dict = {
                        f"stage_{l}": np.zeros((H_raw // (2**l), W_raw // (2**l)), dtype=np.float32)
                        for l in range(self.stages)
                    }
                    ref_mask_dict = {
                        f"stage_{l}": np.zeros((H_raw // (2**l), W_raw // (2**l)), dtype=np.float32)
                        for l in range(self.stages)
                    }

            # 3. 构造 4 个尺度的投影矩阵 P = K [R|t] 与内参矩阵
            # stage 3 (1/8)
            K_3 = intrinsics_base.copy()
            K_3[:2, :] *= 0.125
            P_3 = extrinsics.copy()
            P_3[:3, :4] = np.matmul(K_3, P_3[:3, :4])
            proj_matrices_3.append(P_3)
            intrinsics_matrices_3.append(K_3)

            # stage 2 (1/4)
            K_2 = intrinsics_base.copy()
            K_2[:2, :] *= 0.25
            P_2 = extrinsics.copy()
            P_2[:3, :4] = np.matmul(K_2, P_2[:3, :4])
            proj_matrices_2.append(P_2)
            intrinsics_matrices_2.append(K_2)

            # stage 1 (1/2)
            K_1 = intrinsics_base.copy()
            K_1[:2, :] *= 0.5
            P_1 = extrinsics.copy()
            P_1[:3, :4] = np.matmul(K_1, P_1[:3, :4])
            proj_matrices_1.append(P_1)
            intrinsics_matrices_1.append(K_1)

            # stage 0 (1/1)
            K_0 = intrinsics_base.copy()
            P_0 = extrinsics.copy()
            P_0[:3, :4] = np.matmul(K_0, P_0[:3, :4])
            proj_matrices_0.append(P_0)
            intrinsics_matrices_0.append(K_0)

        # 4. 堆叠多视角图像 [V, 3, H, W]
        imgs_dict = {
            "stage_0": np.stack(imgs_0).transpose([0, 3, 1, 2]),
            "stage_1": np.stack(imgs_1).transpose([0, 3, 1, 2]),
            "stage_2": np.stack(imgs_2).transpose([0, 3, 1, 2]),
            "stage_3": np.stack(imgs_3).transpose([0, 3, 1, 2]),
        }

        # 5. 堆叠多尺度投影与内参矩阵 [V, 4, 4] / [V, 3, 3]
        proj_dict = {
            "stage_0": np.stack(proj_matrices_0),
            "stage_1": np.stack(proj_matrices_1),
            "stage_2": np.stack(proj_matrices_2),
            "stage_3": np.stack(proj_matrices_3),
        }
        intrinsics_dict = {
            "stage_0": np.stack(intrinsics_matrices_0),
            "stage_1": np.stack(intrinsics_matrices_1),
            "stage_2": np.stack(intrinsics_matrices_2),
            "stage_3": np.stack(intrinsics_matrices_3),
        }

        # 6. 处理深度与掩码维度 [1, H, W]
        for l in range(self.stages):
            ref_depth_dict[f'stage_{l}'] = np.expand_dims(ref_depth_dict[f'stage_{l}'], 0)
            ref_mask_dict[f'stage_{l}'] = np.expand_dims(ref_mask_dict[f'stage_{l}'], 0)

        # 7. 加载参考视角的 CDT 三角剖分几何数据
        cdt_path = os.path.join(scan_dir, "triangulation", "CDTinfo", f"CDT_info_vlf_{ref_view:08d}.txt")
        H_ref, W_ref = imgs_0[0].shape[:2]
        if os.path.exists(cdt_path):
            cdt_data = get_cdt_datas(cdt_path, H=H_ref, W=W_ref)
            vertexs = np.asarray(cdt_data.vertexs, dtype=np.float32)
            lines = np.asarray(cdt_data.lines, dtype=np.int64)
            triangles = []
            for t in cdt_data.triangles:
                triangles.append({
                    'vertex_ids': np.asarray(t.vertex_ids, dtype=np.int64),
                    'line_ids': np.asarray(t.line_ids, dtype=np.int64),
                    'valid_points': np.asarray(t.valid_points, dtype=np.int64)
                })
        else:
            # 零回退保护
            vertexs = np.zeros((0, 2), dtype=np.float32)
            lines = np.zeros((0, 4), dtype=np.int64)
            triangles = []

        return {
            "imgs": imgs_dict,
            "proj_matrices": proj_dict,
            "intrinsics_mats": intrinsics_dict,
            "depth": ref_depth_dict,
            "mask": ref_mask_dict,
            "depth_min": ref_depth_min,
            "depth_max": ref_depth_max,
            "vertexs": vertexs,
            "lines": lines,
            "triangles": triangles,
            "scan": scan,
            "file_id": f"{ref_view:08d}"
        }


def collate_keep_list(batch):
    """
    自适应批次拼接函数：保留变长拓扑结构 (triangles, vertexs, lines) 与字符串标识
    """
    out = {}
    keys = batch[0].keys()
    for k in keys:
        if k in ['triangles', 'vertexs', 'lines', 'scan', 'file_id']:
            out[k] = [b[k] for b in batch]
        else:
            out[k] = default_collate([b[k] for b in batch])
    return out
