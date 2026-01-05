from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from datasets.data_io import *
import cv2
import random

from torch.utils.data._utils.collate import default_collate
from datasets.triangulation import *


class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, robust_train = False):
        super(MVSDataset, self).__init__()

        self.stages = 4
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews # 每个样本使用的视图数量（参考视图+源视图）
        self.robust_train = robust_train # 是否使用鲁棒训练策略（随机选择源视图）
        

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()

    def build_list(self):
        """
        return:metas 里面装着 各种类似于 ('scan2', 0, 0, [10, 1, 9, 12, 11, 13, 2, 8, 14, 27])
        分别对应的是，数据集种类，光度值，参考图的编号，源图的编号
        """
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        for scan in scans:
            # 存储着一共多少视角图，以及选中该视角图为参考图之后对应的源图
            pair_file = os.path.join(self.datapath,"{}/cams/pair.txt".format(scan))
            
            with open(pair_file) as f:
                self.num_viewpoint = int(f.readline()) # 视图总数
                # viewpoints (49)
                for view_idx in range(self.num_viewpoint):
                    # ref_view：直接从那一行读取参考视角 id
                    ref_view = int(f.readline().rstrip())
                    # 把第二行按空格拆成token列表，形式像['k', 'view1', 'score1',  ...]
                    # [1::2] 这个意味着从索引 1 开始、步长 2，取出 view1, view2
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    metas.append((scan, ref_view, src_views))
        print("dataset", self.mode, "metas:", len(metas))
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename,override_scale=None):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))

        # 3. Depth Min ym-issue
        depth_min_raw = float(lines[11].split()[0])

        # --- 【修改逻辑】 ---
        if override_scale is not None:
            # 如果外界指定了缩放因子（源图），直接用
            scale_factor = override_scale
        else:
            # 如果没指定（参考图），现场计算
            # 这里的 100.0 是为了把深度拉到 PatchMatchNet 喜欢的范围，也可以设为 1.0
            scale_factor = 100.0 / (depth_min_raw + 1e-5)

            # 4. 应用缩放
        depth_min = depth_min_raw * scale_factor

        # 只缩放平移向量 (x, y, z)，旋转矩阵 R 不变
        extrinsics[:3, 3] *= scale_factor

        return intrinsics, extrinsics, depth_min, scale_factor

    def read_img(self, filename):
        """
        读取图像并生成 4 个尺度（用于多尺度模型）：
        """
        img = Image.open(filename)
        # scale 0~255 to 0~1
        
        np_img = np.array(img, dtype=np.float32) / 255.
        h, w, _ = np_img.shape
        np_img_ms = {
            "stage_3": cv2.resize(np_img, (w//8, h//8), interpolation=cv2.INTER_LINEAR), 
            "stage_2": cv2.resize(np_img, (w//4, h//4), interpolation=cv2.INTER_LINEAR),
            "stage_1": cv2.resize(np_img, (w//2, h//2), interpolation=cv2.INTER_LINEAR),
            "stage_0": np_img
        }
        return np_img_ms


    def prepare_img(self, hr_img):
        """
        对高分辨率图像预处理（下采样 1/2 + 裁剪到 512x640），统一输入尺寸
        """
        # original w,h: 1600, 1200; downsample -> 800, 600 ; crop -> 640, 512
        #downsample
        h, w = hr_img.shape
        hr_img_ds = cv2.resize(hr_img, (w//2, h//2), interpolation=cv2.INTER_NEAREST)
        #crop
        h, w = hr_img_ds.shape
        target_h, target_w = 512, 640
        start_h, start_w = (h - target_h)//2, (w - target_w)//2
        hr_img_crop = hr_img_ds[start_h: start_h + target_h, start_w: start_w + target_w]

        return hr_img_crop

    def read_depth_hr(self, filename, scale_factor,depth_min):
        """
              读取高分辨率深度图，应用缩放，生成多尺度掩码，并计算 depth_max
              """
        # 1. 读取 PFM
        depth_hr = np.array(read_pfm(filename)[0], dtype=np.float32)
        # 2. 维度处理
        if depth_hr.ndim == 3:
            depth_hr = np.squeeze(depth_hr, 2)

        # 3. 【核心修改】应用缩放因子
        depth_hr = depth_hr * scale_factor

        # 4. 数据清洗 (NaN/Inf -> 0)
        depth_hr = np.nan_to_num(depth_hr, nan=0.0, posinf=0.0, neginf=0.0)

        # 5. 生成掩码 (基于缩放后的深度)
        mask_hr = (depth_hr > depth_min).astype(np.float32)

        # 6. 【核心修改】从真实的深度图中计算 depth_max
        # 我们只计算有效区域的最大值
        if mask_hr.sum() > 0:
            depth_max = depth_hr.max()
        else:
            # 如果全黑，给一个默认值防止报错
            depth_max = 100.0

        # 7. 多尺度下采样 (保持不变)
        h, w = depth_hr.shape
        depth_lr_ms = {
            "stage_3": cv2.resize(depth_hr, (w // 8, h // 8), interpolation=cv2.INTER_NEAREST),
            "stage_2": cv2.resize(depth_hr, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage_1": cv2.resize(depth_hr, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage_0": depth_hr
        }

        mask_lr_ms = {
            "stage_3": cv2.resize(mask_hr, (w // 8, h // 8), interpolation=cv2.INTER_NEAREST),
            "stage_2": cv2.resize(mask_hr, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage_1": cv2.resize(mask_hr, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage_0": mask_hr
        }

        return depth_lr_ms, mask_lr_ms, depth_max


    def __getitem__(self, idx):
        # 这里是对应的一组数据包括一张参考图加几张源图
        meta = self.metas[idx]
        scan, ref_view, src_views = meta

        # robust training strategy
        if self.robust_train:
            num_src_views = len(src_views)
            # 在10个源图中，随机选几个源图，使得参考图加上源图等于nviews数
            index = random.sample(range(num_src_views), self.nviews - 1)
            view_ids = [ref_view] + [src_views[i] for i in index]

        else:
            view_ids = [ref_view] + src_views[:self.nviews - 1]

        imgs_0 = []
        imgs_1 = []
        imgs_2 = []
        imgs_3 = []

        mask = None
        depth = None
        depth_min = None
        depth_max = None
        # BlendedMVS 的场景尺度跨度极大（有的场景是几米的物体，有的是几百米的城市），直接使用原始数值会导致网络难以收敛。
        current_scale=1.0

        proj_matrices_0 = []
        proj_matrices_1 = []
        proj_matrices_2 = []
        proj_matrices_3 = []

        # 装载cdt三角剖分数据
        # 顶点坐标集合：[(x1, y1), (x2, y2), ...]
        # 线集合：[Line(p1, p2, face1, face2), ...]
        # 三角形集合：[Triangle(vertex_ids, line_ids, valid_points), ...]
        cdt_data=[]
        # 读取源图和参考图的信息
        for i, vid in enumerate(view_ids):
            # 转为字符串后补前导零到8位
            vid_str=str(vid).zfill(8)
            img_filename = os.path.join(self.datapath,
                                        '{}/blended_images/{}.jpg'.format(scan, vid_str))
            depth_filename_hr = os.path.join(self.datapath, '{}/rendered_depth_maps/{}.pfm'.format(scan, vid_str))
            proj_mat_filename = os.path.join(self.datapath, '{}/cams/{}_cam.txt').format(scan,vid_str)

            # 暂时不需要三角剖分
            # ym-modify 因为vid是从零开始 而图片是从1开始所以要加个1
            # triangulation_filename = os.path.join(self.datapath,
            #                                       'Rectified/{}_train/triangulation/CDTinfo/CDT_info_vlf_rect_{:03d}_1_r5000.txt'.format(
            #                                           scan, vid+1))

            imgs = self.read_img(img_filename)
            imgs_0.append(imgs['stage_0'])
            imgs_1.append(imgs['stage_1'])
            imgs_2.append(imgs['stage_2'])
            imgs_3.append(imgs['stage_3'])

            intrinsics =[]
            extrinsics =[]
            # here, the intrinsics from file is already adjusted to the downsampled size of feature 1/4H0 * 1/4W0
            # BlendedMVS 的场景尺度跨度极大（有的场景是几米的物体，有的是几百米的城市），直接使用原始数值会导致网络难以收敛
            # 所以使深度进行一个缩放
            if i == 0:  # reference view
                # 参考图：不传 override_scale，让它自己算，并返回算出来的 scale_factor
                # 保证参考图和源图都进行一个缩放
                intrinsics, extrinsics, depth_min_, current_scale = self.read_cam_file(proj_mat_filename, override_scale=None)
                # 对掩码和深度图进行一个下采样，深度掩码通过pfm得到
                depth,mask,depth_max_ = self.read_depth_hr(depth_filename_hr,current_scale,depth_min_)
                depth_min = depth_min_
                depth_max = depth_max_

                for l in range(self.stages):
                    mask[f'stage_{l}'] = np.expand_dims(mask[f'stage_{l}'],2)
                    mask[f'stage_{l}'] = mask[f'stage_{l}'].transpose([2,0,1])
                    depth[f'stage_{l}'] = np.expand_dims(depth[f'stage_{l}'],2)
                    depth[f'stage_{l}'] = depth[f'stage_{l}'].transpose([2,0,1])

                # ym-add 获取参考图三角网数据 这里vid需要加1 因为是从零开始
                # W=imgs_0[0].shape[1]
                # H=imgs_0[0].shape[0]
                # # print("{}--------{}".format(scan,vid+1))
                # cdt_data = get_cdt_datas(triangulation_filename,H=H,W=W)

            else: # source view
                # 强制源图使用和参考图完全一样的缩放比例，否则几何关系就断了！
                intrinsics, extrinsics, _, _ = self.read_cam_file(proj_mat_filename,override_scale=current_scale)

            # 对矩阵进行一个处理，分别求得不同大小图片的投影矩阵
            proj_mat = extrinsics.copy()
            # 将1，2行的系数*scale
            intrinsics[:2,:] *= 0.125
            # 求得是投影矩阵 P = K [R|t]  外参矩阵是取三行四列大小的数据
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_3.append(proj_mat)

            proj_mat = extrinsics.copy()
            intrinsics[:2,:] *= 2
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_2.append(proj_mat)

            proj_mat = extrinsics.copy()
            intrinsics[:2,:] *= 2
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_1.append(proj_mat)

            proj_mat = extrinsics.copy()
            intrinsics[:2,:] *= 2
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_0.append(proj_mat)


        # 对数据进行一个处理，因为多批次数处理需要保证每个样本的该字段的形状一致
        # imgs: N*3*H0*W0, N is number of images
        imgs_0 = np.stack(imgs_0).transpose([0, 3, 1, 2])
        imgs_1 = np.stack(imgs_1).transpose([0, 3, 1, 2])
        imgs_2 = np.stack(imgs_2).transpose([0, 3, 1, 2])
        imgs_3 = np.stack(imgs_3).transpose([0, 3, 1, 2])
        
        imgs = {}
        imgs['stage_0'] = imgs_0
        imgs['stage_1'] = imgs_1
        imgs['stage_2'] = imgs_2
        imgs['stage_3'] = imgs_3
        
        # proj_matrices: N*4*4
        proj_matrices_0 = np.stack(proj_matrices_0)
        proj_matrices_1 = np.stack(proj_matrices_1)
        proj_matrices_2 = np.stack(proj_matrices_2)
        proj_matrices_3 = np.stack(proj_matrices_3)
        
        proj={}
        proj['stage_3']=proj_matrices_3
        proj['stage_2']=proj_matrices_2
        proj['stage_1']=proj_matrices_1
        proj['stage_0']=proj_matrices_0

        # todo：将数据转化为list or ndarray，为的是后续可以使用，如果之后要进行并行运算还需要修改
        # vertexs = np.asarray(cdt_data.vertexs, dtype=np.int64)
        # lines = np.asarray(cdt_data.lines, dtype=np.int64)
        # # 每个 triangle 分开处理，保留 list，
        # triangles = []
        # for t in cdt_data.triangles:
        #     tri_v = np.asarray(t.vertex_ids, dtype=np.int64)
        #     tri_l = np.asarray(t.line_ids, dtype=np.int64)
        #     tri_pts = np.asarray(t.valid_points, dtype=np.int64)  # 变长，允许不同长度
        #     triangles.append({'vertex_ids': tri_v, 'line_ids': tri_l, 'valid_points': tri_pts})

        vertexs=[]
        lines=[]
        triangles=[]

        # data is numpy array
        return {"imgs": imgs,                   # N*3*H0*W0
                "proj_matrices": proj,          # N*4*4
                "depth": depth,                 # 1*H0 * W0
                "depth_min": depth_min,         # scalar
                "depth_max": depth_max,         # scalar
                "mask": mask,                   # 1*H0 * W0
                "vertexs": vertexs,         # ndarray (Nv, ..)
                "lines": lines,             # ndarray (Nl, ..)
                "triangles": triangles      # list of ndarrays
                }

def collate_keep_list(batch):
    """
    跳过cdt-data，后面单独进行一个处理
    """
    out = {}
    keys = batch[0].keys()
    for k in keys:
        if k in ['triangles', 'vertexs', 'lines']:
            out[k] = [b[k] for b in batch]
        else:
            out[k] = default_collate([b[k] for b in batch])
    return out
