from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from datasets.data_io import *
import cv2

from datasets.triangulation import get_cdt_datas


class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, img_wh=(2688, 1792), **kwargs):
        super(MVSDataset, self).__init__()

        self.stages = 4
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.img_wh = img_wh

        assert self.mode == "test"
        self.metas = self.build_list()

    def build_list(self):
        """
        return:metas 里面装着 子数据集的名称和里面每个图片的id
        """
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        for scan in scans:
            # 储存着每个照片的id
            list_file = os.path.join(self.datapath, "{}/Images/{}/list.txt".format(self.mode,scan))

            with open(list_file) as f:
                for line in f:
                    # 1. 去除行尾的换行符和空白
                    clean_line = line.strip()
                    # 2. 确保行不为空
                    if clean_line:
                        # 3. 去除后缀 (例如 .png)
                        file_id = clean_line.split('.')[0]
                        # 4. 加入集合,只能传一个参数可以传一个元组数
                        metas.append((scan,file_id))

        print("dataset", self.mode, "metas:", len(metas))
        return metas

    def __len__(self):
        return len(self.metas)

    def read_whu_cam_big(self, filename):
        try:
            with open(filename, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]

            if lines[0] != 'intrinsic':
                raise ValueError(f"文件 {filename} 并非大场景参数格式（缺少 intrinsic 关键字）。")

            # --- 1. 读取内参 ---
            int_params = list(map(float, lines[1].split()))
            f_val = int_params[0]
            x0 = int_params[1]
            y0 = int_params[2]
            # 动态获取当前切片的原始物理尺寸 (用于后续计算内参缩放比例)
            original_w = int(int_params[4])
            original_h = int(int_params[3])

            intrinsics = np.array([
                [-f_val, 0, x0],
                [0, f_val, y0],
                [0, 0, 1]
            ], dtype=np.float32)

            # --- 2. 读取外参 ---
            idx_ext = lines.index('extrinsic')
            extrinsics = []
            for i in range(idx_ext + 1, idx_ext + 5):
                extrinsics.append(list(map(float, lines[i].split())))
            extrinsics = np.array(extrinsics, dtype=np.float32)

            # --- 3. 读取深度范围 (最后一行) ---
            depth_line = list(map(float, lines[-1].split()))
            depth_min = depth_line[0]
            depth_max = depth_line[1]
            # 大场景最后一行可能没有 interval，做个安全处理
            depth_interval = depth_line[2] if len(depth_line) > 2 else (depth_max - depth_min) / 192.0

            return intrinsics, extrinsics, depth_min, depth_max, depth_interval, original_w, original_h


        except Exception as e:
            # --- 这里的代码会在出错时执行 ---
            print("\n" + "=" * 50)
            print(f"❌ [严重错误] 读取相机参数文件失败！")
            print(f"📂 出错文件路径 filename: {filename}")
            print(f"⚠️ 错误具体信息: {e}")
            print("建议: 请检查该文件是否为空，或内容是否损坏。")
            print("=" * 50 + "\n")
            # 抛出异常终止程序，防止使用错误数据继续训练
            raise e


    def read_cam_file(self, filename,depth_filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))

        depth_min = float(lines[11].split()[0])

        # 1. 读取 PFM
        depth_hr = np.array(read_pfm(depth_filename)[0], dtype=np.float32)
        # 2. 维度处理
        if depth_hr.ndim == 3:
            depth_hr = np.squeeze(depth_hr, 2)

        depth_max = depth_hr.max()

        return intrinsics, extrinsics, depth_min, depth_max

    def read_img(self, filename):
        img = Image.open(filename)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        np_img = cv2.resize(np_img, self.img_wh, interpolation=cv2.INTER_LINEAR)

        h, w, _ = np_img.shape

        np_img_ms = {
            "stage_3": cv2.resize(np_img, (w // 8, h // 8), interpolation=cv2.INTER_LINEAR),
            "stage_2": cv2.resize(np_img, (w // 4, h // 4), interpolation=cv2.INTER_LINEAR),
            "stage_1": cv2.resize(np_img, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR),
            "stage_0": np_img
        }
        return np_img_ms

    def read_depth_hr(self, filename, depth_min=0.0):
        """
        读取 whuMVS 数据集的 PNG 深度图，应用 whuMVS 规则转换，生成多尺度掩码。
        """
        # 1. 【核心修改】读取 16-bit PNG 深度图
        # 注意：必须使用 cv2.IMREAD_UNCHANGED (-1) 才能正确读取 16位 uint
        depth_png = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

        if depth_png is None:
            raise FileNotFoundError(f"Depth file not found: {filename}")

        # 2. 【核心修改】转换数值
        # 原始数据是 uint16，先转为 float32
        depth_hr = depth_png.astype(np.float32)

        # 根据 whuMVS 规则: TRUE_DEPTH = STORED_DEPTH / 64.0
        depth_hr = depth_hr / 64.0

        # 3. 维度处理：极度鲁棒版
        if depth_hr.ndim == 3:
            if depth_hr.shape[2] == 1:
                # 如果是 [H, W, 1]，正常挤压
                depth_hr = np.squeeze(depth_hr, 2)
            else:
                # 兼容处理：如果深度图被错误地保存成了RGB(3通道)或RGBA(4通道)
                # 强制只取第一个通道 (因为灰度图的 R=G=B)
                depth_hr = depth_hr[:, :, 0]

        # 4. 数据清洗 (NaN/Inf -> 0)
        # 虽然 PNG 转 float 不太会出现 NaN，但保留此步作为防御性编程
        depth_hr = np.nan_to_num(depth_hr, nan=0.0, posinf=0.0, neginf=0.0)

        # 5. 生成掩码 (whuMVS 中 0 通常代表无效值，结合 depth_min 使用)
        mask_hr = (depth_hr > depth_min).astype(np.float32)

        # 6. 计算 depth_max
        # 这里计算的是当前视图的最大有效深度，用于后续可能的归一化或范围设定
        depth_max = depth_hr.max()

        # ------------------------------------------------------------------
        # 以下部分保持你原有的逻辑不变，进行多尺度下采样
        # ------------------------------------------------------------------
        h, w = depth_hr.shape

        # 注意：OpenCV 的 resize 传入尺寸是 (width, height)
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
        # 这里是对应的一组数据子数据集名称和里面的图片名
        # 只测试 图1
        meta = self.metas[idx]
        scan, file_id = meta
        # whu数据集每个参考图和源图已经固定了

        view_ids=[1,0,2,3,4]


        imgs_0 = []
        imgs_1 = []
        imgs_2 = []
        imgs_3 = []

        mask = None
        depth = None
        depth_min = None
        depth_max = None


        proj_matrices_0 = []
        proj_matrices_1 = []
        proj_matrices_2 = []
        proj_matrices_3 = []

        intrinsics_matrices_0 = []
        intrinsics_matrices_1 = []
        intrinsics_matrices_2 = []
        intrinsics_matrices_3 = []

        # 装载cdt三角剖分数据
        # 顶点坐标集合：[(x1, y1), (x2, y2), ...]
        # 线集合：[Line(p1, p2, face1, face2), ...]
        # 三角形集合：[Triangle(vertex_ids, line_ids, valid_points), ...]
        cdt_data=[]

        for i, vid in enumerate(view_ids):
            # 转为字符串后补前导零到8位
            # 只有参考图的图片路径有所变化其他的都是一个逻辑
            img_filename=[]
            triangulation_filename=[]
            # 对于参考图进行不一样的处理
            if(vid==1):
                img_filename = os.path.join(self.datapath,"{}/Images/{}/urd/{}.png".format(self.mode,scan,file_id))
                triangulation_filename = os.path.join(self.datapath,
                                                      '{}/Images/{}/triangulation/CDTinfo/CDT_info_vlf_{}.txt'.format(
                                                          self.mode, scan, file_id))
            else:
                img_filename = os.path.join(self.datapath,"{}/Images/{}/{}/{}.png".format(self.mode,scan,vid,file_id))

            depth_filename_hr = os.path.join(self.datapath,
                                             "{}/Depths/{}/{}/{}.png".format(self.mode, scan, vid, file_id))
            proj_mat_filename = os.path.join(self.datapath, "{}/Cams/{}/{}/{}.txt".format(self.mode,scan,vid,file_id))



            imgs = self.read_img(img_filename)
            imgs_0.append(imgs['stage_0'])
            imgs_1.append(imgs['stage_1'])
            imgs_2.append(imgs['stage_2'])
            imgs_3.append(imgs['stage_3'])

            intrinsics, extrinsics, depth_min_, depth_max_, depth_interval, original_w, original_h = self.read_whu_cam_big(proj_mat_filename)

            if vid == 1:  # reference view
                depth_min = depth_min_
                depth_max = depth_max_

                depth, mask, _ = self.read_depth_hr(depth_filename_hr)

                for l in range(self.stages):
                    mask[f'stage_{l}'] = np.expand_dims(mask[f'stage_{l}'], 2)
                    mask[f'stage_{l}'] = mask[f'stage_{l}'].transpose([2, 0, 1])
                    depth[f'stage_{l}'] = np.expand_dims(depth[f'stage_{l}'], 2)
                    depth[f'stage_{l}'] = depth[f'stage_{l}'].transpose([2, 0, 1])

                # ym-add 获取参考图三角网数据
                W = imgs_0[0].shape[1]
                H = imgs_0[0].shape[0]

                cdt_data = get_cdt_datas(triangulation_filename, H=H, W=W)

            intrinsics[0] *= self.img_wh[0] / original_w
            intrinsics[1] *= self.img_wh[1] / original_h

            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat = extrinsics.copy()

            intrinsics[:2, :] *= 0.125
            # 复制在加入不然加入的是同一个元素
            intrs_mat = intrinsics.copy()
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_3.append(proj_mat)
            intrinsics_matrices_3.append(intrs_mat)

            proj_mat = extrinsics.copy()
            intrinsics[:2, :] *= 2
            intrs_mat = intrinsics.copy()
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_2.append(proj_mat)
            intrinsics_matrices_2.append(intrs_mat)

            proj_mat = extrinsics.copy()
            intrinsics[:2, :] *= 2
            intrs_mat = intrinsics.copy()
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_1.append(proj_mat)
            intrinsics_matrices_1.append(intrs_mat)

            proj_mat = extrinsics.copy()
            intrinsics[:2, :] *= 2
            intrs_mat = intrinsics.copy()
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_0.append(proj_mat)
            intrinsics_matrices_0.append(intrs_mat)


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
        proj = {}
        proj['stage_3'] = proj_matrices_3
        proj['stage_2'] = proj_matrices_2
        proj['stage_1'] = proj_matrices_1
        proj['stage_0'] = proj_matrices_0

        # 相机内参矩阵 N*3*3

        intrinsics_matrices_0 = np.stack(intrinsics_matrices_0)
        intrinsics_matrices_1 = np.stack(intrinsics_matrices_1)
        intrinsics_matrices_2 = np.stack(intrinsics_matrices_2)
        intrinsics_matrices_3 = np.stack(intrinsics_matrices_3)

        intrinsics_mats={}
        intrinsics_mats['stage_3'] = intrinsics_matrices_3
        intrinsics_mats['stage_2'] = intrinsics_matrices_2
        intrinsics_mats['stage_1'] = intrinsics_matrices_1
        intrinsics_mats['stage_0'] = intrinsics_matrices_0

        # todo：将数据转化为list or ndarray，为的是后续可以使用，如果之后要进行并行运算还需要修改
        vertexs = np.asarray(cdt_data.vertexs, dtype=np.int64)
        lines = np.asarray(cdt_data.lines, dtype=np.int64)
        # 每个 triangle 分开处理，保留 list，
        triangles = []
        for t in cdt_data.triangles:
            tri_v = np.asarray(t.vertex_ids, dtype=np.int64)
            tri_l = np.asarray(t.line_ids, dtype=np.int64)
            tri_pts = np.asarray(t.valid_points, dtype=np.int64)  # 变长，允许不同长度
            triangles.append({'vertex_ids': tri_v, 'line_ids': tri_l, 'valid_points': tri_pts})


        return {"imgs": imgs,  # N*3*H0*W0
                "proj_matrices": proj,  # N*4*4
                "intrinsics_mats": intrinsics_mats,# N*3*3
                "depth": depth,  # 1*H0 * W0
                "mask": mask,  # 1*H0 * W0
                "depth_min": depth_min,  # scalar
                "depth_max": depth_max,  # scalar
                "filename": scan +'/{}/' + '{}'.format(file_id) + "{}",
                "vertexs": vertexs,  # ndarray (Nv, ..)
                "lines": lines,  # ndarray (Nl, ..)
                "triangles": triangles  # list of ndarrays
                }

