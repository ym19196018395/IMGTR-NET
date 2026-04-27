from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from datasets.data_io import *
import cv2

from datasets.triangulation import get_cdt_datas


class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, img_wh=(768, 384), **kwargs):
        super(MVSDataset, self).__init__()

        self.stages = 4
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.img_wh = img_wh

        assert self.mode == "test"
        self.metas = self.build_list()

    # def build_list(self):
    #     """
    #     return:metas 里面装着 子数据集的名称和里面每个图片的id
    #     """
    #     metas = []
    #     with open(self.listfile) as f:
    #         scans = f.readlines()
    #         scans = [line.rstrip() for line in scans]
    #
    #     # todo：ym_add 1.2 因为之前只将1作为参考图，现在将每一个图片都作为一次参考图
    #     pair_file = os.path.join(self.datapath, "{}/pair.txt".format(self.mode))
    #
    #     # read the pair file,这是公共的每一个scan都是一样的
    #     pair_lines = []
    #     with open(pair_file) as pair:
    #         pair_lines = pair.readlines()
    #         # 过滤掉空行，防止报错
    #         pair_lines = [line.strip() for line in pair_lines if line.strip()]
    #
    #     for scan in scans:
    #         # 储存着每个照片的id
    #         list_file = os.path.join(self.datapath, "{}/Images/{}/list.txt".format(self.mode,scan))
    #
    #         self.num_viewpoint = len(pair_lines)  # 直接用行数作为视图总数
    #         # viewpoints (5)
    #         for pairline in pair_lines:
    #             values = pairline.split()
    #             # 1. 解析参考视图 (每行的第一个数)
    #             ref_view = int(values[0])
    #             # 2. 解析源视图 (每行剩下的数)
    #             # 你的文件格式只有ID没有分数，所以直接取 [1:] 即可
    #             src_views = [int(x) for x in values[1:]]
    #
    #             # 读取照片的编号
    #             with open(list_file) as f:
    #                 for line in f:
    #                     # 1. 去除行尾的换行符和空白
    #                     clean_line = line.strip()
    #                     # 2. 确保行不为空
    #                     if clean_line:
    #                         # 3. 去除后缀 (例如 .png)
    #                         file_id = clean_line.split('.')[0]
    #                         # 4. 加入集合,只能传一个参数可以传一个元组数
    #                         metas.append((scan, ref_view, src_views, file_id))
    #
    #     print("dataset", self.mode, "metas:", len(metas))
    #     return metas

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

    def read_whu_cam(self,filename):

        try:
            with open(filename, 'r') as f:
                lines = f.readlines()

            # --- 保护措施 1: 预先检查行数 ---
            # 根据 whuMVS 格式，相机文件至少需要包含外参、内参、深度等信息，通常 > 9 行
            # 这里至少保证能读完外参 (前5行)
            if len(lines) < 5:
                raise IndexError(f"文件{filename}行数不足，仅有 {len(lines)} 行，无法解析外参。")

            # --- 1. 读取外参 (Lines 2-5) ---
            extrinsics = []
            for i in range(1, 5):
                # 去除首尾空格
                line_str = lines[i].strip()

                # --- 保护措施 2: 检查是不是空行 ---
                if not line_str:
                    raise ValueError(f"文件{filename}第 {i + 1} 行内容为空，无法解析数值。")

                extrinsics.append(list(map(float, line_str.split())))

            extrinsics = np.array(extrinsics, dtype=np.float32)

            # --- 2. 读取原始内参 (Line 7) ---
            # f, x0, y0
            vals = list(map(float, lines[6].strip().split()))
            f_val = vals[0]
            x0 = vals[1]
            y0 = vals[2]

            # --- 3. 读取深度范围 (Line 9) ---
            depth_line = list(map(float, lines[8].strip().split()))
            depth_min = depth_line[0]
            depth_max = depth_line[1]
            depth_interval = depth_line[2]

            # 对相机内参不需要进行处理
            # --- 4. 【关键步骤】读取切片信息并修正内参 (Line 10) ---
            # 格式: INDEX ROW_CUT COL_CUT ROW_IDX COL_IDX ROW_SHIFT COL_SHIFT
            # crop_info = list(lines[9].strip().split())  # 注意第一个可能是字符串
            #
            # # 转换为数值 (跳过第一个名字)
            # row_cutoff = float(crop_info[1])
            # col_cutoff = float(crop_info[2])
            # row_index = float(crop_info[3])
            # col_index = float(crop_info[4])
            # row_shift = float(crop_info[5])
            # col_shift = float(crop_info[6])
            #
            # # 计算子图左上角在原图的坐标 (Offset)
            # offset_row = row_index * row_shift + row_cutoff  # Y方向偏移
            # offset_col = col_index * col_shift + col_cutoff  # X方向偏移
            #
            # # 修正主点坐标
            # x0_new = x0_ori - offset_col
            # y0_new = y0_ori - offset_row

            # 构建内参矩阵 (注意 whuMVS 的 -f 定义)
            intrinsics = np.array([
                [-f_val, 0, x0],
                [0, f_val, y0],
                [0, 0, 1]
            ], dtype=np.float32)

            return intrinsics, extrinsics, depth_min, depth_max, depth_interval

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

        # 3. 维度处理 (OpenCV 读取灰度图通常是 2D 的，但为了兼容性保留此检查)
        if depth_hr.ndim == 3:
            depth_hr = np.squeeze(depth_hr, 2)

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
        meta = self.metas[idx]
        scan, file_id = meta
        # whu数据集每个参考图和源图已经固定了

        view_ids=[1,0,2,3,4]

        img_w = 768
        img_h = 384

        imgs_0 = []
        imgs_1 = []
        imgs_2 = []
        imgs_3 = []
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
                # 暂时不需要三角剖分数据
                triangulation_filename = os.path.join(self.datapath,
                            '{}/Images/{}/triangulation/CDTinfo/CDT_info_vlf_{}.txt'.format(self.mode,scan,file_id))
            else:
                img_filename = os.path.join(self.datapath,"{}/Images/{}/{}/{}.png".format(self.mode,scan,vid,file_id))

            proj_mat_filename = os.path.join(self.datapath, "{}/Cams/{}/{}/{}.txt".format(self.mode,scan,vid,file_id))
            depth_filename_hr = os.path.join(self.datapath,
                                             "{}/Depths/{}/{}/{}.png".format(self.mode, scan, vid, file_id))


            imgs = self.read_img(img_filename)
            imgs_0.append(imgs['stage_0'])
            imgs_1.append(imgs['stage_1'])
            imgs_2.append(imgs['stage_2'])
            imgs_3.append(imgs['stage_3'])

            intrinsics, extrinsics, depth_min_, depth_max_, depth_interval = self.read_whu_cam(proj_mat_filename)

            if vid == 1:  # reference view
                depth_min = depth_min_
                depth_max = depth_max_

                depth, mask, depth_max = self.read_depth_hr(depth_filename_hr)

                for l in range(self.stages):
                    mask[f'stage_{l}'] = np.expand_dims(mask[f'stage_{l}'], 2)
                    mask[f'stage_{l}'] = mask[f'stage_{l}'].transpose([2, 0, 1])
                    depth[f'stage_{l}'] = np.expand_dims(depth[f'stage_{l}'], 2)
                    depth[f'stage_{l}'] = depth[f'stage_{l}'].transpose([2, 0, 1])

                # ym-add 获取参考图三角网数据
                W = imgs_0[0].shape[1]
                H = imgs_0[0].shape[0]

                cdt_data = get_cdt_datas(triangulation_filename, H=H, W=W)

            intrinsics[0] *= self.img_wh[0] / img_w
            intrinsics[1] *= self.img_wh[1] / img_h

            # 对矩阵进行一个处理，分别求得不同大小图片的投影矩阵
            proj_mat = extrinsics.copy()

            # 将1，2行的系数*scale
            intrinsics[:2, :] *= 0.125
            # 复制在加入不然加入的是同一个元素
            intrs_mat = intrinsics.copy()
            # 求得是投影矩阵 P = K [R|t]  外参矩阵是取三行四列大小的数据
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

        intrinsics_mats = {}
        intrinsics_mats['stage_3'] = intrinsics_matrices_3
        intrinsics_mats['stage_2'] = intrinsics_matrices_2
        intrinsics_mats['stage_1'] = intrinsics_matrices_1
        intrinsics_mats['stage_0'] = intrinsics_matrices_0

        vertexs = np.asarray(cdt_data.vertexs, dtype=np.float32)
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
                "intrinsics_mats": intrinsics_mats,  # N*3*3
                "depth_min": depth_min,  # scalar
                "depth_max": depth_max,  # scalar
                "filename": scan +'/{}/' + '{}'.format(file_id) + "{}",
                "vertexs": vertexs,  # ndarray (Nv, ..)
                "lines": lines,  # ndarray (Nl, ..)
                "triangles": triangles  # list of ndarrays
                }

