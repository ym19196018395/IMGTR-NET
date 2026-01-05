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

    def build_list(self):
        """
        return:metas 里面装着 子数据集的名称和里面每个图片的id
        """
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        # todo：ym_add 1.2 因为之前只将1作为参考图，现在将每一个图片都作为一次参考图
        pair_file = os.path.join(self.datapath, "{}/pair.txt".format(self.mode))

        # read the pair file,这是公共的每一个scan都是一样的
        pair_lines = []
        with open(pair_file) as pair:
            pair_lines = pair.readlines()
            # 过滤掉空行，防止报错
            pair_lines = [line.strip() for line in pair_lines if line.strip()]

        for scan in scans:
            # 储存着每个照片的id
            list_file = os.path.join(self.datapath, "{}/Images/{}/list.txt".format(self.mode,scan))

            self.num_viewpoint = len(pair_lines)  # 直接用行数作为视图总数
            # viewpoints (5)
            for pairline in pair_lines:
                values = pairline.split()
                # 1. 解析参考视图 (每行的第一个数)
                ref_view = int(values[0])
                # 2. 解析源视图 (每行剩下的数)
                # 你的文件格式只有ID没有分数，所以直接取 [1:] 即可
                src_views = [int(x) for x in values[1:]]

                # 读取照片的编号
                with open(list_file) as f:
                    for line in f:
                        # 1. 去除行尾的换行符和空白
                        clean_line = line.strip()
                        # 2. 确保行不为空
                        if clean_line:
                            # 3. 去除后缀 (例如 .png)
                            file_id = clean_line.split('.')[0]
                            # 4. 加入集合,只能传一个参数可以传一个元组数
                            metas.append((scan, ref_view, src_views, file_id))

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

    def __getitem__(self, idx):
        # 这里是对应的一组数据子数据集名称和里面的图片名
        meta = self.metas[idx]
        scan, ref_view, src_views,file_id = meta
        # whu数据集每个参考图和源图已经固定了
        view_ids=[ref_view] + src_views[:self.nviews - 1]

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
                # triangulation_filename = os.path.join(self.datapath,
                #             '{}/Images/{}/triangulation/CDTinfo/CDT_info_vlf_{}.txt'.format(self.mode,scan,file_id))
            else:
                img_filename = os.path.join(self.datapath,"{}/Images/{}/{}/{}.png".format(self.mode,scan,vid,file_id))

            proj_mat_filename = os.path.join(self.datapath, "{}/Cams/{}/{}/{}.txt".format(self.mode,scan,vid,file_id))



            imgs = self.read_img(img_filename)
            imgs_0.append(imgs['stage_0'])
            imgs_1.append(imgs['stage_1'])
            imgs_2.append(imgs['stage_2'])
            imgs_3.append(imgs['stage_3'])

            intrinsics, extrinsics, depth_min_, depth_max_, depth_interval = self.read_whu_cam(proj_mat_filename)

            if vid == 1:  # reference view
                depth_min = depth_min_
                depth_max = depth_max_

                # ym-add 获取参考图三角网数据
                # W = imgs_0[0].shape[1]
                # H = imgs_0[0].shape[0]
                # print("{}--------{}".format(scan,vid+1))
                # cdt_data = get_cdt_datas(triangulation_filename, H=H, W=W)

            intrinsics[0] *= self.img_wh[0] / img_w
            intrinsics[1] *= self.img_wh[1] / img_h

            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat = extrinsics.copy()
            intrinsics[:2, :] *= 0.125
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_3.append(proj_mat)

            proj_mat = extrinsics.copy()
            intrinsics[:2, :] *= 2
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_2.append(proj_mat)

            proj_mat = extrinsics.copy()
            intrinsics[:2, :] *= 2
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_1.append(proj_mat)

            proj_mat = extrinsics.copy()
            intrinsics[:2, :] *= 2
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_0.append(proj_mat)


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

        # vertexs = np.asarray(cdt_data.vertexs, dtype=np.int64)
        # lines = np.asarray(cdt_data.lines, dtype=np.int64)
        # # 每个 triangle 分开处理，保留 list，
        # triangles = []
        # for t in cdt_data.triangles:
        #     tri_v = np.asarray(t.vertex_ids, dtype=np.int64)
        #     tri_l = np.asarray(t.line_ids, dtype=np.int64)
        #     tri_pts = np.asarray(t.valid_points, dtype=np.int64)  # 变长，允许不同长度
        #     triangles.append({'vertex_ids': tri_v, 'line_ids': tri_l, 'valid_points': tri_pts})

        vertexs = []
        lines = []
        triangles = []

        return {"imgs": imgs,  # N*3*H0*W0
                "proj_matrices": proj,  # N*4*4
                "depth_min": depth_min,  # scalar
                "depth_max": depth_max,  # scalar
                "filename": scan + '/{}'.format(ref_view)+'/{}/' + '{}'.format(file_id) + "{}",
                "vertexs": vertexs,  # ndarray (Nv, ..)
                "lines": lines,  # ndarray (Nl, ..)
                "triangles": triangles  # list of ndarrays
                }

