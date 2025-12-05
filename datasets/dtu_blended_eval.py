from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from datasets.data_io import *
import cv2


class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, img_wh=(768, 576), **kwargs):
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
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        for scan in scans:
            pair_file = os.path.join(self.datapath,"{}/cams/pair.txt".format(scan))
            # read the pair file
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
        meta = self.metas[idx]
        scan, ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]
        img_w = 768
        img_h = 576

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

            intrinsics, extrinsics, depth_min_, depth_max_ = self.read_cam_file(proj_mat_filename,depth_filename_hr)

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

            if i == 0:  # reference view
                depth_min = depth_min_
                depth_max = depth_max_

                # ym-add 获取参考图三角网数据 这里vid需要加1 因为是从零开始
                # img_id="CDT_info_vlf_rect_{:03d}_1_r5000".format(vid+1)
                # W=imgs_0[0].shape[1]
                # H=imgs_0[0].shape[0]
                # # print("{}--------{}".format(scan,vid+1))
                # cdt_data = get_cdt_datas(img_id, triangulation_filename,H=H,W=W)

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
                "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}",
                "vertexs": vertexs,  # ndarray (Nv, ..)
                "lines": lines,  # ndarray (Nl, ..)
                "triangles": triangles  # list of ndarrays
                }

