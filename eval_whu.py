import argparse
import os

from tensorboard.plugins.hparams.metadata import NULL_TENSOR

os.environ["CUDA_VISIBLE_DEVICES"] = "3" #ym_add 要在torch之前因为要让服务器只看得见第二张卡
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
from datasets import find_dataset_def
from models import *
from utils import *
import sys
from datasets.data_io import read_pfm, save_pfm
import cv2
from plyfile import PlyData, PlyElement
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Predict depth, filter, and fuse')
parser.add_argument('--model', default='PatchmatchNet', help='select model')

parser.add_argument('--dataset', default='dtu_yao_eval', help='select dataset')
parser.add_argument('--testpath', help='testing data path')
parser.add_argument('--testlist', help='testing scan list')

parser.add_argument('--batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--n_views', type=int, default=5, help='num of view')


parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--outdir', default='./outputs', help='output dir')
parser.add_argument('--display', action='store_true', help='display depth images and masks')

parser.add_argument('--patchmatch_iteration', nargs='+', type=int, default=[1,2,2], 
        help='num of iteration of patchmatch on stages 1,2,3')
parser.add_argument('--patchmatch_num_sample', nargs='+', type=int, default=[8,8,16], 
        help='num of generated samples in local perturbation on stages 1,2,3')
parser.add_argument('--patchmatch_interval_scale', nargs='+', type=float, default=[0.005, 0.0125, 0.025], 
        help='normalized interval in inverse depth range to generate samples in local perturbation')
parser.add_argument('--patchmatch_range', nargs='+', type=int, default=[6,4,2], 
        help='fixed offset of sampling points for propogation of patchmatch on stages 1,2,3')
parser.add_argument('--propagate_neighbors', nargs='+', type=int, default=[0,8,16], 
        help='num of neighbors for adaptive propagation on stages 1,2,3')
parser.add_argument('--evaluate_neighbors', nargs='+', type=int, default=[9,9,9], 
        help='num of neighbors for adaptive matching cost aggregation of adaptive evaluation on stages 1,2,3')

parser.add_argument('--geo_pixel_thres', type=float, default=1, help='pixel threshold for geometric consistency filtering')
parser.add_argument('--geo_depth_thres', type=float, default=0.01, help='depth threshold for geometric consistency filtering')
parser.add_argument('--photo_thres', type=float, default=0.8, help='threshold for photometric consistency filtering')

# parse arguments and check
args = parser.parse_args()
print("argv:", sys.argv[1:])
print_args(args)


# read intrinsics and extrinsics
def read_camera_parameters(filename):
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

    # 构建内参矩阵 (注意 whuMVS 的 -f 定义)
    intrinsics = np.array([
        [-f_val, 0, x0],
        [0, f_val, y0],
        [0, 0, 1]
    ], dtype=np.float32)

    return intrinsics, extrinsics



# read an image
def read_img(filename, img_wh):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.float32) / 255.
    np_img = cv2.resize(np_img, img_wh, interpolation=cv2.INTER_LINEAR)
    return np_img


# save a binary mask
def save_mask(filename, mask):
    assert mask.dtype == bool
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)

def save_depth_img(filename, depth):
    # assert mask.dtype == np.bool
    depth = depth.astype(np.float32) * 255
    Image.fromarray(depth).save(filename)


def read_list_file(filename):
    data = []
    with open(filename) as f:
        for line in f:
            # 1. 去除行尾的换行符和空白
            clean_line = line.strip()
            # 2. 确保行不为空
            if clean_line:
                # 3. 去除后缀 (例如 .png)
                file_id = clean_line.split('.')[0]
                # 4. 加入集合,只能传一个参数可以传一个元组数
                data.append(file_id)
    return data

def read_pair_file(filename):
    data = []
    with open(filename) as pair:
        pair_lines = pair.readlines()
        # 过滤掉空行，防止报错
        pair_lines = [line.strip() for line in pair_lines if line.strip()]

        for pairline in pair_lines:
            values = pairline.split()
            # 1. 解析参考视图 (每行的第一个数)
            ref_view = int(values[0])
            # 2. 解析源视图 (每行剩下的数)
            # 你的文件格式只有ID没有分数，所以直接取 [1:] 即可
            src_views = [int(x) for x in values[1:]]
            data.append((ref_view, src_views))
    return data

# run MVS model to save depth maps
def save_depth():
    # dataset, dataloader
    MVSDataset = find_dataset_def(args.dataset)
    test_dataset = MVSDataset(args.testpath, args.testlist, "test", args.n_views)
    # 忘记加入collate_fn
    TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    # model
    model = PatchmatchNet(patchmatch_interval_scale=args.patchmatch_interval_scale,
                propagation_range = args.patchmatch_range, patchmatch_iteration=args.patchmatch_iteration, 
                patchmatch_num_sample = args.patchmatch_num_sample, 
                propagate_neighbors=args.propagate_neighbors, evaluate_neighbors=args.evaluate_neighbors)
    # ym_modify 取消并行处理
    # model = nn.DataParallel(model)
    # model.cuda()
    model.to(device)

    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])
    model.eval()
    
    with torch.no_grad():
        for batch_idx, sample in enumerate(TestImgLoader):
            start_time = time.time()

            skip = ["vertexs", "lines", "triangles"]
            sample_cuda = tocuda(sample, device=device, skip_keys=skip)

            outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], 
                            sample_cuda["depth_min"], sample_cuda["depth_max"],
                            NULL_TENSOR,NULL_TENSOR,NULL_TENSOR)
            
            outputs = tensor2numpy(outputs)
            del sample_cuda
            print('Iter {}/{}, time = {:.3f}'.format(batch_idx, len(TestImgLoader), time.time() - start_time))
            filenames = sample["filename"]

            # save depth maps and confidence maps
            for filename, depth_est, photometric_confidence in zip(filenames, outputs["refined_depth"]['stage_0'],
                                                                outputs["photometric_confidence"]):
                depth_filename = os.path.join(args.outdir, filename.format('depth_est', '.pfm'))
                # 输出一个概率图（Confidence Map），表示“网络觉得这个深度准不准”。
                confidence_filename = os.path.join(args.outdir, filename.format('confidence', '.pfm'))
                os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(confidence_filename.rsplit('/', 1)[0], exist_ok=True)
                # save depth maps
                depth_est = np.squeeze(depth_est, 0)
                save_pfm(depth_filename, depth_est)
                # save confidence maps
                save_pfm(confidence_filename, photometric_confidence)
                


# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src,
                                geo_pixel_thres, geo_depth_thres):
    '''
    几何一致性过滤 (Geometric Consistency),核心部分
    '''
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref,
                                                     depth_src, intrinsics_src, extrinsics_src)
    # print(depth_ref.shape)
    # print(depth_reprojected.shape)
    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    # depth_ref = np.squeeze(depth_ref, 2)
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    mask = np.logical_and(dist < geo_pixel_thres, relative_depth_diff < geo_depth_thres)
    depth_reprojected[~mask] = 0

    return mask, depth_reprojected, x2d_src, y2d_src


def filter_depth(scan_folder, out_folder, plyfilename, geo_pixel_thres, geo_depth_thres, photo_thres, img_wh):
    # the pair file
    pair_file = os.path.join(scan_folder, "cams/pair.txt")
    # for the final point cloud
    vertexs = []
    vertex_colors = []

    pair_data = read_pair_file(pair_file)
    nviews = len(pair_data)
    original_w = 768
    original_h = 384
    

    # for each reference view and the corresponding source views
    for ref_view, src_views in pair_data:
        # load the camera parameters
        ref_intrinsics, ref_extrinsics = read_camera_parameters(
            os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(ref_view)))
        ref_intrinsics[0] *= img_wh[0]/original_w
        ref_intrinsics[1] *= img_wh[1]/original_h
        # load the reference image
        ref_img = read_img(os.path.join(scan_folder, 'blended_images/{:0>8}.jpg'.format(ref_view)), img_wh)
        # load the estimated depth of the reference view
        ref_depth_est = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(ref_view)))[0]
        ref_depth_est = np.squeeze(ref_depth_est, 2)
        # load the photometric mask of the reference view
        confidence = read_pfm(os.path.join(out_folder, 'confidence/{:0>8}.pfm'.format(ref_view)))[0]
        # 如果某点的置信度低于阈值 photo_thres,置信度掩码图
        photo_mask = confidence > photo_thres
        photo_mask = np.squeeze(photo_mask, 2)
        

        all_srcview_depth_ests = []
        

        # compute the geometric mask
        geo_mask_sum = 0
        for src_view in src_views:
            # camera parameters of the source view
            src_intrinsics, src_extrinsics = read_camera_parameters(
                os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(src_view)))
            src_intrinsics[0] *= img_wh[0]/original_w
            src_intrinsics[1] *= img_wh[1]/original_h
            # the estimated depth of the source view
            src_depth_est = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(src_view)))[0]
            

            geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(ref_depth_est, ref_intrinsics, ref_extrinsics,
                                                                      src_depth_est,
                                                                      src_intrinsics, src_extrinsics,
                                                                      geo_pixel_thres, geo_depth_thres)
            geo_mask_sum += geo_mask.astype(np.int32)
            all_srcview_depth_ests.append(depth_reprojected)
            

        depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)
        # at least 3 source views matched
        # large threshold, high accuracy, low completeness
        geo_mask = geo_mask_sum >= 1
        final_mask = np.logical_and(photo_mask, geo_mask)
        

        os.makedirs(os.path.join(out_folder, "mask"), exist_ok=True)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_photo.png".format(ref_view)), photo_mask)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_geo.png".format(ref_view)), geo_mask)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_final.png".format(ref_view)), final_mask)
        os.makedirs(os.path.join(out_folder, "depth_img"), exist_ok=True)


        print("processing {}, ref-view{:0>2}, geo_mask:{:3f} photo_mask:{:3f} final_mask: {:3f}".format(scan_folder, ref_view,
                                                                geo_mask.mean(), photo_mask.mean(), final_mask.mean()))

        if args.display:
            import cv2
            cv2.imshow('ref_img', ref_img[:, :, ::-1])
            cv2.imshow('ref_depth', ref_depth_est / 800)
            cv2.imshow('ref_depth * photo_mask', ref_depth_est * photo_mask.astype(np.float32) / 800)
            cv2.imshow('ref_depth * geo_mask', ref_depth_est * geo_mask.astype(np.float32) / 800)
            cv2.imshow('ref_depth * mask', ref_depth_est * final_mask.astype(np.float32) / 800)
            cv2.waitKey(1)

        height, width = depth_est_averaged.shape[:2]
        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
        
        valid_points = final_mask
        # print("valid_points", valid_points.mean())
        x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]
        
        color = ref_img[valid_points]
        xyz_ref = np.matmul(    np.linalg.inv(ref_intrinsics),
                            np.vstack((x, y, np.ones_like(x))) * depth)
        xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                              np.vstack((xyz_ref, np.ones_like(x))))[:3]
        vertexs.append(xyz_world.transpose((1, 0)))
        vertex_colors.append((color * 255).astype(np.uint8))

        
    vertexs = np.concatenate(vertexs, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)
    print("saving the final model to", plyfilename)


def filter_depth_new( scans ,plyfilename, geo_pixel_thres, geo_depth_thres, photo_thres, img_wh):
    '''
    没有用几何一致性过滤，只用置信度过滤
    '''

    # 用来存放所有子单元生成的点，最后再一次性写入 plyfilename
    all_vertexs = []
    all_colors = []

    for scan in scans:
        scan_folder = os.path.join(args.testpath, "test/Images/{}".format(scan))
        out_folder = os.path.join(args.outdir, scan)

        pair_file = os.path.join(scan_folder, "list.txt")
        # 进行了修改适配whu
        pair_data = read_pair_file(pair_file)

        # 提高光度阈值！因为没有几何校验了，必须把这个调高，否则噪点很多
        # 建议设为 0.8 或 0.9
        FINAL_CONFIDENCE_THRES = 0.8
        print(f"注意：几何一致性已关闭，仅使用置信度过滤 (阈值: {FINAL_CONFIDENCE_THRES})")

        for ref_view in pair_data:
            # ref_view 现在是类似 "006_8/images/1.png" 这样的路径
            # 我们需要解析出它是哪个子文件夹的

            # --- 1. 加载 Ref 数据 ---
            # 路径处理需要根据你的 pair.txt 格式微调
            # 假设 ref_view 是相对路径
            ref_img_path = os.path.join(scan_folder, "urd/{}.png".format(ref_view))
            # 对应的深度图路径 (PatchMatchNet 输出的)
            ref_depth_path = os.path.join(out_folder, "depth_est/{}.pfm".format(ref_view))
            # 对应的置信度路径
            ref_conf_path = os.path.join(out_folder, "confidence/{}.pfm".format(ref_view))
            # 对应的相机参数
            ref_cam_path = os.path.join(args.testpath, "test/Cams/{}/1/{}.txt".format(scan,ref_view))

            if not os.path.exists(ref_depth_path):
                print(f"跳过: 找不到深度图 {ref_depth_path}")
                continue

            ref_img = read_img(ref_img_path, img_wh)
            # 读取方式和dataloader保持一致
            ref_intrinsics, ref_extrinsics = read_camera_parameters(ref_cam_path)

            # 调整内参比例 (如果图片缩放过)
            original_w, original_h = 768, 384  # 假设原图尺寸
            ref_intrinsics[0] *= img_wh[0] / original_w
            ref_intrinsics[1] *= img_wh[1] / original_h

            ref_depth_est = read_pfm(ref_depth_path)[0]
            ref_depth_est = np.squeeze(ref_depth_est, 2)
            confidence = read_pfm(ref_conf_path)[0]
            confidence = np.squeeze(confidence, 2)

            # --- 2. 关键修改：只进行光度过滤 ---
            # 放弃 geometric consistency，因为我们没有 src_views 的深度图

            # 生成掩码：只保留置信度高的点
            final_mask = confidence > FINAL_CONFIDENCE_THRES

            # 保存一下掩码图片方便检查 (可选)
            # save_mask_path = os.path.join(out_folder, "mask", os.path.dirname(ref_view))
            # os.makedirs(save_mask_path, exist_ok=True)
            # save_mask(os.path.join(save_mask_path, "mask_final.png"), final_mask)

            print(f"Processing {ref_view}, valid points: {final_mask.mean():.4f}")

            # --- 3. 反投影生成点云 ---
            height, width = ref_depth_est.shape[:2]
            x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))

            # 应用掩码
            valid_points = final_mask
            x, y, depth = x[valid_points], y[valid_points], ref_depth_est[valid_points]
            color = ref_img[valid_points]

            # 2D -> 3D
            xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                                np.vstack((x, y, np.ones_like(x))) * depth)
            xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                                  np.vstack((xyz_ref, np.ones_like(x))))[:3]

            # 收集当前这块砖的点
            all_vertexs.append(xyz_world.transpose((1, 0)))
            all_colors.append((color * 255).astype(np.uint8))


    # --- 4. 合并保存所有点云 ---
    if not all_vertexs:
        print("没有生成任何有效点云！")
        return

    all_vertexs = np.concatenate(all_vertexs, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)

    # 转换为 PLY 格式所需的结构化数组
    vertexs_tuple = np.array([tuple(v) for v in all_vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    colors_tuple = np.array([tuple(v) for v in all_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs_tuple), vertexs_tuple.dtype.descr + colors_tuple.dtype.descr)
    for prop in vertexs_tuple.dtype.names:
        vertex_all[prop] = vertexs_tuple[prop]
    for prop in colors_tuple.dtype.names:
        vertex_all[prop] = colors_tuple[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)
    print(f"所有碎片已合并，最终模型保存至: {plyfilename}")


def filter_depth_new2( scans ,plyfilename, geo_pixel_thres, geo_depth_thres, photo_thres, img_wh):
    '''
    不计算几何一致性，将参考图和源图深度图直接转化为点云给一个极高的阈值0.9
    '''

    # 用来存放所有子单元生成的点，最后再一次性写入 plyfilename
    all_vertexs = []
    all_colors = []

    for scan in scans:
        # 图片路径
        scan_folder = os.path.join(args.testpath, "test/Images/{}".format(scan))
        # 模型输出结果的路径
        out_folder = os.path.join(args.outdir, scan)
        # 图片ids
        list_file = os.path.join(scan_folder, "list.txt")
        # 进行了修改适配whu
        list_data = read_pair_file(list_file)

        # 提高光度阈值！因为没有几何校验了，必须把这个调高，否则噪点很多
        FINAL_CONFIDENCE_THRES = 0.9
        print(f"注意：几何一致性已关闭，仅使用置信度过滤 (阈值: {FINAL_CONFIDENCE_THRES})")
        # 将各个视角的图片都加入进来
        view_ids=[0,1,2,3,4]
        for ref_view in view_ids:
            for list_id in list_data:

                # --- 1. 加载 Ref 数据 ---
                ref_img_path=[]
                # 因为参考图读取路径不一样
                if ref_view==1:
                    ref_img_path= os.path.join(scan_folder, "urd/{}.png".format(list_id))
                else:
                    ref_img_path = os.path.join(scan_folder, "{}/{}.png".format(ref_view,list_id))

                # 对应的深度图路径 (PatchMatchNet 输出的)
                ref_depth_path = os.path.join(out_folder, "{}/depth_est/{}.pfm".format(ref_view,list_id))
                # 对应的置信度路径
                ref_conf_path = os.path.join(out_folder, "{}/confidence/{}.pfm".format(ref_view,list_id))
                # 对应的相机参数
                ref_cam_path = os.path.join(args.testpath, "test/Cams/{}/{}/{}.txt".format(scan, ref_view,list_id))

                if not os.path.exists(ref_depth_path):
                    print(f"跳过: 找不到深度图 {ref_depth_path}")
                    continue

                ref_img = read_img(ref_img_path, img_wh)
                # 读取方式和dataloader保持一致
                ref_intrinsics, ref_extrinsics = read_camera_parameters(ref_cam_path)

                # 调整内参比例 (如果图片缩放过)
                original_w, original_h = 768, 384  # 假设原图尺寸
                ref_intrinsics[0] *= img_wh[0] / original_w
                ref_intrinsics[1] *= img_wh[1] / original_h

                ref_depth_est = read_pfm(ref_depth_path)[0]
                ref_depth_est = np.squeeze(ref_depth_est, 2)
                confidence = read_pfm(ref_conf_path)[0]
                confidence = np.squeeze(confidence, 2)

                # --- 2. 关键修改：只进行光度过滤 ---
                # 放弃 geometric consistency，因为我们没有 src_views 的深度图

                # 生成掩码：只保留置信度高的点
                final_mask = confidence > FINAL_CONFIDENCE_THRES

                # 保存一下掩码图片方便检查 (可选)
                save_mask_path = os.path.join(out_folder, "mask/{}".format(ref_view), os.path.dirname(list_id))
                os.makedirs(save_mask_path, exist_ok=True)
                save_mask(os.path.join(save_mask_path, "{}.png".format(list_id)), final_mask)

                print(f"Processing {ref_view}, valid points: {final_mask.mean():.4f}")

                # --- 3. 反投影生成点云 ---
                height, width = ref_depth_est.shape[:2]
                x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))

                # 应用掩码
                valid_points = final_mask
                x, y, depth = x[valid_points], y[valid_points], ref_depth_est[valid_points]
                color = ref_img[valid_points]

                # 2D -> 3D
                xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                                    np.vstack((x, y, np.ones_like(x))) * depth)
                xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                                      np.vstack((xyz_ref, np.ones_like(x))))[:3]

                # 收集当前这块砖的点
                all_vertexs.append(xyz_world.transpose((1, 0)))
                all_colors.append((color * 255).astype(np.uint8))


    # --- 4. 合并保存所有点云 ---
    if not all_vertexs:
        print("没有生成任何有效点云！")
        return

    all_vertexs = np.concatenate(all_vertexs, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)

    # 转换为 PLY 格式所需的结构化数组
    vertexs_tuple = np.array([tuple(v) for v in all_vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    colors_tuple = np.array([tuple(v) for v in all_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs_tuple), vertexs_tuple.dtype.descr + colors_tuple.dtype.descr)
    for prop in vertexs_tuple.dtype.names:
        vertex_all[prop] = vertexs_tuple[prop]
    for prop in colors_tuple.dtype.names:
        vertex_all[prop] = colors_tuple[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)
    print(f"所有碎片已合并，最终模型保存至: {plyfilename}")


def filter_depth_new3( scans ,plyfilename, geo_pixel_thres, geo_depth_thres, photo_thres, img_wh):
    '''
    不计算几何一致性，将参考图和源图深度图直接转化为点云给一个极高的阈值0.9
    '''

    # 用来存放所有子单元生成的点，最后再一次性写入 plyfilename
    all_vertexs = []
    all_colors = []

    # pair list
    pair_file = os.path.join(args.testpath, "test/pair.txt")
    pair_data = read_pair_file(pair_file)
    nviews = len(pair_data)

    for scan in scans:
        # 图片路径
        scan_folder = os.path.join(args.testpath, "test/Images/{}".format(scan))
        # 模型输出结果的路径
        out_folder = os.path.join(args.outdir, scan)
        # 图片ids
        list_file = os.path.join(scan_folder, "list.txt")
        # 进行了修改适配whu
        list_data = read_list_file(list_file)

        # 将各个视角的图片都加入进来
        for ref_view, src_views in pair_data:
            for list_id in list_data:

                # --- 1. 加载 Ref 数据 ---
                ref_img_path=[]
                # 因为参考图读取路径不一样
                if ref_view==1:
                    ref_img_path= os.path.join(scan_folder, "urd/{}.png".format(list_id))
                else:
                    ref_img_path = os.path.join(scan_folder, "{}/{}.png".format(ref_view,list_id))

                # 对应的深度图路径 (PatchMatchNet 输出的)
                ref_depth_path = os.path.join(out_folder, "{}/depth_est/{}.pfm".format(ref_view,list_id))
                # 对应的置信度路径
                ref_conf_path = os.path.join(out_folder, "{}/confidence/{}.pfm".format(ref_view,list_id))
                # 对应的相机参数
                ref_cam_path = os.path.join(args.testpath, "test/Cams/{}/{}/{}.txt".format(scan, ref_view,list_id))

                if not os.path.exists(ref_depth_path):
                    print(f"跳过: 找不到深度图 {ref_depth_path}")
                    continue

                ref_img = read_img(ref_img_path, img_wh)
                # 读取方式和dataloader保持一致
                ref_intrinsics, ref_extrinsics = read_camera_parameters(ref_cam_path)

                # 调整内参比例 (如果图片缩放过)
                original_w, original_h = 768, 384  # 假设原图尺寸
                ref_intrinsics[0] *= img_wh[0] / original_w
                ref_intrinsics[1] *= img_wh[1] / original_h

                ref_depth_est = read_pfm(ref_depth_path)[0]
                ref_depth_est = np.squeeze(ref_depth_est, 2)
                # 读取光度置信度
                confidence = read_pfm(ref_conf_path)[0]

                # --- 2. 获取source图,并计算几何和光度一致性 ---

                # 生成掩码：只保留置信度高的点
                photo_mask = confidence > photo_thres
                photo_mask = np.squeeze(photo_mask, 2)

                all_srcview_depth_ests = []

                # compute the geometric mask
                geo_mask_sum = 0

                for src_view in src_views:
                    # camera parameters of the source view
                    source_cam_path = os.path.join(args.testpath, "test/Cams/{}/{}/{}.txt".format(scan, src_view,list_id))
                    src_intrinsics, src_extrinsics = read_camera_parameters(source_cam_path)
                    src_intrinsics[0] *= img_wh[0] / original_w
                    src_intrinsics[1] *= img_wh[1] / original_h
                    # the estimated depth of the source view
                    source_depth_path=os.path.join(out_folder, "{}/depth_est/{}.pfm".format(src_view,list_id))
                    src_depth_est = read_pfm(source_depth_path)[0]

                    geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(ref_depth_est,
                                                                                                ref_intrinsics,
                                                                                                ref_extrinsics,
                                                                                                src_depth_est,
                                                                                                src_intrinsics,
                                                                                                src_extrinsics,
                                                                                                geo_pixel_thres,
                                                                                                geo_depth_thres)
                    geo_mask_sum += geo_mask.astype(np.int32)
                    all_srcview_depth_ests.append(depth_reprojected)

                depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)
                # at least 3 source views matched
                # large threshold, high accuracy, low completeness
                geo_mask = geo_mask_sum >= 1
                final_mask = np.logical_and(photo_mask, geo_mask)

                # 保存一下掩码图片方便检查 (可选)

                os.makedirs(os.path.join(out_folder, "mask/{}".format(ref_view)), exist_ok=True)
                save_mask(os.path.join(out_folder, "mask/{}/{}_photo.png".format(ref_view,list_id)), photo_mask)
                save_mask(os.path.join(out_folder, "mask/{}/{}_geo.png".format(ref_view,list_id)), geo_mask)
                save_mask(os.path.join(out_folder, "mask/{}/{}_final.png".format(ref_view,list_id)), final_mask)
                os.makedirs(os.path.join(out_folder, "depth_img"), exist_ok=True)

                print("processing {}, ref-view{:0>2}, geo_mask:{:3f} photo_mask:{:3f} final_mask: {:3f}".format(
                    scan_folder, ref_view,
                    geo_mask.mean(), photo_mask.mean(), final_mask.mean()))

                # --- 3. 反投影生成点云 ---
                height, width = depth_est_averaged.shape[:2]
                x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))

                # 应用掩码
                valid_points = final_mask
                x, y, depth = x[valid_points], y[valid_points], ref_depth_est[valid_points]
                color = ref_img[valid_points]

                # 2D -> 3D
                xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                                    np.vstack((x, y, np.ones_like(x))) * depth)
                xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                                      np.vstack((xyz_ref, np.ones_like(x))))[:3]

                # 收集当前这块砖的点
                all_vertexs.append(xyz_world.transpose((1, 0)))
                all_colors.append((color * 255).astype(np.uint8))


    # --- 4. 合并保存所有点云 ---
    if not all_vertexs:
        print("没有生成任何有效点云！")
        return

    all_vertexs = np.concatenate(all_vertexs, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)

    # 转换为 PLY 格式所需的结构化数组
    vertexs_tuple = np.array([tuple(v) for v in all_vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    colors_tuple = np.array([tuple(v) for v in all_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs_tuple), vertexs_tuple.dtype.descr + colors_tuple.dtype.descr)
    for prop in vertexs_tuple.dtype.names:
        vertex_all[prop] = vertexs_tuple[prop]
    for prop in colors_tuple.dtype.names:
        vertex_all[prop] = colors_tuple[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)
    print(f"所有碎片已合并，最终模型保存至: {plyfilename}")

if __name__ == '__main__':
    # step1. save all the depth maps and the masks in outputs directory
    # save_depth()
    img_wh=(768, 384)
    
    with open(args.testlist) as f:
        scans = f.readlines()
        scans = [line.rstrip() for line in scans]
        
    # 将每一张图都作为参考图，然后经过光度一致性和几何一致性过滤
    # for scan in scans:
    #     scan_folder = os.path.join(args.testpath, scan)
    #     out_folder = os.path.join(args.outdir, scan)
    #     # step2. filter saved depth maps with geometric constraints
    #     filter_depth(scan_folder, out_folder, os.path.join(args.outdir, '{}.ply'.format(scan)),
    #                 args.geo_pixel_thres, args.geo_depth_thres, args.photo_thres, img_wh)

    # now-1 不计算几何一致性，简单的用参考图深度图转化为点云
    # filter_depth_new(scans, os.path.join(args.outdir, 'full.ply'),
    #                 args.geo_pixel_thres, args.geo_depth_thres, args.photo_thres, img_wh)

    # now-2 不计算几何一致性，将参考图和源图深度图直接转化为点云给一个极高的阈值0.9
    # filter_depth_new2(scans, os.path.join(args.outdir, 'full_2.ply'),
    #                 args.geo_pixel_thres, args.geo_depth_thres, args.photo_thres, img_wh)

    # now-3 全部都计算
    filter_depth_new3(scans, os.path.join(args.outdir, 'full_3.ply'),
                    args.geo_pixel_thres, args.geo_depth_thres, args.photo_thres, img_wh)