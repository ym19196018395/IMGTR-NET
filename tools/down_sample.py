import open3d as o3d
import numpy as np
import os


def downsample_ply_point_cloud(
        input_ply_path: str,
        output_ply_path: str,
        voxel_size: float = 0.05,  # 体素大小（核心参数，越大压缩率越高）
        use_random_sampling: bool = False,  # 备用：是否用随机采样
        random_sample_num: int = 1000000  # 随机采样目标点数（默认100万）
) -> None:
    """
    对PLY格式的点云进行下采样，优先体素下采样（保留几何结构），随机采样为备用

    参数说明：
    - input_ply_path: 输入PLY文件路径（如 "/data/raw_point_cloud.ply"）
    - output_ply_path: 输出下采样后的PLY文件路径（如 "/data/downsampled_point_cloud.ply"）
    - voxel_size: 体素大小（单位：米，需根据点云尺度调整）
                  示例：0.01→轻度压缩，0.05→中度压缩，0.1→重度压缩
    - use_random_sampling: 若为True，忽略体素大小，直接随机采样到指定点数
    - random_sample_num: 随机采样的目标点数（建议10万~1000万，根据需求调整）
    """
    # 1. 检查输入文件是否存在
    if not os.path.exists(input_ply_path):
        raise FileNotFoundError(f"输入文件不存在：{input_ply_path}")

    # 2. 加载点云（Open3D支持大文件流式加载，适配2G文件）
    print(f"正在加载点云文件：{input_ply_path}")
    pcd = o3d.io.read_point_cloud(input_ply_path)
    if not pcd.has_points():
        raise ValueError("加载的点云为空，请检查文件格式是否为PLY")

    # 3. 统计原始点云信息
    original_points_num = len(pcd.points)
    print(f"原始点云点数：{original_points_num:,}（{original_points_num / 1e6:.2f}百万）")

    # 4. 执行下采样
    if use_random_sampling:
        # 随机采样（快速缩减，适合极度压缩，几何特征保留稍差）
        print(f"执行随机下采样，目标点数：{random_sample_num:,}")
        # 若原始点数少于目标点数，直接保存
        if original_points_num <= random_sample_num:
            downsampled_pcd = pcd
        else:
            downsampled_pcd = pcd.random_down_sample(sampling_ratio=random_sample_num / original_points_num)
    else:
        # 体素下采样（推荐！均匀压缩，保留点云整体结构）
        print(f"执行体素下采样，体素大小：{voxel_size}米")
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    # 5. 统计下采样后信息
    downsampled_points_num = len(downsampled_pcd.points)
    compression_ratio = (1 - downsampled_points_num / original_points_num) * 100
    print(f"下采样后点数：{downsampled_points_num:,}（{downsampled_points_num / 1e6:.2f}百万）")
    print(f"压缩比例：{compression_ratio:.2f}%（减少了{original_points_num - downsampled_points_num:,}个点）")

    # 6. 保存下采样后的点云
    o3d.io.write_point_cloud(output_ply_path, downsampled_pcd)
    print(f"下采样后的点云已保存至：{output_ply_path}")

    # 7. 可选：输出文件大小（服务器端验证）
    if os.path.exists(output_ply_path):
        output_size = os.path.getsize(output_ply_path) / (1024 * 1024)  # 转换为MB
        print(f"输出文件大小：{output_size:.2f} MB")


# ------------------------------
# 函数使用示例（服务器端运行）
# ------------------------------
if __name__ == "__main__":
    # 配置参数（根据你的实际路径调整）
    INPUT_PLY = "../outputs/full_3.ply"  # 2G的原始点云路径
    OUTPUT_PLY = "/home/ym/Experiment/PatchmatchNet-new/outputs/downsampled_point_cloud.ply"  # 输出路径

    # 方案1：推荐体素下采样（优先选这个！）
    # 先试voxel_size=0.05，若仍大，调大到0.1；若想保留更多细节，调小到0.01
    downsample_ply_point_cloud(
        input_ply_path=INPUT_PLY,
        output_ply_path=OUTPUT_PLY,
        voxel_size=0.2  # 核心参数，按需调整
    )

    # 方案2：备用随机下采样（极度压缩时用，比如只想保留100万点）
    # downsample_ply_point_cloud(
    #     input_ply_path=INPUT_PLY,
    #     output_ply_path=OUTPUT_PLY,
    #     use_random_sampling=True,
    #     random_sample_num=1000000  # 100万点，文件大小约几十MB
    # )