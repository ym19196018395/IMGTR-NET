#!/usr/bin/env bash

# ==============================================================================
# WHU-MVS 基准对比评测运行脚本: GeoMVSNet (CVPR 2023)
# 遵循 whu-mvs-benchmark-adapter 专家规范 (模式 A: 包装器接入)
# ==============================================================================

# 自动解析当前脚本目录与 PatchmatchNet 项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCHMATCHNET_DIR="${PATCHMATCHNET_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

# 1. 严格进入项目根目录执行，确保工作空间与路径基准完全统一
cd "${PATCHMATCHNET_DIR}" || exit 1

# ==============================================================================
# 2. 关键路径配置 (显式绝对路径锚定，绝不因脚本位置改变而错位)
# ==============================================================================

# 外部 GeoMVSNet 源码根目录 (优先环境变量，自动兼容服务器多用户路径)
if [ -z "${GEOMVSNET_DIR}" ]; then
    if [ -d "/home/myao/GeoMVSNet-master" ]; then
        GEOMVSNET_DIR="/home/myao/GeoMVSNet-master"
    elif [ -d "/home/ym/Experiment/GeoMVSNet-master" ]; then
        GEOMVSNET_DIR="/home/ym/Experiment/GeoMVSNet-master"
    else
        GEOMVSNET_DIR="/home/myao/GeoMVSNet-master"
    fi
fi

# GeoMVSNet 预训练模型权重 (.ckpt)
DEFAULT_CKPT="${PATCHMATCHNET_DIR}/checkpoints/geomvsnet_whu_train/model_000013.ckpt"
CKPT_FILE="${1:-$DEFAULT_CKPT}"

# WHU 数据集根目录与测试切片列表
MVS_TESTING="${MVS_TESTING:-/home/ym/Experiment/Datas/WHU_MVS_dataset}"
TEST_LIST="${TEST_LIST:-${PATCHMATCHNET_DIR}/lists/whu/newtest.txt}"

# 本项目已提前导出的同源平面掩码目录 (用于 Table 2 提取同源平面区 MAE)
MASK_DIR="${MASK_DIR:-${PATCHMATCHNET_DIR}/outputs_minitest}"

# 评测输出目录 (保存 Markdown 双表报表和 json 记录，统一生成在项目根目录下)
OUT_DIR="${OUT_DIR:-${PATCHMATCHNET_DIR}/outputs_geomvsnet}"

# ==============================================================================
# 3. GPU 设备与运行参数配置
# ==============================================================================
export GPU_ID="${GPU_ID:-3}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

mkdir -p "${PATCHMATCHNET_DIR}/txt_logs"
mkdir -p "${OUT_DIR}"
timestamp=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${PATCHMATCHNET_DIR}/txt_logs/test_whu_geomvsnet_${timestamp}.log"

echo "=============================================================================="
echo "启动 WHU-MVS GeoMVSNet 基准适配独立评测系统"
echo "GPU ID               : ${GPU_ID}"
echo "PatchmatchNet 目录   : ${PATCHMATCHNET_DIR}"
echo "GeoMVSNet 代码目录   : ${GEOMVSNET_DIR}"
echo "GeoMVSNet 权重路径   : ${CKPT_FILE}"
echo "WHU 测试集路径       : ${MVS_TESTING}"
echo "测试列表             : ${TEST_LIST}"
echo "同源平面掩码目录     : ${MASK_DIR}"
echo "输出结果目录         : ${OUT_DIR}"
echo "终端日志文件         : ${LOG_FILE}"
echo "=============================================================================="

# 消费第一个参数 (CKPT_FILE)，其余参数透传给 python
if [ "$#" -gt 0 ]; then
    shift
fi

# 4. 执行适配评测
python "${SCRIPT_DIR}/test_whu_geomvsnet.py" \
    --dataset dtu_whu \
    --testpath "${MVS_TESTING}" \
    --testlist "${TEST_LIST}" \
    --loadckpt "${CKPT_FILE}" \
    --geomvsnet_code_dir "${GEOMVSNET_DIR}" \
    --mask_dir "${MASK_DIR}" \
    --outdir "${OUT_DIR}" \
    --batch_size 1 \
    --n_views 5 \
    --num_workers 4 \
    "$@" 2>&1 | tee "${LOG_FILE}"

echo "=============================================================================="
echo "评测完成！完整报表已生成至: ${OUT_DIR}/geomvsnet_evaluation_report.md"
echo "=============================================================================="
