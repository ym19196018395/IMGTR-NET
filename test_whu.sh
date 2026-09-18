#!/usr/bin/env bash

# ==============================================================================
# WHU MVS 独立精度评测与平面掩码导出运行脚本 (minitest.txt)
# ==============================================================================

# 1. 基础路径配置
MVS_TESTING="/home/ym/Experiment/Datas/WHU_MVS_dataset"
TEST_LIST="lists/whu/newtest.txt"
OUT_DIR="./outputs_minitest"

# 2. Checkpoint 配置（优先使用命令行第一个参数，若无则使用默认模型权重）
# 例如: ./test.sh ./checkpoints/tensorboard_train19/model_000033.ckpt
DEFAULT_CKPT="./checkpoints/tensorboard_train24/model_000033.ckpt"
CKPT_FILE="${1:-$DEFAULT_CKPT}"

# 3. GPU 设备配置（默认使用 3 号 GPU，也可通过外部环境变量临时指定：GPU_ID=2 ./test.sh）
export GPU_ID="${GPU_ID:-3}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# 4. 掩码导出配置 (默认 false 开启快速评测模式：不写掩码图像到磁盘以极速跑完测试；若需导出掩码可设置 SAVE_MASKS=true ./test_whu.sh)
SAVE_MASKS="${SAVE_MASKS:-false}"
if [ "${SAVE_MASKS}" = "true" ]; then
    MASK_OPT="--save_masks"
else
    MASK_OPT="--no_masks"
fi

# 5. 日志记录
mkdir -p txt_logs
mkdir -p "${OUT_DIR}"
timestamp=$(date +%Y%m%d_%H%M%S)
LOG_FILE="txt_logs/test_whu_${timestamp}.log"

echo "=============================================================================="
echo "启动 WHU MVS 独立评测系统"
echo "GPU ID         : ${GPU_ID}"
echo "测试数据集路径 : ${MVS_TESTING}"
echo "测试列表       : ${TEST_LIST}"
echo "模型权重       : ${CKPT_FILE}"
echo "输出目录       : ${OUT_DIR}"
echo "掩码生成模式   : $([ "${SAVE_MASKS}" = "true" ] && echo "导出掩码图像 (--save_masks)" || echo "快速测试模式 [不生成掩码] (--no_masks)")"
echo "日志文件       : ${LOG_FILE}"
echo "=============================================================================="

# 6. 执行测试（shift 移除已消费的第一个参数，其余参数透传给 python 脚本）
if [ "$#" -gt 0 ]; then
    shift
fi

python test_whu.py \
    --dataset dtu_whu \
    --testpath "${MVS_TESTING}" \
    --testlist "${TEST_LIST}" \
    --loadckpt "${CKPT_FILE}" \
    --outdir "${OUT_DIR}" \
    --batch_size 1 \
    --n_views 5 \
    ${MASK_OPT} \
    "$@" \
    2>&1 | tee "${LOG_FILE}"
