#!/usr/bin/env bash
# ==============================================================================
# ENRICH-Aerial_Data 航空遥感数据集独立评测执行脚本
# 适配 2112x1408 超大分辨率与 1 参 2 源 (严格 3 视角)
# ==============================================================================
set -e

# 获取 PatchmatchNet-new 项目根目录
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

# GPU 设定 (默认 GPU 0，可通过外部环境变量 GPU_ID=1 bash ... 灵活指定)
GPU_ID=${GPU_ID:-3}
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# 基础路径配置
DATA_PATH=${DATA_PATH:-"/home/myao/Crop_Test_Dataset"}
LIST_FILE=${LIST_FILE:-"/home/myao/Crop_Test_Dataset/scan_list.txt"}
OUT_DIR=${OUT_DIR:-"./outputs_enrich_aerial_crop"}

# 模型 Checkpoint 路径 (优先取第一个参数，否则使用默认权重)
DEFAULT_CKPT="./checkpoints/tensorboard_train35/model_000033.ckpt"
CKPT_FILE="${1:-$DEFAULT_CKPT}"

# 掩码导出与深度保存配置
SAVE_MASKS="${SAVE_MASKS:-true}"
if [ "${SAVE_MASKS}" = "true" ]; then
    MASK_OPT="--save_masks"
else
    MASK_OPT="--no_masks"
fi

SAVE_DEPTH="${SAVE_DEPTH:-true}"
if [ "${SAVE_DEPTH}" = "true" ]; then
    DEPTH_OPT="--save_depth"
else
    DEPTH_OPT="--no_depth"
fi

mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/test_enrich_aerial.log"

echo "=============================================================================="
echo "启动 ENRICH-Aerial_Data 航空遥感大场景评测系统"
echo "当前主机: $(hostname) | 物理 GPU: ${GPU_ID}"
echo "数据集路径: ${DATA_PATH}"
echo "场景列表:   ${LIST_FILE}"
echo "模型权重:   ${CKPT_FILE}"
echo "输出目录:   ${OUT_DIR}"
echo "日志文件:   ${LOG_FILE}"
echo "=============================================================================="

# 若传入了第一个参数作为 ckpt，则 shift 移除
if [ "$#" -gt 0 ]; then
    shift
fi

python "${PROJECT_DIR}/test_enrich_aerial.py" \
    --dataset enrich_aerial \
    --testpath "${DATA_PATH}" \
    --testlist "${LIST_FILE}" \
    --loadckpt "${CKPT_FILE}" \
    --outdir "${OUT_DIR}" \
    --batch_size 1 \
    --n_views 3 \
    --num_workers 2 \
    ${MASK_OPT} \
    ${DEPTH_OPT} \
    "$@" 2>&1 | tee "${LOG_FILE}"

echo "=============================================================================="
echo "ENRICH-Aerial_Data 评测已顺利完成！"
echo "  - Markdown 评估报表: ${OUT_DIR}/test_summary.md"
echo "  - 详细 JSON 数据:     ${OUT_DIR}/metrics.json"
echo "  - 完整执行日志:       ${LOG_FILE}"
echo "=============================================================================="
