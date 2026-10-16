#!/usr/bin/env bash
# ==============================================================================
# ENRICH-Aerial 航空遥感大场景深度精度诊断与区域 MAE 排序评测脚本
# 适配统一 eval() 机制 | 5 视角 | BatchSize 2 | 输出各拍摄组/区域排名
# ==============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# GPU 设定 (默认 GPU 0，可通过外部环境变量 GPU_ID=1 bash ... 指定)
GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# 数据集路径候选 (自动探查)
DATA_PATH="${DATA_PATH:-/home/myao/ENRICH-Aerial_Cropped_0.6x_Overlap50/}"
if [[ ! -d "${DATA_PATH}" ]]; then
  if [[ -d "/home/myao/ENRICH-Aerial_Data/" ]]; then
    DATA_PATH="/home/myao/ENRICH-Aerial_Data/"
  elif [[ -d "/home/myao/Crop_Test_Dataset/" ]]; then
    DATA_PATH="/home/myao/Crop_Test_Dataset/"
  fi
fi

TRAIN_LIST="${TRAIN_LIST:-lists/enrich_aerial/train.txt}"
TEST_LIST="${TEST_LIST:-lists/enrich_aerial/test.txt}"
OUT_DIR="${OUT_DIR:-./outputs_enrich_aerial_diag}"
BATCH_SIZE="${BATCH_SIZE:-2}"
N_VIEWS="${N_VIEWS:-5}"
SPLIT="${SPLIT:-both}"   # 'both' (同时评测并对比训练集与测试集), 'test', 'train'
BN_MODE="${BN_MODE:-eval}"   # 'eval' | 'batch' | 'recalib'
RECALIB_BATCHES="${RECALIB_BATCHES:-200}"

# 权重 Checkpoint 寻找机制
CKPT_FILE="${1:-}"
if [[ -z "${CKPT_FILE}" ]]; then
  # 自动寻找 checkpoints 目录下最新的 .ckpt 文件
  if [[ -d "./checkpoints" ]]; then
    LATEST_CKPT=$(find ./checkpoints -name "*.ckpt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -f2- -d" ")
    if [[ -n "${LATEST_CKPT}" && -f "${LATEST_CKPT}" ]]; then
      CKPT_FILE="${LATEST_CKPT}"
    fi
  fi
fi

if [[ -z "${CKPT_FILE}" || ! -f "${CKPT_FILE}" ]]; then
  CKPT_FILE="./checkpoints/tensorboard_enrich_ours_6/model_000033.ckpt"
fi

mkdir -p "${OUT_DIR}"
mkdir -p txt_logs
timestamp=$(date +%Y%m%d_%H%M%S)
LOG_FILE="txt_logs/${timestamp}_eval_diag.log"

echo "=============================================================================="
echo "启动 ENRICH-Aerial 航空遥感深度精度诊断评测系统"
echo "  主机名称:   $(hostname)"
echo "  物理 GPU:   ${GPU_ID}"
echo "  数据集路径: ${DATA_PATH}"
echo "  评估划分:   ${SPLIT} (both=同时同构评测训练集与测试集)"
echo "  测试场景表: ${TEST_LIST}"
echo "  训练场景表: ${TRAIN_LIST}"
echo "  模型权重:   ${CKPT_FILE}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  视角数量:   ${N_VIEWS}"
echo "  输出目录:   ${OUT_DIR}"
echo "  日志保存:   ${LOG_FILE}"
echo "=============================================================================="

# 移去已经读取的 $1 参数
if [ "$#" -gt 0 ]; then
  shift
fi

python "${PROJECT_DIR}/test_enrich_aerial.py" \
  --dataset enrich_aerial \
  --testpath "${DATA_PATH}" \
  --trainlist "${TRAIN_LIST}" \
  --testlist "${TEST_LIST}" \
  --loadckpt "${CKPT_FILE}" \
  --outdir "${OUT_DIR}" \
  --split "${SPLIT}" \
  --batch_size "${BATCH_SIZE}" \
  --n_views "${N_VIEWS}" \
  --num_workers 4 \
  --bn_mode "${BN_MODE}" \
  --recalib_batches "${RECALIB_BATCHES}" \
  "$@" 2>&1 | tee "${LOG_FILE}"

echo "=============================================================================="
echo "评测报告已生成！(BN 模式: ${BN_MODE})"
echo "  - 报表: ${OUT_DIR}/${BN_MODE}/test_summary.md"
echo "  - JSON: ${OUT_DIR}/${BN_MODE}/metrics_diagnostic.json"
echo "  - 日志: ${LOG_FILE}"
echo "=============================================================================="
