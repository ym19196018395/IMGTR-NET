#!/usr/bin/env bash
# ==============================================================================
# ENRICH-Aerial 测试集 A / B / D 三路对比与 BN 重校准诊断一键启动脚本
# ==============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# 数据集路径探查
DATA_PATH="${DATA_PATH:-/home/myao/ENRICH-Aerial_Cropped_0.6x_Overlap50/}"
if [[ ! -d "${DATA_PATH}" ]]; then
  if [[ -d "/home/myao/ENRICH-Aerial_Data/" ]]; then
    DATA_PATH="/home/myao/ENRICH-Aerial_Data/"
  elif [[ -d "/home/myao/Crop_Test_Dataset/" ]]; then
    DATA_PATH="/home/myao/Crop_Test_Dataset/"
  fi
fi

TRAIN_LIST="${TRAIN_LIST:-lists/enrich_aerial/train.txt}"
TEST_LIST="${TEST_LIST:-lists/enrich_aerial/test_vis.txt}"
OUT_DIR="${OUT_DIR:-./outputs_enrich_aerial_diag}"
SPLIT="${SPLIT:-test}"
RECALIB_SAMPLES="${RECALIB_SAMPLES:-200}"
N_VIEWS="${N_VIEWS:-5}"

# Checkpoint 探查
CKPT_FILE="${1:-}"
if [[ -z "${CKPT_FILE}" ]]; then
  if [[ -f "./checkpoints/tensorboard_enrich_ours_14/model_000032.ckpt" ]]; then
    CKPT_FILE="./checkpoints/tensorboard_enrich_ours_14/model_000032.ckpt"
  elif [[ -d "./checkpoints" ]]; then
    LATEST_CKPT=$(find ./checkpoints -name "*.ckpt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -f2- -d" ")
    if [[ -n "${LATEST_CKPT}" && -f "${LATEST_CKPT}" ]]; then
      CKPT_FILE="${LATEST_CKPT}"
    fi
  fi
fi

if [[ -z "${CKPT_FILE}" || ! -f "${CKPT_FILE}" ]]; then
  echo "🚨 未找到有效的 Checkpoint 文件，请传入参数: bash diagnose_bn_enrich.sh <ckpt_path>"
  exit 1
fi

mkdir -p "${OUT_DIR}"
mkdir -p txt_logs
timestamp=$(date +%Y%m%d_%H%M%S)
LOG_FILE="txt_logs/${timestamp}_ab_diag.log"

echo "=============================================================================="
echo "启动 ENRICH-Aerial 测试集 A / B 对照与精确门控融合评估系统"
echo "  主机名称:       $(hostname)"
echo "  物理 GPU:       ${GPU_ID}"
echo "  测试列表:       ${TEST_LIST}"
echo "  输入视角数:     ${N_VIEWS} 视角"
echo "  Batch Size:     1 (严格单卡单切片)"
echo "  模型权重:       ${CKPT_FILE}"
echo "  输出目录:       ${OUT_DIR}"
echo "  日志保存:       ${LOG_FILE}"
echo "=============================================================================="

# 移去 $1
if [ "$#" -gt 0 ]; then
  shift
fi

python "${PROJECT_DIR}/diagnose_bn_enrich.py" \
  --dataset enrich_aerial \
  --datapath "${DATA_PATH}" \
  --testlist "${TEST_LIST}" \
  --loadckpt "${CKPT_FILE}" \
  --nviews "${N_VIEWS}" \
  --outdir "${OUT_DIR}" \
  --pre_base_depth_thresh 0.18 \
  --post_base_depth_thresh 0.12 \
  "$@" 2>&1 | tee "${LOG_FILE}"

echo "=============================================================================="
echo "评测全部完成！"
echo "  - 报表: ${OUT_DIR}/test_ab_diagnostic_report.md"
echo "  - JSON: ${OUT_DIR}/test_ab_metrics.json"
echo "  - 日志: ${LOG_FILE}"
echo "=============================================================================="
