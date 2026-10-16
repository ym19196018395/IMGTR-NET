#!/usr/bin/env bash
set -euo pipefail

# 可在启动时覆盖：
# GPU_ID=1 MVS_TRAINING=/data/ENRICH-Aerial_Data bash train_enrich_aerial.sh
if [[ -z "${MVS_TRAINING:-}" ]]; then
  DATASET_CANDIDATES=(
    "/home/myao/ENRICH-Aerial_Cropped_0.6x_Overlap50/"
    "/home/myao/ENRICH-Aerial_Data/"
  )
  MVS_TRAINING=""
  for candidate in "${DATASET_CANDIDATES[@]}"; do
    if [[ -d "${candidate}" ]]; then
      MVS_TRAINING="${candidate}"
      break
    fi
  done
fi
GPU_ID="${GPU_ID:-3}"
RUN_NAME="${RUN_NAME:-enrich_ours_15}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-34}"

if [[ -z "${MVS_TRAINING}" || ! -d "${MVS_TRAINING}" ]]; then
  echo "ENRICH-Aerial dataset not found in the known local/server locations."
  echo "Set its real path explicitly, for example:"
  echo "MVS_TRAINING=/path/to/ENRICH-Aerial_Data bash train_enrich_aerial.sh"
  exit 1
fi

mkdir -p txt_logs
timestamp=$(date +%Y%m%d_%H%M%S)
export GPU_ID
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

python train_enrich_aerial.py \
  --trainpath "${MVS_TRAINING}" \
  --valpath "${MVS_TRAINING}" \
  --trainlist lists/enrich_aerial/train.txt \
  --vallist lists/enrich_aerial/test.txt \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --patchmatch_iteration 1 2 2 \
  --patchmatch_range 6 4 2 \
  --patchmatch_num_sample 8 8 16 \
  --propagate_neighbors 0 8 16 \
  --evaluate_neighbors 9 9 9 \
  --patchmatch_interval_scale 0.005 0.0125 0.025 \
  --lrepochs "22,28,30:2" \
  --logdir "./checkpoints/tensorboard_${RUN_NAME}" \
  --clip_mode independent \
  --normal_gt_mode soft \
  --conf_gt_mode hard \
  --pre_base_depth_thresh 0.18 \
  --post_base_depth_thresh 0.12 \
  "$@" \
  2>&1 | tee "txt_logs/${timestamp}_调整学习率,修改数据集平面置信度阈值0.5_0.09${RUN_NAME}.log"
