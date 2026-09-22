#!/usr/bin/env bash

# test on DTU's evaluation set
#DTU_TESTING="/home/ym/Experiment/Datas/dtu_eval"
#CKPT_FILE="./checkpoints/tensorboard/model_000001.ckpt"
#python eval.py --dataset=dtu_yao_eval --batch_size=1 --n_views 5 \
#--patchmatch_iteration 1 2 2 --patchmatch_range 6 4 2 \
#--patchmatch_num_sample 8 8 16 --propagate_neighbors 0 8 16 --evaluate_neighbors 9 9 9 \
#--patchmatch_interval_scale 0.005 0.0125 0.025 \
#--testpath=$DTU_TESTING --geo_pixel_thres=1 --geo_depth_thres=0.01 --photo_thres 0.8 \
#--outdir=./outputs --testlist lists/dtu/minitest.txt --loadckpt $CKPT_FILE $@




# -------------------------------------------------------------------------------------
# test on eth3d benchmark
#ETH3d_TESTING="/home/ym/Experiment/Datas/eth3d"
# python eval_eth.py --dataset=eth3d --split train --batch_size=1 --n_views 7 \
# --patchmatch_iteration 1 2 2 --patchmatch_range 6 4 2 \
# --patchmatch_num_sample 8 8 16 --propagate_neighbors 0 8 16 --evaluate_neighbors 9 9 9 \
# --patchmatch_interval_scale 0.005 0.0125 0.025 \
# --testpath=$ETH3d_TESTING --geo_pixel_thres=1 --geo_depth_thres=0.01 --photo_thres=0.6 \
# --outdir ./outputs_eth --loadckpt $CKPT_FILE $@

# -------------------------------------------------------------------------------------
# test on tanks & temples
TANK_TESTING="/home/TankandTemples/"
# python eval_tank.py --dataset=tanks --split intermediate --batch_size=1 --n_views 7 \
# --patchmatch_iteration 1 2 2 --patchmatch_range 6 4 2 \
# --patchmatch_num_sample 8 8 16 --propagate_neighbors 0 8 16 --evaluate_neighbors 9 9 9 \
# --patchmatch_interval_scale 0.005 0.0125 0.025 \
# --testpath=$TANK_TESTING --geo_pixel_thres=1 --geo_depth_thres=0.01 \
# --outdir ./outputs_tanks --loadckpt $CKPT_FILE $@

# -------------------------------------------------------------------------------------
# test on your custom dataset
CUSTOM_TESTING="/home/custom/"
# python eval_custom.py --dataset=custom --batch_size=1 --n_views 5 \
# --patchmatch_iteration 1 2 2 --patchmatch_range 6 4 2 \
# --patchmatch_num_sample 8 8 16 --propagate_neighbors 0 8 16 --evaluate_neighbors 9 9 9 \
# --patchmatch_interval_scale 0.005 0.0125 0.025 \
# --testpath=$CUSTOM_TESTING --geo_pixel_thres=1 --geo_depth_thres=0.01 --photo_thres 0.8 \
# --outdir ./outputs_custom --loadckpt $CKPT_FILE $@


# -------------------------------------------------------------------------------------
# WHU Stage-1 融合法向量可视化
#
# 调用方式：
#   bash eval.sh [checkpoint]
# 示例：
#   bash eval.sh ./checkpoints/tensorboard_train40/model_000033.ckpt
# 可调平面门限：
#   NORMAL_PLANAR_THRES=0.85 bash eval.sh [checkpoint]

WHU_TESTING="${WHU_TESTING:-/home/ym/Experiment/Datas/WHU_MVS_dataset}"
WHU_TEST_LIST="${WHU_TEST_LIST:-lists/whu/evaltest.txt}"
WHU_NORMAL_OUT="${WHU_NORMAL_OUT:-./outputs_whu_normal_vis}"
NORMAL_PLANAR_THRES="${NORMAL_PLANAR_THRES:-0.80}"
GPU_ID="${GPU_ID:-2}"
DEFAULT_CKPT="./checkpoints/tensorboard_train40/model_000033.ckpt"
CKPT_FILE="${1:-$DEFAULT_CKPT}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
mkdir -p "${WHU_NORMAL_OUT}"

python eval_whu.py \
    --dataset dtu_whu_eval \
    --testpath "${WHU_TESTING}" \
    --testlist "${WHU_TEST_LIST}" \
    --loadckpt "${CKPT_FILE}" \
    --outdir "${WHU_NORMAL_OUT}" \
    --batch_size 1 \
    --n_views 5 \
    --patchmatch_iteration 1 2 2 \
    --patchmatch_range 6 4 2 \
    --patchmatch_num_sample 8 8 16 \
    --propagate_neighbors 0 8 16 \
    --evaluate_neighbors 9 9 9 \
    --patchmatch_interval_scale 0.005 0.0125 0.025 \
    --geo_pixel_thres 1 \
    --geo_depth_thres 0.01 \
    --photo_thres 0.8 \
    --normal_planar_thres "${NORMAL_PLANAR_THRES}" \
    "${@:2}"
