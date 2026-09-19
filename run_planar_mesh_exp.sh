#!/usr/bin/env bash

# ==============================================================================
# WHU MVS 大场景 (dtu_whu_eval_big, bigtest.txt) 平面点云、Mesh 面片导出与浮动诊断脚本
# ==============================================================================

MVS_TESTING="/home/ym/Experiment/Datas/WHU_MVS_dataset"
TEST_LIST="lists/whu/bigtest.txt"
DEFAULT_CKPT="/home/ym/Experiment/PatchmatchNet-new/checkpoints/tensorboard_train26/model_000037.ckpt"
CKPT_FILE="${1:-$DEFAULT_CKPT}"

export GPU_ID="${GPU_ID:-3}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

OUT_DIR="./outputs_big_planar_comparison"
mkdir -p "${OUT_DIR}"

echo "=============================================================================="
echo "启动 WHU MVS 大场景平面几何与深度浮动对比实验"
echo "GPU ID         : ${GPU_ID}"
echo "数据集类型     : dtu_whu_eval_big"
echo "数据集路径     : ${MVS_TESTING}"
echo "测试列表       : ${TEST_LIST}"
echo "模型权重       : ${CKPT_FILE}"
echo "输出目录       : ${OUT_DIR}"
echo "=============================================================================="

python eval_whu_big.py \
    --dataset dtu_whu_eval_big \
    --testpath "${MVS_TESTING}" \
    --testlist "${TEST_LIST}" \
    --loadckpt "${CKPT_FILE}" \
    --outdir "${OUT_DIR}" \
    --batch_size 1 \
    --n_views 5 \
    --geo_pixel_thres 3.0 \
    --geo_depth_thres 0.05 \
    --photo_thres 0.8 \
    "${@:2}"
