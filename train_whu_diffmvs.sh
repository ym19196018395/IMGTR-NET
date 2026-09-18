#!/usr/bin/env bash

# ==============================================================================
# WHU-MVS 从零训练 DiffMVS / CasDiffMVS (IEEE TPAMI 2025) 运行脚本
# 严格遵守 whu-mvs-benchmark-adapter 规范 (轻量过程监控 + 极小验证集 minitest)
# ==============================================================================

# ==============================================================================
# 1. 服务器实际路径配置 (可在执行前修改或通过外部环境变量传入)
# ==============================================================================
# 本项目代码根目录
PATCHMATCHNET_DIR="${PATCHMATCHNET_DIR:-/home/ym/Experiment/PatchmatchNet-new}"

# DiffMVS 源码根目录
DIFFMVS_DIR="${DIFFMVS_DIR:-/home/ym/Experiment/diffmvs-main}"

# WHU 数据集根目录
MVS_TRAINING="${MVS_TRAINING:-/home/ym/Experiment/Datas/WHU_MVS_dataset}"
TRAIN_LIST="lists/whu/newtrain.txt"

# 验证集 (强制绑定极小集 80 张 minitest.txt，严禁全量评测拖慢训练)
VAL_LIST="lists/whu/minitest.txt"

# Checkpoint 与 TensorBoard 保存目录
LOG_DIR="./checkpoints/diffmvs_whu_train"

# ==============================================================================
# 2. GPU 与训练超参数配置
# ==============================================================================
export GPU_ID="${GPU_ID:-2}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# 训练 Batch Size: DiffMVS 包含扩散去噪迭代，建议设为 2 (若显存吃紧如 16G 显存可设为 1)
BATCH_SIZE="${BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-16}"
LR="${LR:-0.001}"

mkdir -p txt_logs
mkdir -p "${LOG_DIR}"
timestamp=$(date +%Y%m%d_%H%M%S)
LOG_FILE="txt_logs/train_whu_diffmvs_${timestamp}.log"

echo "=============================================================================="
echo "启动 WHU-MVS DiffMVS / CasDiffMVS 从零训练流程"
echo "GPU ID             : ${GPU_ID}"
echo "PatchmatchNet 目录 : ${PATCHMATCHNET_DIR}"
echo "DiffMVS 源码目录   : ${DIFFMVS_DIR}"
echo "训练集列表         : ${TRAIN_LIST}"
echo "验证集列表         : ${VAL_LIST} (极速过程监控)"
echo "Epochs             : ${EPOCHS}"
echo "Batch Size         : ${BATCH_SIZE}"
echo "学习率 (LR)        : ${LR}"
echo "权重保存目录       : ${LOG_DIR}"
echo "日志输出文件       : ${LOG_FILE}"
echo "=============================================================================="

# 3. 检查并进入工作目录
if [ ! -d "${PATCHMATCHNET_DIR}" ]; then
    echo "⚠️ 警告: 未找到指定的 PatchmatchNet 目录: ${PATCHMATCHNET_DIR}"
    echo "请检查并在脚本中配置正确的 PATCHMATCHNET_DIR 路径！"
else
    cd "${PATCHMATCHNET_DIR}" || exit 1
fi

# 4. 执行训练 (后台运行或前台打印)
python train_whu_diffmvs.py \
    --dataset dtu_whu \
    --trainpath "${MVS_TRAINING}" \
    --trainlist "${TRAIN_LIST}" \
    --vallist "${VAL_LIST}" \
    --diffmvs_code_dir "${DIFFMVS_DIR}" \
    --logdir "${LOG_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --lr_sche onecycle \
    --n_views 5 \
    --eval_freq 2 \
    --save_freq 1 \
    --numdepth_initial 48 \
    --numdepth 384 \
    --scale 0.0 0.5 0.1 \
    --sampling_timesteps 0 1 1 \
    --ddim_eta 0 1 1 \
    --stage_iters 1 3 3 \
    --cost_dim_stage 4 4 4 \
    --CostNum 0 4 4 \
    --hidden_dim 0 32 20 \
    --context_dim 32 32 16 \
    --unet_dim 0 16 8 \
    --min_radius 0.125 \
    --max_radius 8.0 \
    --conf_weight 0.05 \
    "$@" 2>&1 | tee "${LOG_FILE}"

echo "=============================================================================="
echo "训练彻底完成！"
echo "最终收敛权重保存在: ${LOG_DIR}/model_$(printf "%06d" $((EPOCHS - 1))).ckpt"
echo "接下来请执行收官独立评测脚本: ./test_whu_diffmvs.sh ${LOG_DIR}/model_$(printf "%06d" $((EPOCHS - 1))).ckpt"
echo "=============================================================================="
