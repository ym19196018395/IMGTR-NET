#!/usr/bin/env bash

# ==============================================================================
# WHU-MVS 从零训练 CasMVSNet (CVPR 2020) 运行脚本
# 严格遵守 whu-mvs-benchmark-adapter 规范 (轻量过程监控 + 极小验证集 minitest)
# ==============================================================================

# 自动解析当前脚本目录与 PatchmatchNet 项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCHMATCHNET_DIR="${PATCHMATCHNET_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

# 1. 严格进入项目根目录执行，确保工作空间与路径基准完全统一
cd "${PATCHMATCHNET_DIR}" || exit 1

# ==============================================================================
# 2. 关键路径配置 (显式绝对路径锚定，绝不因脚本位置改变而错位)
# ==============================================================================

# CasMVSNet 源码根目录 (优先环境变量，自动兼容服务器多用户路径)
if [ -z "${CASMVSNET_DIR}" ]; then
    CANDIDATES=(
        "/home/myao/CasMVSNet_pl-master"
        "/home/ym/Experiment/CasMVSNet_pl-master"
        "/home/myao/CasMVSNet"
        "/home/ym/Experiment/CasMVSNet"
        "${PATCHMATCHNET_DIR}/../CasMVSNet_pl-master"
        "${PATCHMATCHNET_DIR}/../CasMVSNet"
    )
    for c in "${CANDIDATES[@]}"; do
        if [ -d "$c" ]; then
            CASMVSNET_DIR="$c"
            break
        fi
    done
fi

if [ -z "${CASMVSNET_DIR}" ]; then
    CASMVSNET_DIR="/home/myao/CasMVSNet_pl-master"
fi

# WHU 数据集根目录
MVS_TRAINING="${MVS_TRAINING:-/home/ym/Experiment/Datas/WHU_MVS_dataset}"
TRAIN_LIST="${TRAIN_LIST:-${PATCHMATCHNET_DIR}/lists/whu/newtrain.txt}"

# 验证集 (强制绑定极小集 80 张 minitest.txt，严禁全量评测拖慢训练)
VAL_LIST="${VAL_LIST:-${PATCHMATCHNET_DIR}/lists/whu/minitest.txt}"

# Checkpoint 与 TensorBoard 保存目录 (统一保存在项目根目录下的 checkpoints/)
LOG_DIR="${LOG_DIR:-${PATCHMATCHNET_DIR}/checkpoints/casmvsnet_whu_train}"

# ==============================================================================
# 3. GPU 与训练超参数配置
# ==============================================================================
export GPU_ID="${GPU_ID:-3}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# 训练 Batch Size: CasMVSNet 包含 3 级代价体，显存占用中等，建议设为 2 (若显存吃紧可设为 1)
BATCH_SIZE="${BATCH_SIZE:-3}"
EPOCHS="${EPOCHS:-16}"
LR="${LR:-0.001}"

mkdir -p "${PATCHMATCHNET_DIR}/txt_logs"
mkdir -p "${LOG_DIR}"
timestamp=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${PATCHMATCHNET_DIR}/txt_logs/train_whu_casmvsnet_${timestamp}.log"

echo "=============================================================================="
echo "启动 WHU-MVS CasMVSNet 从零训练流程"
echo "GPU ID             : ${GPU_ID}"
echo "PatchmatchNet 目录 : ${PATCHMATCHNET_DIR}"
echo "CasMVSNet 源码目录 : ${CASMVSNET_DIR}"
echo "训练集列表         : ${TRAIN_LIST}"
echo "验证集列表         : ${VAL_LIST} (极速过程监控)"
echo "Epochs             : ${EPOCHS}"
echo "Batch Size         : ${BATCH_SIZE}"
echo "学习率 (LR)        : ${LR}"
echo "权重保存目录       : ${LOG_DIR}"
echo "日志输出文件       : ${LOG_FILE}"
echo "=============================================================================="

# 4. 执行训练
python "${SCRIPT_DIR}/train_whu_casmvsnet.py" \
    --dataset dtu_whu \
    --trainpath "${MVS_TRAINING}" \
    --trainlist "${TRAIN_LIST}" \
    --vallist "${VAL_LIST}" \
    --casmvsnet_code_dir "${CASMVSNET_DIR}" \
    --logdir "${LOG_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --lr_sche onecycle \
    --n_views 5 \
    --eval_freq 2 \
    --save_freq 1 \
    --n_depths 8 32 48 \
    --interval_ratios 1.0 2.0 4.0 \
    --num_groups 1 \
    "$@" 2>&1 | tee "${LOG_FILE}"

echo "=============================================================================="
echo "CasMVSNet 训练流程已全部顺利完成！"
echo "  - 模型权重保存目录: ${LOG_DIR}"
echo "  - 完整执行日志: ${LOG_FILE}"
echo "=============================================================================="
