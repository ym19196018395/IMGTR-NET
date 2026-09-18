#!/usr/bin/env bash

# ==============================================================================
# WHU-MVS 基准对比评测运行脚本: DiffMVS / CasDiffMVS (IEEE TPAMI 2025)
# 遵循 whu-mvs-benchmark-adapter 专家规范 (模式 A: 包装器接入与同源切片)
# ==============================================================================

# ==============================================================================
# 1. 关键路径配置 (可在执行前通过环境变量覆盖，或直接在此修改默认值)
# ==============================================================================

# 本项目 PatchmatchNet 代码根目录 (服务器实际路径)
PATCHMATCHNET_DIR="${PATCHMATCHNET_DIR:-/home/ym/Experiment/PatchmatchNet-new}"

# 外部 DiffMVS 源码根目录 (服务器实际路径)
DIFFMVS_DIR="${DIFFMVS_DIR:-/home/ym/Experiment/diffmvs-main}"

# DiffMVS 模型权重 (.ckpt)
DEFAULT_CKPT="./checkpoints/diffmvs_whu_train/model_000015.ckpt"
CKPT_FILE="${1:-$DEFAULT_CKPT}"

# WHU 数据集根目录与测试切片列表 (默认绑定 640 张标准全量测试集)
MVS_TESTING="${MVS_TESTING:-/home/ym/Experiment/Datas/WHU_MVS_dataset}"
TEST_LIST="${TEST_LIST:-lists/whu/newtest.txt}"

# 本项目已提前导出的同源平面掩码目录 (用于 Table 2 提取同源平面区 MAE)
# 若之前运行过 test_whu.sh 并开启了 --save_masks，可直接指定其 outputs 目录
MASK_DIR="${MASK_DIR:-${PATCHMATCHNET_DIR}/outputs_minitest}"

# 评测输出目录 (保存 Markdown 双表报表和 json 详细记录)
OUT_DIR="./outputs_diffmvs"

# ==============================================================================
# 2. GPU 设备与运行参数配置
# ==============================================================================
export GPU_ID="${GPU_ID:-3}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

mkdir -p txt_logs
mkdir -p "${OUT_DIR}"
timestamp=$(date +%Y%m%d_%H%M%S)
LOG_FILE="txt_logs/test_whu_diffmvs_${timestamp}.log"

echo "=============================================================================="
echo "启动 WHU-MVS DiffMVS / CasDiffMVS 基准适配独立评测系统"
echo "GPU ID               : ${GPU_ID}"
echo "PatchmatchNet 目录   : ${PATCHMATCHNET_DIR}"
echo "DiffMVS 代码目录     : ${DIFFMVS_DIR}"
echo "DiffMVS 权重路径     : ${CKPT_FILE}"
echo "WHU 测试集路径       : ${MVS_TESTING}"
echo "测试列表             : ${TEST_LIST}"
echo "同源平面掩码目录     : ${MASK_DIR}"
echo "输出结果目录         : ${OUT_DIR}"
echo "终端日志文件         : ${LOG_FILE}"
echo "=============================================================================="

# 3. 检查并进入工作目录
if [ ! -d "${PATCHMATCHNET_DIR}" ]; then
    echo "⚠️ 警告: 未找到指定的 PatchmatchNet 目录: ${PATCHMATCHNET_DIR}"
    echo "请检查并在脚本中配置正确的 PATCHMATCHNET_DIR 路径！"
else
    cd "${PATCHMATCHNET_DIR}" || exit 1
fi

# 消费第一个参数 (CKPT_FILE)，其余参数透传给 python
if [ "$#" -gt 0 ]; then
    shift
fi

# 4. 执行适配评测
python test_whu_diffmvs.py \
    --dataset dtu_whu \
    --testpath "${MVS_TESTING}" \
    --testlist "${TEST_LIST}" \
    --loadckpt "${CKPT_FILE}" \
    --diffmvs_code_dir "${DIFFMVS_DIR}" \
    --mask_dir "${MASK_DIR}" \
    --outdir "${OUT_DIR}" \
    --batch_size 1 \
    --n_views 5 \
    --num_workers 4 \
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
    "$@" 2>&1 | tee "${LOG_FILE}"

echo "=============================================================================="
echo "评测完成！完整报表已生成至: ${OUT_DIR}/diffmvs_evaluation_report.md"
echo "=============================================================================="
