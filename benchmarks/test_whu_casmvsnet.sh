#!/usr/bin/env bash
# ==============================================================================
# WHU-MVS 基准评测执行脚本: CasMVSNet (Cascade Cost Volume MVSNet, CVPR 2020)
# 遵循 whu-mvs-benchmark-adapter 规范 (Mode A: 外挂包装器模式)
# ==============================================================================
set -e

# 获取 PatchmatchNet-new 项目根目录绝对路径 (基于当前脚本所在 benchmarks/ 目录锚定)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCHMATCHNET_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 将主仓目录加入 PYTHONPATH
export PYTHONPATH="${PATCHMATCHNET_DIR}:${PYTHONPATH}"

# GPU 设定 (默认 GPU 0，可通过外部环境变量 GPU_ID=1 bash ... 灵活指定)
GPU_ID=${GPU_ID:-0}
export CUDA_VISIBLE_DEVICES=${GPU_ID}

echo "=============================================================================="
echo "WHU-MVS 基准适配评测启动器: CasMVSNet (CVPR 2020)"
echo "当前执行主机: $(hostname) | 物理 GPU: ${GPU_ID}"
echo "项目主目录: ${PATCHMATCHNET_DIR}"
echo "=============================================================================="

# 1. 自动探测 CasMVSNet 源码路径
CASMVSNET_DIR=${CASMVSNET_DIR:-""}
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

if [ -n "${CASMVSNET_DIR}" ] && [ -d "${CASMVSNET_DIR}" ]; then
    echo "  -> 挂载 CasMVSNet 源码目录: ${CASMVSNET_DIR}"
    export PYTHONPATH="${CASMVSNET_DIR}:${PYTHONPATH}"
else
    echo "  -> [Warning] 未找到 CasMVSNet 源码目录，请通过环境变量 CASMVSNET_DIR 指定！"
fi

# 2. 检查 Checkpoint 路径
# 优先从命令行首个参数 $1 传入，否则自动探测常见路径
DEFAULT_CKPT=${1:-""}
if [ -z "${DEFAULT_CKPT}" ]; then
    CKPT_CANDIDATES=(
        "/home/myao/CasMVSNet_pl-master/ckpts/exp2/_ckpt_epoch_10.ckpt"
        "/home/ym/Experiment/CasMVSNet_pl-master/ckpts/exp2/_ckpt_epoch_10.ckpt"
        "/home/myao/CasMVSNet_pl-master/ckpts/_ckpt_epoch_10.ckpt"
        "${CASMVSNET_DIR}/ckpts/exp2/_ckpt_epoch_10.ckpt"
        "${CASMVSNET_DIR}/ckpts/_ckpt_epoch_10.ckpt"
    )
    for ck in "${CKPT_CANDIDATES[@]}"; do
        if [ -f "$ck" ]; then
            DEFAULT_CKPT="$ck"
            break
        fi
    done
fi

if [ -z "${DEFAULT_CKPT}" ] || [ ! -f "${DEFAULT_CKPT}" ]; then
    echo "🚨 [错误] 未找到 CasMVSNet 权重文件！"
    echo "请在命令行传入权重路径，例如:"
    echo "  bash benchmarks/test_whu_casmvsnet.sh /path/to/casmvsnet_checkpoint.ckpt"
    exit 1
fi

echo "  -> 加载权重 Checkpoint: ${DEFAULT_CKPT}"

# 3. 数据集与输出路径
WHU_DATA_DIR=${WHU_DATA_DIR:-"/home/ym/Experiment/Datas/WHU_MVS_dataset"}
TEST_LIST=${TEST_LIST:-"${PATCHMATCHNET_DIR}/lists/whu/newtest.txt"}
MASK_DIR=${MASK_DIR:-"${PATCHMATCHNET_DIR}/outputs_minitest"}
OUT_DIR=${OUT_DIR:-"${PATCHMATCHNET_DIR}/outputs_casmvsnet"}

LOG_FILE="${OUT_DIR}/test_whu_casmvsnet.log"
mkdir -p "${OUT_DIR}"

echo "  -> 测试数据集路径: ${WHU_DATA_DIR}"
echo "  -> 测试扫描列表: ${TEST_LIST}"
echo "  -> 同源平面掩码目录: ${MASK_DIR}"
echo "  -> 评测结果输出目录: ${OUT_DIR}"
echo "  -> 实时日志文件: ${LOG_FILE}"
echo "=============================================================================="

# 执行 Python 评测脚本
python "${SCRIPT_DIR}/test_whu_casmvsnet.py" \
    --testpath "${WHU_DATA_DIR}" \
    --testlist "${TEST_LIST}" \
    --loadckpt "${DEFAULT_CKPT}" \
    --casmvsnet_code_dir "${CASMVSNET_DIR}" \
    --mask_dir "${MASK_DIR}" \
    --outdir "${OUT_DIR}" \
    --batch_size 1 \
    --n_views 5 \
    --num_workers 4 \
    --n_depths 8 32 48 \
    --interval_ratios 1.0 2.0 4.0 \
    --num_groups 1 \
    2>&1 | tee "${LOG_FILE}"

echo "=============================================================================="
echo "CasMVSNet 评测已顺利完成！"
echo "  - Markdown 评估报告: ${OUT_DIR}/casmvsnet_evaluation_report.md"
echo "  - 详细 JSON 数据: ${OUT_DIR}/casmvsnet_records.json"
echo "  - 完整执行日志: ${LOG_FILE}"
echo "=============================================================================="
