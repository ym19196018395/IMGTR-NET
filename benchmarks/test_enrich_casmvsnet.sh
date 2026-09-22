#!/usr/bin/env bash
# ==============================================================================
# ENRICH-Aerial_Data 航空遥感大场景 CasMVSNet 独立评测执行脚本
# 适配 2112x1408 超大分辨率与 1 参 2 源 (严格 3 视角)
# 自动读取 outputs_enrich_aerial/ 下的同源平面掩码，生成对齐的 Table 1 与 Table 2
# ==============================================================================
set -e
set -o pipefail

# 获取 PatchmatchNet-new 项目根目录绝对路径 (基于 benchmarks/ 目录锚定)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCHMATCHNET_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 将主仓目录加入 PYTHONPATH
export PYTHONPATH="${PATCHMATCHNET_DIR}:${PYTHONPATH}"

# GPU 设定 (默认 GPU 3，可通过外部环境变量 GPU_ID=0 bash ... 灵活指定)
GPU_ID=${GPU_ID:-3}
export CUDA_VISIBLE_DEVICES=${GPU_ID}

echo "=============================================================================="
echo "ENRICH-Aerial_Data 航空遥感大场景 CasMVSNet 评测启动器"
echo "当前执行主机: $(hostname) | 物理 GPU: ${GPU_ID}"
echo "项目主目录:   ${PATCHMATCHNET_DIR}"
echo "=============================================================================="

# 1. 自动探测外部 CasMVSNet 源码路径
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
    echo "  -> 挂载外部 CasMVSNet 源码目录: ${CASMVSNET_DIR}"
else
    echo "  -> [Warning] 未自动找到 CasMVSNet 源码目录，请确保已通过环境变量 CASMVSNET_DIR 指定！"
fi

# 2. 检查 Checkpoint 路径
# 优先从命令行首个参数 $1 传入，否则自动探测常见权重路径
DEFAULT_CKPT=${1:-""}
if [ -z "${DEFAULT_CKPT}" ]; then
    CKPT_CANDIDATES=(
        "checkpoints/casmvsnet_whu_train/model_15.ckpt"
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
    echo "  bash benchmarks/test_enrich_casmvsnet.sh /path/to/casmvsnet_checkpoint.ckpt"
    exit 1
fi

echo "  -> 加载权重 Checkpoint: ${DEFAULT_CKPT}"

# 若传入了第一个参数作为 ckpt，则 shift 移除以便后续参数传递
if [ "$#" -gt 0 ]; then
    shift
fi

# 3. 数据集与输出路径配置
DATA_PATH=${DATA_PATH:-"/home/myao/ENRICH-Aerial_Data"}
LIST_FILE=${LIST_FILE:-"/home/myao/ENRICH-Aerial_Data/scan_list.txt"}
MASK_DIR=${MASK_DIR:-"${PATCHMATCHNET_DIR}/outputs_enrich_aerial"}
OUT_DIR=${OUT_DIR:-"${PATCHMATCHNET_DIR}/outputs_casmvsnet_enrich"}

LOG_FILE="${OUT_DIR}/test_enrich_casmvsnet.log"
mkdir -p "${OUT_DIR}"

echo "  -> 测试数据集路径: ${DATA_PATH}"
echo "  -> 测试场景列表:   ${LIST_FILE}"
echo "  -> 同源平面掩码:   ${MASK_DIR}"
echo "  -> 评测输出目录:   ${OUT_DIR}"
echo "  -> 终端执行日志:   ${LOG_FILE}"
echo "=============================================================================="

# 4. 执行 CasMVSNet 评测主程序
python "${SCRIPT_DIR}/test_enrich_casmvsnet.py" \
    --dataset enrich_aerial \
    --testpath "${DATA_PATH}" \
    --testlist "${LIST_FILE}" \
    --loadckpt "${DEFAULT_CKPT}" \
    --casmvsnet_code_dir "${CASMVSNET_DIR}" \
    --mask_dir "${MASK_DIR}" \
    --outdir "${OUT_DIR}" \
    --batch_size 1 \
    --n_views 3 \
    --num_workers 2 \
    --n_depths 8 32 48 \
    --interval_ratios 1.0 2.0 4.0 \
    --num_groups 1 \
    "$@" 2>&1 | tee "${LOG_FILE}"

echo "=============================================================================="
echo "CasMVSNet ENRICH-Aerial_Data 评测已顺利完成！"
echo "  - Markdown 评估报告: ${OUT_DIR}/casmvsnet_enrich_summary.md"
echo "  - 详细 JSON 数据:     ${OUT_DIR}/casmvsnet_enrich_records.json"
echo "  - 完整执行日志:       ${LOG_FILE}"
echo "=============================================================================="
