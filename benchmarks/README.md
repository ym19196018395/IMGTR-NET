# WHU-MVS 对比实验与模型适配基准库 (Benchmarks)

本目录收纳所有外部 Baseline 模型的 WHU-MVS 训练与独立评测适配器，严格遵循 `whu-mvs-benchmark-adapter` 规范（Mode A：包装器接入模式），保证物理单位（meters）、内外参代数流以及同源平面掩码（Table 1 / Table 2）的绝对公平性。

---

## 📂 文件结构与模型映射

```
benchmarks/
├── README.md                     # 本说明文档
├── test_whu_diffmvs.py           # DiffMVS / CasDiffMVS (IEEE TPAMI 2025) 独立评测与双表导出
├── test_whu_diffmvs.sh           # DiffMVS 评测启动脚本
├── train_whu_diffmvs.py          # DiffMVS 从零训练适配器
├── train_whu_diffmvs.sh          # DiffMVS 训练启动脚本
├── test_whu_geomvsnet.py         # GeoMVSNet (CVPR 2023) 独立评测与双表导出
├── test_whu_geomvsnet.sh         # GeoMVSNet 评测启动脚本
├── train_whu_geomvsnet.py        # GeoMVSNet 从零训练适配器
└── train_whu_geomvsnet.sh        # GeoMVSNet 训练启动脚本
```

---

## 🚀 运行方法

所有 Shell 脚本内部均已封装 `SCRIPT_DIR` 与 `PATCHMATCHNET_DIR` 动态解析逻辑：
- 无论是在**项目根目录**运行：`./benchmarks/test_whu_diffmvs.sh <ckpt>`
- 还是在 **`benchmarks/` 目录内**运行：`./test_whu_diffmvs.sh <ckpt>`
均可自动定位项目根目录，日志输出在 `txt_logs/`，报表保存在 `outputs_*/`，绝不污染代码库。

### 1. DiffMVS (IEEE TPAMI 2025)
```bash
# 训练 (默认批次 2, 16 轮)
./benchmarks/train_whu_diffmvs.sh

# 评测 (全量 640 张 newtest.txt)
./benchmarks/test_whu_diffmvs.sh ./checkpoints/diffmvs_whu_train/model_000015.ckpt
```

### 2. GeoMVSNet (CVPR 2023)
```bash
# 训练 (默认批次 2, 16 轮)
./benchmarks/train_whu_geomvsnet.sh

# 评测 (全量 640 张 newtest.txt)
./benchmarks/test_whu_geomvsnet.sh ./checkpoints/geomvsnet_whu_train/model_000013.ckpt
```
