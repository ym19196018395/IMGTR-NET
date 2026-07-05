# 🌟 全局评估配置 🌟
# 用于控制所有评估脚本中的平面判定阈值，保证 Dataset 和 Eval 脚本绝对同步

# 1. 全局(GT)真值平面置信度阈值 (用于提取/筛选高质量真值平面, 参与所有全局平面指标计算)
GT_PLANAR_CONF_THRESHOLD = 0.8

# 2. 全局(GT)真值平面最大容忍重投影误差 (大于此误差即使置信度高也会被降级)
GT_PLANAR_ERR95_THRESHOLD = 0.08

# 3. 预测平面置信度阈值 (专门用于单独可视化网络预测平面的 MAE 图, 如 _dif_pred_planar.png)
PRED_PLANAR_CONF_THRESHOLD = 0.8
