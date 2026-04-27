#!/usr/bin/env bash

# train on DTU's training set
#MVS_TRAINING="/home/ym/Experiment/Datas/mvs_training/dtu"
#timestamp=$(date +%Y%m%d_%H%M%S)
#
#python train.py --dataset dtu_yao --batch_size 4 --epochs 2 \
#--patchmatch_iteration 1 2 2 --patchmatch_range 6 4 2 \
#--patchmatch_num_sample 8 8 16 --propagate_neighbors 0 8 16 --evaluate_neighbors 9 9 9 \
#--patchmatch_interval_scale 0.005 0.0125 0.025 \
#--trainpath=$MVS_TRAINING --trainlist lists/dtu/train.txt --vallist lists/dtu/test.txt \
#--logdir ./checkpoints/tensorboard \
#2>&1 | tee txt_logs/train_old_${timestamp}.log \
# "$@"


# train on train_blended training set 主要用这个

#MVS_TRAINING="/home/ym/Experiment/Datas/blendmvs-data"
#timestamp=$(date +%Y%m%d_%H%M%S)

#python train_blended.py --dataset dtu_blended --batch_size 3 --epochs 8 \
#--patchmatch_iteration 1 2 2 --patchmatch_range 6 4 2 \
#--patchmatch_num_sample 8 8 16 --propagate_neighbors 0 8 16 --evaluate_neighbors 9 9 9 \
#--patchmatch_interval_scale 0.005 0.0125 0.025 \
#--trainpath=$MVS_TRAINING --trainlist lists/blended/train.txt --vallist lists/blended/test.txt \
#--logdir ./checkpoints/tensorboard \
#2>&1 | tee txt_logs/train_old_${timestamp}.log \
# "$@"


 # train on train_whu

MVS_TRAINING="/home/ym/Experiment/Datas/WHU_MVS_dataset"
timestamp=$(date +%Y%m%d_%H%M%S)

python train_whu.py --dataset dtu_whu --batch_size 6 --epochs 20 \
--patchmatch_iteration 1 2 2 --patchmatch_range 6 4 2 \
--patchmatch_num_sample 8 8 16 --propagate_neighbors 0 8 16 --evaluate_neighbors 9 9 9 \
--patchmatch_interval_scale 0.005 0.0125 0.025 \
--trainpath=$MVS_TRAINING --trainlist lists/whu/newtrain.txt --vallist lists/whu/minitest.txt \
--logdir ./checkpoints/tensorboard_train2 \
2>&1 | tee txt_logs/${timestamp}_2_端到端,将边预测头仍然是一开始初始化,将传播学习四个参数变为学习三个参数,没有光滑性和连续性和法向量损失,.log \
 "$@"

