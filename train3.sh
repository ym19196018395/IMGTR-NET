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

python train_whu.py --dataset dtu_whu --batch_size 6 --epochs 18 \
--patchmatch_iteration 1 2 2 --patchmatch_range 6 4 2 \
--patchmatch_num_sample 8 8 16 --propagate_neighbors 0 8 16 --evaluate_neighbors 9 9 9 \
--patchmatch_interval_scale 0.005 0.0125 0.025 \
--trainpath=$MVS_TRAINING --trainlist lists/whu/newtrain.txt --vallist lists/whu/newtest.txt \
--logdir ./checkpoints/tensorboard_cp \
2>&1 | tee txt_logs/${timestamp}开启连续性,光滑,法向量,0.9才阻断传播,进行硬传播,nosimilarity.log \
 "$@"

