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

python train_whu.py --dataset dtu_whu --batch_size 5 --epochs 10 \
--patchmatch_iteration 1 2 2 --patchmatch_range 6 4 2 \
--patchmatch_num_sample 8 8 16 --propagate_neighbors 0 8 16 --evaluate_neighbors 9 9 9 \
--patchmatch_interval_scale 0.005 0.0125 0.025 \
--trainpath=$MVS_TRAINING --trainlist lists/whu/train.txt --vallist lists/whu/test.txt \
--logdir ./checkpoints/tensorboard_train1 \
2>&1 | tee txt_logs/${timestamp}_1_正常边预测头损失,三次,开启连续,间隙0.4.log \
 "$@"

