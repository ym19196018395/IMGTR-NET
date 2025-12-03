#!/usr/bin/env bash

# train on DTU's training set
MVS_TRAINING="/home/ym/Experiment/Datas/mvs_training/dtu"
timestamp=$(date +%Y%m%d_%H%M%S)

python train.py --dataset dtu_yao --batch_size 4 --epochs 2 \
--patchmatch_iteration 1 2 2 --patchmatch_range 6 4 2 \
--patchmatch_num_sample 8 8 16 --propagate_neighbors 0 8 16 --evaluate_neighbors 9 9 9 \
--patchmatch_interval_scale 0.005 0.0125 0.025 \
--trainpath=$MVS_TRAINING --trainlist lists/dtu/train.txt --vallist lists/dtu/test.txt \
--logdir ./checkpoints/tensorboard \
2>&1 | tee txt_logs/train_old_${timestamp}.log \
 "$@"