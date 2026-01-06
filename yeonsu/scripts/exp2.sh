#!/bin/bash

# Exp2: Focal Loss + 고급 데이터 증강 + Cutout + Mixup
# GPU 3번 사용

export CUDA_VISIBLE_DEVICES=3

python train.py \
    --epochs 150 \
    --lr 0.1 \
    --momentum 0.9 \
    --weight_decay 5e-4 \
    --scheduler step \
    --step_size 30 \
    --gamma 0.1 \
    --batch_size 128 \
    --num_workers 4 \
    --save_freq 10 \
    --loss_type focal \
    --focal_alpha 1.0 \
    --focal_gamma 2.0 \
    --use_advanced_aug \
    --use_cutout \
    --cutout_length 16 \
    --use_mixup \
    --mixup_alpha 1.0 \
    --data_dir ./data

