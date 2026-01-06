#!/bin/bash

# Exp3: Weighted Loss (cat/dog 집중) + 고급 데이터 증강 + Cutout
# GPU 4번 사용

export CUDA_VISIBLE_DEVICES=4

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
    --loss_type weighted \
    --class_weight 2.0 \
    --use_advanced_aug \
    --use_cutout \
    --cutout_length 16 \
    --data_dir ./data

