#!/bin/bash

# Baseline 실험: 기본 설정 (CrossEntropy Loss, 기본 데이터 증강)
# GPU 1번 사용

export CUDA_VISIBLE_DEVICES=1

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
    --loss_type ce \
    --data_dir ./data

