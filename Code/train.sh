#!/bin/bash

# Configuration
MODEL="pointtransformer50" # Change to pointtransformer38 to test the other model
BATCH_SIZE=16  # Reduce if OOM on Kaggle
EPOCH=150
LR=1e-3
GPU=0

# Run training
python train.py \
    --model $MODEL \
    --batch_size $BATCH_SIZE \
    --epoch $EPOCH \
    --lr $LR \
    --opt sgd \
    --gamma 0.1 \
    --decay_epoch 70 120 \
    --val_epoch 5 \
    --weight_decay 5e-4 \
    --gpu $GPU \
    --checkpoint_dir "./checkpoints_$MODEL" \
    --resume