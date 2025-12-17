#!/bin/bash

# Settings
MODEL="pointtransformer38" # Options: pointtransformer38, pointtransformer50
GPU_ID=0
BATCH_SIZE=16
EPOCHS=150

echo "Starting training for $MODEL on GPU $GPU_ID..."

# Make sure data link exists
if [ ! -d "data" ]; then
  mkdir -p data
  # Assuming dataset_root is available in environment or hardcoded
  # ln -sfn /path/to/dataset data/modelnet40_normal_resampled
fi

python train.py \
    --model $MODEL \
    --batch_size $BATCH_SIZE \
    --epoch $EPOCHS \
    --lr 1e-3 \
    --opt sgd \
    --gamma 0.1 \
    --decay_epoch 70 120 \
    --val_epoch 1 \
    --weight_decay 5e-4 \
    --gpu $GPU_ID \
    --checkpoint_dir "./checkpoints_$MODEL" \
    --normal # Remove this flag if you want XYZ only, keep it for XYZ+Normals

echo "Training finished."