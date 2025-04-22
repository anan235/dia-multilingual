#!/bin/bash
set -e

export HF_HOME=/workspace/cache/huggingface
export WANDB_PROJECT=multilang-tts

DATA_DIR=/workspace/data
MANIFEST_PATH=/workspace/train_manifest.json
LANG_VOCAB=/workspace/lang_vocab.json

python3 scripts/train_dia.py \
  --manifest $MANIFEST_PATH \
  --lang_vocab $LANG_VOCAB \
  --data_root $DATA_DIR \
  --output_dir /workspace/checkpoints \
  --epochs 50 \
  --batch_size 16 \
  --lr 1e-4 \
  --num_workers 4