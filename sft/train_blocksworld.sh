#!/bin/bash
# Training script for Qwen3-1.7B on blocksworld state-action dataset using HuggingFace

set -e  # Exit on error

# Environment setup
export HF_HOME="${HF_HOME:-/nas-ssd2/joykirat/.cache/huggingface}"
export HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-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}"
export HF_TOKEN="${HF_TOKEN:-$HUGGINGFACE_TOKEN}"
export TOKENIZERS_PARALLELISM="true"

# CUDA device configuration
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Count number of GPUs
IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#GPUS[@]}

echo "========================================"
echo "SFT Training for Blocksworld State-Action"
echo "========================================"
echo "Using $NUM_GPUS GPU(s): $CUDA_VISIBLE_DEVICES"
echo "Model: Qwen/Qwen3-1.7B"
echo "Dataset: apiTest/o4-mini_responses_with_state_action_train.json"
echo "========================================"

# Change to the repository root directory
cd "$(dirname "$0")/.."

# Training arguments
MODEL_NAME="Qwen/Qwen3-1.7B"
TRAIN_FILE="apiTest/o4-mini_responses_with_state_action_train.json"
OUTPUT_DIR="./checkpoints/blocksworld_state_action_sft"
MAX_SEQ_LENGTH=""  # Leave empty for auto-detection from data
BATCH_SIZE=1
GRAD_ACCUM=16
LEARNING_RATE=2e-5
NUM_EPOCHS=3
WARMUP_RATIO=0.1

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build max_seq_length argument (only if set)
MAX_SEQ_LENGTH_ARG=""
if [ -n "$MAX_SEQ_LENGTH" ]; then
    MAX_SEQ_LENGTH_ARG="--max_seq_length $MAX_SEQ_LENGTH"
    echo "Using specified max_seq_length: $MAX_SEQ_LENGTH"
else
    echo "Auto-detecting optimal max_seq_length from data..."
fi

# Training command
if [ "$NUM_GPUS" -eq 1 ]; then
    # Single GPU training
    echo "Running single GPU training..."
    python3 sft/train.py \
        --model_name_or_path "$MODEL_NAME" \
        --train_file "$TRAIN_FILE" \
        --output_dir "$OUTPUT_DIR" \
        $MAX_SEQ_LENGTH_ARG \
        --per_device_train_batch_size "$BATCH_SIZE" \
        --per_device_eval_batch_size "$BATCH_SIZE" \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        --learning_rate "$LEARNING_RATE" \
        --num_train_epochs "$NUM_EPOCHS" \
        --warmup_ratio "$WARMUP_RATIO" \
        --lr_scheduler_type cosine \
        --weight_decay 0.01 \
        --max_grad_norm 1.0 \
        --bf16 \
        --gradient_checkpointing \
        --logging_steps 10 \
        --save_strategy epoch \
        --save_total_limit 3 \
        --evaluation_strategy epoch \
        --report_to tensorboard \
        --seed 42 \
        --trust_remote_code \
        --use_flash_attention \
        --dataloader_num_workers 4 \
        "$@"
else
    # Multi-GPU training with torchrun
    echo "Running multi-GPU training with $NUM_GPUS GPUs..."
    torchrun \
        --standalone \
        --nproc_per_node=$NUM_GPUS \
        sft/train.py \
        --model_name_or_path "$MODEL_NAME" \
        --train_file "$TRAIN_FILE" \
        --output_dir "$OUTPUT_DIR" \
        $MAX_SEQ_LENGTH_ARG \
        --per_device_train_batch_size "$BATCH_SIZE" \
        --per_device_eval_batch_size "$BATCH_SIZE" \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        --learning_rate "$LEARNING_RATE" \
        --num_train_epochs "$NUM_EPOCHS" \
        --warmup_ratio "$WARMUP_RATIO" \
        --lr_scheduler_type cosine \
        --weight_decay 0.01 \
        --max_grad_norm 1.0 \
        --bf16 \
        --gradient_checkpointing \
        --logging_steps 10 \
        --save_strategy epoch \
        --save_total_limit 3 \
        --evaluation_strategy epoch \
        --report_to tensorboard \
        --seed 42 \
        --trust_remote_code \
        --use_flash_attention \
        --dataloader_num_workers 4 \
        --ddp_find_unused_parameters False \
        "$@"
fi

echo "========================================"
echo "Training completed!"
echo "Model saved to: $OUTPUT_DIR"
echo "========================================"
