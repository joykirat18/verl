#!/bin/bash
# Example training commands for different scenarios

# ===== QUICK START (AUTO-DETECT MAX_SEQ_LENGTH) =====
# Simple single GPU training with auto-detected max_seq_length
# The script will analyze your data and choose optimal length
python sft/train.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --train_file apiTest/o4-mini_responses_with_state_action_train.json \
    --output_dir ./checkpoints/blocksworld_sft \
    --num_train_epochs 3 \
    --bf16 \
    --gradient_checkpointing


# ===== WITH CUSTOM MAX_SEQ_LENGTH =====
# If you want to specify a fixed max_seq_length instead of auto-detection
python sft/train.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --train_file apiTest/o4-mini_responses_with_state_action_train.json \
    --output_dir ./checkpoints/blocksworld_sft \
    --max_seq_length 8192 \
    --num_train_epochs 3 \
    --bf16 \
    --gradient_checkpointing


# ===== CUSTOM AUTO-DETECTION PARAMETERS =====
# Fine-tune the auto-detection: use 99th percentile with 256 buffer
python sft/train.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --train_file apiTest/o4-mini_responses_with_state_action_train.json \
    --output_dir ./checkpoints/blocksworld_sft \
    --max_seq_length_percentile 99.0 \
    --max_seq_length_buffer 256 \
    --num_train_epochs 3 \
    --bf16 \
    --gradient_checkpointing


# ===== SINGLE GPU (with all common options, auto-detect length) =====
python sft/train.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --train_file apiTest/o4-mini_responses_with_state_action_train.json \
    --output_dir ./checkpoints/blocksworld_sft \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --warmup_ratio 0.1 \
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
    --use_flash_attention


# ===== MULTI-GPU (4 GPUs, auto-detect length) =====
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 sft/train.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --train_file apiTest/o4-mini_responses_with_state_action_train.json \
    --output_dir ./checkpoints/blocksworld_sft_4gpu \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --bf16 \
    --gradient_checkpointing \
    --logging_steps 10 \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --ddp_find_unused_parameters False


# ===== LOW MEMORY SETUP (for GPUs with <24GB) =====
# Force a shorter max_seq_length to save memory
python sft/train.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --train_file apiTest/o4-mini_responses_with_state_action_train.json \
    --output_dir ./checkpoints/blocksworld_sft_lowmem \
    --max_seq_length 4096 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --bf16 \
    --gradient_checkpointing \
    --logging_steps 10 \
    --save_strategy epoch


# ===== WITH SEPARATE VALIDATION FILE =====
python sft/train.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --train_file apiTest/o4-mini_responses_with_state_action_train.json \
    --validation_file apiTest/o4-mini_responses_with_state_action_val.json \
    --output_dir ./checkpoints/blocksworld_sft \
    --num_train_epochs 3 \
    --bf16 \
    --gradient_checkpointing


# ===== WITH WANDB LOGGING =====
python sft/train.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --train_file apiTest/o4-mini_responses_with_state_action_train.json \
    --output_dir ./checkpoints/blocksworld_sft \
    --num_train_epochs 3 \
    --bf16 \
    --gradient_checkpointing \
    --report_to wandb \
    --run_name "qwen3-1.7b-blocksworld-sft"


# ===== RESUME FROM CHECKPOINT =====
python sft/train.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --train_file apiTest/o4-mini_responses_with_state_action_train.json \
    --output_dir ./checkpoints/blocksworld_sft \
    --num_train_epochs 3 \
    --bf16 \
    --gradient_checkpointing \
    --resume_from_checkpoint ./checkpoints/blocksworld_sft/checkpoint-100


# ===== CUSTOM LEARNING RATE SCHEDULE =====
python sft/train.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --train_file apiTest/o4-mini_responses_with_state_action_train.json \
    --output_dir ./checkpoints/blocksworld_sft \
    --learning_rate 1e-5 \
    --num_train_epochs 5 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type linear \
    --bf16 \
    --gradient_checkpointing


# ===== MORE FREQUENT CHECKPOINTING =====
python sft/train.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --train_file apiTest/o4-mini_responses_with_state_action_train.json \
    --output_dir ./checkpoints/blocksworld_sft \
    --num_train_epochs 3 \
    --bf16 \
    --gradient_checkpointing \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 5 \
    --evaluation_strategy steps \
    --eval_steps 100
