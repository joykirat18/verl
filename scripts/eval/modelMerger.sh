python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /nas-ssd2/joykirat/code/verl-fork/verl/examples/grpo_trainer/verlCheckpoint/mathbaseRun/qwen4b_lora_function_rm_dapo_math_10k_context/global_step_400/actor \
    --target_dir /nas-ssd2/joykirat/code/verl-fork/verl/scripts/train/verlCheckpoint/NonSummary/MergedModel/qwen4b_lora_dapo_math_10k_context_normal