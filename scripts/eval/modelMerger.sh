python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /nas-ssd2/joykirat/code/verl-fork/customBaselines/l1/checkpoints/deepscaler/l1_max_v1/global_step_180/actor \
    --target_dir /nas-ssd2/joykirat/code/verl-fork/customBaselines/l1/checkpoints/deepscaler/l1_max_v1/global_step_180/actor/HF