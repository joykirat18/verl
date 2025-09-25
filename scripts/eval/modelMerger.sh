python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /nas-ssd2/joykirat/code/verl-fork/customBaselines/l1/checkpoints/deepscaler/l1_max_v2/global_step_200/actor \
    --target_dir /nas-ssd2/joykirat/code/verl-fork/customBaselines/l1/checkpoints/deepscaler/l1_max_v2/global_step_200/actor/HF