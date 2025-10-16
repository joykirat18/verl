set -x

export HF_HOME="/nas-ssd2/joykirat/.cache/huggingface"
export UV_CACHE_DIR="/nas-ssd2/joykirat/.cache/uv"
export RAY_TMPDIR="/nas-hdd/joykirat/tmp_ray"
export HUGGINGFACE_TOKEN='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'
export HF_TOKEN='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'

export CUDA_VISIBLE_DEVICES=0,3
EXPERIMENT_NAME=qwen06b_math_10k_softThinking_gumble
WANDB_API_KEY='c8f694b1460eaf8f06beec994e5aa1bb56183688'
SAVE_PATH=checkpoints/SoftThinking/$EXPERIMENT_NAME
wandb_path=wandb/$EXPERIMENT_NAME
if [ "$WANDB_API_KEY" != "None" ]; then
    export WANDB_DIR=${wandb_path}
    mkdir -p $WANDB_DIR
    mkdir -p $WANDB_DIR/wandb
    chmod -R u+w $WANDB_DIR
    chmod -R u+w $WANDB_DIR/wandb
    wandb login --relogin $WANDB_API_KEY
fi

BATCH_SIZE=8
MICRO_BATCH_SIZE=4

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/nas-ssd2/joykirat/code/latent-reasoning-project/verl/scripts/data/dapo-17k/train.parquet \
    data.val_files=/nas-ssd2/joykirat/code/latent-reasoning-project/verl/scripts/data/aime_gpqa_test_dataset/test.parquet \
    data.train_batch_size=$BATCH_SIZE \
    data.max_prompt_length=1024 \
    data.max_response_length=8000 \
    data.shuffle=False \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.min_p=0.001 \
    actor_rollout_ref.rollout.repetition_penalty=1.0 \
    actor_rollout_ref.rollout.after_thinking_temperature=1.0 \
    actor_rollout_ref.rollout.after_thinking_top_p=1.0 \
    actor_rollout_ref.rollout.after_thinking_top_k=-1 \
    actor_rollout_ref.rollout.early_stopping_entropy_threshold=0.01 \
    actor_rollout_ref.rollout.early_stopping_length_threshold=256 \
    actor_rollout_ref.rollout.max_topk=30 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.min_p=0.001 \
    actor_rollout_ref.rollout.val_kwargs.repetition_penalty=1.0 \
    actor_rollout_ref.rollout.val_kwargs.after_thinking_temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.after_thinking_top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.after_thinking_top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.early_stopping_entropy_threshold=0.01 \
    actor_rollout_ref.rollout.val_kwargs.early_stopping_length_threshold=256 \
    actor_rollout_ref.rollout.val_kwargs.gumbel_softmax_temperature=0.5 \
    actor_rollout_ref.rollout.val_kwargs.max_topk=30 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_soft_thinking=True \
    actor_rollout_ref.rollout.add_noise_gumbel_softmax=True \
    actor_rollout_ref.rollout.gumbel_softmax_temperature=0.5 \
    actor_rollout_ref.rollout.n=8 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.rollout_data_dir=${SAVE_PATH}/train_rollout \
    trainer.validation_data_dir=${SAVE_PATH}/val_rollout \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='latent-reasoning' \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.default_local_dir=${SAVE_PATH} \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.val_before_train=True \
    trainer.total_epochs=2 $@