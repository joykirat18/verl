# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# It outperforms the Qwen2 7B base model by two percentage points on the test set of GSM8K.

set -x

export HF_HOME="/nas-ssd2/joykirat/.cache/huggingface"
export UV_CACHE_DIR="/nas-ssd2/joykirat/.cache/uv"
export RAY_TMPDIR="/nas-ssd2/joykirat/tmp_ray"
export HUGGINGFACE_TOKEN='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'
export HF_TOKEN='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'
export CUDA_VISIBLE_DEVICES=6,7
EXPERIMENT_NAME=qwen4b_dapo_math_10k_context_linear_reward_attention_no_difficulty
WANDB_API_KEY='c8f694b1460eaf8f06beec994e5aa1bb56183688'
SAVE_PATH=/nas-ssd2/joykirat/code/verl-fork/verl/scripts/train/verlCheckpoint/Summary/$EXPERIMENT_NAME
if [ "$WANDB_API_KEY" != "None" ]; then
    export WANDB_DIR=${SAVE_PATH}
    mkdir -p $WANDB_DIR
    mkdir -p $WANDB_DIR/wandb
    chmod -R u+w $WANDB_DIR
    chmod -R u+w $WANDB_DIR/wandb
    wandb login --relogin $WANDB_API_KEY
fi

ROLLOUT_SAVE_PATH=${SAVE_PATH}/rollout
if [ ! -d "$ROLLOUT_SAVE_PATH" ]; then
    mkdir -p $ROLLOUT_SAVE_PATH
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/nas-ssd2/joykirat/code/verl-fork/verl/scripts/data/dapo-17k/train.parquet \
    data.val_files=/nas-ssd2/joykirat/code/verl-fork/verl/scripts/data/underthink_test_dataset/test.parquet \
    data.train_batch_size=8 \
    data.max_prompt_length=1024 \
    data.max_response_length=10000 \
    data.shuffle=False \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.summary_mode=compression_no_difficulty \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=sumLinearNoDifficulty \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.default_local_dir=${SAVE_PATH} \
    trainer.rollout_save_path=${ROLLOUT_SAVE_PATH} \
    trainer.save_freq=10 \
    trainer.test_freq=20 \
    trainer.val_before_train=True \
    trainer.total_epochs=0 $@