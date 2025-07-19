# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# It outperforms the Qwen2 7B base model by two percentage points on the test set of GSM8K.

set -x

export HF_HOME="/nas-ssd2/joykirat/.cache/huggingface"
export UV_CACHE_DIR="/nas-ssd2/joykirat/.cache/uv"
export RAY_TMPDIR="/nas-ssd2/joykirat/tmp_ray"

export CUDA_VISIBLE_DEVICES=2,3,5,6
PROMPT_KEY=prompt
TRAIN_BATCH_SIZE=4
PPO_MINI_BATCH_SIZE=1
PPO_MICRO_BATCH_SIZE_PER_GPU=1
LR=1e-6
MAX_PROMPT_LENGTH=1000
MAX_RESPONSE_LENGTH=15000
ACTOR_MODEL_PATH=Qwen/Qwen3-4B
ROLLOUT_NAME=vllm
ROLLOUT_N=8
ROLLOUT_TP=1
ROLLOUT_GPU_UTIL=0.75
PROJECT_NAME=TMI
EXPERIMENT_NAME=qwen-3-4B-summary-15k-combined-data-17-07
NNODES=1
N_GPUS_PER_NODE=4
SAVE_FREQ=10
TEST_FREQ=10
TOTAL_EPOCHS=2
WANDB_API_KEY='c8f694b1460eaf8f06beec994e5aa1bb56183688'
SAVE_PATH=verlCheckpoint/SummarizationVerlCheckpoint/$EXPERIMENT_NAME
TRAIN_FILES=../data/combined_train_dataset/train.parquet
TEST_FILES=../data/combined_test_dataset/test.parquet
REWARD_MANAGER=sum

if [ "$WANDB_API_KEY" != "None" ]; then
    export WANDB_DIR=${SAVE_PATH}
    mkdir -p $WANDB_DIR
    mkdir -p $WANDB_DIR/wandb
    chmod -R u+w $WANDB_DIR
    chmod -R u+w $WANDB_DIR/wandb
    wandb login --relogin $WANDB_API_KEY
fi

if [ ! -d "$SAVE_PATH" ]; then
    mkdir -p $SAVE_PATH
fi

ROLLOUT_SAVE_PATH=${SAVE_PATH}/rollout
if [ ! -d "$ROLLOUT_SAVE_PATH" ]; then
    mkdir -p $ROLLOUT_SAVE_PATH
fi


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILES"  \
    data.val_files="$TEST_FILES" \
    data.prompt_key="$PROMPT_KEY" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$ACTOR_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.name=$ROLLOUT_NAME \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_UTIL \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=$REWARD_MANAGER \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.val_before_train=True \
    trainer.resume_mode=auto \
    trainer.default_local_dir=${SAVE_PATH} \
    trainer.rollout_save_path=${ROLLOUT_SAVE_PATH} \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=$TOTAL_EPOCHS $@