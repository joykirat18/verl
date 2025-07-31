set -x

data_path=/nas-ssd2/joykirat/code/verl-fork/verl/scripts/data/full_test_dataset/test.parquet
save_path=/nas-ssd2/joykirat/code/verl-fork/verl/scripts/data/eval_result/qwen_3-4b_lora_dapo_math_10k_context_linear_reward_no_summary_170.parquet
model_path=/nas-ssd2/joykirat/code/verl-fork/verl/scripts/train/verlCheckpoint/NonSummary/MergedModel/qwen4b_lora_dapo_math_10k_context_linear_reward_no_summary_170
model_lora_path=$model_path/lora_adapter

export HF_HOME="/nas-ssd2/joykirat/.cache/huggingface"
export UV_CACHE_DIR="/nas-ssd2/joykirat/.cache/uv"
export RAY_TMPDIR="/nas-ssd2/joykirat/tmp_ray"

export CUDA_VISIBLE_DEVICES=1,2,3,4

python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=4 \
    data.path=$data_path \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.output_path=$save_path \
    +data.is_lora=True \
    +data.model_lora_path=$model_lora_path \
    model.path=Qwen/Qwen3-4B \
    +model.trust_remote_code=True \
    rollout.temperature=1.0 \
    rollout.top_k=-1 \
    rollout.top_p=1.0 \
    rollout.prompt_length=1024 \
    rollout.response_length=10000 \
    rollout.tensor_model_parallel_size=2 \
    rollout.gpu_memory_utilization=0.8

# Evaluation
python3 -m scripts.eval.main_eval \
    data.path=$save_path \
    data.prompt_key=prompt \
    data.response_key=responses \
    data.model_name=Qwen/Qwen3-4B \
    custom_reward_function.path=/nas-ssd2/joykirat/code/verl-fork/verl/scripts/eval/reward_score.py \
    custom_reward_function.name=reward_func