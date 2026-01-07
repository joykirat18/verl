export HF_HOME="/nas-ssd2/joykirat/.cache/huggingface"
export UV_CACHE_DIR="/nas-ssd2/joykirat/.cache/uv"
export RAY_TMPDIR="/nas-ssd2/joykirat/tmp_ray"
export HUGGINGFACE_TOKEN='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'
export HF_TOKEN='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'
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3-1.7B \
    --port 8005 \
    --tensor-parallel-size 4
