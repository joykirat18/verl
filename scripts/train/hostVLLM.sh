export HF_HOME="/nas-ssd2/joykirat/.cache/huggingface"
export UV_CACHE_DIR="/nas-ssd2/joykirat/.cache/uv"
export RAY_TMPDIR="/nas-ssd2/joykirat/tmp_ray"
export HUGGINGFACE_TOKEN='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'
export HF_TOKEN='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'
CUDA_VISIBLE_DEVICES=2,3 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-14B-Instruct \
  --port 8005 \
  --tensor-parallel-size 2