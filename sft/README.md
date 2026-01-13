# SFT Training for Blocksworld State-Action Reasoning

This directory contains a simple supervised fine-tuning (SFT) script for training Qwen3-1.7B on the blocksworld state-action reasoning dataset using HuggingFace Transformers.

## Files

- `train.py` - Main training script using HuggingFace Trainer
- `train_blocksworld.sh` - Convenient shell script to launch training
- `README.md` - This file

## Requirements

Install required packages:

```bash
pip install transformers datasets accelerate torch tensorboard
pip install flash-attn --no-build-isolation  # Optional but recommended for faster training
```

## Dataset Format

The training script expects a JSON file with a list of examples, where each example has:
- `question`: The problem prompt
- `response`: The expected response/solution

Example:
```json
[
  {
    "question": "I am playing with a set of blocks...",
    "response": "<think>...</think>\n<answer>...</answer>"
  }
]
```

## Automatic Sequence Length Detection

By default, the script automatically analyzes your training data to determine the optimal `max_seq_length`. This:

- **Saves Memory**: Uses only what's needed, not arbitrary large values
- **Prevents Truncation**: Ensures most sequences fit without being cut off
- **Optimizes Training**: Faster training with appropriate sequence lengths

The script samples your data, calculates sequence lengths, and uses the 99.5th percentile + 128 token buffer. You'll see output like:

```
============================================================
Sequence Length Analysis:
  Min length:     458 tokens
  Mean length:    2,341 tokens
  Median length:  2,198 tokens
  Max length:     5,892 tokens
  99.5th percentile: 5,234 tokens
  Optimal (p99.5 + 128 buffer): 5,362 tokens
  Sequences that will fit: 99.50%
  Sequences that will be truncated: 5 / 1000
============================================================
```

To override and use a fixed length:
```bash
python sft/train.py --max_seq_length 8192 ...
```

## Quick Start

### Single GPU Training

```bash
# Using the convenience script (auto-detects max_seq_length)
./sft/train_blocksworld.sh

# Or directly with Python (auto-detects max_seq_length)
python sft/train.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --train_file apiTest/o4-mini_responses_with_state_action_train.json \
    --output_dir ./checkpoints/blocksworld_sft \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --bf16 \
    --gradient_checkpointing \
    --logging_steps 10 \
    --save_strategy epoch \
    --evaluation_strategy epoch

# Or with a fixed max_seq_length
python sft/train.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --train_file apiTest/o4-mini_responses_with_state_action_train.json \
    --output_dir ./checkpoints/blocksworld_sft \
    --max_seq_length 8192 \
    --num_train_epochs 3 \
    --bf16 \
    --gradient_checkpointing
```

### Multi-GPU Training

```bash
# Set GPUs to use
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Run training
./sft/train_blocksworld.sh

# Or with torchrun directly
torchrun --nproc_per_node=4 sft/train.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --train_file apiTest/o4-mini_responses_with_state_action_train.json \
    --output_dir ./checkpoints/blocksworld_sft \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --bf16 \
    --gradient_checkpointing
```

## Training Arguments

### Key Arguments

- `--model_name_or_path`: Model to fine-tune (default: Qwen/Qwen3-1.7B)
- `--train_file`: Path to training JSON file
- `--validation_file`: Optional validation file (if not provided, splits from train)
- `--output_dir`: Where to save the trained model
- `--max_seq_length`: Maximum sequence length (default: auto-detect from data)
  - If not specified, automatically calculates optimal length from your data
  - Uses 99.5th percentile + 128 token buffer by default
- `--max_seq_length_percentile`: Percentile for auto-detection (default: 99.5)
- `--max_seq_length_buffer`: Buffer tokens to add (default: 128)
- `--num_train_epochs`: Number of training epochs (default: 3)
- `--learning_rate`: Learning rate (default: 2e-5)

### Batch Size and Memory

Adjust these based on your GPU memory:

- `--per_device_train_batch_size`: Batch size per GPU (default: 1)
- `--gradient_accumulation_steps`: Accumulate gradients over N steps (default: 16)
- `--gradient_checkpointing`: Enable to save memory (recommended)
- `--bf16`: Use bfloat16 mixed precision (recommended)

**Effective batch size** = `per_device_train_batch_size` × `gradient_accumulation_steps` × `num_gpus`

Example: 1 × 16 × 1 = 16 (for single GPU)

### Advanced Options

```bash
python sft/train.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --train_file apiTest/o4-mini_responses_with_state_action_train.json \
    --validation_file apiTest/o4-mini_responses_with_state_action_val.json \
    --output_dir ./checkpoints/blocksworld_sft \
    --max_seq_length 8192 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
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
```

## Monitoring Training

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir ./checkpoints/blocksworld_sft/runs

# Open browser to http://localhost:6006
```

### Weights & Biases

To use W&B instead of TensorBoard:

```bash
pip install wandb
wandb login

# Then run with --report_to wandb
python sft/train.py --report_to wandb ...
```

## Output

After training, the model will be saved to the output directory with:

- `pytorch_model.bin` or `model.safetensors` - Model weights
- `config.json` - Model configuration
- `tokenizer_config.json` - Tokenizer configuration
- `trainer_state.json` - Training state
- `training_args.bin` - Training arguments
- Checkpoints saved at each epoch (if using `--save_strategy epoch`)

## Using the Trained Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model_path = "./checkpoints/blocksworld_sft"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Generate
prompt = "Your blocksworld problem here..."
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=4096, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Memory Requirements

Approximate GPU memory requirements for Qwen3-1.7B:

- **Training (bf16 + gradient checkpointing)**:
  - Batch size 1: ~20-25 GB
  - Batch size 2: ~30-35 GB
  
- **Inference**: ~5-8 GB

For limited GPU memory, consider:
1. Reduce `per_device_train_batch_size` to 1
2. Enable `--gradient_checkpointing`
3. Use `--bf16` for mixed precision
4. Increase `--gradient_accumulation_steps` to maintain effective batch size
5. Reduce `--max_seq_length` if sequences are shorter

## Troubleshooting

### Out of Memory (OOM)

1. Reduce batch size: `--per_device_train_batch_size 1`
2. Enable gradient checkpointing: `--gradient_checkpointing`
3. Reduce sequence length: `--max_seq_length 4096`
4. Increase gradient accumulation: `--gradient_accumulation_steps 32`

### Slow Training

1. Enable Flash Attention: `--use_flash_attention`
2. Use bf16: `--bf16` (faster than fp16 on modern GPUs)
3. Increase batch size if memory allows
4. Use multiple GPUs

### Model Not Learning

1. Check learning rate (try 1e-5 to 5e-5)
2. Increase warmup: `--warmup_ratio 0.1`
3. Check data preprocessing
4. Verify labels are not all -100

## License

Same as the parent project.
