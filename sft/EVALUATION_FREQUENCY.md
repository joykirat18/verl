# Understanding Evaluation Frequency with Gradient Accumulation

## The Issue

When using gradient accumulation, **steps** are counted in terms of **optimizer steps**, not forward passes. This affects how frequently evaluation happens.

## Current Configuration

In `train_blocksworld.sh`:
- `BATCH_SIZE=2` (per device)
- `GRAD_ACCUM=16` (gradient accumulation steps)
- `eval_steps=10` (evaluate every 10 optimizer steps)

## How Step Counting Works

### Without Gradient Accumulation:
- 1 forward pass = 1 optimizer step
- `eval_steps=10` → Evaluate after 10 batches

### With Gradient Accumulation (GRAD_ACCUM=16):
- 16 forward passes = 1 optimizer step
- `eval_steps=10` → Evaluate after 160 batches (10 × 16)

## Your Effective Settings

With your current config:
- **Effective batch size**: 2 × 16 = 32
- **Optimizer steps per evaluation**: 10
- **Batches per evaluation**: 160 (10 × 16)
- **Examples per evaluation**: 320 (2 × 16 × 10)

With 429 training examples:
- **Optimizer steps per epoch**: ~13 (429 ÷ 32)
- **Evaluations per epoch**: ~1.3 (13 ÷ 10)

## Solutions for More Frequent Evaluation

### Option 1: Reduce eval_steps (Recommended)
Change `eval_steps` to a smaller value:

```bash
# Evaluate every 5 optimizer steps
--eval_steps 5

# Evaluate every 2 optimizer steps (very frequent)
--eval_steps 2
```

### Option 2: Reduce Gradient Accumulation
⚠️ This changes your effective batch size!

```bash
GRAD_ACCUM=8  # Instead of 16
# Now: eval_steps=10 → Evaluate after 80 batches (10 × 8)
```

### Option 3: Use eval_strategy="epoch"
Evaluate after each epoch:

```bash
--evaluation_strategy epoch
# Remove --eval_steps (not needed with epoch strategy)
```

### Option 4: Hybrid Approach
Combine steps and epoch strategies by using `load_best_model_at_end`:

```bash
--evaluation_strategy steps \
--eval_steps 5 \
--eval_on_start \
--load_best_model_at_end \
--metric_for_best_model eval_loss \
--greater_is_better False
```

## What I've Added

### 1. Initial Evaluation (`--eval_on_start`)
Now runs evaluation before training starts, so you can see baseline metrics immediately.

### 2. Enhanced Logging
The training script now prints:
```
Training Steps Calculation:
  Per-device batch size: 2
  Gradient accumulation: 16
  Number of GPUs: 1
  Effective batch size: 32
  Estimated optimizer steps per epoch: 13
  Total optimizer steps: ~39
  Evaluation frequency: Every 10 optimizer steps
  Expected evaluations: ~3
```

This helps you understand when to expect evaluations.

## Recommended Configuration for Your Dataset

With 429 training examples, I recommend:

```bash
# In train_blocksworld.sh
BATCH_SIZE=2
GRAD_ACCUM=16
# ... other settings ...

# In the training command:
--eval_steps 5 \           # Evaluate every 5 optimizer steps
--logging_steps 1 \         # Log every optimizer step
--save_steps 5 \            # Save checkpoint every 5 steps (aligned with eval)
--eval_on_start \           # See baseline before training
--load_best_model_at_end \  # Load best model after training
```

This gives you:
- **2-3 evaluations per epoch** (13 steps ÷ 5)
- **Logging every optimizer step** for detailed tracking
- **Initial baseline evaluation**
- **Best model selection** at the end

## Monitoring in Wandb

With these settings, you'll see in Wandb:
- Initial evaluation loss (step 0)
- Training loss logged every step
- Evaluation loss every 5 steps
- Learning rate schedule
- Gradient norms

## Quick Test

To verify evaluations are working, you can temporarily set:

```bash
--eval_steps 2 \
--logging_steps 1 \
--num_train_epochs 1
```

This will give you ~6 evaluations in one epoch, making it easy to verify the evaluation is running.
