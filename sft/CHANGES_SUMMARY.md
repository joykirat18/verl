# Summary of Changes - Evaluation Frequency Fix

## Problem
Evaluation was not happening as frequently as expected because with `GRAD_ACCUM=16`, the `eval_steps` parameter counts **optimizer steps** (after accumulation), not individual forward passes.

## Root Cause
- With `GRAD_ACCUM=16` and `eval_steps=10`:
  - Evaluation happens every **10 optimizer steps**
  - Each optimizer step = 16 forward passes
  - So evaluation happens every **160 batches**, not 10

## Changes Made

### 1. Updated Training Configuration
**File: `sft/train_blocksworld.sh`**

#### New Variables (lines 52-57):
```bash
# Evaluation settings
EVAL_STEPS=5      # Evaluate every 5 optimizer steps (80 batches)
SAVE_STEPS=5      # Save aligned with evaluation
LOGGING_STEPS=1   # Log every optimizer step
```

#### New Training Arguments:
- `--eval_on_start` - Run evaluation before training starts (see baseline)
- `--load_best_model_at_end` - Load the best checkpoint after training
- `--metric_for_best_model eval_loss` - Use validation loss for best model
- `--greater_is_better False` - Lower loss is better

### 2. Enhanced Logging in train.py
**File: `sft/train.py`**

Added detailed step calculation output:
```
Training Steps Calculation:
  Per-device batch size: 2
  Gradient accumulation: 16
  Number of GPUs: 1
  Effective batch size: 32
  Estimated optimizer steps per epoch: 13
  Total optimizer steps: ~39
  Evaluation frequency: Every 5 optimizer steps
  Expected evaluations: ~7
```

### 3. Removed Debug Code
- Removed `breakpoint()` that was pausing training

## What to Expect Now

### With Current Settings (429 training examples, 3 epochs):

| Metric | Value |
|--------|-------|
| Effective batch size | 32 (2 × 16 × 1 GPU) |
| Optimizer steps per epoch | ~13 (429 ÷ 32) |
| Total optimizer steps | ~39 (13 × 3 epochs) |
| **Evaluations per epoch** | **~2-3** (every 5 steps) |
| **Total evaluations** | **~7-8** (plus initial) |
| Logs per epoch | 13 (every step) |
| Checkpoints saved | ~7-8 (every 5 steps) |

### Timeline Example:
```
Step 0:  Initial evaluation (baseline)
Step 1:  Training begins, log metrics
Step 5:  Evaluation + Checkpoint save
Step 10: Evaluation + Checkpoint save
Step 13: End of epoch 1
Step 15: Evaluation + Checkpoint save
...
Step 39: End of training
         Load best checkpoint based on eval_loss
```

## How to Adjust

### Want MORE frequent evaluation?
```bash
# In train_blocksworld.sh
EVAL_STEPS=2    # Evaluate every 2 optimizer steps
SAVE_STEPS=2
```

### Want LESS frequent evaluation?
```bash
# In train_blocksworld.sh
EVAL_STEPS=10   # Evaluate every 10 optimizer steps
SAVE_STEPS=10
```

### Want evaluation per epoch only?
```bash
# In train_blocksworld.sh, change the training arguments:
--evaluation_strategy epoch \
# Remove --eval_steps line
```

## Monitoring in Wandb

You'll now see in your Wandb dashboard:
- ✅ Initial evaluation metrics (step 0)
- ✅ Training loss every optimizer step
- ✅ Evaluation loss every 5 optimizer steps
- ✅ Learning rate schedule
- ✅ Best model selection at the end
- ✅ Checkpoint saves aligned with evaluations

## Files Modified
1. ✅ `sft/train.py` - Added eval_on_start flag, enhanced logging, removed breakpoint
2. ✅ `sft/train_blocksworld.sh` - Updated evaluation configuration, added variables
3. 📄 `sft/EVALUATION_FREQUENCY.md` - Detailed explanation (new)
4. 📄 `sft/CHANGES_SUMMARY.md` - This file (new)

## Run Training

```bash
# Run with new configuration
./sft/train_blocksworld.sh

# Override evaluation frequency
EVAL_STEPS=2 ./sft/train_blocksworld.sh

# Monitor in wandb
# Check your dashboard at: https://wandb.ai/<your-username>/blocksworld-sft
```

## Verify It's Working

Look for these in the terminal output:

1. **Initial evaluation:**
   ```
   ***** Running Evaluation *****
     Num examples = 47
   ```

2. **Step calculation output:**
   ```
   Training Steps Calculation:
     Evaluation frequency: Every 5 optimizer steps
     Expected evaluations: ~7
   ```

3. **Evaluation logs during training:**
   ```
   {'eval_loss': 1.234, 'eval_runtime': 2.5, 'eval_steps_per_second': 18.8, 'epoch': 0.38}
   ```

4. **Best model loaded at end:**
   ```
   Loading best model from checkpoint-X (score: Y)
   ```

## Tips

- Check Wandb dashboard to see evaluation points clearly marked
- The script now saves the **best** model (lowest eval_loss), not just the last one
- With 429 examples and batch size 32, you'll see ~2-3 evals per epoch
- Adjust `EVAL_STEPS` at the top of the script to change frequency

---

**Ready to train!** 🚀

The evaluation will now happen every 5 optimizer steps (every 80 batches), giving you much more frequent feedback on model performance.
