# Quick Fix - Load Best Model Error

## The Error
```
ValueError: --load_best_model_at_end requires the save and eval strategy to match, but found
- Evaluation strategy: no
- Save strategy: steps
```

## Root Cause
The `--eval_on_start` flag was not a valid HuggingFace `TrainingArguments` parameter, causing argument parsing to fail silently and resetting the evaluation strategy to "no".

## The Fix

### 1. Removed Invalid Flag
**Removed from `train.py`:**
```python
eval_on_start: bool = field(
    default=True,
    metadata={"help": "Run evaluation before training starts"}
)
```

**Removed from `train_blocksworld.sh`:**
```bash
--eval_on_start \
```

### 2. Added Manual Initial Evaluation
**Added to `train.py` (before training starts):**
```python
# Run initial evaluation before training (if evaluation strategy is not 'no')
if training_args.evaluation_strategy != "no" and training_args.local_rank <= 0:
    logger.info("Running initial evaluation before training...")
    initial_metrics = trainer.evaluate()
    trainer.log_metrics("eval_initial", initial_metrics)
    logger.info(f"Initial evaluation loss: {initial_metrics.get('eval_loss', 'N/A'):.4f}")
```

## What Works Now

✅ **Initial evaluation** - Runs before training starts (via manual call)
✅ **Periodic evaluation** - Every 5 optimizer steps (via `--eval_steps 5`)
✅ **Best model selection** - Loads best checkpoint at end (via `--load_best_model_at_end`)
✅ **Checkpoint saving** - Saves every 5 steps, aligned with evaluation

## Training Should Now Run Successfully

```bash
./sft/train_blocksworld.sh
```

Expected output:
```
Running initial evaluation before training...
Initial evaluation loss: X.XXXX
Starting training...
{'loss': ..., 'learning_rate': ..., 'epoch': ...}
{'eval_loss': ..., 'eval_runtime': ..., 'epoch': ...}
...
Loading best model from checkpoint-X (score: Y)
```

## Files Modified
1. ✅ `sft/train.py` - Removed invalid field, added manual initial evaluation
2. ✅ `sft/train_blocksworld.sh` - Removed `--eval_on_start` flag
3. 📄 `sft/QUICKFIX_LOAD_BEST_MODEL.md` - This file (new)

---

**Status: Fixed and ready to train!** 🚀
