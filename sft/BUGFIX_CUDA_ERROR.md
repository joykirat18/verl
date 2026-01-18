# CUDA Device-Side Assert Error - FIXED

## Problem
Training was failing with:
```
RuntimeError: CUDA error: device-side assert triggered
```

## Root Cause
The preprocessing function in `train.py` was using a **shallow copy** when creating labels from input_ids:

```python
tokenized["labels"] = tokenized["input_ids"].copy()  # WRONG - shallow copy!
```

When masking prompt tokens with `-100` in the labels, this also modified the input_ids because they referenced the same list objects. This caused invalid token IDs (-100) in the input_ids, which triggered the CUDA error.

## Solution
Changed to a **deep copy**:

```python
tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]  # CORRECT - deep copy!
```

## Files Modified
1. ✅ `sft/train.py` - Fixed the preprocessing function
2. ✅ `sft/train_blocksworld.sh` - Added wandb support + virtual environment activation
3. ✅ `sft/validate_data.py` - New validation script (optional tool)
4. ✅ `sft/test_preprocessing.py` - New preprocessing test script (optional tool)

## Verification
Run the preprocessing test to verify the fix:

```bash
source .venv/bin/activate
python3 sft/test_preprocessing.py \
    apiTest/o4-mini_responses_with_state_action_train_filtered.json \
    Qwen/Qwen3-1.7B \
    4096
```

Expected output:
```
✅ Preprocessing successful!
✅ All validation checks passed!
```

## Training Now
The training script should now work correctly:

```bash
./sft/train_blocksworld.sh
```

### With Wandb logging:
```bash
wandb login  # First time only
WANDB_PROJECT="my-project" ./sft/train_blocksworld.sh
```

### Debug Mode (if needed):
Uncomment these lines in `train_blocksworld.sh`:
```bash
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
```

## Additional Tools Created

### 1. Data Validation Script
Checks for invalid token IDs before training:
```bash
python3 sft/validate_data.py <train_file> <model_name>
```

### 2. Preprocessing Test Script
Tests the actual preprocessing pipeline:
```bash
python3 sft/test_preprocessing.py <train_file> <model_name> <max_seq_length>
```

These scripts are useful for debugging data issues and can be enabled in the training script if needed.
