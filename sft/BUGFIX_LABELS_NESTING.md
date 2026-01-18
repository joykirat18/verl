# Bug Fix - Labels Excessive Nesting Error

## The Error
```
ValueError: Unable to create tensor, you should probably activate truncation and/or padding 
with 'padding=True' 'truncation=True' to have batched tensors with the same length. 
Perhaps your features (`labels` in this case) have excessive nesting 
(inputs type `list` where type `int` is expected).
```

## Root Cause

The preprocessing function was using **slice assignment** to mask prompt tokens:

```python
# PROBLEMATIC CODE:
tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
tokenized["labels"][i][:prompt_length] = [-100] * prompt_length  # Slice assignment
```

When Python does slice assignment with a list on the right side, it can create nested structures or modify references in unexpected ways that confuse the HuggingFace data collator.

## The Fix

Rewrote the labels creation to **explicitly build each label list** element by element:

```python
# FIXED CODE:
labels = []

for i, (prompt, response) in enumerate(zip(prompts, responses)):
    input_ids = tokenized["input_ids"][i]
    
    # ... calculate prompt_length ...
    
    # Create labels: mask prompt tokens with -100, keep response tokens
    example_labels = []
    for j, token_id in enumerate(input_ids):
        if j < prompt_length:
            example_labels.append(-100)  # Mask prompt
        else:
            example_labels.append(token_id)  # Keep response
    
    labels.append(example_labels)

tokenized["labels"] = labels
```

## Why This Works

1. **No slice assignment** - We build fresh lists explicitly
2. **Clear ownership** - Each example gets its own distinct label list
3. **No reference sharing** - No risk of shallow copy issues
4. **Type correctness** - `labels` is `list[list[int]]` as expected

## Verification

```python
Type of labels: <class 'list'>
Type of labels[0]: <class 'list'>
Type of labels[0][0]: <class 'int'>  # ✅ Correct!
Length of input_ids[0]: 815
Length of labels[0]: 815  # ✅ Same length
First 10 labels[0]: [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100]  # ✅ Masked
```

## Files Modified
1. ✅ `sft/train.py` - Fixed preprocess_function
2. ✅ `sft/test_preprocessing.py` - Applied same fix for consistency

## Testing

Run preprocessing test:
```bash
cd /home/joykirat18/verl
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

---

**Status: FIXED! Training should now work.** 🚀
