#!/usr/bin/env python3
"""
Test the actual preprocessing pipeline to identify CUDA error causes.
"""

import json
import sys
import torch
from transformers import AutoTokenizer
from datasets import Dataset

def preprocess_function(examples, tokenizer, max_seq_length, prompt_key="question", response_key="response"):
    """Preprocess examples for SFT training (same as train.py)."""
    prompts = examples[prompt_key]
    responses = examples[response_key]
    
    # Format with chat template
    formatted_texts = []
    for prompt, response in zip(prompts, responses):
        messages = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        full_text = prompt_text + response + tokenizer.eos_token
        formatted_texts.append(full_text)
    
    # Tokenize
    tokenized = tokenizer(
        formatted_texts,
        truncation=True,
        max_length=max_seq_length,
        padding=False,
        return_tensors=None,
    )
    
    # Create labels by masking prompt tokens
    labels = []
    
    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        input_ids = tokenized["input_ids"][i]
        
        messages = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        
        prompt_tokens = tokenizer(
            prompt_text,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_tensors=None,
        )
        
        prompt_length = len(prompt_tokens["input_ids"])
        
        # Create labels: mask prompt tokens with -100, keep response tokens
        example_labels = []
        for j, token_id in enumerate(input_ids):
            if j < prompt_length:
                example_labels.append(-100)
            else:
                example_labels.append(token_id)
        
        labels.append(example_labels)
    
    tokenized["labels"] = labels
    
    return tokenized

def test_preprocessing(train_file: str, model_name: str = "Qwen/Qwen3-1.7B", max_seq_length: int = 4096):
    """Test preprocessing pipeline."""
    
    print("=" * 80)
    print("Testing Preprocessing Pipeline")
    print("=" * 80)
    print(f"Train file: {train_file}")
    print(f"Model: {model_name}")
    print(f"Max seq length: {max_seq_length}")
    print()
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    vocab_size = len(tokenizer)
    print(f"Vocabulary size: {vocab_size}")
    print()
    
    # Load data
    print("Loading dataset...")
    with open(train_file, "r") as f:
        data = json.load(f)
    
    dataset = Dataset.from_list(data)
    print(f"Total examples: {len(dataset)}")
    print()
    
    # Process dataset
    print("Testing preprocessing on first 10 examples...")
    test_dataset = dataset.select(range(min(10, len(dataset))))
    
    try:
        processed = preprocess_function(
            test_dataset.to_dict(),
            tokenizer,
            max_seq_length,
            "question",
            "response"
        )
        
        print("✅ Preprocessing successful!")
        print()
        
        # Validate processed data
        print("Validating processed data...")
        issues = 0
        
        for i in range(len(processed["input_ids"])):
            input_ids = processed["input_ids"][i]
            labels = processed["labels"][i]
            
            # Check input_ids range
            for token_id in input_ids:
                if token_id < 0 or token_id >= vocab_size:
                    print(f"❌ Example {i}: Invalid input token ID {token_id}")
                    issues += 1
            
            # Check labels range (should be -100 or valid token ID)
            for label in labels:
                if label != -100 and (label < 0 or label >= vocab_size):
                    print(f"❌ Example {i}: Invalid label {label}")
                    issues += 1
            
            # Check lengths match
            if len(input_ids) != len(labels):
                print(f"❌ Example {i}: Length mismatch - inputs: {len(input_ids)}, labels: {len(labels)}")
                issues += 1
            
            # Print stats for first example
            if i == 0:
                num_masked = sum(1 for l in labels if l == -100)
                num_trained = sum(1 for l in labels if l != -100)
                print(f"   Example 0 stats:")
                print(f"   - Total tokens: {len(input_ids)}")
                print(f"   - Masked (prompt) tokens: {num_masked}")
                print(f"   - Training (response) tokens: {num_trained}")
        
        print()
        if issues == 0:
            print("✅ All validation checks passed!")
        else:
            print(f"❌ Found {issues} issue(s)")
        
        print("=" * 80)
        return issues == 0
        
    except Exception as e:
        print(f"❌ Preprocessing failed with error:")
        print(f"   {type(e).__name__}: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    train_file = sys.argv[1] if len(sys.argv) > 1 else "apiTest/o4-mini_responses_with_state_action_train_filtered.json"
    model_name = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3-1.7B"
    max_seq_length = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
    
    success = test_preprocessing(train_file, model_name, max_seq_length)
    sys.exit(0 if success else 1)
