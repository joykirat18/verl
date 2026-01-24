#!/usr/bin/env python3
"""
Data validation script to check for issues before training.
This helps identify problems like invalid token IDs that cause CUDA errors.
"""

import json
import sys
from transformers import AutoTokenizer

def validate_dataset(train_file: str, model_name: str = "Qwen/Qwen3-1.7B"):
    """Validate dataset for potential issues."""
    
    print("=" * 80)
    print("Dataset Validation")
    print("=" * 80)
    print(f"Train file: {train_file}")
    print(f"Model: {model_name}")
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
    print(f"EOS token ID: {tokenizer.eos_token_id}")
    print(f"PAD token ID: {tokenizer.pad_token_id}")
    print()
    
    # Load data
    print("Loading dataset...")
    with open(train_file, "r") as f:
        data = json.load(f)
    
    print(f"Total examples: {len(data)}")
    print()
    
    # Validate each example
    print("Validating examples...")
    issues_found = 0
    
    for idx, example in enumerate(data):
        # Check for required keys
        if "question" not in example or "response" not in example:
            print(f"❌ Example {idx}: Missing 'question' or 'response' key")
            issues_found += 1
            continue
        
        prompt = example["question"]
        response = example["response"]
        
        # Format with chat template
        try:
            messages = [{"role": "user", "content": prompt}]
            prompt_text = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            
            # Combine prompt and response
            full_text = prompt_text + response + tokenizer.eos_token
            
            # Tokenize
            tokens = tokenizer(full_text, add_special_tokens=False)
            token_ids = tokens["input_ids"]
            
            # Check for invalid token IDs
            invalid_tokens = [tid for tid in token_ids if tid < 0 or tid >= vocab_size]
            
            if invalid_tokens:
                print(f"❌ Example {idx}: Found invalid token IDs: {set(invalid_tokens)}")
                print(f"   Prompt length: {len(prompt)}, Response length: {len(response)}")
                print(f"   Token count: {len(token_ids)}")
                issues_found += 1
            
            # Check for extremely long sequences
            if len(token_ids) > 32768:  # Qwen3 max context
                print(f"⚠️  Example {idx}: Very long sequence ({len(token_ids)} tokens)")
                print(f"   This will be truncated during training")
        
        except Exception as e:
            print(f"❌ Example {idx}: Error during tokenization: {e}")
            issues_found += 1
    
    print()
    print("=" * 80)
    if issues_found == 0:
        print("✅ Validation passed! No issues found.")
    else:
        print(f"❌ Validation failed! Found {issues_found} issue(s).")
        print("   Please fix the data before training.")
    print("=" * 80)
    
    return issues_found == 0

if __name__ == "__main__":
    train_file = sys.argv[1] if len(sys.argv) > 1 else "apiTest/o4-mini_responses_with_state_action_train_filtered.json"
    model_name = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3-1.7B"
    
    success = validate_dataset(train_file, model_name)
    sys.exit(0 if success else 1)
