#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) script for Qwen3-1.7B using HuggingFace Transformers.

This script trains a Qwen3-1.7B model on the blocksworld state-action reasoning dataset
using the standard HuggingFace Trainer API.

Usage:
    # Single GPU training
    python sft/train.py
    
    # Multi-GPU training with accelerate
    accelerate launch sft/train.py
    
    # Multi-GPU training with torchrun
    torchrun --nproc_per_node=4 sft/train.py
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    
    model_name_or_path: str = field(
        default="Qwen/Qwen3-1.7B",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading the model"}
    )
    use_flash_attention: bool = field(
        default=True,
        metadata={"help": "Whether to use Flash Attention 2"}
    )
    torch_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Torch dtype for model weights (float32, float16, bfloat16)"}
    )


@dataclass
class DataArguments:
    """Arguments for data loading and preprocessing."""
    
    train_file: str = field(
        default="apiTest/o4-mini_responses_with_state_action_train.json",
        metadata={"help": "Path to training data JSON file"}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to validation data JSON file (if None, uses a split of train_file)"}
    )
    validation_split_percentage: int = field(
        default=10,
        metadata={"help": "Percentage of training data to use for validation if validation_file is None"}
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum sequence length for training (if None, auto-calculated from data)"}
    )
    max_seq_length_buffer: int = field(
        default=128,
        metadata={"help": "Buffer to add to auto-calculated max_seq_length"}
    )
    max_seq_length_percentile: float = field(
        default=99.5,
        metadata={"help": "Percentile to use for auto-calculating max_seq_length (e.g., 99.5 means use 99.5th percentile)"}
    )
    prompt_key: str = field(
        default="question",
        metadata={"help": "Key for prompts in the JSON data"}
    )
    response_key: str = field(
        default="response",
        metadata={"help": "Key for responses in the JSON data"}
    )
    preprocessing_num_workers: int = field(
        default=4,
        metadata={"help": "Number of processes for data preprocessing"}
    )


@dataclass
class SFTTrainingArguments(TrainingArguments):
    """Extended training arguments for SFT."""
    
    output_dir: str = field(
        default=".checkpoints/blocksworld_state_action_sft",
        metadata={"help": "Directory to save model checkpoints"}
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "Number of training epochs"}
    )
    per_device_train_batch_size: int = field(
        default=2,
        metadata={"help": "Batch size per GPU for training"}
    )
    per_device_eval_batch_size: int = field(
        default=2,
        metadata={"help": "Batch size per GPU for evaluation"}
    )
    gradient_accumulation_steps: int = field(
        default=8,
        metadata={"help": "Number of gradient accumulation steps"}
    )
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "Initial learning rate"}
    )
    warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "Warmup ratio for learning rate scheduler"}
    )
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "Learning rate scheduler type"}
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay"}
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Maximum gradient norm for clipping"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "Log every X updates steps"}
    )
    save_strategy: str = field(
        default="epoch",
        metadata={"help": "Save checkpoint strategy"}
    )
    save_total_limit: int = field(
        default=3,
        metadata={"help": "Maximum number of checkpoints to keep"}
    )
    evaluation_strategy: str = field(
        default="epoch",
        metadata={"help": "Evaluation strategy"}
    )
    bf16: bool = field(
        default=True,
        metadata={"help": "Use bfloat16 mixed precision training"}
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Enable gradient checkpointing to save memory"}
    )
    report_to: str = field(
        default="tensorboard",
        metadata={"help": "Reporting tool (tensorboard, wandb, none)"}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed"}
    )


def load_json_dataset(file_path: str, prompt_key: str = "question", response_key: str = "response"):
    """Load dataset from JSON file.
    
    Args:
        file_path: Path to JSON file
        prompt_key: Key for prompts in the data
        response_key: Key for responses in the data
        
    Returns:
        Dataset object
    """
    logger.info(f"Loading dataset from {file_path}")
    
    with open(file_path, "r") as f:
        data = json.load(f)
    
    # Ensure data is a list
    if not isinstance(data, list):
        raise ValueError("JSON data must be a list of examples")
    
    # Create dataset
    dataset = Dataset.from_list(data)
    
    # Verify required keys exist
    if prompt_key not in dataset.column_names or response_key not in dataset.column_names:
        raise ValueError(f"Dataset must contain '{prompt_key}' and '{response_key}' columns. Found: {dataset.column_names}")
    
    logger.info(f"Loaded {len(dataset)} examples")
    return dataset


def calculate_optimal_max_seq_length(
    dataset: Dataset,
    tokenizer,
    prompt_key: str = "question",
    response_key: str = "response",
    percentile: float = 99.5,
    buffer: int = 128,
    sample_size: int = 1000,
):
    """Calculate optimal max_seq_length based on actual data.
    
    Args:
        dataset: Dataset to analyze
        tokenizer: Tokenizer to use for length calculation
        prompt_key: Key for prompts in the data
        response_key: Key for responses in the data
        percentile: Percentile to use (e.g., 99.5 means 99.5% of sequences will fit)
        buffer: Additional tokens to add as buffer
        sample_size: Number of examples to sample for calculation (None = use all)
        
    Returns:
        Optimal max_seq_length
    """
    import numpy as np
    
    logger.info("Calculating optimal max_seq_length from data...")
    
    # Sample dataset if it's large
    if sample_size and len(dataset) > sample_size:
        indices = np.random.choice(len(dataset), sample_size, replace=False)
        sample_dataset = dataset.select(indices)
        logger.info(f"Sampling {sample_size} examples from {len(dataset)} total")
    else:
        sample_dataset = dataset
        logger.info(f"Analyzing all {len(dataset)} examples")
    
    # Calculate lengths
    lengths = []
    for example in sample_dataset:
        prompt = example[prompt_key]
        response = example[response_key]
        
        # Format with chat template
        messages = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        
        # Combine prompt and response
        full_text = prompt_text + response + tokenizer.eos_token
        
        # Tokenize and get length
        tokens = tokenizer(full_text, add_special_tokens=False)
        length = len(tokens["input_ids"])
        lengths.append(length)
    
    # Calculate statistics
    lengths = np.array(lengths)
    max_length = int(np.max(lengths))
    mean_length = int(np.mean(lengths))
    median_length = int(np.median(lengths))
    percentile_length = int(np.percentile(lengths, percentile))
    
    # Calculate optimal length with buffer
    optimal_length = percentile_length + buffer
    
    logger.info("=" * 60)
    logger.info("Sequence Length Analysis:")
    logger.info(f"  Min length:     {int(np.min(lengths)):,} tokens")
    logger.info(f"  Mean length:    {mean_length:,} tokens")
    logger.info(f"  Median length:  {median_length:,} tokens")
    logger.info(f"  Max length:     {max_length:,} tokens")
    logger.info(f"  {percentile}th percentile: {percentile_length:,} tokens")
    logger.info(f"  Optimal (p{percentile} + {buffer} buffer): {optimal_length:,} tokens")
    logger.info(f"  Sequences that will fit: {(lengths <= optimal_length).mean() * 100:.2f}%")
    logger.info(f"  Sequences that will be truncated: {(lengths > optimal_length).sum()} / {len(lengths)}")
    logger.info("=" * 60)
    
    return optimal_length


def preprocess_function(examples, tokenizer, max_seq_length, prompt_key="question", response_key="response"):
    """Preprocess examples for SFT training.
    
    Args:
        examples: Batch of examples
        tokenizer: Tokenizer to use
        max_seq_length: Maximum sequence length
        prompt_key: Key for prompts
        response_key: Key for responses
        
    Returns:
        Tokenized examples with labels
    """
    prompts = examples[prompt_key]
    responses = examples[response_key]
    
    # Format with chat template
    formatted_texts = []
    for prompt, response in zip(prompts, responses):
        # Create messages
        messages = [{"role": "user", "content": prompt}]
        
        # Apply chat template to prompt
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        
        # Combine prompt and response
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
    
    # Create labels (copy of input_ids for causal LM)
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    # Mask out prompt tokens in labels (we only want to train on the response)
    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        messages = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        
        # Tokenize just the prompt to get its length
        prompt_tokens = tokenizer(
            prompt_text,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_tensors=None,
        )
        
        prompt_length = len(prompt_tokens["input_ids"])
        
        # Mask prompt tokens in labels
        if prompt_length > 0:
            tokenized["labels"][i][:prompt_length] = [-100] * prompt_length
    
    return tokenized


def main():
    """Main training function."""
    
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, SFTTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Set seed
    set_seed(training_args.seed)
    
    # Setup logging
    logger.setLevel(logging.INFO if training_args.local_rank <= 0 else logging.WARN)
    
    if training_args.local_rank <= 0:
        logger.info("=" * 80)
        logger.info("Training Configuration")
        logger.info("=" * 80)
        logger.info(f"Model: {model_args.model_name_or_path}")
        logger.info(f"Train file: {data_args.train_file}")
        logger.info(f"Output dir: {training_args.output_dir}")
        logger.info(f"Num epochs: {training_args.num_train_epochs}")
        logger.info(f"Per device train batch size: {training_args.per_device_train_batch_size}")
        logger.info(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
        logger.info(f"Learning rate: {training_args.learning_rate}")
        if data_args.max_seq_length is None:
            logger.info(f"Max sequence length: Auto-detect from data (p{data_args.max_seq_length_percentile} + {data_args.max_seq_length_buffer} buffer)")
        else:
            logger.info(f"Max sequence length: {data_args.max_seq_length}")
        logger.info("=" * 80)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=True,
    )
    
    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load datasets
    train_dataset = load_json_dataset(
        data_args.train_file,
        prompt_key=data_args.prompt_key,
        response_key=data_args.response_key
    )
    
    if data_args.validation_file:
        eval_dataset = load_json_dataset(
            data_args.validation_file,
            prompt_key=data_args.prompt_key,
            response_key=data_args.response_key
        )
    else:
        # Split training data for validation
        logger.info(f"Splitting {data_args.validation_split_percentage}% of training data for validation")
        split_dataset = train_dataset.train_test_split(
            test_size=data_args.validation_split_percentage / 100,
            seed=training_args.seed
        )
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
    
    logger.info(f"Training examples: {len(train_dataset)}")
    logger.info(f"Validation examples: {len(eval_dataset)}")
    
    # Calculate optimal max_seq_length if not provided
    if data_args.max_seq_length is None:
        data_args.max_seq_length = calculate_optimal_max_seq_length(
            dataset=train_dataset,
            tokenizer=tokenizer,
            prompt_key=data_args.prompt_key,
            response_key=data_args.response_key,
            percentile=data_args.max_seq_length_percentile,
            buffer=data_args.max_seq_length_buffer,
            sample_size=1000,  # Sample 1000 examples for efficiency
        )
        logger.info(f"Using auto-calculated max_seq_length: {data_args.max_seq_length}")
    else:
        logger.info(f"Using provided max_seq_length: {data_args.max_seq_length}")
    
    # Preprocess datasets
    logger.info("Preprocessing datasets...")
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(
            x,
            tokenizer,
            data_args.max_seq_length,
            data_args.prompt_key,
            data_args.response_key
        ),
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=train_dataset.column_names,
        desc="Preprocessing training data",
    )
    
    eval_dataset = eval_dataset.map(
        lambda x: preprocess_function(
            x,
            tokenizer,
            data_args.max_seq_length,
            data_args.prompt_key,
            data_args.response_key
        ),
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=eval_dataset.column_names,
        desc="Preprocessing validation data",
    )
    
    # Load model
    logger.info(f"Loading model from {model_args.model_name_or_path}")
    
    # Get torch dtype
    torch_dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = torch_dtype_map.get(model_args.torch_dtype, torch.bfloat16)
    
    # Load config
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=torch_dtype,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation="flash_attention_2" if model_args.use_flash_attention else "eager",
    )
    
    # Enable gradient checkpointing if requested
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    
    # Resize token embeddings if needed
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Resized token embeddings to {len(tokenizer)}")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
        pad_to_multiple_of=8,
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train
    logger.info("Starting training...")
    train_result = trainer.train()
    
    # Save final model
    logger.info(f"Saving final model to {training_args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    # Final evaluation
    logger.info("Running final evaluation...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    
    logger.info("=" * 80)
    logger.info("Training completed successfully!")
    logger.info(f"Model saved to: {training_args.output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
