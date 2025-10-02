#!/usr/bin/env python3
"""
Phase 1 only - Generate and store all rollouts.

This file processes the dataset in phase 1 only:
1. Generate all rollouts for all prompts
2. Save the rollouts to disk
"""

import logging
import math
import re
import random
import requests
import json
import time
from typing import Dict, List, Tuple, Any, Optional
from contextlib import contextmanager

import torch
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Phase1OnlyInference:
    """
    Phase 1 only inference class that generates and stores all rollouts.
    """
    
    def __init__(
        self,
        model_path: str,
        config: DictConfig,
        tokenizer,
        vllm_server_url: str = "http://localhost:8000"
    ):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the model
            config: Configuration dictionary
            tokenizer: Tokenizer instance
            vllm_server_url: Base URL for the vLLM server
        """
        self.model_path = model_path
        self.config = config
        self.tokenizer = tokenizer
        self.vllm_server_url = vllm_server_url
        
        # Initialize sampling parameters
        self.sampling_params = {
            "n": 5,
            "max_tokens": config.get("response_length", 10000),
            "temperature": config.get("temperature", 1.0),
            "top_p": config.get("top_p", 1.0),
            "top_k": config.get("top_k", -1),
        }
        
        # Add any additional sampling params from config
        for k in config.keys():
            if k in ["n", "logprobs", "max_tokens", "temperature", "top_p", "top_k", "stop", "stop_token_ids"]:
                self.sampling_params[k] = config.get(k)
                
        print(f"Initialized with sampling params: {self.sampling_params}")
    
    def _make_vllm_request(self, prompts: List[str], sampling_params: Dict = None, max_retries: int = 3) -> List[Dict]:
        """
        Make API request to vLLM server with retry mechanism.
        
        Args:
            prompts: List of input prompts
            sampling_params: Sampling parameters override
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of response dictionaries
        """
        endpoint = f"{self.vllm_server_url}/v1/completions"
        
        # Use provided sampling params or default ones
        params = sampling_params or self.sampling_params.copy()
        
        payload = {
            "prompt": prompts[0] if len(prompts) == 1 else prompts,
            **params
        }
        
        # Calculate timeout for long generations
        timeout = max(300, self.sampling_params.get('max_tokens', 512) * 0.1)  # 0.1s per token minimum
        
        for attempt in range(max_retries + 1):
            try:
                logger.info(f"Making vLLM request (attempt {attempt + 1}/{max_retries + 1})")
                response = requests.post(endpoint, json=payload, timeout=timeout)
                response.raise_for_status()
                
                if response.status_code == 200:
                    data = response.json()
                    return data['choices'] if 'choices' in data else [data]
                else:
                    raise ValueError(f"vLLM request failed with status {response.status_code}")
                    
            except (requests.exceptions.RequestException, TimeoutError) as e:
                logger.warning(f"vLLM request failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                
                if attempt < max_retries:
                    # Exponential backoff: wait 2^attempt seconds
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {max_retries + 1} attempts failed. Returning empty response.")
                    # Return empty response only after all retries failed
                    return [{'text': '', 'logprobs': None}]
        
        # This should never be reached, but just in case
        raise ValueError(f"All {max_retries + 1} attempts failed. Returning empty response.")
    
    def phase1_generate_all_rollouts(self, prompts: List[str], data_sources: List[str], reward_models: List[Dict], batch_size: int = 10) -> List[Dict]:
        """
        Phase 1: Generate all rollouts for all prompts in batches.
        
        Args:
            prompts: List of input prompts
            data_sources: List of data source identifiers
            reward_models: List of reward model dictionaries
            batch_size: Number of prompts to process in each batch
            
        Returns:
            List of all generated responses
        """
        print(f"Phase 1: Generating all rollouts in batches of {batch_size}...")
        all_responses = []
        
        # Process prompts in batches to avoid timeouts
        for batch_start in tqdm(range(0, len(prompts), batch_size), desc="Processing prompt batches"):
            batch_end = min(batch_start + batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]
            
            print(f"Processing batch {batch_start//batch_size + 1}: prompts {batch_start}-{batch_end-1}")
            
            # Generate initial responses for this batch
            choices = self._make_vllm_request(batch_prompts)
            
            # Process outputs for this batch
            for batch_idx, prompt in enumerate(batch_prompts):
                prompt_idx = batch_start + batch_idx
                # Get choices for this prompt (n choices per prompt)
                prompt_choices = choices[batch_idx * self.sampling_params['n']:(batch_idx + 1) * self.sampling_params['n']]
                
                for choice in prompt_choices:
                    # Extract response text and tokenize
                    response_text = choice['text']
                    response_ids = self.tokenizer.encode(response_text, add_special_tokens=False)
                    prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
                    
                    all_responses.append({
                        "prompt_idx": prompt_idx,
                        "prompt_ids": prompt_ids,
                        "response_ids": response_ids,
                        "response_text": response_text,
                        "data_source": data_sources[prompt_idx],
                        "reward_model": reward_models[prompt_idx],
                    })
        
        print(f"Generated {len(all_responses)} rollouts for {len(prompts)} prompts")
        return all_responses


def load_dataset(path_to_parquet):
    import pandas as pd
    df = pd.read_parquet(path_to_parquet)
    return df


def main():
    """Generate Phase 1 rollouts only and save them."""
    import argparse
    from omegaconf import OmegaConf
    
    parser = argparse.ArgumentParser(description="Phase 1 only: Generate and store rollouts")
    parser.add_argument("--model_path", type=str, help="Path to model", default="Qwen/Qwen3-4B")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--vllm_server_url", type=str, default="http://localhost:8000", help="vLLM server URL")
    parser.add_argument("--data_source", type=str, default="math-amc", help="Data source")
    parser.add_argument("--dataset_name", type=str, default="full", help="Dataset name")
    parser.add_argument("--output_dir", type=str, default="/nas-ssd2/joykirat/code/verl-fork/verl/TTS_attention_phase1", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size for processing")
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = OmegaConf.load(args.config)
    else:
        config = OmegaConf.create({
            "response_length": 10000,
            "temperature": 1.0,
            "top_p": 1.0,
            "tensor_model_parallel_size": 1,
            "gpu_memory_utilization": 0.9,
        })
    
    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize inference engine
    inference_engine = Phase1OnlyInference(
        model_path=args.model_path,
        config=config,
        tokenizer=tokenizer,
        vllm_server_url=args.vllm_server_url
    )
    
    dataset_name = args.dataset_name
    selected_data_source = args.data_source
    dataset = load_dataset(f"/nas-ssd2/joykirat/code/verl-fork/verl/scripts/data/{dataset_name}_test_dataset/test.parquet")

    # Filter dataset to only include selected data source
    filtered_dataset = dataset[dataset['data_source'] == selected_data_source].copy()
    # Reset index to ensure consecutive integer indices starting from 0
    filtered_dataset = filtered_dataset.reset_index(drop=True)
    print(f"Filtered dataset from {len(dataset)} to {len(filtered_dataset)} samples for data source: {selected_data_source}")
    
    # Process all prompts at once
    print(f"Processing {len(filtered_dataset)} samples...")
    
    # Prepare all inputs
    all_prompts = []
    all_data_sources = []
    all_reward_models = []
    
    for i in range(len(filtered_dataset)):
        prompt = filtered_dataset['prompt'].iloc[i]
        data_source = filtered_dataset['data_source'].iloc[i]
        reward_model = filtered_dataset['reward_model'].iloc[i]
        
        # Apply chat template
        formatted_prompt = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
        
        all_prompts.append(formatted_prompt)
        all_data_sources.append(data_source)
        all_reward_models.append(reward_model)
    
    # Run Phase 1 only
    phase1_results = inference_engine.phase1_generate_all_rollouts(
        prompts=all_prompts,
        data_sources=all_data_sources,
        reward_models=all_reward_models,
        batch_size=args.batch_size
    )
    
    # Initialize responses column
    filtered_dataset['response'] = None
    
    # Organize results by prompt index
    results_by_prompt = {}
    for result in phase1_results:
        prompt_idx = result['prompt_idx']
        if prompt_idx not in results_by_prompt:
            results_by_prompt[prompt_idx] = []
        results_by_prompt[prompt_idx].append(result['response_text'])
    
    # Save results to filtered dataset
    for prompt_idx, responses in results_by_prompt.items():
        # Store the list of responses for each prompt using at[] for single cell assignment
        filtered_dataset.at[filtered_dataset.index[prompt_idx], 'response'] = responses
    
    # Save results as parquet
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    output_path = f"{args.output_dir}/{dataset_name}_{selected_data_source}_phase1_rollouts.parquet"
    filtered_dataset.to_parquet(output_path)
    print(f"Saved Phase 1 results to {output_path}")
    
    # Print summary statistics
    print("\n=== Phase 1 Summary ===")
    print(f"Total rollouts generated: {len(phase1_results)}")
    print(f"Number of prompts: {len(filtered_dataset)}")
    print(f"Rollouts per prompt: {len(phase1_results) // len(filtered_dataset) if len(filtered_dataset) > 0 else 0}")
    
    # Print sample results
    print("\nSample Results:")
    for i in range(min(3, len(filtered_dataset))):
        print(f"\nPrompt {i+1}:")
        responses = filtered_dataset.loc[filtered_dataset.index[i], 'response']
        if isinstance(responses, list):
            print(f"Number of responses: {len(responses)}")
            for j, response in enumerate(responses):
                print(f"Response {j+1}: {response[:200]}...")  # Show first 200 chars
        else:
            print(f"Response: {responses}")


if __name__ == "__main__":
    main()

