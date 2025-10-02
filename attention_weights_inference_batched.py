#!/usr/bin/env python3
"""
Batched test-time inference with attention_weights summarization mode.

This file processes the entire dataset in three phases:
1. Generate all rollouts for all prompts
2. Summarize all thinking content
3. Complete all generations with summarized content
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

from verl.utils.reward_score import default_compute_score

logger = logging.getLogger(__name__)


class BatchedAttentionWeightsInference:
    """
    Batched inference class that processes all prompts in three phases.
    """
    
    def __init__(
        self,
        model_path: str,
        config: DictConfig,
        tokenizer,
        vllm_server_url: str = "http://localhost:8000",
        api_base_url: str = "http://localhost:8008"
    ):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the model
            config: Configuration dictionary
            tokenizer: Tokenizer instance
            vllm_server_url: Base URL for the vLLM server
            api_base_url: Base URL for the summarization API
        """
        self.model_path = model_path
        self.config = config
        self.tokenizer = tokenizer
        self.vllm_server_url = vllm_server_url
        self.api_base_url = api_base_url
        
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
    
    def check_format(self, response_str: str) -> Tuple[bool, str]:
        """
        Check if response is in correct format and extract thinking content.
        
        Args:
            response_str: Response string
            
        Returns:
            Tuple of (is_correct_format, think_str)
        """
        
        # Check for <think>...</think> pattern
        if self.model_path in ["Qwen/Qwen3-4B", "Qwen/Qwen3-8B"]:
            pattern = r"<think>(.*?)</think>"
            match = re.findall(pattern, response_str, re.DOTALL)
            if match and len(match) == 1:
                think_str = match[0].strip()
                return True, think_str
            else:
                return False, response_str
                
        elif self.model_path in ["deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"]:
            # DeepSeek models use different format
            pattern = r"<think>\n(.*?)</think>"
            match = re.findall(pattern, response_str, re.DOTALL)
            if match and len(match) == 1:
                think_str = match[0].strip()
                return True, think_str
            else:
                return False, response_str
        
        return False, response_str
    
    def remove_answer_from_think_str(self, think_str, answer):
        import re
        # Remove the last reasoning step if it contains the answer (either as a substring or as \boxed{answer})
        if answer is not None and isinstance(think_str, str) and think_str.strip():
            # Split into reasoning steps (by double newlines)
            steps = think_str.strip().split('\n\n')
            answer_str = str(answer).strip()
            
            # Recursively remove last step if it contains the answer or \boxed{answer}
            while steps:
                last_step = steps[-1]
                boxed_match = re.search(r'\\boxed\{([A-Za-z0-9]+)\}', last_step)
                if (answer_str and answer_str in last_step) or boxed_match:
                    steps = steps[:-1]
                else:
                    break
            think_str = '\n\n'.join(steps)
        return think_str
    
    def _make_api_request(self, payload: dict, max_retries: int = 3, timeout: int = 30) -> str:
        """
        Make API request to summarization service with retry mechanism.
        
        Args:
            payload: Request payload
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
            
        Returns:
            Summarized reasoning chain
        """
        api_endpoint = f"{self.api_base_url}/get_reduced_reasoning_chain"
        
        for attempt in range(max_retries + 1):
            try:
                logger.info(f"Making API request (attempt {attempt + 1}/{max_retries + 1})")
                response = requests.post(api_endpoint, json=payload, timeout=timeout)
                response.raise_for_status()
                
                if response.status_code == 200:
                    data = response.json()
                    return data['new_reasoning_chain']
                else:
                    raise ValueError(f"API request failed with status {response.status_code}")
                    
            except (requests.exceptions.RequestException, TimeoutError) as e:
                logger.warning(f"API request failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                
                if attempt < max_retries:
                    # Exponential backoff: wait 2^attempt seconds
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {max_retries + 1} attempts failed.")
                    raise ValueError(f"Failed to get model output after {max_retries + 1} attempts: {e}")
    
    def summarize_attention_weights(self, summary_inputs: List[Dict]) -> List[str]:
        """
        Summarize thinking using attention weights approach.
        
        Args:
            summary_inputs: List of input dictionaries with 'think_str', etc.
            
        Returns:
            List of summarized reasoning chains
        """
        summary_outputs = []
        from tqdm import tqdm
        for input_data in tqdm(summary_inputs):
            text = input_data['think_str']
            apply_summarization = input_data['apply_summarization']

            # Prepare text for API
            if self.model_path in ["Qwen/Qwen3-4B", "Qwen/Qwen3-8B"]:
                text = f"<think>{text}Time is up. Given the time I've spent and the approaches I've tried, I should stop thinking and now write summarization in one sentence.</think>"
            elif self.model_path in ["deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"]:
                text = f"<think>\n{text}Time is up. Given the time I've spent and the approaches I've tried, I should stop thinking and now write summarization in one sentence.</think>"
            
            payload = {
                "text": text,
                "reduction_score": 0.4
            }

            if apply_summarization:
                new_reasoning_chain = self._make_api_request(payload)
            else:
                new_reasoning_chain = text
            
            # Clean up the response
            if "Time is up. Given the time I've spent and the approaches I've tried, I should stop thinking and now write summarization in one sentence." in new_reasoning_chain:
                new_reasoning_chain = new_reasoning_chain.replace(
                    "Time is up. Given the time I've spent and the approaches I've tried, I should stop thinking and now write summarization in one sentence.", 
                    ""
                )
            if "<think>" in new_reasoning_chain:
                new_reasoning_chain = new_reasoning_chain.replace("<think>", "")
            if "</think>" in new_reasoning_chain:
                new_reasoning_chain = new_reasoning_chain.replace("</think>", "")
            
            # breakpoint()
            summary_outputs.append(new_reasoning_chain)
                
            # except Exception as e:
            #     logger.error(f"Error processing input: {e}")
            #     # Fallback to original text
            #     summary_outputs.append(text)
        
        return summary_outputs
    
    def create_summary_inputs(self, prompt: str, summarized_think: str) -> List[int]:
        """
        Create input for generating summarized response.
        
        Args:
            prompt: Original prompt
            summarized_think: Summarized thinking content
            
        Returns:
            New prompt token IDs with summarized thinking
        """
        prompt_str = prompt
        
        if self.model_path in ["deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"]:
            new_prompt = prompt_str + summarized_think + "</think>"
        elif self.model_path in ["Qwen/Qwen3-4B", "Qwen/Qwen3-8B"]:
            new_prompt = prompt_str + "<think>" + summarized_think + "</think>"
        else:
            raise ValueError(f"Model path {self.model_path} not supported")
        
        return self.tokenizer.encode(new_prompt, add_special_tokens=True)
    
    @contextmanager
    def update_sampling_params(self, **kwargs):
        """Context manager to temporarily update sampling parameters."""
        old_params = self.sampling_params.copy()
        new_params = self.sampling_params.copy()
        new_params.update(kwargs)
        self.sampling_params = new_params
        try:
            yield
        finally:
            self.sampling_params = old_params
    
    def phase1_generate_all_rollouts(self, prompts: List[str], data_sources: List[str], batch_size: int = 10) -> List[Dict]:
        """
        Phase 1: Generate all rollouts for all prompts in batches.
        
        Args:
            prompts: List of input prompts
            data_sources: List of data source identifiers
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
                        "data_source": data_sources[prompt_idx],
                    })
        
        print(f"Generated {len(all_responses)} rollouts for {len(prompts)} prompts")
        return all_responses
    
    def phase2_summarize_all(self, all_prompts: List[str], all_responses: List[str]) -> List[Dict]:
        """
        Phase 2: Summarize all thinking content.
        
        Args:
            all_prompts: List of all prompts
            all_responses: List of all generated responses
            
        Returns:
            List of responses with summarized thinking
        """
        print("Phase 2: Summarizing all thinking content...")
        
        # Filter and prepare for summarization
        correct_format_responses = []
        summary_inputs = []
        
        for prompt, res in zip(all_prompts, all_responses):
            
            # Check format and extract thinking
            correct_format, think_str = self.check_format(res)
            
            
            summary_inputs.append({
                'think_str': think_str,
                'apply_summarization': correct_format
            })
            correct_format_responses.append({
                "prompt": prompt,
                "response": res,
                "think_str": think_str,
                "apply_summarization": correct_format
            })
        
        print(f"Summarizing {len(summary_inputs)} responses")
        
        # Perform summarization
        summary_outputs = self.summarize_attention_weights(summary_inputs)
        
        # Add summary outputs to responses
        for i, summary_output in enumerate(summary_outputs):
            correct_format_responses[i]["summary_output"] = summary_output
        
        print(f"Completed summarization for {len(correct_format_responses)} responses")
        return correct_format_responses
    
    def phase3_complete_all_generations(self, summarized_responses: List[Dict], batch_size: int = 1) -> List[Dict]:
        """
        Phase 3: Complete all generations with summarized content (batched processing).
        
        Args:
            summarized_responses: List of responses with summarized thinking
            batch_size: Number of responses to process in each batch
            
        Returns:
            List of final completed responses
        """
        print(f"Phase 3: Completing all generations in batches of {batch_size}...")
        
        final_responses = []
        
        # Process responses in batches
        for batch_start in tqdm(range(0, len(summarized_responses), batch_size), desc="Processing completion batches"):
            batch_end = min(batch_start + batch_size, len(summarized_responses))
            batch_responses = summarized_responses[batch_start:batch_end]
            
            # Prepare batch inputs
            batch_vllm_inputs = []
            batch_prompts = []
            batch_metadata = []
            
            for i in range(len(batch_responses)):
                prompt = batch_responses[i]['prompt']
                res = batch_responses[i]['response']
                think_str = batch_responses[i]['think_str']
                summary_output = batch_responses[i]['summary_output']
                
                # Create summarized prompt for this response
                vllm_input = self.create_summary_inputs(prompt, summary_output)
                # breakpoint()
                summarized_prompt = self.tokenizer.decode(vllm_input)
                
                batch_vllm_inputs.append(vllm_input)
                batch_prompts.append(summarized_prompt)
                batch_metadata.append({
                    'prompt': prompt,
                    'response': res
                })
            
            # Calculate max_tokens for the batch (use minimum to ensure all fit)
            max_tokens_list = [self.config.get("response_length", 10000) - len(vllm_input) for vllm_input in batch_vllm_inputs]
            max_tokens = max(1, min(max_tokens_list))
            
            print(f"Batch {batch_start//batch_size + 1}: Processing {len(batch_prompts)} prompts, max_tokens: {max_tokens}")
            
            # Generate summarized responses for the entire batch
            with self.update_sampling_params(
                n=1,
                temperature=1.0,
                top_p=1.0,
                top_k=-1,
                max_tokens=max_tokens
            ):
                summarized_choices = self._make_vllm_request(batch_prompts)
            
            # Process the batch responses
            for idx, (choice, vllm_input, metadata) in enumerate(zip(summarized_choices, batch_vllm_inputs, batch_metadata)):
                if choice and choice.get('text'):
                    response_text = choice['text']
                    response_ids = self.tokenizer.encode(response_text, add_special_tokens=False)
                    final_response_ids = vllm_input + response_ids
                    # breakpoint()
                    final_responses.append({
                        'prompt': metadata['prompt'], 
                        'response': self.tokenizer.decode(final_response_ids)
                    })
                else:
                    # Fallback to original response if summarization fails
                    final_responses.append({
                        'prompt': metadata['prompt'], 
                        'response': metadata['response']
                    })
        
        print(f"Completed {len(final_responses)} final generations")
        return final_responses
    
    def infer_with_summarization_batched(
        self,
        prompts: List[str],
        data_sources: List[str],
        responses: List[str],
        **kwargs
    ) -> List[Dict]:
        """
        Main batched inference method with attention_weights summarization.
        
        Args:
            prompts: List of input prompts
            data_sources: List of data source identifiers
            responses: List of responses
            **kwargs: Additional sampling parameters
            
        Returns:
            List of final completed responses
        """
        # Phase 1: Generate all rollouts
        # all_responses = self.phase1_generate_all_rollouts(prompts, data_sources, batch_size=10)
        # print(f"Phase 1: Generated {len(all_responses)} rollouts")
        
        # Phase 2: Summarize all thinking content
        summarized_responses = self.phase2_summarize_all(prompts, responses)
        print(f"Phase 2: Summarized {len(summarized_responses)} responses")
        # breakpoint()
        # Phase 3: Complete all generations
        final_responses = self.phase3_complete_all_generations(summarized_responses)
        
        return final_responses


def load_dataset(path_to_parquet):
    import pandas as pd
    df = pd.read_parquet(path_to_parquet)
    return df


def main():
    """Example usage of the BatchedAttentionWeightsInference class."""
    import argparse
    from omegaconf import OmegaConf
    
    parser = argparse.ArgumentParser(description="Test batched attention weights inference")
    parser.add_argument("--model_path", type=str, help="Path to model", default="Qwen/Qwen3-4B")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--vllm_server_url", type=str, default="http://localhost:8000", help="vLLM server URL")
    parser.add_argument("--api_url", type=str, default="http://localhost:8008", help="API base URL")
    parser.add_argument("--dataset_name", type=str, default="full", help="Dataset name")
    parser.add_argument("--data_source", type=str, default="math-amc", help="Data source")
    
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
    inference_engine = BatchedAttentionWeightsInference(
        model_path=args.model_path,
        config=config,
        tokenizer=tokenizer,
        vllm_server_url=args.vllm_server_url,
        api_base_url=args.api_url
    )
    
    dataset_name = args.dataset_name
    selected_data_source = args.data_source
    import json
    with open('/nas-ssd2/joykirat/code/verl-fork/verl/TTS_attention/rollouts.jsonl', 'r') as f:
        dataset = [json.loads(line) for line in f]
    
    filtered_dataset = [d for d in dataset if d['data_source'] == selected_data_source]

    
    # dataset = load_dataset(f"/nas-ssd2/joykirat/code/verl-fork/verl/scripts/data/{dataset_name}_test_dataset/test.parquet")

    # Filter dataset to only include selected data source
    # filtered_dataset = dataset[dataset['data_source'] == selected_data_source].copy()
    # Reset index to ensure consecutive integer indices starting from 0
    # filtered_dataset = filtered_dataset.reset_index(drop=True)
    # print(f"Filtered dataset from {len(dataset)} to {len(filtered_dataset)} samples for data source: {selected_data_source}")

    # Initialize responses column
    # filtered_dataset['response'] = None
    
    # Process all prompts at once
    # print(f"Processing {len(filtered_dataset)} samples...")
    
    # Prepare all inputs
    all_prompts = []
    all_data_sources = []
    all_responses = []
    
    for i in range(len(filtered_dataset)):
        prompt = filtered_dataset[i]['prompt'].replace('user', '').replace('assistant', '').strip()
        prompt = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
        data_source = filtered_dataset[i]['data_source']
        response = filtered_dataset[i]['response']
        all_prompts.append(prompt)
        all_responses.append(response)
        all_data_sources.append(data_source)
    
    # Run batched inference
    results = inference_engine.infer_with_summarization_batched(
        prompts=all_prompts,
        data_sources=all_data_sources,
        responses=all_responses,
    )

    all_summarized_responses = [r['response'] for r in results]
    
    # # Organize results by prompt index
    # results_by_prompt = {}
    # for result in results:
    #     prompt_idx = result['prompt_idx']
    #     if prompt_idx not in results_by_prompt:
    #         results_by_prompt[prompt_idx] = []
    #     results_by_prompt[prompt_idx].append(result['response'])

    for i in range(len(filtered_dataset)):
        filtered_dataset[i]['summarized_response'] = all_summarized_responses[i]

    # save dataset to jsonl
    with open(f"/nas-ssd2/joykirat/code/verl-fork/verl/TTS_attention/{dataset_name}_{selected_data_source}_test_with_responses_batched_v5.jsonl", 'w') as f:
        for i in range(len(filtered_dataset)):
            f.write(json.dumps(filtered_dataset[i], ensure_ascii=False) + '\n')
    
    # # Save results to filtered dataset
    # for prompt_idx, responses in results_by_prompt.items():
    #     # Store the list of responses for each prompt using at[] for single cell assignment
    #     filtered_dataset.at[filtered_dataset.index[prompt_idx], 'response'] = responses
    
    # # Save final filtered dataset
    # filtered_dataset.to_parquet(f"/nas-ssd2/joykirat/code/verl-fork/verl/TTS_attention/{dataset_name}_{selected_data_source}_test_with_responses_batched_v2.parquet")
    # print(f"Completed processing {len(filtered_dataset)} samples")
    
    # # Print sample results
    # print("\nSample Results:")
    # for i in range(min(3, len(filtered_dataset))):
    #     print(f"\nPrompt {i+1}:")
    #     responses = filtered_dataset.loc[filtered_dataset.index[i], 'response']
    #     if isinstance(responses, list):
    #         print(f"Number of responses: {len(responses)}")
    #         for j, response in enumerate(responses):
    #             print(f"Response {j+1}: {response[:200]}...")  # Show first 200 chars
    #     else:
    #         print(f"Response: {responses}")


if __name__ == "__main__":
    main()
