#!/usr/bin/env python3
"""
Test-time inference with attention_weights summarization mode.

This file replicates the attention_weights summary mode behavior from the rollout code
for standalone inference without the full training pipeline.
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


class AttentionWeightsInference:
    """
    Standalone inference class that replicates the attention_weights summary mode
    from the vLLM rollout implementation.
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
    
    def token_confidence(self, top_logprobs: Dict[str, float]) -> float:
        """
        Compute confidence score based on entropy of top-k logprobs.
        
        Args:
            top_logprobs: Dictionary of token_id -> logprob
            
        Returns:
            Confidence score (higher = more confident)
        """
        if not top_logprobs:
            return 0.0
            
        # Extract logprobs and convert to probabilities
        items = [v.logprob for v in top_logprobs.values()]
        probs = [math.exp(lp) for lp in items]  # logprob -> prob
        Z = sum(probs)
        probs = [p / Z for p in probs]  # Normalize
        
        # Calculate entropy: -sum(p * log(p))
        entropy = -sum(p * math.log(p + 1e-12) for p in probs)
        
        # Return confidence (inverse of entropy)
        return 1.0 / (1.0 + entropy)
    
    def check_format(self, response_ids: List[int]) -> Tuple[bool, str]:
        """
        Check if response is in correct format and extract thinking content.
        
        Args:
            response_ids: List of token IDs
            
        Returns:
            Tuple of (is_correct_format, think_str)
        """
        response_str = self.tokenizer.decode(response_ids)
        
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

    
    def _make_api_request(self, payload: dict) -> str:
        """
        Make API request to summarization service.
        
        Args:
            payload: Request payload
            
        Returns:
            Summarized reasoning chain
        """
        api_endpoint = f"{self.api_base_url}/get_reduced_reasoning_chain"
        
        try:
            response = requests.post(api_endpoint, json=payload, timeout=30)
            response.raise_for_status()
            
            if response.status_code == 200:
                data = response.json()
                return data['new_reasoning_chain']
            else:
                raise ValueError(f"API request failed with status {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise ValueError(f"Failed to get model output: {e}")
    
    def summarize_attention_weights(self, summary_inputs: List[Dict]) -> List[str]:
        """
        Summarize thinking using attention weights approach.
        
        Args:
            summary_inputs: List of input dictionaries with 'think_str', etc.
            
        Returns:
            List of summarized reasoning chains
        """
        summary_outputs = []
        
        for input_data in summary_inputs:
            text = input_data['think_str']
            
            
            # Prepare text for API
            if self.model_path in ["Qwen/Qwen3-4B", "Qwen/Qwen3-8B"]:
                text = f"<think>{text}Time is up. Given the time I've spent and the approaches I've tried, I should stop thinking and now write summarization in one sentence.</think>"
            elif self.model_path in ["deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"]:
                text = f"<think>\n{text}Time is up. Given the time I've spent and the approaches I've tried, I should stop thinking and now write summarization in one sentence.</think>"
            
            payload = {
                "text": text,
                "reduction_score": 0.5
            }
            
            try:
                new_reasoning_chain = self._make_api_request(payload)
                
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
                
                summary_outputs.append(new_reasoning_chain)
                
            except Exception as e:
                logger.error(f"Error processing input: {e}")
                # Fallback to original text
                summary_outputs.append(text)
        
        return summary_outputs
    
    def create_summary_inputs(self, prompt_ids: List[int], summarized_think: str) -> List[int]:
        """
        Create input for generating summarized response.
        
        Args:
            prompt_ids: Original prompt token IDs
            summarized_think: Summarized thinking content
            
        Returns:
            New prompt token IDs with summarized thinking
        """
        prompt_str = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
        
        if self.model_path in ["deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"]:
            new_prompt = prompt_str + summarized_think + "</think>"
        elif self.model_path in ["Qwen/Qwen3-4B", "Qwen/Qwen3-8B"]:
            new_prompt = prompt_str + "<think>" + summarized_think + "</think>"
        else:
            raise ValueError(f"Model path {self.model_path} not supported")
        
        return self.tokenizer.encode(new_prompt, add_special_tokens=True)
    
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
        return [{'text': '', 'logprobs': None}]
    
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
    
    def infer_with_summarization(
        self,
        prompts: List[str],
        data_sources: List[str],
        reward_models: List[Dict],
        rollout_n: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Main inference method with attention_weights summarization.
        
        Args:
            prompts: List of input prompts
            data_sources: List of data source identifiers
            ground_truths: List of ground truth answers
            reward_models: List of reward model dictionaries
            rollout_n: Number of rollouts per prompt
            **kwargs: Additional sampling parameters
            
        Returns:
            Dictionary containing inference results
        """
        # Generate initial responses
        with self.update_sampling_params(**kwargs):
            choices = self._make_vllm_request(prompts)
        
        # Process outputs
        response = []
        counter = 0
        
        # Handle multiple choices per prompt
        for prompt_idx, prompt in enumerate(prompts):
            # Get choices for this prompt (n choices per prompt)
            prompt_choices = choices[prompt_idx * self.sampling_params['n']:(prompt_idx + 1) * self.sampling_params['n']]
            
            for choice in prompt_choices:
                # Extract response text and tokenize
                response_text = choice['text']
                response_ids = self.tokenizer.encode(response_text, add_special_tokens=False)
                prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
                
                response.append({
                    "prompt_ids": prompt_ids,
                    "response_ids": response_ids,
                    "summarized": False,
                    "data_source": data_sources[prompt_idx],
                    "reward_model": reward_models[prompt_idx],
                })
                counter += 1
        
        # Filter and prepare for summarization
        correct_format_response = []
        summary_inputs = []
        
        for i, res in enumerate(response):
            prompt_ids = res["prompt_ids"]
            response_ids = res["response_ids"]
            data_source = res['data_source']
            reward_model = res['reward_model']
            
            # Check format and extract thinking
            correct_format, think_str = self.check_format(response_ids)

            # Remove answer from think_str
            think_str = self.remove_answer_from_think_str(think_str, reward_model['ground_truth'])
            
            
            summary_inputs.append({
                'think_str': think_str,
                'reward_model': reward_model,
            })
            correct_format_response.append({
                "prompt_ids": prompt_ids,
                "response_ids": response_ids,
                "think_str": think_str,
                "data_source": data_source,
                "reward_model": reward_model
            })
        
        print(f"Summarizing {len(summary_inputs)} responses")
        

        # Perform summarization
        summary_outputs = self.summarize_attention_weights(summary_inputs)

        # breakpoint()
        
        # Add summary outputs to responses
        for i, summary_output in enumerate(summary_outputs):
            correct_format_response[i]["summary_output"] = summary_output
        
        # Generate summarized responses
        vllm_inputs_summarized = []
        for res in correct_format_response:
            prompt_ids = res["prompt_ids"]
            summary_output = res["summary_output"]
            
            vllm_input = self.create_summary_inputs(prompt_ids, summary_output)
            vllm_inputs_summarized.append({
                'original_prompt_ids': prompt_ids,
                'original_response_ids': res["response_ids"],
                'summarized_prompt_ids': vllm_input,
            })
        
        # Calculate max tokens for summarized generation
        max_tokens = self.config.get("response_length", 512) - max([
            len(vllm_input['summarized_prompt_ids']) 
            for vllm_input in vllm_inputs_summarized
        ])
        max_tokens = max(1, max_tokens)
        
        print(f"Max tokens for summarized generation: {max_tokens}")
        
        # Generate summarized responses
        with self.update_sampling_params(
            n=1,
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=max_tokens
        ):
            summarized_prompts = [self.tokenizer.decode(vllm_input['summarized_prompt_ids']) 
                                for vllm_input in vllm_inputs_summarized]
            summarized_choices = self._make_vllm_request(summarized_prompts)
        # breakpoint()

        final_response = []
        # Process summarized outputs
        for i, choice in enumerate(summarized_choices):
            response_text = choice['text']
            response_ids = self.tokenizer.encode(response_text, add_special_tokens=False)
            vllm_inputs_summarized[i]["summarized_response_ids"] = response_ids
            vllm_inputs_summarized[i]["summarized"] = True

            prompt_ids = vllm_inputs_summarized[i]["original_prompt_ids"]
            response_ids = vllm_inputs_summarized[i]["summarized_prompt_ids"] + vllm_inputs_summarized[i]["summarized_response_ids"]

            final_response.append({'prompt': self.tokenizer.decode(prompt_ids), 'response': self.tokenizer.decode(response_ids)})
        
            
        
        return final_response


def load_dataset(path_to_parquet):
    import pandas as pd

    df = pd.read_parquet(path_to_parquet)

    return df

def main():
    """Example usage of the AttentionWeightsInference class."""
    import argparse
    from omegaconf import OmegaConf
    
    parser = argparse.ArgumentParser(description="Test attention weights inference")
    parser.add_argument("--model_path", type=str, help="Path to model", default="Qwen/Qwen3-4B")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--vllm_server_url", type=str, default="http://localhost:8000", help="vLLM server URL")
    parser.add_argument("--api_url", type=str, default="http://localhost:8008", help="API base URL")
    parser.add_argument("--prompts", nargs="+", help="Input prompts")
    parser.add_argument("--data_sources", nargs="+", help="Data sources")
    parser.add_argument("--ground_truths", nargs="+", help="Ground truth answers")
    
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
    inference_engine = AttentionWeightsInference(
        model_path=args.model_path,
        config=config,
        tokenizer=tokenizer,
        vllm_server_url=args.vllm_server_url,
        api_base_url=args.api_url
    )
    
    dataset_name = 'full'
    dataset = load_dataset(f"/nas-ssd2/joykirat/code/verl-fork/verl/scripts/data/{dataset_name}_test_dataset/test.parquet")

    # Initialize responses column
    dataset['response'] = None
    from tqdm import tqdm
    for i in tqdm(range(len(dataset))):
        prompts = dataset['prompt'][i]
        data_sources = [dataset['data_source'][i]]
        reward_models = [dataset['reward_model'][i]]

        prompts = [tokenizer.apply_chat_template(prompts, add_generation_prompt=True, tokenize=False)]
        
        results = inference_engine.infer_with_summarization(
            prompts=prompts,
            data_sources=data_sources,
            reward_models=reward_models,
        )
        # breakpoint()

        responses = [result['response'] for result in results]
        dataset.at[i, 'response'] = responses

        # Save progress periodically
        if i % 10 == 0:
            dataset.to_parquet(f"/nas-ssd2/joykirat/code/verl-fork/verl/TTS_attention/{dataset_name}_test_with_responses.parquet")
    
    # Final save
    dataset.to_parquet(f"/nas-ssd2/joykirat/code/verl-fork/verl/TTS_attention/{dataset_name}_test_with_responses.parquet")
    print(f"Completed processing {len(dataset)} samples")
    
    


if __name__ == "__main__":
    main()
