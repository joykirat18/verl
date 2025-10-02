#!/usr/bin/env python3
"""
Test script for the attention_weights inference implementation.
"""

import os
import sys
from omegaconf import OmegaConf
from transformers import AutoTokenizer

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from attention_weights_inference import AttentionWeightsInference


def test_attention_weights_inference():
    """Test the attention weights inference with sample data."""
    
    # Configuration
    model_path = "Qwen/Qwen3-4B"  # Change this to your model path
    config = OmegaConf.create({
        "response_length": 512,
        "temperature": 1.0,
        "top_p": 1.0,
        "tensor_model_parallel_size": 1,
        "gpu_memory_utilization": 0.8,
        "max_model_len": 4096,
        "trust_remote_code": True,
    })
    
    # Sample test data
    test_prompts = [
        "What is 2 + 2? Think step by step.",
        "Solve: 3x + 5 = 14. Show your work.",
        "What is the capital of France? Explain your reasoning."
    ]
    
    test_data_sources = ["math-numina", "math-numina", "commonsense"]
    test_ground_truths = ["4", "3", "Paris"]
    
    # Create reward models
    reward_models = [{"ground_truth": gt} for gt in test_ground_truths]
    
    try:
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Initialize inference engine
        print("Initializing inference engine...")
        inference_engine = AttentionWeightsInference(
            model_path=model_path,
            config=config,
            tokenizer=tokenizer,
            api_base_url="http://localhost:8008"  # Make sure this API is running
        )
        
        print("Running inference...")
        results = inference_engine.infer_with_summarization(
            prompts=test_prompts,
            data_sources=test_data_sources,
            ground_truths=test_ground_truths,
            reward_models=reward_models,
            rollout_n=1
        )
        
        # Print results
        print("\n" + "="*50)
        print("INFERENCE RESULTS")
        print("="*50)
        
        for i, (prompt, response, is_summarized, difficulty) in enumerate(zip(
            test_prompts,
            results['responses'], 
            results['is_summarized'], 
            results['difficulties']
        )):
            response_text = tokenizer.decode(response, skip_special_tokens=True)
            print(f"\nPrompt {i+1}: {prompt}")
            print(f"Response: {response_text}")
            print(f"Summarized: {is_summarized}")
            print(f"Difficulty: {difficulty:.2f}")
            print("-" * 30)
        
        print(f"\nTotal responses: {len(results['responses'])}")
        print(f"Summarized responses: {sum(results['is_summarized'])}")
        print(f"Average difficulty: {sum(results['difficulties']) / len(results['difficulties']):.2f}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


def test_individual_methods():
    """Test individual methods without full inference."""
    
    print("Testing individual methods...")
    
    # Mock configuration
    config = OmegaConf.create({
        "response_length": 512,
        "temperature": 1.0,
        "top_p": 1.0,
    })
    
    # Mock tokenizer (you would use a real one in practice)
    class MockTokenizer:
        def __init__(self):
            self.pad_token = "<pad>"
            self.eos_token = "<eos>"
        
        def encode(self, text, add_special_tokens=True):
            # Simple mock encoding
            return [1, 2, 3, 4, 5]
        
        def decode(self, token_ids, skip_special_tokens=True):
            # Simple mock decoding
            return f"Mock response for tokens {token_ids}"
        
        def convert_ids_to_tokens(self, token_id):
            return f"token_{token_id}"
    
    # Create inference engine with mock components
    inference_engine = AttentionWeightsInference(
        model_path="Qwen/Qwen3-4B",
        config=config,
        tokenizer=MockTokenizer(),
        api_base_url="http://localhost:8008"
    )
    
    # Test difficulty classification
    print("Testing difficulty classification...")
    difficulties = [0.9, 0.5, 0.1, None]
    for diff in difficulties:
        category = inference_engine.get_difficulty_class(diff)
        print(f"Difficulty {diff} -> {category}")
    
    # Test reduction scores
    print("\nTesting reduction scores...")
    categories = ["easy", "medium", "hard", "no_difficulty"]
    for cat in categories:
        score = inference_engine.get_reduction_score(cat)
        print(f"Category {cat} -> {score}")
    
    # Test confidence calculation
    print("\nTesting confidence calculation...")
    mock_logprobs = {
        "token1": type('obj', (object,), {'logprob': -0.1})(),
        "token2": type('obj', (object,), {'logprob': -0.2})(),
        "token3": type('obj', (object,), {'logprob': -0.3})(),
    }
    confidence = inference_engine.token_confidence(mock_logprobs)
    print(f"Confidence score: {confidence:.3f}")
    
    print("Individual method tests completed!")


if __name__ == "__main__":
    print("Attention Weights Inference Test")
    print("=" * 40)
    
    # Test individual methods first
    test_individual_methods()
    
    print("\n" + "=" * 40)
    print("Note: Full inference test requires:")
    print("1. A valid model path")
    print("2. GPU resources")
    print("3. Running summarization API at http://localhost:8008")
    print("=" * 40)
    
    # Uncomment the line below to run full inference test
    # test_attention_weights_inference()
