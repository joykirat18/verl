#!/usr/bin/env python3
"""
Test client for the FastAPI model host API.
This script demonstrates how to use the API endpoints.
"""

import requests
import json
import time
import torch
import numpy as np

# API configuration
BASE_URL = "http://localhost:8000"
API_ENDPOINTS = {
    "health": f"{BASE_URL}/health",
    "model_output": f"{BASE_URL}/model_output",
}

def test_health():
    """Test the health endpoint."""
    print("Testing health endpoint...")
    try:
        response = requests.get(API_ENDPOINTS["health"])
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the API server. Make sure it's running.")
        return False

def convert_to_torch_format(serialized_data: dict) -> dict:
    """
    Convert serialized data back to PyTorch tensor format.
    
    Args:
        serialized_data: Dictionary containing serialized input_ids and attention_weights
        
    Returns:
        Dictionary with PyTorch tensors
    """
    try:
        # Convert input_ids back to tensor
        input_ids = torch.tensor(serialized_data["input_ids"], dtype=torch.long)
        
        # Convert attention weights back to tensors
        attention_weights = []
        for layer_attentions in serialized_data["attention_weights"]:
            layer_tensor = torch.tensor(layer_attentions, dtype=torch.float32)
            attention_weights.append(layer_tensor)
        
        return {
            "input_ids": input_ids,
            "attention_weights": attention_weights
        }
    except Exception as e:
        print(f"Error converting to torch format: {e}")
        raise ValueError(f"Failed to convert data to torch format: {str(e)}")

def test_model_output():
    """Test getting model output with attention weights."""
    print("\nTesting model output endpoint...")
    
    test_text = "<think>\nOkay, let's try to figure out this problem. So, Maureen is tracking her mean quiz scores. The problem says that if she scores an 11 on the next quiz, her mean will increase by 1. Also, if she scores 11 on each of the next three quizzes, her mean will increase by 2. We need to find her current mean.\n</think>"
    
    payload = {
        "text": test_text,
        "return_attention": True
    }
    
    try:
        response = requests.post(API_ENDPOINTS["model_output"], json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model output received successfully")
            print(f"   Input length: {len(data['input_ids'])} tokens")
            print(f"   Attention weights: {len(data['attention_weights'])} layers")
            
            # Convert back to PyTorch tensors
            print("\n🔄 Converting data back to PyTorch tensors...")
            torch_data = convert_to_torch_format(data)
            
            # Display tensor information
            print(f"✅ Conversion successful!")
            print(f"   Input IDs: {torch_data['input_ids'].shape} ({torch_data['input_ids'].dtype})")
            print(f"   Attention weights: {len(torch_data['attention_weights'])} layers")
            for i, attn in enumerate(torch_data['attention_weights']):
                print(f"     Layer {i}: {attn.shape} ({attn.dtype})")
            
            return torch_data
        else:
            print(f"❌ Model output failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error testing model output: {e}")
        return None

def main():
    """Main test function."""
    print("🚀 Starting API client tests...")
    print(f"📡 API Base URL: {BASE_URL}")
    
    # Wait a bit for the server to be ready
    print("\n⏳ Waiting for server to be ready...")
    time.sleep(2)
    
    # Test health endpoint
    if not test_health():
        print("❌ Health check failed. Exiting.")
        return
    
    # Test model output
    model_output = test_model_output()
    if model_output is None:
        print("❌ Model output test failed. Exiting.")
        return


if __name__ == "__main__":
    main() 