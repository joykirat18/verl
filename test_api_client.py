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
BASE_URL = "http://localhost:8008"
API_ENDPOINTS = {
    "health": f"{BASE_URL}/health",
    "get_reduced_reasoning_chain": f"{BASE_URL}/get_reduced_reasoning_chain",
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

def convert_to_torch_format(attention_weights: dict) -> dict:
    """
    Convert serialized data back to PyTorch tensor format.
    
    Args:
        serialized_data: Dictionary containing serialized input_ids and attention_weights
        
    Returns:
        Dictionary with PyTorch tensors
    """
    try:
        
        # Convert attention weights back to tensors
        attention_weights = []
        for layer_attentions in serialized_data["attention_weights"]:
            layer_tensor = torch.tensor(layer_attentions, dtype=torch.float32)
            attention_weights.append(layer_tensor)
        
        return {
            "attention_weights": attention_weights
        }
    except Exception as e:
        print(f"Error converting to torch format: {e}")
        raise ValueError(f"Failed to convert data to torch format: {str(e)}")

def test_model_output():
    """Test getting model output with attention weights."""
    print("\nTesting model output endpoint...")
    
    test_text = "<think>\nOkay, let's see. Maureen is tracking her quiz scores' mean. The problem says that if she scores an 11 on the next quiz, her mean will increase by 1. Also, if she scores an 11 on each of the next three quizzes, her mean will increase by 2. We need to find her current mean. Hmm, let me think.\n\nFirst, let me recall what mean is. The mean is the sum of all scores divided by the number of scores. So, if she has some current scores, let's denote the current number of quizzes as, say, n, and the current total sum of scores as S. Then her current mean is S/n.\n\nNow, the first scenario: if she scores an 11 on the next quiz, her mean increases by 1. Let me write that down.\n\nAfter scoring an 11 on the next quiz, the total sum becomes S + 11, and the number of quizzes becomes n + 1. The new mean is (S + 11)/(n + 1). According to the problem, this new mean is equal to her current mean plus 1. So:\n\n(S + 11)/(n + 1) = S/n + 1\n\nOkay, that's one equation. Let's note that down.\n\nSecond scenario: if she scores 11 on each of the next three quizzes, her mean will increase by 2. So, after three quizzes, the total sum is S + 11*3 = S + 33, and the number of quizzes is n + 3. The new mean is (S + 33)/(n + 3). According to the problem, this is equal to her current mean plus 2:\n\n(S + 33)/(n + 3) = S/n + 2\n\nSo now we have two equations:\n\n1) (S + 11)/(n + 1) = S/n + 1\n\n2) (S + 33)/(n + 3) = S/n + 2\n\nSo, we need to solve these two equations for S and n. Then, once we have those, we can find the current mean S/n.\n\nLet me try solving the first equation first.\n\nStarting with equation 1:\n\n(S + 11)/(n + 1) = S/n + 1\n\nLet me multiply both sides by n(n + 1) to eliminate denominators.\n\nn(S + 11) = (S/n + 1) * n(n + 1)\n\nWait, maybe better to first subtract S/n + 1 from both sides and then combine terms. Alternatively, cross multiply.\n\nAlternatively:\n\n(S + 11)/(n + 1) - S/n = 1\n\nLet me compute the left side:\n\n[ (S + 11) * n - S(n + 1) ] / [n(n + 1)] = 1\n\nExpanding numerator:\n\n(Sn + 11n - Sn - S) / [n(n + 1)] = 1\n\nSimplify numerator:\n\nSn cancels with -Sn, so left with 11n - S.\n\nTherefore:\n\n(11n - S)/[n(n + 1)] = 1\n\nMultiply both sides by n(n + 1):\n\n11n - S = n(n + 1)\n\nSo:\n\n11n - S = n² + n\n\nRearranged:\n\nS = 11n - n² - n = 10n - n²\n\nSo S = -n² + 10n\n\nSo that's equation 1. Let me write that.\n\nNow, moving to equation 2:\n\n(S + 33)/(n + 3) = S/n + 2\n\nAgain, let's do similar steps.\n\nFirst, subtract S/n + 2 from both sides:\n\n(S + 33)/(n + 3) - S/n - 2 = 0\n\nAlternatively, let's rearrange:\n\n(S + 33)/(n + 3) = S/n + 2\n\nMultiply both sides by n(n + 3):\n\nn(S + 33) = (S/n + 2) * n(n + 3)\n\nWait, similar to before, maybe better to compute the left side minus the right side equals zero.\n\nAlternatively, let's do:\n\n(S + 33)/(n + 3) - S/n = 2\n\nCompute the left side:\n\n[ (S + 33)n - S(n + 3) ] / [n(n + 3)] = 2\n\nExpanding numerator:\n\nSn + 33n - Sn - 3S = 33n - 3S\n\nTherefore:\n\n(33n - 3S)/[n(n + 3)] = 2\n\nMultiply both sides by n(n + 3):\n\n33n - 3S = 2n(n + 3)\n\nSimplify right side: 2n² + 6n\n\nSo:\n\n33n - 3S = 2n² + 6n\n\nLet me rearrange terms:\n\n-3S = 2n² + 6n - 33n\n\nWhich is:\n\n-3S = 2n² - 27n\n\nMultiply both sides by -1:\n\n3S = -2n² + 27n\n\nTherefore:\n\nS = (-2n² + 27n)/3 = (-2/3)n² + 9n\n\nSo now we have two expressions for S:\n\nFrom equation 1: S = -n² + 10n\n\nFrom equation 2: S = (-2/3)n² + 9n\n\nSet them equal:\n\n-n² + 10n = (-2/3)n² + 9n\n\nLet me bring all terms to one side:\n\n-n² + 10n + (2/3)n² - 9n = 0\n\nCombine like terms:\n\n(-n² + (2/3)n²) + (10n - 9n) = 0\n\nCalculating coefficients:\n\n(-1 + 2/3) n² + 1n = 0 => (-1/3)n² + n = 0\n\nMultiply both sides by 3 to eliminate fraction:\n\n- n² + 3n = 0\n\nSo:\n\n- n² + 3n = 0 => n² - 3n = 0 => n(n - 3) = 0\n\nTherefore, n = 0 or n = 3.\n\nBut n is the number of quizzes Maureen has taken so far. It can't be 0 because she has taken some quizzes already. So n = 3.\n\nSo the current number of quizzes is 3. Then, we can find S from equation 1: S = -n² + 10n.\n\nPlugging in n = 3:\n\nS = - (3)^2 + 10*3 = -9 + 30 = 21.\n\nSo the total current sum of scores is 21, and the number of quizzes is 3. Therefore, the current mean is 21 / 3 = 7.\n\nWait, so is the answer 7? Let me check if that works with both equations.\n\nFirst check equation 1: if she scores 11 on the next quiz, the new total is 21 + 11 = 32, and the number of quizzes is 4. The new mean is 32 / 4 = 8. Current mean is 7, so increase by 1. That works.\n\nNow equation 2: if she scores 11 on each of the next three quizzes, total becomes 21 + 3*11 = 21 + 33 = 54. Number of quizzes is 3 + 3 = 6. New mean is 54 / 6 = 9. Current mean was 7, so increase by 2. That also works. Perfect.\n\nSo the current mean is 7. So the answer is 7.\n\nBut let me just make sure there are no other possible solutions. We had n(n - 3) = 0. n = 0 is invalid, so only n = 3. So that seems correct.\n\nTherefore, the mean of her quiz scores currently is 7.\n\n**Final Answer**\nThe mean of her quiz scores currently is \\boxed{7}.\n$$. Time is up. I should stop thinking and now write a summary containing all key steps required to solve the problem.</think>"
#    Time is up. I should stop thinking and now write a summary containing all key steps required to solve the problem. 
    payload = {
        "text": test_text,
        "reduction_score": 0.7
    }
    
    try:
        print("Sending request to get reduced reasoning chain...")

        print("Original reasoning chain: ", test_text)
        
        response = requests.post(API_ENDPOINTS["get_reduced_reasoning_chain"], json=payload)
        if response.status_code == 200:
            data = response.json()
            print("New reasoning chain: ", data['new_reasoning_chain'])
            # print(f"✅ Model output received successfully")
            
            # attention_weights = data['attention_weights']

            # ordered_layer_indices = sorted(attention_weights.keys())
            # attention_weights = torch.stack([attention_weights[i] for i in ordered_layer_indices], dim=0)

            # breakpoint()
            # print("\n🔄 Converting data back to PyTorch tensors...")
            # torch_data = convert_to_torch_format(attention_weights)
            
            # # Display tensor information
            # print(f"✅ Conversion successful!")
            # print(f"   Input IDs: {torch_data['input_ids'].shape} ({torch_data['input_ids'].dtype})")
            # print(f"   Attention weights: {len(torch_data['attention_weights'])} layers")
            # for i, attn in enumerate(torch_data['attention_weights']):
            #     print(f"     Layer {i}: {attn.shape} ({attn.dtype})")
            
            # return torch_data
            new_reasoning_chain = data['new_reasoning_chain']
            breakpoint()
            return None
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