#!/usr/bin/env python3
"""
Test script for the Redundant Token Eviction via Self-summarization algorithm.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from redundant_token_eviction import run_eviction_algorithm
import os

os.environ['HF_HOME'] = '/nas-ssd2/joykirat/.cache/huggingface'
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

def test_with_example():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", attn_implementation="eager")
    model.to("cuda")

    sentence = "<think>\nOkay, let's try to figure out this problem. So, Maureen is tracking her mean quiz scores. The problem says that if she scores an 11 on the next quiz, her mean will increase by 1. Also, if she scores 11 on each of the next three quizzes, her mean will increase by 2. We need to find her current mean. Hmm, let's break this down step by step.\n\nFirst, let me recall that the mean is the total sum of all scores divided by the number of quizzes. Let me denote her current mean as m, and the number of quizzes she has taken so far as n. Therefore, the total sum of her current scores is n * m.\n\nNow, let's consider the first scenario: if she scores an 11 on the next quiz, her mean increases by 1. So, after taking one more quiz, the total number of quizzes becomes n + 1, and the total sum becomes n * m + 11. The new mean would be (n * m + 11) / (n + 1). According to the problem, this new mean is equal to her current mean plus 1. So, we can write the equation:\n\n(n * m + 11) / (n + 1) = m + 1.\n\nLet me simplify this equation. Multiply both sides by (n + 1):\n\nn * m + 11 = (m + 1)(n + 1)\n\nExpanding the right side:\n\nn * m + 11 = m * n + m + n + 1\n\nSubtract n * m from both sides:\n\n11 = m + n + 1\n\nSo, m + n = 10. Let me note that down as equation (1): m + n = 10.\n\nNow, moving on to the second scenario: if she scores 11 on each of the next three quizzes, her mean increases by 2. So, after three more quizzes, the total number of quizzes is n + 3, and the total sum is n * m + 11 * 3 = n * m + 33. The new mean is (n * m + 33) / (n + 3). According to the problem, this is equal to her current mean plus 2. Therefore:\n\n(n * m + 33) / (n + 3) = m + 2.\n\nAgain, multiply both sides by (n + 3):\n\nn * m + 33 = (m + 2)(n + 3)\n\nExpanding the right side:\n\nn * m + 33 = m * n + 3m + 2n + 6\n\nSubtract n * m from both sides:\n\n33 = 3m + 2n + 6\n\nSimplify:\n\n33 - 6 = 3m + 2n => 27 = 3m + 2n. Let me note this as equation (2): 3m + 2n = 27.\n\nNow, we have two equations:\n\n1) m + n = 10\n\n2) 3m + 2n = 27\n\nWe can solve these equations simultaneously. Let me solve equation (1) for n: n = 10 - m. Substitute this into equation (2):\n\n3m + 2*(10 - m) = 27\n\nSimplify:\n\n3m + 20 - 2m = 27 => m + 20 = 27 => m = 7.\n\nSo, the current mean is 7. Let me check if this is correct.\n\nIf m = 7, then n = 10 - 7 = 3. So, she has taken 3 quizzes so far, with a total of 21 points (3*7). \n\nFirst scenario: adding an 11. Total becomes 21 + 11 = 32, over 4 quizzes. 32 / 4 = 8, which is 7 + 1. Correct.\n\nSecond scenario: adding three 11s. Total becomes 21 + 33 = 54, over 6 quizzes. 54 / 6 = 9, which is 7 + 2. Correct.\n\nSo, the answer is 7. Therefore, the mean of her quiz scores currently is 7.\n\n**Final Answer**\nThe mean of her quiz scores currently is \\boxed{7}.\n</think>"
    input_tokenized = tokenizer(sentence, return_tensors="pt").to("cuda")

    with torch.no_grad():
        model_output = model(**input_tokenized, output_attentions=True)

    # Run the eviction algorithm to get reasoning steps to evict and new reasoning chain
    evicted_steps, new_reasoning_chain = run_eviction_algorithm(
        model_output, 
        tokenizer, 
        input_tokenized, 
        target_reduction=0.25
    )
    
    print(f"\nEvicted reasoning steps: {evicted_steps}")
    print(f"\nNew reasoning chain:")
    print(new_reasoning_chain)
    
    return evicted_steps, new_reasoning_chain

if __name__ == "__main__":
    test_with_example() 