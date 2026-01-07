# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        # Extract all numbers from the equation
        numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]

        # Check if all numbers in equation are available
        available_numbers = sorted(available_numbers)
        numbers_in_eq = sorted(numbers_in_eq)

        # Each number should be used exactly once
        return numbers_in_eq == available_numbers
    except Exception:
        return False

def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    
    # if '</think>' in solution_str:
        # solution_str = solution_str.split('</think>')[-1]

    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer

def evaluate_equation(equation_str):
    """Safely evaluate the arithmetic equation using eval() with precautions."""
    try:
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")

        # Evaluate the equation with restricted globals and locals
        result = eval(equation_str, {"__builtins__": None}, {})
        return result
    except Exception:
        return None


def checkFormat(response):
        response = response.strip()

        # Rule 1: Must start with <think> and end with </answer>
        if not response.startswith("<think>") or not response.endswith("</answer>"):
            return False

        # Rule 2: Must contain exactly one of each tag.
        if response.count("<think>") != 1 or response.count("</think>") != 1:
            return False
        if response.count("<answer>") != 1 or response.count("</answer>") != 1:
            return False

        # Find indices for each tag.
        think_open = response.find("<think>")
        think_close = response.find("</think>")
        plan_open = response.find("<answer>")
        plan_close = response.find("</answer>")

        # Rule 3: The order should be: <think> ... </think> then <answer> ... </answer>
        if think_open != 0:  # Should start with <think>
            return False
        if (think_close==-1) or (plan_open==-1) or (plan_close==-1):
            return False
        if think_close > plan_open:
            return False

        # Rule 4: Check non-empty content between tags.
        think_content = response[len("<think>"):think_close].strip()
        plan_content = response[plan_open + len("<answer>"):plan_close].strip()
        if not think_content or not plan_content:
            return False

        # Rule 5: Check <answer> immedietly follows </think>
        if not (response[think_close+len("</think>"):plan_open].strip() == ''):
            return False

        return True


def compute_score(solution_str, ground_truth, method="strict", format_score=0.1, score=1.0):
    """The scoring function for countdown task.

    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """
    final_reward = 0
    format_reward = 0
    correctness_reward = 0
    if checkFormat(solution_str):
        format_reward = 0.5

    
        target = ground_truth["target"]
        numbers = ground_truth["numbers"]

        equation = extract_solution(solution_str=solution_str)

        if equation is None:
            correctness_reward = 0.0
        elif not validate_equation(equation, numbers):
            correctness_reward = 0.0
        else:
            result = evaluate_equation(equation)
            if result is None:
                correctness_reward = 0.0
            else:
                if abs(result - target) < 1e-5:  # Account for floating point precision
                    correctness_reward = 1.0
                else:
                    correctness_reward = 0.0
    
    final_reward = format_reward + correctness_reward
    
    return {"score": final_reward, "format_reward": format_reward, "correctness_reward": correctness_reward}