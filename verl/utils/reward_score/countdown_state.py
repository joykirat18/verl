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

        # Rule 2: Must have exactly one <answer> and one </answer>
        if response.count("<answer>") != 1 or response.count("</answer>") != 1:
            return False

        # Rule 3: Must have matching pairs of <think> and </think>
        if response.count("<think>") != response.count("</think>"):
            return False
        if response.count("<think>") == 0:
            return False

        # Rule 4: Must have matching pairs of <state> and </state> (can be zero or more)
        if response.count("<state>") != response.count("</state>"):
            return False

        # Rule 5: Find all tag positions
        import re
        reasoning_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
        state_pattern = re.compile(r'<state>(.*?)</state>', re.DOTALL)
        answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)

        reasoning_matches = list(reasoning_pattern.finditer(response))
        state_matches = list(state_pattern.finditer(response))
        answer_matches = list(answer_pattern.finditer(response))

        if len(answer_matches) != 1:
            return False

        answer_match = answer_matches[0]
        answer_start = answer_match.start()
        answer_end = answer_match.end()

        # Rule 6: All reasoning and state tags must come before <answer>
        for match in reasoning_matches + state_matches:
            if match.end() > answer_start:
                return False

        # Rule 7: Check that the structure follows the pattern: reasoning, (state), reasoning, (state), ..., answer
        # Verify all reasoning blocks have non-empty content
        for match in reasoning_matches:
            content = match.group(1).strip()
            if not content:
                return False

        # Rule 8: Verify all state blocks have non-empty content (if any exist)
        for match in state_matches:
            content = match.group(1).strip()
            if not content:
                return False

        # Rule 9: Verify answer has non-empty content
        answer_content = answer_match.group(1).strip()
        if not answer_content:
            return False

        # Rule 10: Verify the pattern: reasoning, (optional state), reasoning, (optional state), ..., answer
        # Build list of all tags in order
        all_tags = []
        for match in reasoning_matches:
            all_tags.append(('reasoning', match.start(), match.end()))
        for match in state_matches:
            all_tags.append(('state', match.start(), match.end()))
        all_tags.append(('answer', answer_start, answer_end))
        all_tags.sort(key=lambda x: x[1])  # Sort by start position

        # Must start with reasoning and end with answer
        if len(all_tags) < 2:
            return False
        
        if all_tags[0][0] != 'reasoning':
            return False
        
        if all_tags[-1][0] != 'answer':
            return False

        # Verify pattern: reasoning, (optional state), reasoning, (optional state), ..., answer
        # Pattern: each reasoning can be followed by 0 or 1 state, then another reasoning or answer
        i = 0
        while i < len(all_tags) - 1:  # Don't check the last one (answer)
            tag_type, tag_start, tag_end = all_tags[i]
            
            if tag_type == 'reasoning':
                # After reasoning, can have: state (then reasoning or answer), reasoning, or answer
                if i + 1 < len(all_tags) - 1:  # Not at answer yet
                    next_tag_type = all_tags[i + 1][0]
                    if next_tag_type == 'state':
                        # After state, must have reasoning or answer
                        if i + 2 < len(all_tags):
                            # There's something after state
                            if all_tags[i + 2][0] not in ['reasoning', 'answer']:
                                return False
                            i += 2  # Skip reasoning and state
                        elif i + 2 == len(all_tags):
                            # State is followed by answer (which is the last tag), which is valid
                            break
                        else:
                            return False
                    elif next_tag_type == 'reasoning':
                        i += 1  # Move to next reasoning
                    elif next_tag_type == 'answer':
                        # Reasoning directly followed by answer is valid
                        break
                    else:
                        return False
                else:
                    # Next is answer, which is valid
                    break
            elif tag_type == 'state':
                # State must follow reasoning
                if i == 0 or all_tags[i - 1][0] != 'reasoning':
                    return False
                # After state, must have reasoning or answer
                if i + 1 < len(all_tags) - 1:
                    if all_tags[i + 1][0] not in ['reasoning', 'answer']:
                        return False
                i += 1
            else:
                return False

        return True


import re
from typing import List, Tuple

STATE_RE = re.compile(r"<state>(.*?)</state>", re.DOTALL)

def extract_states(model_output: str) -> List[str]:
    """
    Returns raw <state> block contents.
    """
    return [s.strip() for s in STATE_RE.findall(model_output)]

def parse_countdown_state(state_text: str) -> Tuple[List[int], int]:
    """
    Parse numbers and target from a state block.
    Raises ValueError if unparseable.
    """
    lines = [l.strip() for l in state_text.splitlines() if l.strip()]

    numbers = None
    target = None

    for line in lines:
        if line.startswith("numbers:"):
            raw = line[len("numbers:"):].strip()
            if not raw.startswith("[") or not raw.endswith("]"):
                raise ValueError("numbers not in list form")
            nums = raw[1:-1].strip()
            numbers = [] if nums == "" else [int(x.strip()) for x in nums.split(",")]

        elif line.startswith("target:"):
            target = int(line[len("target:"):].strip())

        else:
            raise ValueError("Unknown field in state")

    if numbers is None or target is None:
        raise ValueError("Missing numbers or target")

    return numbers, target

def is_valid_countdown_state(state_text: str) -> bool:
    try:
        numbers, target = parse_countdown_state(state_text)
    except Exception:
        return False

    # numbers must be non-empty
    if len(numbers) == 0:
        return False

    # all integers (already ensured by parsing)
    if not all(isinstance(x, int) for x in numbers):
        return False

    # must be sorted ascending
    if numbers != sorted(numbers):
        return False

    # target must be integer
    if not isinstance(target, int):
        return False

    return True

def intermediate_countdown_state_rewards(model_output: str) -> List[float]:
    """
    Returns one reward per <state>.
    1.0 if parseable & valid, else 0.0
    """
    states = extract_states(model_output)
    return [1.0 if is_valid_countdown_state(s) else 0.0 for s in states]

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
    state_reward = 0
    if checkFormat(solution_str):
        format_reward = 0.5

        intermediate_state_reward = intermediate_countdown_state_rewards(solution_str)
        if len(intermediate_state_reward) == 0:
            state_reward = 0.0
        else:
            state_reward = 0.5
            state_reward += sum(intermediate_state_reward) / len(intermediate_state_reward)
    
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
    
    final_reward = format_reward + correctness_reward + state_reward
    
    return {"score": final_reward, "format_reward": format_reward, "correctness_reward": correctness_reward, "state_reward": state_reward}