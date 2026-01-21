import re
import sympy
import logging
from sympy.parsing.latex import parse_latex
from functools import partial, update_wrapper
import signal
import json
from typing import List
import pandas as pd

class BlocksworldCorrectnessReward:
    @staticmethod
    def normalize_input_string(s: str) -> str:
        return re.sub(r'\bblock (\d+)\b', r'\1 block', s, flags=re.IGNORECASE)
    
    @staticmethod
    def normalize_block_name(name: str) -> str:
        name = name.strip().lower()
        if name.endswith(" block"):
            name = name[:-6].strip()
        return name
    
    @staticmethod
    def denormalize_block_name(name: str) -> str:
        return f"{name} block"
    
    @staticmethod
    def is_clear(block: str, state: dict, hand: str) -> bool:
        """
        A block is clear if no other block is resting on top of it.
        (A block held by the hand is not considered clear.)
        """
        if state.get(block) == "hand":
            return False
        for b, loc in state.items():
            if loc == block:
                return False
        return True
    
    @classmethod
    def parse_initial_state(cls, state_str: str):
        """
        Parses the initial state string into:
          - state: a dictionary mapping each normalized block to its location
                   ("table", another block, or "hand")
          - hand: None if empty, or the block currently held
          - blocks_order: a list of block names (normalized) in order of appearance.
        """
        state_str = cls.normalize_input_string(state_str)
        normalized = state_str.replace(" and ", ", ")
        facts = [fact.strip() for fact in normalized.split(",")]
        
        state = {}  # maps block -> location ("table", another block, or "hand")
        hand = None
        blocks_order = []

        def add_block(b: str) -> str:
            bn = cls.normalize_block_name(b)
            if bn not in blocks_order:
                blocks_order.append(bn)
            return bn

        for fact in facts:
            # Hand facts.
            if re.fullmatch(r"the hand is empty", fact, re.IGNORECASE):
                hand = None
            elif m := re.match(r"the hand is holding the (.+?)(?: block)?$", fact, re.IGNORECASE):
                block = cls.normalize_block_name(m.group(1))
                hand = block
                state[block] = "hand"
                add_block(block)
            # Clear facts.
            elif m := re.match(r"the (.+?)(?: block)? is clear$", fact, re.IGNORECASE):
                block = cls.normalize_block_name(m.group(1))
                state.setdefault(block, None)
                add_block(block)
            # "On top of" relations.
            elif m := re.match(r"the (.+?)(?: block)? is on top of the (.+?)(?: block)?$", fact, re.IGNORECASE):
                block = cls.normalize_block_name(m.group(1))
                support = cls.normalize_block_name(m.group(2))
                state[block] = support
                add_block(block)
                add_block(support)
            # "On the table" relations.
            elif m := re.match(r"the (.+?)(?: block)? is on the table$", fact, re.IGNORECASE):
                block = cls.normalize_block_name(m.group(1))
                state[block] = "table"
                add_block(block)
            # Unrecognized facts are ignored.
        return state, hand, blocks_order
    
    @classmethod
    def parse_action(cls, action_str: str):
        """
        Parses an action string into a tuple describing the action.
        Supported actions:
          - "unstack the <block> from on top of the <support>"
          - "pick up the <block>"
          - "stack the <block> on top of the <support>"
          - "put down the <block>"
        The action string is normalized before parsing.
        """
        action_str = cls.normalize_input_string(action_str)
        action_str = action_str.lower().strip()
        # print(action_str)
        patterns = [
            (
                r"^unstack the (\w+) block from on top of the (\w+) block[^\w\s]*\s*$",
                lambda m: ("unstack", cls.normalize_block_name(m.group(1)), cls.normalize_block_name(m.group(2)))
            ),
            (
                r"^pick up the (\w+) block[^\w\s]*\s*$",
                lambda m: ("pickup", cls.normalize_block_name(m.group(1)))
            ),
            (
                r"^stack the (\w+) block on top of the (\w+) block[^\w\s]*\s*$",
                lambda m: ("stack", cls.normalize_block_name(m.group(1)), cls.normalize_block_name(m.group(2)))
            ),
            (
                r"^put down the (\w+) block[^\w\s]*\s*$",
                lambda m: ("putdown", cls.normalize_block_name(m.group(1)))
            ),
        ]
        for pattern, action_fn in patterns:
            m = re.match(pattern, action_str)
            if m:
                return action_fn(m)
        raise ValueError("Action not recognized or unsupported.")
    

    @classmethod
    def simulate_action(cls, state: dict, hand: str, blocks_order: list, action_tuple: tuple):
        """
        Applies the given action to the state.
        Supported actions: "unstack", "pickup", "stack", "putdown".
        Returns the updated state and hand.
        """
        action = action_tuple[0]
        if action == "unstack":
            block, support = action_tuple[1], action_tuple[2]
            if state.get(block) != support:
                raise Exception(f"Precondition failed: {cls.denormalize_block_name(block)} is not on {cls.denormalize_block_name(support)}.")
            if not cls.is_clear(block, state, hand):
                raise Exception(f"Precondition failed: {cls.denormalize_block_name(block)} is not clear.")
            if hand is not None:
                raise Exception("Precondition failed: hand is not empty.")
            state[block] = "hand"
            hand = block
        elif action == "pickup":
            block = action_tuple[1]
            if state.get(block) != "table":
                raise Exception(f"Precondition failed: {cls.denormalize_block_name(block)} is not on the table.")
            if not cls.is_clear(block, state, hand):
                raise Exception(f"Precondition failed: {cls.denormalize_block_name(block)} is not clear.")
            if hand is not None:
                raise Exception("Precondition failed: hand is not empty.")
            state[block] = "hand"
            hand = block
        elif action == "stack":
            block, support = action_tuple[1], action_tuple[2]
            if hand != block:
                raise Exception(f"Precondition failed: hand is not holding {cls.denormalize_block_name(block)}.")
            if not cls.is_clear(support, state, hand):
                raise Exception(f"Precondition failed: {cls.denormalize_block_name(support)} is not clear.")
            state[block] = support
            hand = None
        elif action == "putdown":
            block = action_tuple[1]
            if hand != block:
                raise Exception(f"Precondition failed: hand is not holding {cls.denormalize_block_name(block)}.")
            state[block] = "table"
            hand = None
        else:
            raise Exception("Action not supported.")
        return state, hand
    
    @classmethod
    def generate_state_string(cls, state: dict, hand: str, blocks_order: list) -> str:
        """
        Generates a state description string in the same format as the input.
        The output includes clear facts, the hand status, and location facts.
        """
        facts = []
        for block in blocks_order:
            if state.get(block) != "hand" and cls.is_clear(block, state, hand):
                facts.append(f"the {cls.denormalize_block_name(block)} is clear")
        if hand is None:
            facts.append("the hand is empty")
        else:
            facts.append(f"the hand is holding the {cls.denormalize_block_name(hand)}")
        for block in blocks_order:
            loc = state.get(block)
            if loc is None or loc == "hand":
                continue
            elif loc == "table":
                facts.append(f"the {cls.denormalize_block_name(block)} is on the table")
            else:
                facts.append(f"the {cls.denormalize_block_name(block)} is on top of the {cls.denormalize_block_name(loc)}")
        if not facts:
            return ""
        if len(facts) == 1:
            return facts[0]
        return ", ".join(facts[:-1]) + " and " + facts[-1]
    
    @classmethod
    def simulate_step(cls, state_str: str, action_str: str) -> str:
        """
        Simulates a single action (step) given an initial state string and an action string.
        Returns the new state as a description string.
        """
        state, hand, blocks_order = cls.parse_initial_state(state_str)
        action_tuple = cls.parse_action(action_str)
        new_state, new_hand = cls.simulate_action(state, hand, blocks_order, action_tuple)
        return cls.generate_state_string(new_state, new_hand, blocks_order)
    
    @classmethod
    def simplify_state_given_reference(cls, reference_state_str: str, final_state_str: str) -> str:
        """
        Compares a reference state with a final state (both given as descriptive strings)
        and returns a simplified description containing only the blocks whose locations
        have changed. If a block is held, it is reported as "the X block is in hand". For
        other changes, it returns "the X block is on top of the Y block" or "on the table".
        
        The resulting facts are returned in an arbitrary (jumbled) order.
        """
        # Use the existing parsing method to obtain state mappings and hand status.
        ref_state, ref_hand, _ = cls.parse_initial_state(reference_state_str)
        final_state, final_hand, _ = cls.parse_initial_state(final_state_str)
        
        # Collect differences: consider every block appearing in either state.
        diff = {}
        all_blocks = set(ref_state.keys()) | set(final_state.keys())
        for block in all_blocks:
            if ref_state.get(block) != final_state.get(block):
                diff[block] = final_state.get(block)
        
        # If the hand status has changed, record that change.
        if final_hand is not None and final_hand != ref_hand:
            diff[final_hand] = "hand"
        
        # Build descriptive facts for each differing block.
        facts = []
        for block, loc in diff.items():
            if loc == "table":
                facts.append(f"the {cls.denormalize_block_name(block)} is on the table")
            elif loc == "hand":
                facts.append(f"the hand is holding the {cls.denormalize_block_name(block)}")
            else:
                facts.append(f"the {cls.denormalize_block_name(block)} is on top of the {cls.denormalize_block_name(loc)}")
        
        if not facts:
            return ""
        elif len(facts) == 1:
            return facts[0]
        else:
            return ", ".join(facts[:-1]) + " and " + facts[-1]
    
    @classmethod
    def simulate_plan(cls, init_state, plan=None, simplify=False):
        """
        Simulates the sequence of actions in the plan starting from the initial state.
        Returns the final internal state, hand, blocks_order, and a description string.
        """
        lines = [line.strip() for line in plan.strip().split("\n")
                 if line.strip() and "[plan end]" not in line.lower()]
        state = init_state
        for action_line in lines:
            state = cls.simulate_step(state, action_line)
        final_state_str = state
        if simplify:
            final_state_str = cls.simplify_state_given_reference(init_state, final_state_str)
        return final_state_str
    
    @classmethod
    def states_equal(cls, state_str1: str, state_str2: str) -> tuple:
        # Parse both states.
        state1, hand1, _ = cls.parse_initial_state(state_str1)
        state2, hand2, _ = cls.parse_initial_state(state_str2)
        
        equal = True
        differences = []
        
        # Compare hand status.
        if hand1 != hand2:
            equal = False
            if hand2 is None:
                differences.append("hand should be empty")
            else:
                differences.append(f"the {cls.denormalize_block_name(hand2)} should be in hand")
        
        # For each expected block relation in state2, verify state1 matches.
        for block, expected in state2.items():
            if block not in state1 or state1[block] != expected:
                equal = False
                differences.append(
                    f"the {cls.denormalize_block_name(block)} should be on {cls.denormalize_block_name(expected)}"
                )
        return equal, differences
    
    @classmethod
    def check_goal(cls, state: str, goal: str):
        return cls.states_equal(state, goal)
    
    
    @classmethod
    def simulate_plan_with_reward(cls, init_state: str, predicted_plan: str, true_plan: str) -> float:
        # Split plan lines and filter out any end markers.
        plan_lines = [
            line.strip() for line in predicted_plan.strip().split("\n")
            if line.strip() and "[plan end]" not in line.lower()
        ]
        if not plan_lines:
            print("Empty plan.")
            return 0.0
        parsed_goal = cls.simulate_plan(init_state=init_state, plan=true_plan)
        goal_state, goal_hand, _ = cls.parse_initial_state(parsed_goal)
        goal_set = {(block, loc) for block, loc in goal_state.items()}

        def compute_iou(state_str: str) -> float:
            # Parse the current state and form its condition set.
            state, hand, _ = cls.parse_initial_state(state_str)
            state_set = {(block, loc) for block, loc in state.items()}
            union = goal_set.union(state_set)
            return len(goal_set.intersection(state_set)) / len(union)

        current_state = init_state
        last_iou = compute_iou(current_state)
        num_true_actions = len([line.strip() for line in true_plan.strip().split("\n") if line.strip() and "[plan end]" not in line.lower()])
        valid_actions_count = 0
        
        for action in plan_lines:
            try:
                current_state = cls.simulate_step(current_state, action)
                valid_actions_count += 1.0
                last_iou = compute_iou(current_state)
            except Exception as e:
                print(f'Error in parsing action - {e}')
                return 0.0
        
        # If no valid actions were performed, set norm_factor to 0.
        if valid_actions_count == 0:
            norm_factor = 0.0
        elif valid_actions_count <= num_true_actions:
            norm_factor = 1.0
        else:
            deviation_ratio = (valid_actions_count - num_true_actions) / num_true_actions
            norm_factor = 1.0 / (1.0 + deviation_ratio)
        final_reward = float(last_iou == 1.0) * (1 + norm_factor)
        print(f"Final reward: {final_reward}")
        return final_reward
    
    @classmethod
    def __call__(cls, predicted_plan, ground_truth):
        try:
            init_state = ground_truth['question']
            _ = ground_truth['answer'] # Goal State 
            true_plan = ground_truth['solution']
            reward = cls.simulate_plan_with_reward(init_state=init_state, predicted_plan=predicted_plan, true_plan=true_plan)
            print(f'BW verifier reward: {reward}, Predicted Plan: {predicted_plan}')
            return reward
        except Exception as e:
            print(f'Error in computing BW verifier reward - {e}')
            return 0.0

# correctness_reward = BlocksworldCorrectnessReward.__call__(predicted_answer, ground_truth)

def softFormatReward(text):
    """
    Relaxed format reward that gives partial credit for matching tags.
    Returns a score between 0.0 and 0.5 based on tag matching.
    """
    count = 0.0
    max_score = 0.5
    
    # Check for matching action tags
    if text.count('<action>') == text.count('</action>') and text.count('<action>') > 0:
        count += 0.2
    
    # Check for matching state tags
    if text.count('<state>') == text.count('</state>') and text.count('<state>') > 0:
        count += 0.2
    
    # Check for answer tags
    if text.count('<answer>') == 1 and text.count('</answer>') == 1:
        count += 0.1
    
    return min(count, max_score)

def hardFormatReward(text: str) -> tuple[bool, str]:
    """
    Hard format reward - checks for basic structure: action, state pairs, then answer.
    Returns 0.5 if format is correct, 0 otherwise.
    """
    if text.count('<action>') == 0 or text.count('</action>') == 0:
        return 0

    if text.count('<state>') == 0 or text.count('</state>') == 0:
        return 0

    if text.count('<answer>') != 1 or text.count('</answer>') != 1:
        return 0

    # Check that number of actions equals number of states
    if text.count('<action>') != text.count('<state>'):
        return 0

    # Check the order: action, state (repeated), then answer
    current_pos = 0
    while True:
        action_pos = text.find('<action>', current_pos)
        if action_pos == -1:
            break
        action_end_pos = text.find('</action>', action_pos)
        if action_end_pos == -1:
            return 0

        state_pos = text.find('<state>', action_end_pos)
        if state_pos == -1:
            return 0

        state_end_pos = text.find('</state>', state_pos)
        if state_end_pos == -1:
            return 0

        if not (action_pos < action_end_pos < state_pos < state_end_pos):
            return 0
        current_pos = state_end_pos

    answer_start = text.find('<answer>')
    answer_end = text.find('</answer>')
    if answer_start == -1 or answer_end == -1:
        return 0
    if answer_start > answer_end:
        return 0
    
    # Verify answer comes after all action/state pairs
    if current_pos > 0 and answer_start < current_pos:
        return 0
    
    return 0.5


import re
from typing import List

STATE_RE = re.compile(r"<state>(.*?)</state>", re.DOTALL)
ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL)

def extract_states(text: str) -> List[List[str]]:
    return [
        [line.strip() for line in block.splitlines() if line.strip()]
        for block in STATE_RE.findall(text)
    ]

def extract_actions(text: str) -> List[str]:
    """
    Extracts all actions from <action> tags.
    Returns a list of action strings (one per tag).
    """
    return [action.strip() for action in ACTION_RE.findall(text)]


def parse_state_and_blocks(state_lines):
    on = {}
    clear = set()
    holding = None
    handempty = False
    blocks = set()

    for line in state_lines:
        if line.startswith("on(") and line.endswith(")"):
            # Extract content between parentheses and verify format: exactly 2 args separated by ", "
            content = line[3:-1]
            if ", " not in content:
                return None  # Must have comma and space
            parts = content.split(", ")
            if len(parts) != 2:
                return None  # Must have exactly 2 arguments
            x, y = parts
            on[x] = y
            blocks.add(x)
            if y != "table":
                blocks.add(y)

        elif line.startswith("clear(") and line.endswith(")"):
            # Extract content between parentheses and verify format: exactly 1 arg (no commas)
            content = line[6:-1]
            if "," in content:
                return None  # Must have exactly 1 argument (no commas)
            x = content
            clear.add(x)
            blocks.add(x)

        elif line.startswith("holding(") and line.endswith(")"):
            # Extract content between parentheses and verify format: exactly 1 arg (no commas)
            content = line[8:-1]
            if "," in content:
                return None  # Must have exactly 1 argument (no commas)
            x = content
            if holding is not None:
                return None
            holding = x
            blocks.add(x)

        elif line == "handempty":
            handempty = True

        else:
            return None  # illegal predicate

    return on, clear, holding, handempty, blocks


def is_valid_state(state_lines: List[str]) -> bool:
    parsed = parse_state_and_blocks(state_lines)
    if parsed is None:
        return False

    on, clear, holding, handempty, blocks = parsed

    # 1. Exactly one hand condition
    if (holding is None) == (not handempty):
        return False

    # 2. Each block appears exactly once
    placed = set(on.keys())
    if holding:
        placed.add(holding)

    if placed != blocks:
        return False

    # 3. clear(X) consistency
    for b in clear:
        if b == holding:
            return False
        if b in on.values():
            return False

    # 4. Table constraints
    if "table" in clear or holding == "table":
        return False
    for x, y in on.items():
        if x == "table":
            return False

    # 5. No cycles in on-relations
    for start in on:
        seen = set()
        cur = start
        while cur in on:
            cur = on[cur]
            if cur == "table":
                break
            if cur in seen:
                return False
            seen.add(cur)

    return True

def intermediate_state_rewards(model_output: str):
    """
    Returns one reward per <state>.
    1.0 if parseable & valid, else 0.0
    """
    states = extract_states(model_output)
    return [1.0 if is_valid_state(s) else 0.0 for s in states]



def checkFormat(response):
        response = response.strip()

        # Rule 1: Must end with </answer>
        if not response.endswith("</answer>"):
            return False

        # Rule 2: Must have exactly one <answer> and one </answer>
        if response.count("<answer>") != 1 or response.count("</answer>") != 1:
            return False

        # Rule 3: Must have matching pairs of <action> and </action> (at least one)
        if response.count("<action>") != response.count("</action>"):
            return False
        if response.count("<action>") == 0:
            return False

        # Rule 4: Must have matching pairs of <state> and </state> (required, at least one)
        if response.count("<state>") != response.count("</state>"):
            return False
        if response.count("<state>") == 0:
            return False

        # Rule 5: Number of actions and states must be equal
        # (each action must be followed by a state)
        if response.count("<action>") != response.count("<state>"):
            return False

        # Rule 6: Find all tag positions
        import re
        action_pattern = re.compile(r'<action>(.*?)</action>', re.DOTALL)
        state_pattern = re.compile(r'<state>(.*?)</state>', re.DOTALL)
        answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)

        action_matches = list(action_pattern.finditer(response))
        state_matches = list(state_pattern.finditer(response))
        answer_matches = list(answer_pattern.finditer(response))

        if len(answer_matches) != 1:
            return False

        answer_match = answer_matches[0]
        answer_start = answer_match.start()
        answer_end = answer_match.end()

        # Rule 7: <answer> must come after all action/state blocks
        for action_match in action_matches:
            if action_match.end() > answer_start:
                return False
        for state_match in state_matches:
            if state_match.end() > answer_start:
                return False
        
        # Rule 8: Verify the pattern: action, state (repeated pairs)
        # All tags must appear in the correct order: action, state, action, state, ...
        all_tags = []
        for match in action_matches:
            all_tags.append(('action', match.start(), match.end()))
        for match in state_matches:
            all_tags.append(('state', match.start(), match.end()))
        
        all_tags.sort(key=lambda x: x[1])  # Sort by start position

        # Check that we have the pattern: action, state (repeated)
        if len(all_tags) == 0:
            return False  # Must have at least one pair
        
        i = 0
        while i < len(all_tags):
            # Each pair should be: action, state
            if i + 1 >= len(all_tags):
                return False  # Not enough tags for a complete pair
            
            if all_tags[i][0] != 'action':
                return False
            if all_tags[i+1][0] != 'state':
                return False
            
            # Verify order: action ends before state starts
            if all_tags[i][2] > all_tags[i+1][1]:
                return False
            
            i += 2

        # Rule 9: Verify all action blocks have non-empty content
        for match in action_matches:
            content = match.group(1).strip()
            if not content:
                return False

        # Rule 10: Verify all state blocks have non-empty content
        for match in state_matches:
            content = match.group(1).strip()
            if not content:
                return False

        # Rule 11: Verify answer has non-empty content
        answer_content = answer_match.group(1).strip()
        if not answer_content:
            return False

        # Rule 12: Verify there's no content between last state block end and <answer>
        if len(state_matches) > 0:
            last_state_end = max(match.end() for match in state_matches)
            between_content = response[last_state_end:answer_start].strip()
            if between_content:
                return False

        return True

def compute_score(model_output: str, ground_truth):
    final_reward = 0.0
    format_reward = 0.0
    state_reward = 0.0
    correctness_reward = 0.0
    actual_correctness_reward = 0.0

    # Always compute relaxed format reward (gives partial credit)
    # format_reward = softFormatReward(model_output)
    
    # Compute strict format reward (only if format is perfect)
    if checkFormat(model_output):
        format_reward = 1.0

    # Try to extract answer and compute correctness reward (even if format fails)
    answer_matches = re.findall(r'<answer>\s*(.*?)\s*</answer>', model_output, re.DOTALL)
    if answer_matches:
        predicted_plan = answer_matches[-1].strip()
        if predicted_plan:
            # Compute correctness reward based on the plan from answer tag
            correctness_reward = BlocksworldCorrectnessReward.__call__(predicted_plan, ground_truth) 

            if correctness_reward == 2.0:
                actual_correctness_reward = 1.0
            else:
                actual_correctness_reward = 0.0
    
    # Compute state reward (even if format fails)
    # intermediate_state_reward = intermediate_state_rewards(model_output)
    # if len(intermediate_state_reward) > 0:
    #     state_reward = sum(intermediate_state_reward) / len(intermediate_state_reward)

    final_reward = format_reward + correctness_reward

    return {
        "score": final_reward, 
        "format_reward": format_reward, 
        "state_reward": state_reward, 
        "correctness_reward": actual_correctness_reward
    }

test_data = "/nas-ssd2/joykirat/code/state-representation/verl/scripts/data/blocksworld_state_action_lazy/train.parquet"

test_data = pd.read_parquet(test_data)

ground_truths = []

for i in range(len(test_data)):
    ground_truths.append(test_data['reward_model'][i]['ground_truth'])


predicted_path = "/nas-ssd2/joykirat/code/state-representation/verl/apiTest/gpt-oss-120b_responses_state_action_lazy_train.json"
# "/nas-ssd2/joykirat/code/state-representation/verl/scripts/train/checkpoints/blocksworld/state/qwen1_7b_blocksworld_with_state_v1/val_rollout/800.jsonl"

def get_data(path):
    # data = []
    # with open(path, "r") as f:
    #     for line in f:
    #         data.append(json.loads(line))
    data = []
    with open(path, "r") as f:
        data = json.load(f)
    return data

def get_accuracy(predicted_data, ground_truths):
    correct = 0
    total = 0

    for i in range(len(predicted_data)):
        ground_truth = ground_truths[i]
        model_output = predicted_data[i]['response']
        total += 1
        try:
            predicted_answer = re.findall(r'<answer>\s*(.*?)\s*</answer>', model_output, re.DOTALL)[-1].strip()
        except Exception as e:
            print(f"Error in parsing predicted answer - {e}")
            continue

        reward = BlocksworldCorrectnessReward.__call__(predicted_answer, ground_truth)
        if reward > 0.0:
            correct += 1
        
    
    return correct / total


data_path = "/nas-ssd2/joykirat/code/state-representation/verl/apiTest/gpt-oss-120b_responses_state_action_lazy_train.json"

data = get_data(data_path)
filtered_data = []

for i in range(len(data)):
    score = compute_score(data[i]['response'], ground_truths[i])

    if score['correctness_reward'] == 1.0 and score['format_reward'] == 1.0:
        filtered_data.append(data[i])

print(len(data))
print(len(filtered_data))

with open("/nas-ssd2/joykirat/code/state-representation/verl/apiTest/gpt-oss-120b_responses_state_action_lazy_train_filtered.json", "w") as f:
    json.dump(filtered_data, f)
    


# better_than_prediction = 0
# from tqdm import tqdm
# for i in tqdm(range(0, 900, 10)):
#     predicted_path = f"/nas-ssd2/joykirat/code/state-representation/verl/scripts/train/checkpoints/blocksworld/base/qwen1_7b_blocksworld_correctness_only_v0/val_rollout/{i}.jsonl"
#     state_path = f"/nas-ssd2/joykirat/code/state-representation/verl/scripts/train/checkpoints/blocksworld/state/qwen1_7b_blocksworld_with_state_v1/val_rollout/{i}.jsonl"
#     inline_path = f"/nas-ssd2/joykirat/code/state-representation/verl/scripts/train/checkpoints/blocksworld/state_inline/qwen1_7b_blocksworld_with_state_inline_v0/val_rollout/{i}.jsonl"

#     predicted_data = get_data(predicted_path)
#     state_data = get_data(state_path)
#     inline_data = get_data(inline_path)

#     prediction_accuracy = get_accuracy(predicted_data, ground_truths)
#     state_accuracy = get_accuracy(state_data, ground_truths)
#     inline_accuracy = get_accuracy(inline_data, ground_truths)

#     if inline_accuracy > prediction_accuracy:
#         print(f"Inline accuracy is better than prediction accuracy for {i}")
#         better_than_prediction += 1

# print(f"Better than prediction: {better_than_prediction}")



# predicted_path = f"/nas-ssd2/joykirat/code/state-representation/verl/scripts/train/checkpoints/blocksworld/state_inline/qwen1_7b_blocksworld_with_state_inline_v0/val_rollout/900.jsonl"

# predicted_data = get_data(predicted_path)


# print(get_accuracy(predicted_data, ground_truths))

# print(f"Accuracy: {correct / total}")

# import re
# def get_length_of_state(data):
#     avg_output_length = 0
#     avg_state_length = 0
#     avg_state = 0
#     avg_think = 0
#     for i in range(len(data)):
#         output = data[i]['output']
#         avg_output_length += get_token_length(output)
#         ## extract all text between <state> and </state>
#         state_strings = re.findall(r'<state>(.*?)</state>', output, re.DOTALL)

#         avg_state += output.count('<state>')
#         avg_think += output.count('<think>')
#         # Sum the lengths of all state strings (not just count them)
#         avg_state_length += sum(get_token_length(s) for s in state_strings)
    
#     avg_output_length /= len(data)
#     avg_state_length /= len(data)
#     avg_state /= len(data)
#     avg_think /= len(data)
#     return avg_output_length, avg_state_length, avg_state, avg_think

# base_data = get_data(base_path)
# state_data = get_data(state_path)

# base_output_length, base_state_length, base_state, base_think = get_length_of_state(base_data)
# state_output_length, state_state_length, state_state, state_think = get_length_of_state(state_data)

# print(f"Base output length: {base_output_length}, Base state length: {base_state_length}")
# print(f"State output length: {state_output_length}, State state length: {state_state_length}, State state: {state_state}, Base state: {base_state}, Base think: {base_think}, State think: {state_think}")