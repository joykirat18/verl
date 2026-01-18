
import re
import sympy
import logging
from sympy.parsing.latex import parse_latex
from functools import partial, update_wrapper
import signal
import json
from typing import List


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

def softFormatReward(text):
    """
    Relaxed format reward that gives partial credit for matching tags.
    Returns a score between 0.0 and 0.5 based on tag matching.
    """
    count = 0.0
    max_score = 0.5
    
    # Check for matching redacted_reasoning tags
    if text.count('<think>') == text.count('</think>') and text.count('<think>') > 0:
        count += 0.1
    
    # Check for matching reasoning tags
    if text.count('<reasoning>') == text.count('</reasoning>') and text.count('<reasoning>') > 0:
        count += 0.1
    
    # Check for matching action tags
    if text.count('<action>') == text.count('</action>') and text.count('<action>') > 0:
        count += 0.1
    
    # Check for matching state tags
    if text.count('<state>') == text.count('</state>') and text.count('<state>') > 0:
        count += 0.1
    
    # Check for answer tags
    if text.count('<answer>') == 1 and text.count('</answer>') == 1:
        count += 0.1
    
    return min(count, max_score)

def hardFormatReward(text: str) -> tuple[bool, str]:


    if text.count('<think>') == 0 or text.count('</think>') == 0:
        return 0

    if text.count('<answer>') != 1 or text.count('</answer>') != 1:
        return 0

    # check the order of search/result
    current_pos = 0
    while True:
        think_pos = text.find('<think>', current_pos)
        if think_pos == -1:
            break
        think_end_pos = text.find('</think>', think_pos)
        if think_end_pos == -1:
            return 0

        state_pos = text.find('<state>', think_pos)
        if state_pos == -1:
            break

        state_end_pos = text.find('</state>', state_pos)
        if state_end_pos == -1:
            return 0

        if not (think_pos < think_end_pos < state_pos < state_end_pos):
            return 0
        current_pos = state_end_pos

    answer_start = text.find('<answer>')
    answer_end = text.find('</answer>')
    if answer_start > answer_end:
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

        # Rule 1: Must start with <think> and end with </answer>
        if not response.startswith("<think>") or not response.endswith("</answer>"):
            return False

        # Rule 2: Must have exactly one <answer> and one </answer>
        if response.count("<answer>") != 1 or response.count("</answer>") != 1:
            return False

        # Rule 3: Must have matching pairs of <think> and </think> (at least one)
        if response.count("<think>") != response.count("</think>"):
            return False
        if response.count("<think>") == 0:
            return False

        # Rule 4: Must have matching pairs of <reasoning> and </reasoning> (at least one)
        if response.count("<reasoning>") != response.count("</reasoning>"):
            return False
        if response.count("<reasoning>") == 0:
            return False

        # Rule 5: Must have matching pairs of <action> and </action> (at least one)
        if response.count("<action>") != response.count("</action>"):
            return False
        if response.count("<action>") == 0:
            return False

        # Rule 6: Must have matching pairs of <state> and </state> (required, at least one)
        if response.count("<state>") != response.count("</state>"):
            return False
        if response.count("<state>") == 0:
            return False

        # Rule 7: Number of reasoning, actions, and states must all be equal
        # (each reasoning must be followed by an action, which is followed by a state)
        if response.count("<reasoning>") != response.count("<action>"):
            return False
        if response.count("<action>") != response.count("<state>"):
            return False

        # Rule 8: Find all tag positions
        import re
        redacted_reasoning_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
        reasoning_pattern = re.compile(r'<reasoning>(.*?)</reasoning>', re.DOTALL)
        action_pattern = re.compile(r'<action>(.*?)</action>', re.DOTALL)
        state_pattern = re.compile(r'<state>(.*?)</state>', re.DOTALL)
        answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)

        redacted_reasoning_matches = list(redacted_reasoning_pattern.finditer(response))
        reasoning_matches = list(reasoning_pattern.finditer(response))
        action_matches = list(action_pattern.finditer(response))
        state_matches = list(state_pattern.finditer(response))
        answer_matches = list(answer_pattern.finditer(response))

        if len(answer_matches) != 1:
            return False

        answer_match = answer_matches[0]
        answer_start = answer_match.start()
        answer_end = answer_match.end()

        # Rule 9: <answer> must come after all reasoning/action/state blocks
        for match in redacted_reasoning_matches:
            if match.end() > answer_start:
                return False
        for match in reasoning_matches:
            if match.end() > answer_start:
                return False
        for action_match in action_matches:
            if action_match.end() > answer_start:
                return False
        for state_match in state_matches:
            if state_match.end() > answer_start:
                return False

        # Rule 10: Verify the pattern: all reasoning/action/state are inside <think>
        # There should be exactly one redacted_reasoning block that contains all reasoning/action/state
        if len(redacted_reasoning_matches) != 1:
            return False
        
        redacted_reasoning_match = redacted_reasoning_matches[0]
        redacted_reasoning_start = redacted_reasoning_match.start()
        redacted_reasoning_end = redacted_reasoning_match.end()
        
        # All reasoning, action, and state tags must be inside the redacted_reasoning block
        for match in reasoning_matches:
            if match.start() < redacted_reasoning_start or match.end() > redacted_reasoning_end:
                return False
        for action_match in action_matches:
            if action_match.start() < redacted_reasoning_start or action_match.end() > redacted_reasoning_end:
                return False
        for state_match in state_matches:
            if state_match.start() < redacted_reasoning_start or state_match.end() > redacted_reasoning_end:
                return False
        
        # Rule 11: Verify the pattern: reasoning, action, state (repeated) inside redacted_reasoning
        # All tags must appear in the correct order: reasoning, action, state, reasoning, action, state, ...
        all_tags = []
        for match in reasoning_matches:
            all_tags.append(('reasoning', match.start(), match.end()))
        for match in action_matches:
            all_tags.append(('action', match.start(), match.end()))
        for match in state_matches:
            all_tags.append(('state', match.start(), match.end()))
        
        all_tags.sort(key=lambda x: x[1])  # Sort by start position

        # Check that we have the pattern: reasoning, action, state (repeated)
        if len(all_tags) == 0:
            return False  # Must have at least one triplet
        
        i = 0
        while i < len(all_tags):
            # Each triplet should be: reasoning, action, state
            if i + 2 >= len(all_tags):
                return False  # Not enough tags for a complete triplet
            
            if all_tags[i][0] != 'reasoning':
                return False
            if all_tags[i+1][0] != 'action':
                return False
            if all_tags[i+2][0] != 'state':
                return False
            
            # Verify order: reasoning ends before action starts, action ends before state starts
            if all_tags[i][2] > all_tags[i+1][1] or all_tags[i+1][2] > all_tags[i+2][1]:
                return False
            
            i += 3

        # Rule 12: Verify all reasoning blocks have non-empty content
        for match in reasoning_matches:
            content = match.group(1).strip()
            if not content:
                return False

        # Rule 13: Verify all action blocks have non-empty content
        for match in action_matches:
            content = match.group(1).strip()
            if not content:
                return False

        # Rule 14: Verify all state blocks have non-empty content
        for match in state_matches:
            content = match.group(1).strip()
            if not content:
                return False

        # Rule 15: Verify answer has non-empty content
        answer_content = answer_match.group(1).strip()
        if not answer_content:
            return False

        # Rule 16: Verify there's no content between redacted_reasoning block end and <answer>
        redacted_reasoning_end = redacted_reasoning_matches[0].end()
        between_content = response[redacted_reasoning_end:answer_start].strip()
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
        # format_reward += 0.5

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
            # state_reward = sum(intermediate_state_reward) / len(intermediate_state_reward)

    final_reward = format_reward + state_reward + correctness_reward

    return {
        "score": final_reward, 
        # "format_reward": format_reward, 
        # "state_reward": state_reward, 
        "correctness_reward": actual_correctness_reward
    }
    
    


# def compute_score(model_output: str, ground_truth):

#     relaxed_format_reward = 0
#     strict_format_reward = 0
#     correctness_reward = 0
#     state_reward = 0

#     relaxed_format_reward = softFormatReward(model_output)
#     strict_format_reward = hardFormatReward(model_output)


#     answer_matches = re.findall(r'<answer>\s*(.*?)\s*</answer>', model_output, re.DOTALL)
#     predicted_answer = answer_matches[-1].strip() if answer_matches else ""  

#     result = BlocksworldCorrectnessReward.__call__(predicted_answer, ground_truth) 

#     correctness_reward = result


#     intermediate_state_reward = intermediate_state_rewards(model_output)
#     if len(intermediate_state_reward) == 0:
#         state_reward = 0
#     else:
#         state_reward = sum(intermediate_state_reward) / len(intermediate_state_reward)
    
#     final_reward = correctness_reward + relaxed_format_reward + strict_format_reward + state_reward

#     return {"score": final_reward, "correctness_reward": correctness_reward, "format_reward": relaxed_format_reward + strict_format_reward, "state_reward": state_reward}


