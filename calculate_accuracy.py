import json
import argparse

data = []

parser = argparse.ArgumentParser()
# parser.add_argument("--checkpoint_name", type=str)
parser.add_argument("--step", type=int)
args = parser.parse_args()

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

def get_token_length(text):
    return len(tokenizer.encode(text))

# checkpoint_name = args.checkpoint_name
step = args.step

# if checkpoint_name == "base":
# elif checkpoint_name == "state":
# else:
#     raise ValueError(f"Invalid checkpoint name: {checkpoint_name}")


## base accuracy




def get_accuracy(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))

    correct = 0
    total = 0
    for i in range(len(data)):
        if data[i]['correctness_reward'] > 0.0:
            correct += 1
        total += 1

    print(f"Correct: {correct}, Total: {total}, Accuracy: {correct/total}")






print("Base accuracy:")
path = f"/nas-ssd2/joykirat/code/state-representation/verl/scripts/train/checkpoints/blocksworld/base/qwen1_7b_blocksworld_correctness_only/val_rollout/{step}.jsonl"
get_accuracy(path)


print("State accuracy:")
path = f"/nas-ssd2/joykirat/code/state-representation/verl/scripts/train/checkpoints/blocksworld/state/qwen1_7b_blocksworld_with_state_v0/val_rollout/{step}.jsonl"
get_accuracy(path)


print("no grpo accuracy:")
path = f"/nas-ssd2/joykirat/code/state-representation/verl/scripts/eval/checkpoints/blocksworld/base/qwen1_7b_blocksworld_base/val_rollout/0.jsonl"
get_accuracy(path)


base_path = "/nas-ssd2/joykirat/code/state-representation/verl/scripts/train/checkpoints/blocksworld/base/qwen1_7b_blocksworld_correctness_only/val_rollout/900.jsonl"
state_path = f"/nas-ssd2/joykirat/code/state-representation/verl/scripts/train/checkpoints/blocksworld/state/qwen1_7b_blocksworld_with_state_v0/val_rollout/900.jsonl"


def get_data(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data
import re
def get_length_of_state(data):
    avg_output_length = 0
    avg_state_length = 0
    avg_state = 0
    avg_think = 0
    for i in range(len(data)):
        output = data[i]['output']
        avg_output_length += get_token_length(output)
        ## extract all text between <state> and </state>
        state_strings = re.findall(r'<state>(.*?)</state>', output, re.DOTALL)

        avg_state += output.count('<state>')
        avg_think += output.count('<think>')
        # Sum the lengths of all state strings (not just count them)
        avg_state_length += sum(get_token_length(s) for s in state_strings)
    
    avg_output_length /= len(data)
    avg_state_length /= len(data)
    avg_state /= len(data)
    avg_think /= len(data)
    return avg_output_length, avg_state_length, avg_state, avg_think

base_data = get_data(base_path)
state_data = get_data(state_path)

base_output_length, base_state_length, base_state, base_think = get_length_of_state(base_data)
state_output_length, state_state_length, state_state, state_think = get_length_of_state(state_data)

print(f"Base output length: {base_output_length}, Base state length: {base_state_length}")
print(f"State output length: {state_output_length}, State state length: {state_state_length}, State state: {state_state}, Base state: {base_state}, Base think: {base_think}, State think: {state_think}")