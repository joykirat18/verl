# MATH: https://huggingface.co/datasets/NovaSky-AI/labeled_numina_difficulty_162K


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


"""
Preprocess the MATH-lighteval dataset to parquet format
"""

import argparse
import os
import uuid

os.environ["HF_HOME"] = "/nas-ssd2/joykirat/.cache/huggingface"
os.environ["UV_CACHE_DIR"] = "/nas-ssd2/joykirat/.cache/uv"
os.environ["RAY_TMPDIR"] = "/nas-ssd2/joykirat/tmp_ray"


import datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))

def curate_numina_dataset():
    huggingface_data = 'AI-MO/NuminaMath-CoT'
    data_source = 'math-numina'
    print(f"Loading the {huggingface_data} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(huggingface_data, trust_remote_code=True)
    dataset = dataset['train']

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("problem")

            question = question + " " + instruction_following

            answer = example.pop("solution")
            solution = extract_solution(answer)
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {"split": split, "index": idx},
                "uuid": str(uuid.uuid4()),
            }
            return data

        return process_fn

    def filter_fn(solution):
        if('\\boxed' not in solution):
            return False
        
        # count of \\boxed in solution
        count = solution.count('\\boxed')
        if count > 1:
            # breakpoint()
            return False
        try:
            extracted = extract_solution(solution)
            if extracted is None or extracted in ['', 'A', 'B', 'C', 'D', '(A)', '(B)', '(C)', '(D)']:
                return False
        except Exception as e:
            print(f"Error in extracting answer: {e}")
            # breakpoint()
            return False
        return True
        
    def batch_filter(batch_example):
        return [filter_fn(solution) for solution in batch_example['solution']]

    dataset = dataset.shuffle(seed=42).select(range(5000))

    dataset = dataset.filter(batch_filter, batched=True)
    
    dataset = dataset.map(function=make_map_fn("train"), with_indices=True, remove_columns=dataset.column_names)

    return dataset

def curate_arc_dataset(data_source):
    huggingface_data = 'allenai/ai2_arc'
    print(f"Loading the {huggingface_data} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(huggingface_data, data_source)
    # combine dataset
    dataset = datasets.concatenate_datasets([dataset['train'], dataset['validation']])

    # TEMPLATE = (
    #     "Answer the following multiple choice question. The last line of your response should be of the following "
    #     "format: 'Answer: $LETTER' (without quotes) where LETTER is one of {letter}. Think step by step before "
    #     "answering.\n\n{Question}\n\n{Choices}"
    # )

    TEMPLATE = "{Question}\n\n{Choices}"


    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("question")
            choices = example.pop("choices")
            number_of_choices = len(choices['text'])
            if 2 <= number_of_choices <= 26:
                letter = ''.join([chr(ord('A') + i) for i in range(number_of_choices)])
            else:
                print("number of choices ", number_of_choices, "not in supported range (2-26)")
                return None


            choices = [f"{choices['label'][i]}) {choices['text'][i]}" for i in range(number_of_choices)]
            choices = "\n".join(choices)

            question = TEMPLATE.format(Question=question, Choices=choices) + "\n\nLet's think step by step and output the final answer within \\boxed{}."

            answerKey = example.pop("answerKey")
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answerKey},
                "extra_info": {"split": split, "index": idx},
                "uuid": str(uuid.uuid4()),
            }

            return data

        return process_fn
    
    dataset = dataset.map(function=make_map_fn("train"), with_indices=True, remove_columns=dataset.column_names)

    return dataset
    

def curate_commonsense_dataset():
    huggingface_data = 'tau/commonsense_qa'
    data_source = 'commonsense'
    print(f"Loading the {huggingface_data} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(huggingface_data)
    # combine dataset
    dataset = dataset['train']

    # TEMPLATE = (
    #     "Answer the following multiple choice question. The last line of your response should be of the following "
    #     "format: 'Answer: $LETTER' (without quotes) where LETTER is one of {letter}. Think step by step before "
    #     "answering.\n\n{Question}\n\n{Choices}"
    # )

    TEMPLATE = "{Question}\n\n{Choices}"


    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("question")
            choices = example.pop("choices")
            number_of_choices = len(choices['text'])
            if 2 <= number_of_choices <= 26:
                letter = ''.join([chr(ord('A') + i) for i in range(number_of_choices)])
            else:
                print("number of choices ", number_of_choices, "not in supported range (2-26)")
                return None


            choices = [f"{choices['label'][i]}) {choices['text'][i]}" for i in range(number_of_choices)]
            choices = "\n".join(choices)

            question = TEMPLATE.format(Question=question, Choices=choices) + "\n\nLet's think step by step and output the final answer within \\boxed{}."

            answerKey = example.pop("answerKey")

            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answerKey},
                "extra_info": {"split": split, "index": idx},
                "uuid": str(uuid.uuid4()),
            }
            # only keep entries in 

            return data

        return process_fn
    
    dataset = dataset.map(function=make_map_fn("train"), with_indices=True, remove_columns=dataset.column_names)

    return dataset
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./combined_train_dataset")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    # commonesesnse_dataset = curate_commonsense_dataset()
    # arc_easy_dataset = curate_arc_dataset('ARC-Easy')
    # arc_challenge_dataset = curate_arc_dataset('ARC-Challenge')

    math_dataset = curate_numina_dataset()

    # arc_combined_dataset = datasets.concatenate_datasets([arc_easy_dataset, arc_challenge_dataset])
    # arc_combined_dataset = arc_combined_dataset.shuffle(seed=42).select(range(len(math_dataset)))

    # commonesesnse_dataset = commonesesnse_dataset.shuffle(seed=42).select(range(len(math_dataset)))
    # print("Total length of commonsense dataset: ", len(commonesesnse_dataset))
    print("Total length of math dataset: ", len(math_dataset))

    dataset = datasets.concatenate_datasets([math_dataset])

    # shuffle the dataset
    dataset = dataset.shuffle(seed=42)

    print("Total length of dataset: ", len(dataset))


    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    dataset.to_parquet(os.path.join(local_dir, "train.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
