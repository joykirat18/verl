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

import datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="numina_dataset")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    # 'lighteval/MATH' is no longer available on huggingface.
    # Use mirror repo: DigitalLearningGmbH/MATH-lighteval
    data_source = "AI-MO/NuminaMath-CoT"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    train_dataset = dataset['train']
    
    # split the train dataset into train and test
    train_dataset = train_dataset.train_test_split(test_size=0.001, seed=43)
    test_dataset = train_dataset['test']
    train_dataset = train_dataset['train']

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

    train_dataset = train_dataset.shuffle(seed=23).select(range(20000))
    test_dataset = test_dataset.shuffle(seed=23).select(range(100))

    train_dataset = train_dataset.filter(batch_filter, batched=True)
    test_dataset = test_dataset.filter(batch_filter, batched=True)


    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)