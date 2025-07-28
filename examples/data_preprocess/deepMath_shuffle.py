import argparse
import os
import random
from collections import defaultdict

import datasets
from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


def group_difficulty(diff):
    if isinstance(diff, str):
        diff = float(diff)
    if diff <= 3.0:
        return "very_easy"
    elif diff <= 5.0:
        return "easy"
    elif diff <= 7.0:
        return "medium"
    elif diff <= 8.5:
        return "hard"
    else:
        return "very_hard"

# # 5k
# curriculum_target_counts = {
#     "very_easy": 375,
#     "easy": 1000,
#     "medium": 1375,
#     "hard": 1125,
#     "very_hard": 500,
# }

# # 4k
# curriculum_target_counts = {
#     "very_easy": 300,
#     "easy": 800,
#     "medium": 1100,
#     "hard": 900,
#     "very_hard": 400,
# }

# 2k
curriculum_target_counts = {
    "very_easy": 150,
    "easy": 400,
    "medium": 550,
    "hard": 450,
    "very_hard": 200,
}

# # 1k
# curriculum_target_counts = {
#     "very_easy": 75,
#     "easy": 200,
#     "medium": 275,
#     "hard": 225,
#     "very_hard": 100,
# }



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="deepMath_dataset/level5")
    parser.add_argument("--hdfs_dir", default=None)
    args = parser.parse_args()

    data_source = "zwhe99/DeepMath-103K"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    train_dataset = dataset["train"].train_test_split(test_size=0.001, seed=43)
    test_dataset = train_dataset["test"]
    train_dataset = train_dataset["train"]

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("question") + " " + instruction_following
            answer = example.pop("final_answer")
            difficulty_dataset = float(example.pop("difficulty"))
            return {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {"split": split, "index": idx},
                "difficulty_dataset": difficulty_dataset,
                "curriculum_group": group_difficulty(difficulty_dataset),
            }
        return process_fn

    train_dataset = train_dataset.map(make_map_fn("train"), with_indices=True, remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(make_map_fn("test"), with_indices=True)

    # remove difficulty < 5
    train_dataset = train_dataset.filter(lambda example: example["difficulty_dataset"] >= 5)


    train_dataset = train_dataset.shuffle(seed=23).select(range(10000))

    # Curriculum grouping and sampling
    # grouped_examples = defaultdict(list)
    # for example in train_dataset:
    #     grouped_examples[example["curriculum_group"]].append(example)

    # final_train_data = []
    # for group, count in curriculum_target_counts.items():
    #     candidates = grouped_examples[group]
    #     print(f"Group: {group} — Available: {len(candidates)}, Sampling: {count}")
    #     sampled = random.sample(candidates, min(count, len(candidates)))
    #     final_train_data.extend(sampled)

    # # breakpoint()
    # final_train_data.sort(key=lambda x: x["difficulty_dataset"])
    # final_train_dataset = datasets.Dataset.from_list(final_train_data)


    # Save as parquet
    os.makedirs(args.local_dir, exist_ok=True)
    train_parquet_path = os.path.join(args.local_dir, "train.parquet")
    test_parquet_path = os.path.join(args.local_dir, "test.parquet")

    train_dataset.to_parquet(train_parquet_path)
    test_dataset.to_parquet(test_parquet_path)

    print("Length of train data: ", len(train_dataset))
    print("Length of test data: ", len(test_dataset))
    print(f"Saved train to {train_parquet_path}")
    print(f"Saved test to {test_parquet_path}")

    # # Optional: Save to disk
    # os.makedirs(args.local_dir, exist_ok=True)
    # train_path = os.path.join(args.local_dir, "train_curriculum.jsonl")
    # test_path = os.path.join(args.local_dir, "test.jsonl")

    # import json
    # with open(train_path, "w") as f:
    #     for ex in final_train_data:
    #         json.dump(ex, f)
    #         f.write("\n")
    # with open(test_path, "w") as f:
    #     for ex in test_dataset:
    #         json.dump(ex, f)
    #         f.write("\n")

    # print(f"Saved train to {train_path}")
    # print(f"Saved test to {test_path}")

    # Optional HDFS copy
    if args.hdfs_dir:
        makedirs(args.hdfs_dir)
        copy(train_parquet_path, os.path.join(args.hdfs_dir, "train.parquet"))
        copy(test_parquet_path, os.path.join(args.hdfs_dir, "test.parquet"))
        print(f"Copied dataset to HDFS: {args.hdfs_dir}")
