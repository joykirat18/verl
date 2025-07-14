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
Preprocess the nq dataset to parquet format
"""

import re
import os
from datasets import load_dataset, Dataset

from verl.utils.hdfs_io import copy, makedirs
import argparse
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed


def extract_solution_boxed(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))

def extract_solution_gsm8k(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetName', type=str, default='math-500')

    args = parser.parse_args()
    datasetName = args.datasetName
    data_source = datasetName


    
    if(datasetName == 'math-500'):
        dataset = load_dataset('HuggingFaceH4/MATH-500')
            
        test_data = dataset['test']
        
        questions = test_data['problem']
        solutions = test_data['answer']
        function = ["None"]*len(questions)
    
    elif(datasetName == 'GSM8K'):
        dataset = load_dataset("openai/gsm8k", "main")
        
        test_data = dataset['test']
        
        questions = test_data['question']
        solutions = test_data['answer']
        function = ["extract_solution_gsm8k"]*len(questions)
    
    elif(datasetName == 'AIME'):
        dataset = load_dataset('AI-MO/aimo-validation-aime')
        
        test_data = dataset['train']
        questions = test_data['problem']
        solutions = test_data['solution']
        function = ["extract_solution_boxed"]*len(questions)
    
    elif(datasetName == 'AMC'):
        dataset = load_dataset('AI-MO/aimo-validation-amc')
        
        test_data = dataset['train']
        questions = test_data['problem']
        solutions = test_data['answer']
        solutions = [str(s) for s in solutions]
        function = ["None"]*len(questions)
    
    elif(datasetName == 'Olympiad'):
        dataset = load_dataset("Hothan/OlympiadBench", "OE_TO_maths_en_COMP")
        
        test_data = dataset['train']
        questions = test_data['question']
        solutions = test_data['final_answer']
        
        solutions = [s[0] for s in solutions]
        function = ["None"]*len(questions)
    
    else:
        raise ValueError("Invalid datasetName")
    
    
    # create dataset from the questions and solutions json
    
    dataset = {
        'problem': questions,
        'answer': solutions,
        'function': function
    }

    # create a dataset from the json
    dataset = Dataset.from_dict(dataset)

    
    # def make_map_fn(split):

    #     def process_fn(example, idx):
    #         example['question'] = example['problem'].strip()
            
    #         solution = {
    #             "target": example['answer'],
    #         }
            
    #         data = {
    #         "data_source": data_source,
    #         "prompt": [{'role': 'system', 'content': sys_prompt},
    #                   {'role': 'user', 'content': example['question']}],
    #         "ability": "fact-reasoning",
    #         "reward_model": {
    #                 "style": "rule",
    #                 "ground_truth": solution
    #             },
    #         "extra_info": {
    #             'split': split,
    #                 'index': idx,
    #         }
    #     }
    #         return data

    #     return process_fn

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("problem")

            question = question + " " + instruction_following

            answer = example.pop("answer")
            function = example.pop("function")
            
            if function == "None":
                solution = answer
            elif function == "extract_solution_gsm8k":
                solution = extract_solution_gsm8k(answer)
            elif function == "extract_solution_boxed":
                solution = extract_solution_boxed(answer)
            else:
                raise ValueError("Invalid function")
                
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {"split": split, "index": idx},
            }
            return data

        return process_fn

    test_dataset = dataset.map(function=make_map_fn('test'), with_indices=True)

    test_dataset.to_parquet(os.path.join('./eval_dataset', f'{datasetName}.parquet'))