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

# try:
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
import re
# except ImportError:
    # print("To use Math-Verify, please install it first by running `pip install math-verify`.")

def get_format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0

def get_acc_reward(predict_str: str, ground_truth: str) -> float:

    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )

    ret_score = 0.0
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"

    ret_score, _ = verify_func([ground_truth_boxed], [predict_str])
    return ret_score


def compute_score(model_output: str, ground_truth: str) -> float:

    format_reward = get_format_reward(model_output)
    acc_reward = get_acc_reward(model_output, ground_truth)
    return 0.9 * acc_reward + 0.1 * format_reward
