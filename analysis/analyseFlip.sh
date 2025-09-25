
# ### base model


# ### only correctness

# ### no summary CA + LA
# python groundTruthExtract.py --input /nas-ssd2/joykirat/code/verl-fork/verl/scripts/train/verlCheckpoint/NonSummary/qwen4b_dapo_math_10k_context_linear_reward_no_summary_no_difficulty/rollout/val_140_v2.jsonl --output qwen4b_10k_no_summary_length_reward_overthink.jsonl --data_source 'overthink-bench' &
# python groundTruthExtract.py --input /nas-ssd2/joykirat/code/verl-fork/verl/scripts/train/verlCheckpoint/NonSummary/qwen4b_dapo_math_10k_context_linear_reward_no_summary_no_difficulty/rollout/val_140_v3.jsonl --output qwen4b_10k_no_summary_length_reward_underthink.jsonl --data_source 'underthink-bench' &
# python groundTruthExtract.py --input /nas-ssd2/joykirat/code/verl-fork/verl/scripts/train/verlCheckpoint/NonSummary/qwen4b_dapo_math_10k_context_linear_reward_no_summary_no_difficulty/rollout/val_140_v4.jsonl --output qwen4b_10k_no_summary_length_reward_bbeh.jsonl --data_source 'bbeh'


# ### with summary CA + LA
python driftDetection_gemini.py --input '/nas-ssd2/joykirat/code/verl-fork/verl/scripts/train/verlCheckpoint/CorrectnessOnly/qwen4b_dapo_math_10k_correctnessOnly/rollout/val_140_v0.jsonl' --output answerDrift/correctnessOnly/qwen4b_10k_correctnessOnly_aime_gemini.jsonl --data_source 'math-aime'

## FULL
BaseModelPath='/nas-ssd2/joykirat/code/verl-fork/verl/scripts/eval/verlCheckpoint/mathbaseRun/qwen4b_10k_base_model/rollout/val_0_v0.jsonl'
CorrectnessOnlyPath='/nas-ssd2/joykirat/code/verl-fork/verl/scripts/train/verlCheckpoint/CorrectnessOnly/qwen4b_dapo_math_10k_correctnessOnly/rollout/val_140_v0.jsonl'
no_summary_path='/nas-ssd2/joykirat/code/verl-fork/verl/scripts/train/verlCheckpoint/NonSummary/qwen4b_dapo_math_10k_context_linear_reward_no_summary_no_difficulty/rollout/val_140_v0.jsonl'
ourMethodPath='/nas-ssd2/joykirat/code/verl-fork/verl/scripts/train/verlCheckpoint/Summary/qwen4b_dapo_math_10k_context_linear_reward_with_summary_attention/rollout/val_140_v0.jsonl'


python driftdetection.py --input $BaseModelPath --output answerDrift/baseModel/qwen4b_10k_base_gpqa.jsonl --data_source 'gpqa' &
python driftdetection.py --input $CorrectnessOnlyPath --output answerDrift/correctnessOnly/qwen4b_10k_correctnessOnly_gpqa.jsonl --data_source 'gpqa'

python driftdetection.py --input $no_summary_path --output answerDrift/noSummary/qwen4b_10k_no_summary_gpqa.jsonl --data_source 'gpqa' &
python driftdetection.py --input $ourMethodPath --output answerDrift/withSummary/qwen4b_10k_with_summary_attention_gpqa.jsonl --data_source 'gpqa'

python driftdetection.py --input $BaseModelPath --output answerDrift/baseModel/qwen4b_10k_base_amc.jsonl --data_source 'math-amc' &
python driftdetection.py --input $CorrectnessOnlyPath --output answerDrift/correctnessOnly/qwen4b_10k_correctnessOnly_amc.jsonl --data_source 'math-amc'

python driftdetection.py --input $no_summary_path --output answerDrift/noSummary/qwen4b_10k_no_summary_amc.jsonl --data_source 'math-amc' &
python driftdetection.py --input $ourMethodPath --output answerDrift/withSummary/qwen4b_10k_with_summary_attention_amc.jsonl --data_source 'math-amc'

python driftdetection.py --input $BaseModelPath --output answerDrift/baseModel/qwen4b_10k_base_aime.jsonl --data_source 'math-aime' &
python driftdetection.py --input $CorrectnessOnlyPath --output answerDrift/correctnessOnly/qwen4b_10k_correctnessOnly_aime.jsonl --data_source 'math-aime'

python driftdetection.py --input $no_summary_path --output answerDrift/noSummary/qwen4b_10k_no_summary_aime.jsonl --data_source 'math-aime' &
python driftdetection.py --input $ourMethodPath --output answerDrift/withSummary/qwen4b_10k_with_summary_attention_aime.jsonl --data_source 'math-aime'

## BBEH
BaseModelPath='/nas-ssd2/joykirat/code/verl-fork/verl/scripts/eval/verlCheckpoint/mathbaseRun/qwen4b_10k_base_model/rollout/val_0_v5.jsonl'
CorrectnessOnlyPath='/nas-ssd2/joykirat/code/verl-fork/verl/scripts/train/verlCheckpoint/CorrectnessOnly/qwen4b_dapo_math_10k_correctnessOnly/rollout/val_140_v5.jsonl'
no_summary_path='/nas-ssd2/joykirat/code/verl-fork/verl/scripts/train/verlCheckpoint/NonSummary/qwen4b_dapo_math_10k_context_linear_reward_no_summary_no_difficulty/rollout/val_140_v4.jsonl'
ourMethodPath='/nas-ssd2/joykirat/code/verl-fork/verl/scripts/train/verlCheckpoint/Summary/qwen4b_dapo_math_10k_context_linear_reward_with_summary_attention/rollout/val_140_v4.jsonl'

python driftdetection.py --input $BaseModelPath --output answerDrift/baseModel/qwen4b_10k_base_bbeh.jsonl --data_source 'bbeh' &
python driftdetection.py --input $CorrectnessOnlyPath --output answerDrift/correctnessOnly/qwen4b_10k_correctnessOnly_bbeh.jsonl --data_source 'bbeh'

python driftdetection.py --input $no_summary_path --output answerDrift/noSummary/qwen4b_10k_no_summary_bbeh.jsonl --data_source 'bbeh' &
python driftdetection.py --input $ourMethodPath --output answerDrift/withSummary/qwen4b_10k_with_summary_attention_bbeh.jsonl --data_source 'bbeh'

# ## UNDERTHINK
BaseModelPath='/nas-ssd2/joykirat/code/verl-fork/verl/scripts/eval/verlCheckpoint/mathbaseRun/qwen4b_10k_base_model/rollout/val_0_v4.jsonl'
CorrectnessOnlyPath='/nas-ssd2/joykirat/code/verl-fork/verl/scripts/train/verlCheckpoint/CorrectnessOnly/qwen4b_dapo_math_10k_correctnessOnly/rollout/val_140_v4.jsonl'
no_summary_path='/nas-ssd2/joykirat/code/verl-fork/verl/scripts/train/verlCheckpoint/NonSummary/qwen4b_dapo_math_10k_context_linear_reward_no_summary_no_difficulty/rollout/val_140_v3.jsonl'
ourMethodPath='/nas-ssd2/joykirat/code/verl-fork/verl/scripts/train/verlCheckpoint/Summary/qwen4b_dapo_math_10k_context_linear_reward_with_summary_attention/rollout/val_140_v3.jsonl'

python driftdetection.py --input $BaseModelPath --output answerDrift/baseModel/qwen4b_10k_base_underthink.jsonl --data_source 'underthink-bench' &
python driftdetection.py --input $CorrectnessOnlyPath --output answerDrift/correctnessOnly/qwen4b_10k_correctnessOnly_underthink.jsonl --data_source 'underthink-bench'

python driftdetection.py --input $no_summary_path --output answerDrift/noSummary/qwen4b_10k_no_summary_underthink.jsonl --data_source 'underthink-bench' &
python driftdetection.py --input $ourMethodPath --output answerDrift/withSummary/qwen4b_10k_with_summary_attention_underthink.jsonl --data_source 'underthink-bench'

## OVERTHINK
BaseModelPath='/nas-ssd2/joykirat/code/verl-fork/verl/scripts/eval/verlCheckpoint/mathbaseRun/qwen4b_10k_base_model/rollout/val_0_v3.jsonl'
CorrectnessOnlyPath='/nas-ssd2/joykirat/code/verl-fork/verl/scripts/train/verlCheckpoint/CorrectnessOnly/qwen4b_dapo_math_10k_correctnessOnly/rollout/val_140_v3.jsonl'
no_summary_path='/nas-ssd2/joykirat/code/verl-fork/verl/scripts/train/verlCheckpoint/NonSummary/qwen4b_dapo_math_10k_context_linear_reward_no_summary_no_difficulty/rollout/val_140_v2.jsonl'
ourMethodPath='/nas-ssd2/joykirat/code/verl-fork/verl/scripts/train/verlCheckpoint/Summary/qwen4b_dapo_math_10k_context_linear_reward_with_summary_attention/rollout/val_140_v2.jsonl'

python driftdetection.py --input $BaseModelPath --output answerDrift/baseModel/qwen4b_10k_base_overthink.jsonl --data_source 'overthink-bench' &
python driftdetection.py --input $CorrectnessOnlyPath --output answerDrift/correctnessOnly/qwen4b_10k_correctnessOnly_overthink.jsonl --data_source 'overthink-bench'

python driftdetection.py --input $no_summary_path --output answerDrift/noSummary/qwen4b_10k_no_summary_overthink.jsonl --data_source 'overthink-bench' &
python driftdetection.py --input $ourMethodPath --output answerDrift/withSummary/qwen4b_10k_with_summary_attention_overthink.jsonl --data_source 'overthink-bench'
