# Direct API Baseline Summary on Belief-R Commitment-Control Test

Date: 2026-04-17 UTC

## Setup

This report compares direct external-model baselines on the fixed Belief-R
commitment-control test split:

- dataset: `data/processed/belief_r_commitment_control_test.jsonl`
- size: `260`
- conditions:
  - `full_info`: `130`
  - `incremental_no_overturn`: `27`
  - `incremental_overturn_reasoning`: `103`

The direct API baselines do **not** reuse the repository's earlier
`source_revision` prompt framework. Each model is asked to directly choose the
final answer label `a/b/c` from the dataset example itself. The predicted
control label is then derived as:

- `preserve` if predicted final answer equals the early commitment label
- `replace` otherwise

## Main Comparison

| method | overall answer | overturn answer | no-overturn answer |
|---|---:|---:|---:|
| direct OpenAI `gpt-5.4-mini` | 0.2077 | 0.0000 | 1.0000 |
| direct DeepSeek `deepseek-chat` | 0.2192 | 0.0097 | 1.0000 |
| frozen Qwen2.5-0.5B MC baseline | 0.7308 | 0.8835 | 0.1481 |
| frozen prompt baseline | 0.3615 | 0.2718 | 0.5926 |
| NumPy CIPC baseline | 0.7846 | 0.7961 | 0.9259 |
| HF highrank v1 | 0.8462 | 0.8641 | 0.7778 |

## Direct API Results

### OpenAI `gpt-5.4-mini`

- overall answer: `0.2077`
- `full_info`: `0.2077`
- `incremental_no_overturn`: `1.0000`
- `incremental_overturn_reasoning`: `0.0000`
- early commitment persistence on change cases: `1.0000`
- late evidence takeover on change cases: `0.0000`

Prediction behavior:

- predicted `preserve`: `260 / 260`
- predicted `replace`: `0 / 260`
- predicted answers:
  - `a`: `142`
  - `b`: `118`
  - `c`: `0`

### DeepSeek `deepseek-chat`

- overall answer: `0.2192`
- `full_info`: `0.2231`
- `incremental_no_overturn`: `1.0000`
- `incremental_overturn_reasoning`: `0.0097`
- early commitment persistence on change cases: `0.9854`
- late evidence takeover on change cases: `0.0146`

Prediction behavior:

- predicted `preserve`: `257 / 260`
- predicted `replace`: `3 / 260`
- predicted answers:
  - `a`: `141`
  - `b`: `116`
  - `c`: `3`

## Interpretation

- Both direct API baselines collapse toward preserving the early commitment.
- They solve `incremental_no_overturn`, but almost completely fail on
  `incremental_overturn_reasoning`.
- This is a different failure mode from the frozen local Qwen multiple-choice
  baseline, which was overly aggressive and over-predicted overturns.
- Together, these baselines strengthen the main claim that the dataset is not
  solved by generic strong LLMs through straightforward prompting. The central
  difficulty is learning the right preserve-vs-revise trade-off.

## Artifacts

- direct OpenAI baseline:
  - `runs/direct_openai_gpt54mini_mc_test_v1/`
- direct DeepSeek baseline:
  - `runs/direct_deepseek_chat_mc_test_v1/`
- direct baseline script:
  - `analysis/evaluate_api_mc_baseline.py`
