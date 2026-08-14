# Experiment configurations / 实验配置

## `main/`

The assignment's primary track. / 课程任务的主实验。

- `qwen25_math_15b_base_lora_sft_v1.yaml`: `Qwen2.5-Math-1.5B` → GSM8K LoRA-SFT.

## `controls/`

Optional controls for separating math-domain pretraining from SFT gains. /
用于区分“数学领域继续预训练收益”和“SFT 收益”的可选对照实验。

- `qwen25_15b_base_lora_sft_v1.yaml`: general `Qwen2.5-1.5B` → GSM8K LoRA-SFT.

## `archive/`

Historical experiments retained for reproducibility. / 为保证可复现性而保留的历史实验。

- `qwen25_15b_instruct_lora_sft_v1.yaml`: continued SFT from the 1.5B Instruct model.
- `qwen25_05b_*`: early 0.5B smoke tests and SFT ablations.

All commands assume they are run from the project root, so `dataset_dir: data`
and `output_dir: outputs/...` continue to resolve correctly.
