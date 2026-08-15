# GSM8K Post-training Experiment

[English](README.md) | [简体中文](README_zh.md)

This repository completes the SFT part of a post-training assignment:

> `Qwen/Qwen2.5-Math-1.5B` (raw math base) → GSM8K LoRA-SFT → held-out
> validation → paired evaluation on the official test set

The detailed experiment report is available in
[`reports/SFT_EXPERIMENT_REPORT_zh.md`](reports/SFT_EXPERIMENT_REPORT_zh.md).

## Final result

| Official GSM8K test (1,319 examples) | Raw Base | SFT v7 |
|---|---:|---:|
| Relaxed numerical accuracy | 71.80% | 71.65% |
| Strict `####` accuracy | 0.00% | 71.65% |
| Format compliance | 0.00% | 98.26% |
| Hit 1,024-token limit | 3.11% | 1.36% |
| Mean generated tokens | 371.74 | 102.30 |
| Full evaluation time | 179m 0s | 64m 4s |

The paired comparison contains 187 Base-only correct answers and 185 SFT-only
correct answers. The exact two-sided McNemar test gives `p=0.958659`. SFT v7
therefore does not improve numerical accuracy, but it preserves accuracy while
substantially improving formatting, termination, generation length, and wall-clock
evaluation time.

## Final setup

- Base checkpoint: `Qwen/Qwen2.5-Math-1.5B`
- Framework: LLaMA-Factory with PEFT LoRA
- Config: [`configs/main/qwen25_math_15b_base_lora_sft_v7.yaml`](configs/main/qwen25_math_15b_base_lora_sft_v7.yaml)
- Data: `data/gsm8k_sft_clean.json`, 7,473 examples
- LoRA: `q_proj,v_proj,lm_head`, rank 8, alpha 16
- Training: one epoch, learning rate `2e-5`, seed 42
- Selected adapter: `outputs/qwen25_math_15b_base_lora_sft_v7/checkpoint-888`
- Evaluator: `gsm8k_numeric_v3`

The v7 dataset removes 23,716 `<<expression=result>>` calculator annotations
while preserving every final `####` answer. This recovers validation accuracy
from 80.75% for v3 to the 85.29% raw-base baseline.

## Layout

```text
configs/main/               # Main v1-v3 and final v7 configs
configs/archive/math_15b/   # Failed v4-v6 ablations
data/                       # Training data and dataset registry
scripts/                    # Data preparation and evaluation utilities
results/dev/                # Validation and diagnostic artifacts
results/final/              # Final official-test artifacts
results/archive/            # Historical continued-SFT experiments
reports/                    # Experiment reports
tests/                      # Unit tests
outputs/                    # Local checkpoints, ignored by Git
```

## Reproduction

```bash
cd /home/sakura/projects/llm/post-training-math
conda activate sft

python3 scripts/prepare_clean_sft_data.py \
  --input data/gsm8k_sft_formal.json \
  --output data/gsm8k_sft_clean.json \
  --overwrite

llamafactory-cli train configs/main/qwen25_math_15b_base_lora_sft_v7.yaml
```

Evaluate native generation without task-specific answer-line truncation:

Existing result files are protected from accidental replacement. Use a new output
name for a reproduction run, or explicitly pass `--overwrite` after confirming it.

```bash
python3 scripts/eval_sft_adapter.py \
  --base-model Qwen/Qwen2.5-Math-1.5B \
  --adapter outputs/qwen25_math_15b_base_lora_sft_v7/checkpoint-888 \
  --eval-split test \
  --max-new-tokens 1024 \
  --no-stop-after-answer-line \
  --output results/final/test_gsm8k_sft_v7_15b_ckpt888_native_v3.json
```

Generate the paired analysis:

```bash
python3 scripts/compare_base_sft.py \
  --base results/final/test_gsm8k_base_15b_v3.json \
  --sft results/final/test_gsm8k_sft_v7_15b_ckpt888_native_v3.json \
  --output results/final/test_base_sft_v7_ckpt888_transition_analysis.json
```

The official test set must not be used for checkpoint or hyperparameter
selection. Token-level validation loss is also not a substitute for free-generation
mathematical accuracy.

## Tests

```bash
python3 -m unittest discover -s tests -v
```
