# GSM8K Post-training Experiment

[简体中文](README.md) | [English](README_en.md)

This repository implements both SFT and On-Policy Distillation (OPD/GKD):

> `Qwen/Qwen2.5-Math-1.5B` (raw math base) → GSM8K LoRA-SFT → on-policy
> distillation with an Instruct teacher → held-out selection → official test

Detailed reports: [SFT (Chinese)](reports/SFT_EXPERIMENT_REPORT_zh.md),
[OPD (English)](reports/OPD_EXPERIMENT_REPORT.md), and
[OPD (Chinese)](reports/OPD_EXPERIMENT_REPORT_zh.md). The interim independent
generalization analysis is documented in the
[SVAMP report (Chinese)](reports/SVAMP_EXPERIMENT_REPORT_zh.md).

## Final result

| Official GSM8K test (1,319 examples) | Raw Base | SFT v7 | OPD seed 42 | OPD 3-run mean ± SD |
|---|---:|---:|---:|---:|
| Relaxed numerical accuracy | 71.80% | 71.65% | 72.33% | **72.91% ± 0.54** |
| Strict `####` accuracy | 0.00% | 71.65% | 72.25% | **72.83% ± 0.54** |
| Format compliance | 0.00% | **98.26%** | 97.73% | 97.68% ± 0.16 |
| Hit 1,024-token limit | 3.11% | **1.36%** | 1.90% | 1.95% ± 0.16 |
| Mean generated tokens | 371.74 | **102.30** | 111.24 | 111.60 ± 1.34 |
| Full evaluation time | 179m 0s | 64m 4s | 75m 0s | 74m 57s ± 2m 48s |

The paired comparison contains 187 Base-only correct answers and 185 SFT-only
correct answers. The exact two-sided McNemar test gives `p=0.958659`. SFT v7
therefore does not improve numerical accuracy, but it preserves accuracy while
substantially improving formatting, termination, generation length, and wall-clock
evaluation time. OPD seeds 42/43/44 reach 72.33%, 73.39%, and 73.01% on test,
respectively, for a mean of `72.91% ± 0.54 pp`; all three point estimates exceed
SFT. Their paired McNemar p-values against SFT are 0.439440, 0.050487, and
0.117213, while all three validation scores are below SFT. The test direction is
consistent across runs, but the evidence is still insufficient to claim a stable,
statistically significant improvement. OPD also slightly worsens formatting,
truncation, and response length.

## Final setup

- Base checkpoint: `Qwen/Qwen2.5-Math-1.5B`
- Framework: LLaMA-Factory with PEFT LoRA
- Config: [`configs/main/qwen25_math_15b_base_lora_sft_v7.yaml`](configs/main/qwen25_math_15b_base_lora_sft_v7.yaml)
- Data: `data/gsm8k_sft_clean.json`, 7,473 examples
- LoRA: `q_proj,v_proj,lm_head`, rank 8, alpha 16
- Training: one epoch, learning rate `2e-5`, seed 42
- Selected adapter: `outputs/qwen25_math_15b_base_lora_sft_v7/checkpoint-888`
- Evaluator: `gsm8k_numeric_v3`
- OPD teacher: `Qwen/Qwen2.5-Math-1.5B-Instruct`, frozen NF4
- OPD objective: TRL GKD, `lmbda=1.0`, `beta=0.5`, 50 steps / 200 rollouts
- Reported OPD checkpoints: step 30 from
  `outputs/opd/qwen25_math_15b_gkd_{pilot50,seed43,seed44}`

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
results/opd/                # Teacher, OPD validation/test, and paired analyses
results/svamp/              # Independent SVAMP generalization protocol and results
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

Run the OPD pilot after producing the SFT v7 adapter:

```bash
python scripts/train_opd_gkd.py \
  --output-dir outputs/opd/qwen25_math_15b_gkd_pilot50 \
  --num-samples 256 \
  --max-steps 50 \
  --gradient-accumulation-steps 4 \
  --learning-rate 5e-6 \
  --max-new-tokens 256 \
  --lmbda 1.0 \
  --beta 0.5 \
  --save-steps 10 \
  --save-total-limit 5 \
  --save-final-adapter
```

The script exactly reproduces and excludes the fixed 374-example SFT validation
split. Model checkpoints stay in the ignored `outputs/` directory; tracked OPD
metrics and paired analyses are under `results/opd/`.

Seeds 42/43/44 use identical hyperparameters and a fixed checkpoint-30. The
current `--seed` controls both selection of the 256 training prompts and trainer
randomness, so the reported variation is end-to-end run variability rather than
fixed-data training-seed variability. The aggregate artifact is
[`opd_ckpt30_multiseed_summary.json`](results/opd/final/opd_ckpt30_multiseed_summary.json).

### Independent SVAMP evaluation

To avoid further iteration on the GSM8K test set, the evaluator also supports all
1,000 examples from
[`MU-NLPC/Calc-svamp`](https://huggingface.co/datasets/MU-NLPC/Calc-svamp/blob/main/README.md)
under `default/test`. The
[original SVAMP release](https://github.com/arkilpatel/SVAMP/blob/main/SVAMP.json)
does not define an official train/test split, so the full collection is treated as
a test set, following the Calc-SVAMP data card. Calc-SVAMP also corrects one
inconsistent equation/answer pair from the original data.

Run a 10-example plumbing smoke test first, without using its accuracy for model
selection:

```bash
python3 scripts/eval_sft_adapter.py \
  --benchmark svamp \
  --base-model Qwen/Qwen2.5-Math-1.5B \
  --eval-split test \
  --num-samples 10 \
  --max-new-tokens 1024 \
  --no-stop-after-answer-line \
  --output results/svamp/smoke/svamp_base_15b_smoke10_v1.json
```

After the smoke test, evaluate Raw Base, SFT v7, and OPD seeds 42/43/44 exactly
once with the fixed protocol in
[`results/svamp/README.md`](results/svamp/README.md). These results use evaluator
version `svamp_numeric_v1` and must not be used for checkpoint or hyperparameter
selection.

As of 2026-08-20, Base, SFT, and OPD seeds 42/43 are complete; seed 44 remains
pending:

| SVAMP test (1,000 examples) | Raw Base | SFT v7 | OPD seed 42 | OPD seed 43 |
|---|---:|---:|---:|---:|
| Numerical accuracy | **85.20%** | 81.50% | 82.30% | 81.70% |
| Format compliance | 0.00% | **98.80%** | 98.70% | 98.40% |
| Hit token limit | 7.00% | **0.90%** | 1.00% | 1.30% |
| Evaluation time | 109m 21s | **27m 32s** | 29m 23s | 34m 38s |

The paired Base-to-SFT change is −3.70 pp (`p=0.00761528`). OPD seeds 42 and
43 recover +0.80 and +0.20 pp over SFT, with `p=0.322236` and `p=0.891923`.
This interim evidence indicates a significant cross-dataset numerical regression
from SFT despite major format, termination, and speed improvements. The first two
OPD runs provide only small, non-significant recoveries; the three-run conclusion
must wait for seed 44.

The official test set must not be used for checkpoint or hyperparameter
selection. Token-level validation loss is also not a substitute for free-generation
mathematical accuracy.

## Tests

```bash
python3 -m unittest discover -s tests -v
```
