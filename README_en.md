# GSM8K Post-training Experiment

[简体中文](README.md) | [English](README_en.md)

This repository implements completed three-seed experiments for SFT,
On-Policy Distillation (OPD/GKD), and GRPO:

> `Qwen/Qwen2.5-Math-1.5B` (raw math base) → GSM8K LoRA-SFT → on-policy
> distillation with an Instruct teacher / GRPO → held-out selection → official
> test → independent SVAMP generalization evaluation

Detailed reports: [SFT (Chinese)](reports/SFT_EXPERIMENT_REPORT_zh.md),
[OPD (English)](reports/OPD_EXPERIMENT_REPORT.md), and
[OPD (Chinese)](reports/OPD_EXPERIMENT_REPORT_zh.md). The independent
generalization analysis is documented in the
[SVAMP report (Chinese)](reports/SVAMP_EXPERIMENT_REPORT_zh.md). The complete
GRPO analysis is documented in the
[GRPO report (Chinese)](reports/GRPO_EXPERIMENT_REPORT_zh.md).

## Final result

| Official GSM8K test (1,319 examples) | Raw Base | SFT v7 | OPD 3-run mean ± SD | GRPO 3-run mean ± SD |
|---|---:|---:|---:|---:|
| Relaxed numerical accuracy | 71.80% | 71.65% | **72.91% ± 0.54** | 72.18% ± 0.35 |
| Strict `####` accuracy | 0.00% | 71.65% | **72.83% ± 0.54** | 72.10% ± 0.40 |
| Format compliance | 0.00% | **98.26%** | 97.68% ± 0.16 | 98.13% ± 0.09 |
| Hit 1,024-token limit | 3.11% | **1.36%** | 1.95% ± 0.16 | 1.42% ± 0.04 |
| Mean generated tokens | 371.74 | **102.30** | 111.60 ± 1.34 | 102.99 ± 1.06 |

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
truncation, and response length. GRPO seeds 42/43/44 reach 72.25%, 72.48%, and
71.80%, for `72.18% ± 0.35 pp`; their paired gains over SFT are all
non-significant. GRPO averages `81.20% ± 0.36 pp` on SVAMP, 0.30 pp below SFT,
so it does not provide evidence of a cross-dataset gain either.

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
- GRPO: native TRL GRPO, numerical/format reward weights 1.0/0.1,
  `beta=0`, 30 steps / 120 rollouts, fixed checkpoint-30 for seeds 42/43/44

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
results/grpo/               # GRPO validation/test, paired analyses, and multi-seed summary
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

As of 2026-08-21, all five runs in the fixed protocol are complete:

| SVAMP test (1,000 examples) | Raw Base | SFT v7 | OPD seeds 42/43/44 mean ± SD |
|---|---:|---:|---:|
| Numerical accuracy | **85.20%** | 81.50% | 81.93% ± 0.32 |
| Format compliance | 0.00% | **98.80%** | 98.47% ± 0.21 |
| Hit token limit | 7.00% | **0.90%** | 1.23% ± 0.21 |
| Mean generated tokens | 290.81 | **55.07** | 61.32 ± 2.12 |

The paired Base-to-SFT change is −3.70 pp (`p=0.00761528`). OPD seeds 42, 43,
and 44 recover +0.80, +0.20, and +0.30 pp over SFT, with `p=0.322236`,
`p=0.891923`, and `p=0.794844`. SFT therefore shows a significant cross-dataset
numerical regression despite major format, termination, and speed improvements.
OPD is directionally consistent across all three runs but recovers only 0.43 pp
on average, with no individually significant paired result, and remains below Raw
Base.

The official test set must not be used for checkpoint or hyperparameter
selection. Token-level validation loss is also not a substitute for free-generation
mathematical accuracy.

### GRPO

The current LLaMA-Factory checkout does not expose a GRPO training stage, so
[`train_grpo.py`](scripts/train_grpo.py) uses native TRL 0.24. It continues the
SFT v7 adapter, reproduces and excludes the fixed 374-example validation split,
and combines exact numerical correctness with a smaller strict-format reward.
The 8 GiB smoke protocol uses native Transformers generation, four completions,
and a 128-token completion limit:

```bash
python3 scripts/train_grpo.py \
  --output-dir outputs/grpo/qwen25_math_15b_grpo_smoke \
  --num-samples 8 \
  --max-steps 1 \
  --num-generations 4 \
  --gradient-accumulation-steps 4 \
  --max-prompt-length 512 \
  --max-completion-length 128
```

This continued-adapter pilot fixes `beta=0`. With a nonzero KL coefficient,
TRL would disable the existing adapter and incorrectly use Raw Base rather than
the initial SFT policy as its PEFT reference.

The smoke run completed successfully in 7.49 seconds with 3.33 GiB peak allocated
memory, finite rewards and metrics, and a verified LoRA parameter update. Seeds
42/43/44 then completed 30 steps / 120 rollouts each in roughly 137–141 seconds.
Their fixed checkpoint-30 adapters were then evaluated under the same formal
1,024-token, native-EOS protocol used for SFT and OPD:

| Formal GRPO validation (374 examples) | seed42 | seed43 | seed44 | 3-run mean ± SD |
|---|---:|---:|---:|---:|
| Numerical accuracy | 85.83% | 85.83% | 85.29% | 85.65% ± 0.31 |
| Strict accuracy | 85.83% | 85.83% | 85.29% | 85.65% ± 0.31 |
| Format compliance | 99.20% | 99.73% | 99.47% | 99.47% ± 0.27 |

Formal GSM8K test accuracy for seeds 42/43/44 is 72.25%, 72.48%, and 71.80%,
for `72.18% ± 0.35 pp`. The paired changes over SFT are +0.61, +0.83, and
+0.15 pp, with McNemar p-values 0.291215, 0.168978, and 0.885433. The direction
is consistent, but no individual result is significant. On SVAMP, the three-run
mean is `81.20% ± 0.36 pp`, 0.30 pp below SFT, so no cross-dataset benefit is
observed. See the complete
[`GRPO report`](reports/GRPO_EXPERIMENT_REPORT_zh.md) and
[`results/grpo/README.md`](results/grpo/README.md) for protocols, artifacts, and
paired analyses.

## Tests

```bash
python3 -m unittest discover -s tests -v
```
