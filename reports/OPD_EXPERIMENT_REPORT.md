# Qwen2.5-Math-1.5B OPD/GKD Experiment on GSM8K

## Summary

This experiment starts from the final SFT v7 adapter and applies on-policy
distillation with TRL `GKDTrainer`. The frozen teacher is
`Qwen/Qwen2.5-Math-1.5B-Instruct`; the student generates fresh rollouts and is
trained against the teacher's token distributions on those student trajectories.
`lmbda=1.0` makes every training batch on-policy.

Checkpoint-30 is fixed and evaluated across seeds 42/43/44. The three official
GSM8K test accuracies are 72.33%, 73.39%, and 73.01%, compared with 71.65% for
SFT v7. Their mean and sample standard deviation are **72.91% ± 0.54 pp**. All
three point estimates improve on SFT, but their exact paired McNemar p-values are
0.439440, 0.050487, and 0.117213; none is below 0.05. All three validation scores
are also below SFT. The test direction is reproducible across runs, but the
evidence is not sufficient to claim a stable, statistically significant gain.

## Setup

| Item | Value |
|---|---|
| Student | Qwen2.5-Math-1.5B + SFT v7 LoRA |
| Teacher | Qwen2.5-Math-1.5B-Instruct, frozen NF4 |
| Trainable parameters | 2,317,312 (0.1499%) |
| Training candidates | 7,099; fixed 374-example validation excluded |
| Subset per run | 256 prompts |
| Optimizer steps / rollouts | 50 / 200 |
| Learning rate | `5e-6`, constant |
| GKD parameters | `lmbda=1.0`, `beta=0.5`, `seq_kd=false` |
| Generation | temperature 0.9, 256 new tokens |
| Hardware | RTX 5060 Laptop GPU, 8 GiB |

The teacher scores `363/374 = 97.06%` on validation, compared with `85.29%` for
SFT v7. The paired teacher-only/SFT-only counts are 49/5 (`p=3.89e-10`), so the
teacher is strong enough to justify distillation.

The seed-42 pilot takes 998.6 seconds and peaks at 6.776 GiB. Checkpoints are
saved every 10 optimizer steps. A fixed 50-example screen selects steps 20 and
30; full validation then selects step 30 without using the official test set.
Seeds 43/44 subsequently use the same hyperparameters and fixed checkpoint-30,
and both are evaluated in full regardless of their validation result.

The current `--seed` controls both selection of the 256 training prompts and
trainer randomness. The three training subsets overlap by only 7, 19, and 9
examples pairwise. This is therefore end-to-end pipeline replication, not a
fixed-data experiment that isolates optimizer/generation randomness.

## Results

| Official GSM8K test | Raw Base | SFT v7 | OPD seed 42 | OPD seed 43 | OPD seed 44 | OPD mean ± SD |
|---|---:|---:|---:|---:|---:|---:|
| Numerical accuracy | 71.80% | 71.65% | 72.33% | **73.39%** | 73.01% | **72.91% ± 0.54** |
| Strict `####` accuracy | 0.00% | 71.65% | 72.25% | **73.31%** | 72.93% | **72.83% ± 0.54** |
| Format compliance | 0.00% | **98.26%** | 97.73% | 97.80% | 97.50% | 97.68% ± 0.16 |
| Hit 1,024-token limit | 3.11% | **1.36%** | 1.90% | 1.82% | 2.12% | 1.95% ± 0.16 |
| Mean generated tokens | 371.74 | **102.30** | 111.24 | 110.48 | 113.09 | 111.60 ± 1.34 |
| Evaluation time | 179m 0s | 64m 4s | 75m 0s | 72m 8s | 77m 43s | 74m 57s ± 2m 48s |

| OPD run vs SFT | SFT-only | OPD-only | Net | Accuracy delta | McNemar p |
|---|---:|---:|---:|---:|---:|
| seed 42 | 49 | 58 | +9 | +0.68 pp | 0.439440 |
| seed 43 | 52 | 75 | +23 | +1.74 pp | 0.050487 |
| seed 44 | 50 | 68 | +18 | +1.36 pp | 0.117213 |

No individual paired test crosses the 0.05 threshold. The three runs reuse the
same test questions and cannot be treated as independent datasets for a simple
combined significance test. Validation accuracies are 84.76%, 84.22%, and
83.69%, all below SFT's 85.29%. The defensible conclusion is reproducible test
direction, not demonstrated capability improvement. Format compliance,
truncation, and response length also degrade consistently relative to SFT.

## Reproduction

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

For robustness runs, keep every hyperparameter and checkpoint choice fixed and
change only the seed and output directory:

```bash
for seed in 43 44; do
  python scripts/train_opd_gkd.py \
    --output-dir "outputs/opd/qwen25_math_15b_gkd_seed${seed}" \
    --num-samples 256 \
    --max-steps 50 \
    --gradient-accumulation-steps 4 \
    --learning-rate 5e-6 \
    --max-new-tokens 256 \
    --lmbda 1.0 \
    --beta 0.5 \
    --save-steps 10 \
    --save-total-limit 5 \
    --seed "$seed" \
    --save-final-adapter
done
```

Checkpoints remain under the ignored `outputs/` directory. Tracked evaluation
artifacts and paired analyses are under `results/opd/`.

The machine-readable three-run aggregate is
[`opd_ckpt30_multiseed_summary.json`](../results/opd/final/opd_ckpt30_multiseed_summary.json).
The official test has now been used for the initial report and two robustness
runs and must not be used for further hyperparameter tuning.
