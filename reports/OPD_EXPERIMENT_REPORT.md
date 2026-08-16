# Qwen2.5-Math-1.5B OPD/GKD Experiment on GSM8K

## Summary

This experiment starts from the final SFT v7 adapter and applies on-policy
distillation with TRL `GKDTrainer`. The frozen teacher is
`Qwen/Qwen2.5-Math-1.5B-Instruct`; the student generates fresh rollouts and is
trained against the teacher's token distributions on those student trajectories.
`lmbda=1.0` makes every training batch on-policy.

The selected checkpoint-30 reaches `954/1319 = 72.33%` on the official GSM8K
test, versus `71.80%` for Raw Base and `71.65%` for SFT v7. The point estimate is
the best of the three, but the paired improvements are not statistically
significant.

## Setup

| Item | Value |
|---|---|
| Student | Qwen2.5-Math-1.5B + SFT v7 LoRA |
| Teacher | Qwen2.5-Math-1.5B-Instruct, frozen NF4 |
| Trainable parameters | 2,317,312 (0.1499%) |
| Training candidates | 7,099; fixed 374-example validation excluded |
| Pilot subset | 256 prompts |
| Optimizer steps / rollouts | 50 / 200 |
| Learning rate | `5e-6`, constant |
| GKD parameters | `lmbda=1.0`, `beta=0.5`, `seq_kd=false` |
| Generation | temperature 0.9, 256 new tokens |
| Hardware | RTX 5060 Laptop GPU, 8 GiB |

The teacher scores `363/374 = 97.06%` on validation, compared with `85.29%` for
SFT v7. The paired teacher-only/SFT-only counts are 49/5 (`p=3.89e-10`), so the
teacher is strong enough to justify distillation.

Training takes 998.6 seconds and peaks at 6.776 GiB. Checkpoints are saved every
10 optimizer steps. A fixed 50-example screen selects steps 20 and 30; full
validation then selects step 30 without using the official test set.

## Results

| Official GSM8K test | Raw Base | SFT v7 | OPD step 30 |
|---|---:|---:|---:|
| Numerical accuracy | 71.80% | 71.65% | **72.33%** |
| Strict `####` accuracy | 0.00% | 71.65% | **72.25%** |
| Format compliance | 0.00% | **98.26%** | 97.73% |
| Hit 1,024-token limit | 3.11% | **1.36%** | 1.90% |
| Mean generated tokens | 371.74 | **102.30** | 111.24 |
| Evaluation time | 179m 0s | 64m 4s | 75m 0s |

Against SFT, OPD changes 49 correct answers to wrong and 58 wrong answers to
correct: a net gain of nine examples or `+0.68 pp`. The exact two-sided McNemar
test gives `p=0.439440`. Against Base, the net gain is seven examples or
`+0.53 pp`, with `p=0.754825`.

The correct interpretation is that OPD obtains a small positive test point
estimate, not a demonstrated stable improvement. It also slightly worsens
format compliance, truncation rate, and response length relative to SFT.

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

Checkpoints remain under the ignored `outputs/` directory. Tracked evaluation
artifacts and paired analyses are under `results/opd/`.
