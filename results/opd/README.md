# OPD/GKD results

The OPD student starts from SFT v7 and uses the frozen
`Qwen/Qwen2.5-Math-1.5B-Instruct` teacher. See the
[Chinese report](../../reports/OPD_EXPERIMENT_REPORT_zh.md) or
[English report](../../reports/OPD_EXPERIMENT_REPORT.md).

## Layout

```text
teacher/  # Teacher smoke, full validation, and SFT/teacher paired analysis
dev/      # 50-example checkpoint screen and full validation candidates
final/    # Official GSM8K test and paired Base/SFT-to-OPD analyses
```

## Main results

| Official test | Accuracy | Strict accuracy | Format | Token cap |
|---|---:|---:|---:|---:|
| Raw Base | 71.80% | 0.00% | 0.00% | 3.11% |
| SFT v7 | 71.65% | 71.65% | **98.26%** | **1.36%** |
| OPD seed 42 checkpoint-30 | 72.33% | 72.25% | 97.73% | 1.90% |
| OPD seed 43 checkpoint-30 | **73.39%** | **73.31%** | 97.80% | 1.82% |
| OPD seed 44 checkpoint-30 | 73.01% | 72.93% | 97.50% | 2.12% |
| OPD three-run mean ± SD | **72.91% ± 0.54** | **72.83% ± 0.54** | 97.68% ± 0.16 | 1.95% ± 0.16 |

All three OPD test point estimates exceed SFT, but the paired McNemar p-values
are 0.439440, 0.050487, and 0.117213; none is below 0.05. All three validation
scores are below SFT. The aggregate and protocol caveat are recorded in
[`final/opd_ckpt30_multiseed_summary.json`](final/opd_ckpt30_multiseed_summary.json).

Model checkpoints are intentionally excluded by `.gitignore` and remain under
`outputs/opd/` on the experiment machine.
