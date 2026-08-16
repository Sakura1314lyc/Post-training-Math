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
| SFT v7 | 71.65% | 71.65% | 98.26% | 1.36% |
| OPD checkpoint-30 | **72.33%** | **72.25%** | 97.73% | 1.90% |

SFT→OPD contains 49 regressions and 58 improvements, for a net gain of nine
questions (`+0.68 pp`); exact two-sided McNemar `p=0.439440`. This is a positive
point estimate, not a statistically significant result.

Model checkpoints are intentionally excluded by `.gitignore` and remain under
`outputs/opd/` on the experiment machine.
