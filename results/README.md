# Experiment results / 实验结果

New Math-Base main results and general-base control results should be written
directly to this directory with descriptive names, for example:

```text
dev_math_base_15b.json
dev_math_base_sft_15b_ckpt100.json
final_math_base_15b_test.json
final_math_base_sft_15b_test.json
final_math_base_sft_analysis.json
```

新的 Math-Base 主实验结果和通用 Base 对照结果直接写入本目录。Checkpoint
选择只使用 validation 结果；完整 test 结果只用于最终报告。

## Current Math-Base validation artifacts / 当前主实验验证结果

- `dev_math_base_15b_v3.json`: 374-example raw-base validation baseline,
  rescored with `gsm8k_numeric_v3`.
- `dev_math_base_sft_15b_ckpt100.json`: v1 checkpoint-100 full validation.
- `dev_math_base_sft_ckpt100_analysis_v1.json`: paired Base/SFT transition
  analysis and exact McNemar test.
- `dev_math_base_sft_*_smoke20_v3.json`: exploratory 20-example checkpoint
  diagnostics; they are not final benchmark results.
- `debug_sft_*`: one-example artifacts retained to document the missing-EOS
  and repeated-output failure and the answer-line stopping fix.

The v1 full-validation point estimate is `85.29% -> 82.35%` (`-2.94` points,
paired exact McNemar `p=0.3049`). This is a negative result, not a statistically
significant improvement or degradation. The stronger finding is behavioral:
checkpoint-100 emitted native EOS on `0/374` examples and produced only one
strictly compliant correct answer.

当前 v1 完整 validation 的点估计为 `85.29% -> 82.35%`（下降 `2.94`
个百分点，配对精确 McNemar `p=0.3049`）。该差异没有达到统计显著，不能表述为
“显著下降”。更明确的发现是终止与格式异常：checkpoint-100 在 `374` 道题中没有
一次原生 EOS，且只有一道题同时满足严格格式与答案正确。

## `archive/instruct_15b/`

Historical 1.5B Instruct → continued-SFT evaluations and paired analyses.

历史 1.5B Instruct continued-SFT 评测及配对分析。

## `archive/instruct_05b/`

Historical 0.5B pipeline tests, checkpoints, rescored files, and ablations.

历史 0.5B 流程测试、checkpoint 评测、重评分文件和消融实验。
