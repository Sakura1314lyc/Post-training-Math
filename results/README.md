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

## `archive/instruct_15b/`

Historical 1.5B Instruct → continued-SFT evaluations and paired analyses.

历史 1.5B Instruct continued-SFT 评测及配对分析。

## `archive/instruct_05b/`

Historical 0.5B pipeline tests, checkpoints, rescored files, and ablations.

历史 0.5B 流程测试、checkpoint 评测、重评分文件和消融实验。
