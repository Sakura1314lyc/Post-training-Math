# Experiment results / 实验结果

## 目录

```text
dev/base/       # 固定 374 题 validation 的 Raw Base 结果
dev/sft_v3/     # v3 validation、smoke 与 Base/v3 配对分析
dev/sft_v7/     # 最终 v7 validation、smoke 与配对分析
dev/ablations/  # v4-v6 原生生成失败诊断
dev/legacy_v1_v2/ # v1/v2、早期 Base、smoke 与 debug 结果
final/          # 官方 GSM8K test 最终结果
archive/        # 历史 Instruct 与 0.5B 实验
```

## 最终官方 Test

| 文件 | 内容 |
|---|---|
| `final/test_gsm8k_base_15b_v3.json` | Raw Base：947/1319，71.80% |
| `final/test_gsm8k_sft_v7_15b_ckpt888_native_v3.json` | SFT v7：945/1319，71.65%，格式 98.26% |
| `final/test_base_sft_v7_ckpt888_transition_analysis.json` | 逐题配对、回复统计与 McNemar 检验 |

最终配对转移为：共同答对 760、Base-only 187、SFT-only 185、共同答错 187；
精确双侧 McNemar `p=0.958659`。

## Validation

- `dev/base/dev_math_base_15b_v3.json`：Raw Base，319/374（85.29%）。
- `dev/sft_v3/dev_math_base_sft_v3_15b_ckpt888.json`：v3，302/374（80.75%）。
- `dev/sft_v7/dev_math_base_sft_v7_15b_ckpt888.json`：v7，319/374（85.29%）。
- `dev/sft_v7/base_sft_v7_ckpt888_transition_analysis.json`：Base/v7 配对，
  Base-only 40、v7-only 40、`p=1`。
- `dev/sft_v7/sft_v3_v7_transition_analysis.json`：v3/v7 配对，v3-only 16、
  v7-only 33、`p=0.0212941`。

`native10`、`smoke20` 与 `slice20_50` 文件只用于检查输出格式、EOS 和明显退化，
不能代替完整 validation 或 test 结果。

## 结果解释规则

- 只用 validation 选择 checkpoint 和超参数；
- 官方 test 只用于最终报告，不能继续指导 v7 调参；
- 所有正式对比必须保证相同 `evaluation_version`、样本集合、prompt 和生成参数；
- 数值准确率、严格格式准确率、格式遵循率与终止行为分别报告；
- `outputs/` 中的 LoRA checkpoint 不提交到 Git。
