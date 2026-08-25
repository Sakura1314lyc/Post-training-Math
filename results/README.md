# Experiment results / 实验结果

## 目录

```text
dev/base/       # 固定 374 题 validation 的 Raw Base 结果
dev/sft_v3/     # v3 validation、smoke 与 Base/v3 配对分析
dev/sft_v7/     # 最终 v7 validation、smoke 与配对分析
dev/ablations/  # v4-v6 原生生成失败诊断
dev/legacy_v1_v2/ # v1/v2、早期 Base、smoke 与 debug 结果
final/          # 官方 GSM8K test 最终结果
opd/            # 教师、OPD checkpoint 选择、validation 与最终 test
svamp/          # 独立 SVAMP 泛化评测协议与结果
grpo/           # GRPO 三随机种子训练、正式评测、配对分析与汇总
confirmatory_v2/ # 多 seed、大预算、冻结 audit 的确认性修正实验
exploratory_v3/ # GRPO 低温 post-hoc 单因素负结果
exploratory_v4/ # GRPO 低学习率 post-hoc 单因素负结果
exploratory_v5/ # GRPO 移除 tied lm_head 的结构消融负结果
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

## OPD/GKD

OPD 从 SFT v7 开始，以冻结的 `Qwen2.5-Math-1.5B-Instruct` 作为教师。固定使用
checkpoint-30，并完成 seed 42/43/44 三次端到端运行：

| 官方 test | Base | SFT v7 | OPD seed 42 | OPD seed 43 | OPD seed 44 | OPD 均值 ± SD |
|---|---:|---:|---:|---:|---:|---:|
| 数值准确率 | 71.80% | 71.65% | 72.33% | **73.39%** | 73.01% | **72.91% ± 0.54** |
| 严格准确率 | 0.00% | 71.65% | 72.25% | **73.31%** | 72.93% | **72.83% ± 0.54** |
| 格式遵循率 | 0.00% | **98.26%** | 97.73% | 97.80% | 97.50% | 97.68% ± 0.16 |
| 达到长度上限 | 3.11% | **1.36%** | 1.90% | 1.82% | 2.12% | 1.95% ± 0.16 |

三次 SFT→OPD 的准确率差为 +0.68、+1.74、+1.36 pp，McNemar `p` 分别为
0.439440、0.050487、0.117213，均未低于 0.05；三次 validation 又都低于 SFT。
因此只能表述为 test 方向可复现，不能声称显著能力提升。文件说明见
[`opd/README.md`](opd/README.md)。

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

## SVAMP 泛化评测

`svamp/` 使用独立的 `svamp_numeric_v1` 协议评测完整 1,000 题 Calc-SVAMP test。
该集合不参与训练、checkpoint 选择或超参数调整；固定命令和最终结果表见
[`svamp/README.md`](svamp/README.md)。

Base 为 85.20%，SFT 为 81.50%（配对 `p=0.00761528`）；OPD seed 42/43/44
为 82.30%/81.70%/81.80%，均值 `81.93% ± 0.32 pp`。三次相对 SFT 均为正向
点估计，但配对 `p` 为 0.322236/0.891923/0.794844，均不显著。GRPO seed 42/43/44
为 81.60%/81.10%/80.90%，均值 `81.20% ± 0.36 pp`；相对 SFT 的方向不一致，说明
GSM8K 上的小幅正收益没有在 SVAMP 复现。

## GRPO

`grpo/` 记录从 SFT v7 继续训练的原生 TRL GRPO 实验。seed 42/43/44 的 30-step 训练
及正式 validation、GSM8K test、SVAMP 均已完成。GSM8K test 三次平均为
`72.18% ± 0.35 pp`，比 SFT 高 0.53 pp；SVAMP 平均为 `81.20% ± 0.36 pp`，比
SFT 低 0.30 pp。完整汇总、协议说明和配对文件见 [`grpo/README.md`](grpo/README.md)。

## 结果解释规则

- 只用 validation 选择 checkpoint 和超参数；
- 官方 test 只用于最终报告，不能继续指导 v7 调参；
- 所有正式对比必须保证相同 `evaluation_version`、样本集合、prompt 和生成参数；
- 数值准确率、严格格式准确率、格式遵循率与终止行为分别报告；
- `outputs/` 中的 LoRA checkpoint 不提交到 Git。

## Confirmatory v2 与最终边界

确认性修正实验已经完成。SFT 三 seed 的 dev-select 均值为 83.78% ± 1.47 pp；规范
seed42 在一次性 dev-audit 上为 303/374（81.02%）。扩大到 200 steps 后，OPD 因格式与
终止行为崩坏被拒绝，GRPO 三 seed 均显著低于规范 SFT，也被拒绝。完整机器汇总见
[`confirmatory_v2/confirmatory_v2_progress.json`](confirmatory_v2/confirmatory_v2_progress.json)。

协议结束后的 exploratory v3--v5 只使用已经消费的 dev-select，分别检验低 temperature、
低 learning rate 和移除 tied `lm_head`，均为负结果。它们不改变确认性结论，也不得据此
继续查看 audit、GSM8K test 或 SVAMP。项目统一结论见
[`../reports/FINAL_EXPERIMENT_SUMMARY_zh.md`](../reports/FINAL_EXPERIMENT_SUMMARY_zh.md)。
