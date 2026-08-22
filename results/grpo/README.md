# GRPO results / GRPO 结果

本目录记录从最终 SFT v7 adapter 继续进行的三随机种子 GRPO 实验。训练脚本为
[`scripts/train_grpo.py`](../../scripts/train_grpo.py)，checkpoint 位于被 Git 忽略的
`outputs/grpo/`；仓库提交评测结果、逐题配对和机器可读汇总。

## 状态（2026-08-22）

- seed42/43/44 的 30-step 训练、正式 validation、GSM8K test 和 SVAMP 已全部完成；
- 报告 checkpoint 固定为 step 30，官方 test 未用于选择 checkpoint 或超参数；
- 正式协议为贪心解码、BF16、1,024-token 上限、原生 EOS；
- 早期 512-token 结果单独保存在 `pilot/`，不参与最终同协议结论。

机器汇总见 [`grpo_multiseed_summary.json`](grpo_multiseed_summary.json)，完整分析见
[`GRPO_EXPERIMENT_REPORT_zh.md`](../../reports/GRPO_EXPERIMENT_REPORT_zh.md)。

## GSM8K Validation

| 374 题 validation | SFT v7 | GRPO 42 | GRPO 43 | GRPO 44 | GRPO 均值 ± SD |
|---|---:|---:|---:|---:|---:|
| 数值/严格准确率 | 85.29% | 85.83% | 85.83% | 85.29% | 85.65% ± 0.31 |
| 格式遵循率 | 99.47% | 99.20% | 99.73% | 99.47% | 99.47% ± 0.27 |
| 达到长度上限 | 0.53% | 0.80% | 0.53% | 1.07% | 0.80% ± 0.27 |

SFT→GRPO seed42/43/44 的净正确数为 +2/+2/0，McNemar `p` 为
0.790527/0.726562/1。

## GSM8K Test

| 1,319 题 test | SFT v7 | OPD 均值 ± SD | GRPO 42 | GRPO 43 | GRPO 44 | GRPO 均值 ± SD |
|---|---:|---:|---:|---:|---:|---:|
| 数值准确率 | 71.65% | **72.91% ± 0.54** | 72.25% | 72.48% | 71.80% | 72.18% ± 0.35 |
| 严格准确率 | 71.65% | **72.83% ± 0.54** | 72.25% | 72.40% | 71.65% | 72.10% ± 0.40 |
| 格式遵循率 | **98.26%** | 97.68% ± 0.16 | 98.18% | 98.18% | 98.03% | 98.13% ± 0.09 |
| 达到长度上限 | **1.36%** | 1.95% ± 0.16 | 1.36% | 1.44% | 1.44% | 1.42% ± 0.04 |

三次 GRPO 均高于 SFT，平均提高 0.53 pp；配对 `p` 为
0.291215/0.168978/0.885433，均不显著。OPD 平均准确率比 GRPO 高 0.73 pp，但 GRPO
的格式、截断和生成长度更接近 SFT。

## SVAMP

| 1,000 题 test | Raw Base | SFT v7 | OPD 均值 ± SD | GRPO 42 | GRPO 43 | GRPO 44 | GRPO 均值 ± SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| 数值准确率 | **85.20%** | 81.50% | 81.93% ± 0.32 | 81.60% | 81.10% | 80.90% | 81.20% ± 0.36 |
| 格式遵循率 | 0.00% | **98.80%** | 98.47% ± 0.21 | 98.90% | 98.60% | 98.40% | 98.63% ± 0.25 |
| 达到长度上限 | 7.00% | **0.90%** | 1.23% ± 0.21 | 0.90% | 1.20% | 1.20% | 1.10% ± 0.17 |

GRPO 相对 SFT 的变化为 +0.10/−0.40/−0.60 pp，配对 `p` 为
1/0.557197/0.361595。GSM8K 的正向结果没有跨数据集复现。

## 目录

```text
dev/         # 三 seed 正式 validation 与 SFT 配对分析
final/       # 三 seed 正式 GSM8K test 与 SFT 配对分析
pilot/dev/   # 早期 512-token slice/validation，仅作工程记录
pilot/test/  # 早期 seed42 512-token test，仅作工程记录
```

SVAMP 原始结果与配对文件位于 [`results/svamp/final/`](../svamp/final/)。

## 实验纪律

- checkpoint 固定为 step 30，不能根据官方 test 或 SVAMP 再调参；
- `--seed` 同时控制训练样本抽取和训练随机性，SD 表示端到端运行波动；
- 配对比较必须具有相同 evaluator、dataset、prompt 和 generation metadata；
- `compare_base_sft.py` 默认拒绝真实协议冲突，同时兼容缺少可选元数据的旧结果；
- `outputs/` 中的 LoRA checkpoint 不提交 Git。
