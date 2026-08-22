# SVAMP 独立泛化评测报告

## 1. 目的与状态

GSM8K 官方 test 已经用于最终 SFT/OPD 报告，不适合继续承担模型选择职责。本实验引入
未参与训练和 checkpoint 选择的 SVAMP，检验 Raw Base、SFT v7、OPD 与 GRPO 的跨数据集
泛化。

截至 2026-08-22，固定协议中的 Raw Base、SFT v7、OPD seed 42/43/44 和 GRPO
seed 42/43/44 已全部完成。多 seed 汇总使用算术均值与样本标准差（SD）。

## 2. 数据与协议

- 数据集：`MU-NLPC/Calc-svamp`，配置 `default`，split `test`，1,000 题。
- 数据说明：原始 SVAMP 没有官方 train/test 划分，因此遵循 Calc-SVAMP 数据卡，将完整
  集合视为独立 test；Calc-SVAMP 修正了原数据中一条方程与答案不一致的样本。
- 评分器：`svamp_numeric_v1`。
- Prompt：与 GSM8K 正式评测相同，要求最终输出 `#### <answer>`。
- 解码：greedy，`max_new_tokens=1024`，禁用答案行强制截断，观察模型原生 EOS。
- checkpoint：SFT v7 checkpoint-888；OPD/GRPO 三次运行均固定使用 checkpoint-30。
- 使用约束：SVAMP 不用于选择 checkpoint、prompt、生成参数或后训练超参数。

数据来源：

- [Calc-SVAMP 数据卡](https://huggingface.co/datasets/MU-NLPC/Calc-svamp/blob/main/README.md)
- [原始 SVAMP 数据](https://github.com/arkilpatel/SVAMP/blob/main/SVAMP.json)

## 3. 最终结果

| SVAMP test（1,000 题） | Raw Base | SFT v7 | OPD 均值 ± SD | GRPO 42 | GRPO 43 | GRPO 44 | GRPO 均值 ± SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| 数值准确率 | **85.20%** | 81.50% | 81.93% ± 0.32 | 81.60% | 81.10% | 80.90% | 81.20% ± 0.36 |
| 严格准确率 | 0.00% | 81.40% | 81.80% ± 0.26 | 81.50% | 80.90% | 80.80% | 81.07% ± 0.38 |
| 格式遵循率 | 0.00% | **98.80%** | 98.47% ± 0.21 | 98.90% | 98.60% | 98.40% | 98.63% ± 0.25 |
| 达到长度上限 | 7.00% | **0.90%** | 1.23% ± 0.21 | 0.90% | 1.20% | 1.20% | 1.10% ± 0.17 |
| 平均生成 token | 290.81 | **55.07** | 61.32 ± 2.12 | 54.92 | 58.54 | 58.28 | 57.25 ± 2.02 |

## 4. 同题配对分析

| 对比 | 前者独自答对 | 后者独自答对 | 准确率差 | 精确双侧 McNemar p |
|---|---:|---:|---:|---:|
| Raw Base → SFT v7 | 110 | 73 | −3.70 pp | **0.00761528** |
| SFT v7 → OPD seed 42 | 21 | 29 | +0.80 pp | 0.322236 |
| SFT v7 → OPD seed 43 | 26 | 28 | +0.20 pp | 0.891923 |
| SFT v7 → OPD seed 44 | 28 | 31 | +0.30 pp | 0.794844 |
| SFT v7 → GRPO seed 42 | 13 | 14 | +0.10 pp | 1 |
| SFT v7 → GRPO seed 43 | 15 | 11 | −0.40 pp | 0.557197 |
| SFT v7 → GRPO seed 44 | 18 | 12 | −0.60 pp | 0.361595 |

Base→SFT 的差异在当前配对检验下低于 0.05，说明下降并非只来自总分的小幅随机波动。
三个 OPD 运行相对 SFT 都是正向点估计，但交换题目很少，且每次均无显著性证据。由于
三次运行的训练样本与训练随机性同时变化，这里分别报告三次配对检验，不把它们简单合并成
一个伪重复的总体显著性检验。

## 5. 解释

SFT v7 在 GSM8K test 上基本保持 Raw Base 的数值准确率，但在独立 SVAMP 上下降
3.70 pp。这暴露出 GSM8K 格式微调的跨数据集代价：模型输出更短、更稳定、几乎总能按
`####` 格式结束，评测速度也约提升到 Base 的四倍，但数值泛化没有完全保持。

OPD seed 42/43/44 分别恢复 0.80/0.20/0.30 pp，三次方向一致，平均为
`+0.43 pp`，样本 SD 为 `0.32 pp`。但每次配对检验都不显著，且 OPD 平均准确率
81.93% 仍比 Raw Base 低 3.27 pp。因此可以说 OPD 出现了可复现的小幅正向点估计，不能
声称它带来了统计上可靠或足以恢复 Base 水平的跨数据集能力提升。

GRPO seed 42/43/44 相对 SFT 分别为 +0.10/−0.40/−0.60 pp，平均为
`−0.30 pp`，样本 SD 为 `0.36 pp`。其 GSM8K 上三次一致的正向点估计没有在 SVAMP
复现。相对 Raw Base，三次 GRPO 分别下降 3.60/4.10/4.30 pp，配对 `p` 为
0.009684/0.003166/0.001935，均低于 0.05。

OPD 在 SVAMP 三次都高于同 seed GRPO 7/6/9 题，但 OPD→GRPO 配对 `p` 为
0.381693/0.470879/0.289244，单次差异仍不显著。总体上，OPD 的数值点估计更高，GRPO
的格式、截断与生成长度更接近 SFT。

## 6. 局限与下一步

- SVAMP 原始发布没有官方划分；本文采用 Calc-SVAMP 的完整 test 约定，结果必须连同这一
  数据选择一起解释。
- Base 的 7.00% 输出达到 token 上限；宽松数值评分可能仍能从截断回复中提取答案，因此
  必须同时报告截断率和严格准确率。
- 当前 `--seed` 同时控制 256 条 OPD/GRPO 训练样本的抽取和训练随机性，因此三次 SD
  表示端到端运行波动，而不是固定训练数据下的纯随机种子方差。
- 若继续研究，应建立新的 validation 或使用另一独立数据集设计改进；不能回头根据 SVAMP
  test 结果选择现有 checkpoint 或调整当前报告协议。

## 7. 结果文件

- `results/svamp/final/svamp_base_15b_v1.json`
- `results/svamp/final/svamp_sft_v7_15b_ckpt888_v1.json`
- `results/svamp/final/svamp_opd_seed42_ckpt30_v1.json`
- `results/svamp/final/svamp_opd_seed43_ckpt30_v1.json`
- `results/svamp/final/svamp_opd_seed44_ckpt30_v1.json`
- `results/svamp/final/svamp_base_sft_v7_transition_analysis_v1.json`
- `results/svamp/final/svamp_sft_v7_opd_seed42_transition_analysis_v1.json`
- `results/svamp/final/svamp_sft_v7_opd_seed43_transition_analysis_v1.json`
- `results/svamp/final/svamp_sft_v7_opd_seed44_transition_analysis_v1.json`
- `results/svamp/final/svamp_opd_multiseed_summary_v1.json`
- `results/svamp/final/svamp_grpo_seed42_ckpt30_v1.json`
- `results/svamp/final/svamp_grpo_seed43_ckpt30_v1.json`
- `results/svamp/final/svamp_grpo_seed44_ckpt30_v1.json`
- `results/svamp/final/svamp_sft_v7_grpo_seed42_ckpt30_transition_analysis.json`
- `results/svamp/final/svamp_sft_v7_grpo_seed43_ckpt30_transition_analysis.json`
- `results/svamp/final/svamp_sft_v7_grpo_seed44_ckpt30_transition_analysis.json`
