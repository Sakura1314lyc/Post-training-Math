# SVAMP 独立泛化评测报告（阶段性）

## 1. 目的与状态

GSM8K 官方 test 已经用于最终 SFT/OPD 报告，不适合继续承担模型选择职责。本实验引入
未参与训练和 checkpoint 选择的 SVAMP，检验 Raw Base、SFT v7 与 OPD 的跨数据集泛化。

截至 2026-08-20，固定协议中的 Raw Base、SFT v7、OPD seed 42 和 seed 43 已完成；
预定的 OPD seed 44 尚未运行。因此本文只给出阶段性结论，不汇报 OPD 三次均值与标准差。

## 2. 数据与协议

- 数据集：`MU-NLPC/Calc-svamp`，配置 `default`，split `test`，1,000 题。
- 数据说明：原始 SVAMP 没有官方 train/test 划分，因此遵循 Calc-SVAMP 数据卡，将完整
  集合视为独立 test；Calc-SVAMP 修正了原数据中一条方程与答案不一致的样本。
- 评分器：`svamp_numeric_v1`。
- Prompt：与 GSM8K 正式评测相同，要求最终输出 `#### <answer>`。
- 解码：greedy，`max_new_tokens=1024`，禁用答案行强制截断，观察模型原生 EOS。
- checkpoint：SFT v7 checkpoint-888；OPD 三次运行固定使用 checkpoint-30。
- 使用约束：SVAMP 不用于选择 checkpoint、prompt、生成参数或 OPD 超参数。

数据来源：

- [Calc-SVAMP 数据卡](https://huggingface.co/datasets/MU-NLPC/Calc-svamp/blob/main/README.md)
- [原始 SVAMP 数据](https://github.com/arkilpatel/SVAMP/blob/main/SVAMP.json)

## 3. 阶段性结果

| SVAMP test（1,000 题） | Raw Base | SFT v7 | OPD seed 42 | OPD seed 43 |
|---|---:|---:|---:|---:|
| 数值准确率 | **85.20%** | 81.50% | 82.30% | 81.70% |
| 严格 `####` 准确率 | 0.00% | 81.40% | 82.10% | 81.60% |
| 格式遵循率 | 0.00% | **98.80%** | 98.70% | 98.40% |
| 达到 1,024 token 上限 | 7.00% | **0.90%** | 1.00% | 1.30% |
| 原生 EOS | 93.00% | **99.10%** | 99.00% | 98.70% |
| 平均生成 token | 290.81 | **55.07** | 59.04 | 61.67 |
| 全量评测时间 | 109m 21s | **27m 32s** | 29m 23s | 34m 38s |

## 4. 同题配对分析

| 对比 | 前者独自答对 | 后者独自答对 | 准确率差 | 精确双侧 McNemar p |
|---|---:|---:|---:|---:|
| Raw Base → SFT v7 | 110 | 73 | −3.70 pp | **0.00761528** |
| SFT v7 → OPD seed 42 | 21 | 29 | +0.80 pp | 0.322236 |
| SFT v7 → OPD seed 43 | 26 | 28 | +0.20 pp | 0.891923 |

Base→SFT 的差异在当前配对检验下低于 0.05，说明下降并非只来自总分的小幅随机波动。
两个 OPD 运行相对 SFT 都是正向点估计，但交换题目很少，且均无显著性证据。

## 5. 解释

SFT v7 在 GSM8K test 上基本保持 Raw Base 的数值准确率，但在独立 SVAMP 上下降
3.70 pp。这暴露出 GSM8K 格式微调的跨数据集代价：模型输出更短、更稳定、几乎总能按
`####` 格式结束，评测速度也约提升到 Base 的四倍，但数值泛化没有完全保持。

OPD seed 42/43 分别恢复 0.80/0.20 pp，方向暂时一致，却不足以抵消 SFT 相对 Base 的
2.90/3.50 pp 差距。现阶段不能声称 OPD 带来稳定的跨数据集能力提升。

## 6. 局限与下一步

- OPD seed 44 尚未完成，不能提前计算或选择性报告三次运行均值。
- SVAMP 原始发布没有官方划分；本文采用 Calc-SVAMP 的完整 test 约定，结果必须连同这一
  数据选择一起解释。
- Base 的 7.00% 输出达到 token 上限；宽松数值评分可能仍能从截断回复中提取答案，因此
  必须同时报告截断率和严格准确率。
- 下一次只需按既定命令完成 OPD seed 44，随后汇总三次均值、样本标准差和配对检验；不再
  根据当前结果修改模型或协议。

## 7. 结果文件

- `results/svamp/final/svamp_base_15b_v1.json`
- `results/svamp/final/svamp_sft_v7_15b_ckpt888_v1.json`
- `results/svamp/final/svamp_opd_seed42_ckpt30_v1.json`
- `results/svamp/final/svamp_opd_seed43_ckpt30_v1.json`
- `results/svamp/final/svamp_base_sft_v7_transition_analysis_v1.json`
- `results/svamp/final/svamp_sft_v7_opd_seed42_transition_analysis_v1.json`
- `results/svamp/final/svamp_sft_v7_opd_seed43_transition_analysis_v1.json`
