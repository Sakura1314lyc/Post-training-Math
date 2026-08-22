# Qwen2.5-Math-1.5B 在 GSM8K 上的 GRPO 实验报告

## 1. 实验摘要

本实验从最终 SFT v7 adapter 出发，使用 TRL `GRPOTrainer` 完成 seed 42/43/44 三次
30-step GRPO。每次从排除固定 validation 后的 7,099 条训练候选中抽取 256 条，生成
120 个 on-policy completion；数值正确性为主奖励，严格 `####` 格式为辅助奖励。

截至 2026-08-22，三次训练、固定 374 题 validation、GSM8K 官方 test 和独立 SVAMP
评测均已完成。主要结论是：

- GSM8K test 三次均高于 SFT v7，平均 `72.18% ± 0.35 pp`，比 SFT 高 0.53 pp；
- 三次 SFT→GRPO 配对检验均不显著，不能声称稳定能力提升；
- SVAMP 三次平均为 `81.20% ± 0.36 pp`，比 SFT 低 0.30 pp，GSM8K 收益未跨数据集复现；
- OPD 在 GSM8K 和 SVAMP 的平均数值准确率都高于 GRPO，但 GRPO 的格式、截断与生成
  长度更接近 SFT；
- GRPO 没有修复 SFT 相对 Raw Base 的 SVAMP 显著退化。

因此严谨表述是：**当前 30-step GRPO 在同分布 GSM8K 上产生方向一致但不显著的小幅
正收益，同时基本维持 SFT 的输出行为；该收益没有在 SVAMP 上复现。**

## 2. 实现与环境兼容性

当前 LLaMA-Factory checkout 没有 GRPO 入口，因此新增
[`scripts/train_grpo.py`](../scripts/train_grpo.py)，直接使用 TRL 0.24：

- 从 SFT v7 checkpoint-888 加载可训练 LoRA policy；
- 精确复现并排除固定 374 题 validation；
- 主奖励为最终数值正确性（权重 1.0）；
- 辅助奖励为严格 `#### <answer>` 格式（权重 0.1）；
- Transformers 原生生成，不使用 vLLM；
- 4 completions / prompt，DAPO loss，group reward scaling；
- 截断 completion 默认从 loss 中屏蔽。

环境为 Transformers 5.8.0、TRL 0.24.0、PEFT 0.18.1。脚本兼容了两处上游差异：
Transformers 5.x 的 optional-package 私有探测返回 tuple，而 TRL 0.24 假设布尔值；TRL
还访问了 Transformers 5.x 已移除的 `warnings_issued`。两项兼容逻辑均有离线测试。

## 3. 为什么固定 `beta=0`

本实验直接继续训练现有 SFT PEFT adapter。TRL 对 PEFT 模型构造 KL reference 时通过禁用
adapter 得到参考策略；这里得到的是 Raw Base，而不是训练开始时的 SFT policy。若使用
非零 `beta`，KL 会约束到错误参考模型。因此脚本显式拒绝 `beta != 0`。

若后续研究标准 KL-GRPO，应冻结一份独立 SFT reference，而不能静默复用禁用 adapter 后
的 Raw Base。本实验应被理解为无显式 KL 的小规模 GRPO。

## 4. Smoke 与训练设置

1-step smoke 在 8 GiB RTX 5060 Laptop GPU 上成功：runtime 7.49 秒，数值奖励均值 0.50，
格式奖励均值 0.75，LoRA 参数最大变化 `5.00e-06`，峰值分配显存 3.33 GiB。

固定训练设置：

| 设置 | 值 |
|---|---:|
| 每次抽取候选 | 256 |
| optimizer steps | 30 |
| generations / prompt | 4 |
| rollout 数 | 120 |
| learning rate | `5e-6` |
| prompt / completion 上限 | 512 / 128 |
| accuracy / format reward | 1.0 / 0.1 |
| beta | 0 |
| 报告 checkpoint | step 30 |

| seed | train loss | runtime | LoRA 参数最大变化 | 峰值显存 |
|---:|---:|---:|---:|---:|
| 42 | −0.0404 | 136.7 s | `1.099e-4` | 3.33 GiB |
| 43 | −0.1080 | 140.8 s | `9.450e-5` | 3.33 GiB |
| 44 | −0.1344 | 140.1 s | `9.720e-5` | 3.33 GiB |

当前 `--seed` 同时控制 256 条训练样本抽取与训练随机性，所以三次 SD 表示端到端运行
波动，而不是固定训练数据下的纯优化 seed。

## 5. 评测协议与一次实验纠错

正式比较统一使用：贪心解码、BF16、`max_new_tokens=1024`、Qwen EOS 列表、禁用答案行
提前停止。GSM8K 评分器为 `gsm8k_numeric_v3`，SVAMP 为 `svamp_numeric_v1`。

早期 pilot 曾误用 512-token 上限和答案行提前停止。发现后没有继续混用，而是：

1. 将旧结果隔离到 `results/grpo/pilot/`；
2. 删除跨协议 transition 文件；
3. 重新评测三个 seed 的 validation 与 test；
4. 加强 `compare_base_sft.py`，要求 evaluator、dataset、prompt、generation metadata 一致。

这一步说明“同一评分器版本”仍不足以保证公平比较，生成协议也必须一致。

## 6. 正式 Validation

| 固定 validation（374 题） | SFT v7 | GRPO 42 | GRPO 43 | GRPO 44 | GRPO 均值 ± SD |
|---|---:|---:|---:|---:|---:|
| 数值准确率 | 85.29% | 85.83% | 85.83% | 85.29% | **85.65% ± 0.31** |
| 严格准确率 | 85.29% | 85.83% | 85.83% | 85.29% | **85.65% ± 0.31** |
| 格式遵循率 | 99.47% | 99.20% | 99.73% | 99.47% | 99.47% ± 0.27 |
| 达到长度上限 | 0.53% | 0.80% | 0.53% | 1.07% | 0.80% ± 0.27 |

| SFT→GRPO | SFT-only | GRPO-only | 净变化 | McNemar p |
|---|---:|---:|---:|---:|
| seed42 | 6 | 8 | +2 | 0.790527 |
| seed43 | 3 | 5 | +2 | 0.726562 |
| seed44 | 8 | 8 | 0 | 1 |

validation 没有出现准确率下降，但平均只增加 1.33 题，证据很弱。

## 7. GSM8K 官方 Test

| GSM8K test（1,319 题） | Raw Base | SFT v7 | OPD 均值 ± SD | GRPO 42 | GRPO 43 | GRPO 44 | GRPO 均值 ± SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| 数值准确率 | 71.80% | 71.65% | **72.91% ± 0.54** | 72.25% | 72.48% | 71.80% | 72.18% ± 0.35 |
| 严格准确率 | 0.00% | 71.65% | **72.83% ± 0.54** | 72.25% | 72.40% | 71.65% | 72.10% ± 0.40 |
| 格式遵循率 | 0.00% | **98.26%** | 97.68% ± 0.16 | 98.18% | 98.18% | 98.03% | 98.13% ± 0.09 |
| 达到长度上限 | 3.11% | **1.36%** | 1.95% ± 0.16 | 1.36% | 1.44% | 1.44% | 1.42% ± 0.04 |
| 平均生成 token | 371.74 | **102.30** | 111.60 ± 1.34 | 101.86 | 103.14 | 103.97 | 102.99 ± 1.06 |

| SFT→GRPO | SFT-only | GRPO-only | 净变化 | 准确率差 | McNemar p |
|---|---:|---:|---:|---:|---:|
| seed42 | 18 | 26 | +8 | +0.61 pp | 0.291215 |
| seed43 | 21 | 32 | +11 | +0.83 pp | 0.168978 |
| seed44 | 23 | 25 | +2 | +0.15 pp | 0.885433 |

三次 test 点估计都高于 SFT，平均提高 0.53 pp，但没有单次显著。GRPO 平均比 OPD 低
0.73 pp，不过平均格式率高 0.45 pp、长度上限命中率低 0.54 pp、平均少生成 8.61 tokens。

同 seed OPD→GRPO 的数值变化分别为 −1、−12、−16 题，对应 `p=1/0.302878/0.170645`；
虽然都未显著，但方向一致支持 OPD 的点估计更高。

## 8. SVAMP 独立泛化

| SVAMP test（1,000 题） | Raw Base | SFT v7 | OPD 均值 ± SD | GRPO 42 | GRPO 43 | GRPO 44 | GRPO 均值 ± SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| 数值准确率 | **85.20%** | 81.50% | 81.93% ± 0.32 | 81.60% | 81.10% | 80.90% | 81.20% ± 0.36 |
| 严格准确率 | 0.00% | 81.40% | 81.80% ± 0.26 | 81.50% | 80.90% | 80.80% | 81.07% ± 0.38 |
| 格式遵循率 | 0.00% | **98.80%** | 98.47% ± 0.21 | 98.90% | 98.60% | 98.40% | 98.63% ± 0.25 |
| 达到长度上限 | 7.00% | **0.90%** | 1.23% ± 0.21 | 0.90% | 1.20% | 1.20% | 1.10% ± 0.17 |
| 平均生成 token | 290.81 | **55.07** | 61.32 ± 2.12 | 54.92 | 58.54 | 58.28 | 57.25 ± 2.02 |

| SFT→GRPO | SFT-only | GRPO-only | 净变化 | 准确率差 | McNemar p |
|---|---:|---:|---:|---:|---:|
| seed42 | 13 | 14 | +1 | +0.10 pp | 1 |
| seed43 | 15 | 11 | −4 | −0.40 pp | 0.557197 |
| seed44 | 18 | 12 | −6 | −0.60 pp | 0.361595 |

GRPO 的 GSM8K 正向结果没有在 SVAMP 复现，三次平均反而比 SFT 低 0.30 pp。相对 Raw
Base，GRPO seed42/43/44 分别低 3.60/4.10/4.30 pp，配对 `p` 为
0.009684/0.003166/0.001935，三次差距都显著。

同 seed OPD→GRPO 在 SVAMP 分别为 −7/−6/−9 题，`p=0.381693/0.470879/0.289244`。
OPD 三次都高于 GRPO，但同样没有单次显著证据。

## 9. 结论与局限

本轮 GRPO 完成了训练、checkpoint 固定、三 seed validation、官方 test、独立泛化和配对
统计闭环。可以支持的结论：

- 8 GiB GPU 可稳定完成 Qwen2.5-Math-1.5B LoRA-GRPO；
- GRPO 在 GSM8K 上三次方向一致，平均比 SFT 高 0.53 pp；
- 该提升小且单次不显著，并且在 SVAMP 上没有复现；
- 相比 OPD，GRPO 更保守地保持了 SFT 的格式、长度和截断行为，但平均数值收益更低；
- 当前证据不支持“GRPO 显著提升数学推理”或“GRPO 修复跨数据集泛化”。

主要局限：训练仅 30 steps / 120 rollouts；`beta=0`；训练 seed 同时改变样本子集；GSM8K
test 和 SVAMP 已用于最终报告，不能继续用来调参。若继续研究，应先建立新的开发集，或实现
独立冻结的 SFT KL reference，再预注册新实验协议。

## 10. 结果文件

- 训练脚本：[`scripts/train_grpo.py`](../scripts/train_grpo.py)
- 多 seed 汇总：[`results/grpo/grpo_multiseed_summary.json`](../results/grpo/grpo_multiseed_summary.json)
- 正式 validation：[`results/grpo/dev/`](../results/grpo/dev/)
- 正式 GSM8K test：[`results/grpo/final/`](../results/grpo/final/)
- SVAMP test：[`results/svamp/final/`](../results/svamp/final/)
- Pilot 协议结果：[`results/grpo/pilot/`](../results/grpo/pilot/)
