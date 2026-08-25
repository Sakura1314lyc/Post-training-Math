# Qwen2.5-Math-1.5B 的 GSM8K 后训练实验

[简体中文](README.md) | [English](README_en.md)

这个仓库记录了我在一张 8 GiB RTX 5060 Laptop GPU 上，对
`Qwen/Qwen2.5-Math-1.5B` 做 SFT、OPD 和 GRPO 的完整过程。除了训练脚本和最终结果，
这里也保留了失败的配置、配对统计和后续诊断，方便回头检查每个结论是怎么来的。

实验没有得到一条持续上涨的曲线。SFT 明显改善了答案格式和生成效率，但没有提高
GSM8K test 的数值准确率，还损失了部分 SVAMP 泛化；当前实现下，OPD 和 GRPO 也没有
形成足够稳定的证据来替换 SFT。

## 先看结论

- 最终保留 Confirmatory v2 的 SFT seed42，而不是 OPD 或 GRPO adapter。
- SFT 最可靠的收益是输出控制：回复更短，`#### <answer>` 格式基本稳定。
- OPD 的宽松数值分数接近 SFT，但回复经常不结束，不能只看抽取出来的数字。
- GRPO 的输出行为比 OPD 正常，不过三个训练 seed 都显著低于对应的 SFT 起点。

最终 checkpoint：

| checkpoint | dev-audit 数值准确率 | 严格准确率 | 格式合规率 | 截断率 |
|---|---:|---:|---:|---:|
| `outputs/confirmatory_v2/sft_seed42/checkpoint-841` | 303/374（81.02%） | 81.02% | 99.47% | 0.53% |

seed42 不是 dev-select 上分数最高的 seed。它是实验前指定的规范 seed，最终只对它打开一次
dev-audit，避免把随机种子搜索包装成模型改进。

## 主要结果

### Confirmatory v2：方法比较

| 方法 | 数值准确率 | 严格准确率 | 格式合规率 | 截断率 | 平均生成 tokens |
|---|---:|---:|---:|---:|---:|
| SFT 三 seed | 83.78% ± 1.47 | 83.69% ± 1.60 | 99.47% ± 0.27 | 0.71% ± 0.15 | 95.48 ± 1.84 |
| OPD 三 seed | 82.80% ± 1.32 | 1.25% ± 0.15 | 2.58% ± 0.15 | 30.93% ± 2.48 | 427.97 ± 22.38 |
| GRPO 三 seed | 78.61% ± 0.53 | 77.90% ± 0.67 | 95.72% ± 0.53 | 4.63% ± 0.67 | 138.49 ± 5.97 |

OPD 的数值准确率看起来只比 SFT 低 0.98 个百分点，但严格准确率只剩 1.25%，近三分之一
的回复撞到 1,024-token 上限。宽松评分仍能从长回复中抽出正确数字，模型却没有按要求
完成回答，因此 OPD 在 audit 前被拒绝。

GRPO 没有同样严重的格式崩坏，但三个 seed 相对规范 SFT 分别下降 4.55、5.61 和 5.08 个
百分点；配对 McNemar `p` 分别为 0.00948、0.00191 和 0.00432。这个结果同样没有进入
audit。

### 早期外部评测

下面的 GSM8K test 和 SVAMP 结果产生于 Confirmatory v2 之前。两套测试集都已用于分析，
所以这些数字只作为外部观察保留，不再用于选择当前方案。

| 数值准确率 | Raw Base | SFT v7 | OPD 三 seed | GRPO 三 seed |
|---|---:|---:|---:|---:|
| GSM8K test（1,319 题） | 71.80% | 71.65% | 72.91% ± 0.54 | 72.18% ± 0.35 |
| SVAMP（1,000 题） | 85.20% | 81.50% | 81.93% ± 0.32 | 81.20% ± 0.36 |

SFT 在 GSM8K test 上与 Base 基本持平（McNemar `p=0.958659`），但格式合规率从 0% 提高
到 98.26%，平均生成长度从 371.74 tokens 降到 102.30。另一方面，它在 SVAMP 上下降
3.70 个百分点（`p=0.00761528`）。因此，这里的 SFT 更像是输出行为校准，而不是已经得到
跨数据集验证的数学能力提升。

## 这个仓库做了什么

实验主线是：

```text
Raw Math Base
  -> 清洗 GSM8K SFT 数据
  -> 多 seed SFT
  -> OPD/GKD 与 GRPO
  -> 数值、格式、长度和配对显著性评测
```

具体包括：

- 清除 GSM8K 答案中的 calculator annotation，同时保留自然语言推理和最终答案；
- 在 8 GiB 显存内完成 1.5B 模型的 LoRA SFT、量化教师 OPD/GKD 和 GRPO；
- 为训练 seed、数据 seed 和生成 seed 分别留档；
- 同时统计数值准确率、严格格式准确率、格式合规率、截断率和生成长度；
- 用逐题 transition 与 McNemar 检验检查点估计背后的得失分变化。

## Confirmatory v2 的实验约束

早期实验主要用来找方向：SFT 只有一个训练 seed，validation 被多次查看，OPD/GRPO 的训练
预算也偏小，而且一个 `seed` 同时改变了训练子集和优化随机性。旧版 GRPO 关闭 adapter 后
得到的还是 Raw Base，也不是正确的 SFT KL reference。

为此，Confirmatory v2 固定了以下规则：

- 7,473 条数据切成互斥的 6,725 train、374 dev-select 和 374 dev-audit；
- SFT 独立训练 seed 42、43、44；
- `data_seed=42` 固定，只改变 training/generation seed；
- OPD 训练 200 steps / 800 个 on-policy microbatches；
- GRPO 训练 200 steps / 800 个 rollouts，并以合并后的 SFT 作为 KL reference，`beta=0.04`；
- dev-audit 只评测预先指定的 SFT seed42。

完整协议、运行路径和 SHA-256 见
[`CONFIRMATORY_V2_PROTOCOL_zh.md`](reports/CONFIRMATORY_V2_PROTOCOL_zh.md) 与
[`confirmatory_v2_progress.json`](results/confirmatory_v2/confirmatory_v2_progress.json)。

## GRPO 补充诊断

确认性实验结束后，又做了三个单因素检查：将 rollout temperature 从 0.9 降到 0.3，
将 learning rate 从 `5e-6` 降到 `1e-6`，以及从 LoRA targets 中移除 tied `lm_head`。
这些实验只查看已使用过的 dev-select，没有再碰 audit、GSM8K test 或 SVAMP。

前两个改动没有挽回准确率。移除 `lm_head` 将 adapter 从约 454 MiB 缩到 4.17 MiB，也避免
保存完整的 `lm_head.base_layer.weight`；数值准确率仍为 78.61%，与旧 GRPO 没有差异。
这解决了存储问题，但没有改变模型选择。

## 数据准备

SFT 数据位于 `data/gsm8k_sft_clean.json`。7,473 条样本共移除了 23,716 处
`<<expression=result>>` 标注。

```bash
python3 scripts/prepare_clean_sft_data.py \
  --input data/gsm8k_sft_formal.json \
  --output data/gsm8k_sft_clean.json \
  --overwrite

python3 scripts/prepare_confirmatory_splits.py
```

切分索引和数据哈希写在 `data/confirmatory_v2/split_manifest.json`。生成的分区副本只保留在
本地，不提交 Git。

## 复现入口

项目默认使用 `sft` Conda 环境：

```bash
cd post-training-math
conda activate sft
```

三个 SFT seed 分别有独立配置：

```bash
llamafactory-cli train configs/confirmatory/qwen25_math_15b_base_lora_sft_v8_seed42.yaml
llamafactory-cli train configs/confirmatory/qwen25_math_15b_base_lora_sft_v8_seed43.yaml
llamafactory-cli train configs/confirmatory/qwen25_math_15b_base_lora_sft_v8_seed44.yaml
```

OPD、GRPO、SFT 合并、评测和 audit 的冻结命令统一放在
[`configs/confirmatory/README.md`](configs/confirmatory/README.md)。正式结果默认不覆盖；检查
环境时请更换输出目录运行 smoke test。

离线测试：

```bash
python3 -m unittest discover -s tests -v
```

## 评分口径

- 数值准确率：最终数值正确，不要求固定格式。
- 严格准确率：数值正确，并以 `#### <answer>` 结束。
- 格式合规率：存在可解析的 `#### <answer>` 结尾。
- 截断率：回复达到 token 上限，没有自然结束。
- McNemar 检验：比较两个模型在同一批题上的正确性变化。

这些指标需要一起看。只报宽松数值准确率，会漏掉“数字能抽出来，但回复没有正常结束”
这类失败。

## 目录与文档

```text
configs/main/          # 早期 SFT 主线与 v7
configs/confirmatory/  # Confirmatory v2 配置和冻结命令
configs/experimental/  # GRPO 单因素诊断
data/                  # SFT 数据、数据注册和切分 manifest
scripts/               # 数据准备、训练、评测、配对统计
results/               # JSON 结果与机器汇总
reports/               # 各阶段实验报告
tests/                 # 离线单元测试
outputs/               # 本地 checkpoint，不提交 Git
```

- [最终实验总结](reports/FINAL_EXPERIMENT_SUMMARY_zh.md)
- [Confirmatory v2 协议与结果](reports/CONFIRMATORY_V2_PROTOCOL_zh.md)
- [SFT 实验报告](reports/SFT_EXPERIMENT_REPORT_zh.md)
- [OPD/GKD 实验报告](reports/OPD_EXPERIMENT_REPORT_zh.md)
- [GRPO 实验报告](reports/GRPO_EXPERIMENT_REPORT_zh.md)
- [SVAMP 泛化报告](reports/SVAMP_EXPERIMENT_REPORT_zh.md)

早期配置、失败消融和调试结果仍保留在仓库中，用于复现实验过程；它们不替代最终结论。
