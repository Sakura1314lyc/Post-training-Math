# Qwen2.5-Math-1.5B 的 GSM8K 后训练实验

[简体中文](README.md) | [English](README_en.md)

我想弄清楚一件事：只用一张 8 GiB 的 RTX 5060 Laptop GPU，SFT、OPD 和 GRPO
分别能把一个 1.5B 数学模型训练到什么程度。这个仓库保存了整套实验，包括走过的弯路，
不只留下最好看的那次结果。

结果并不是一条一路上涨的曲线。SFT 把输出格式和生成长度收拾得很好，却没有提高 GSM8K
test 的数值准确率，还损失了部分 SVAMP 泛化；OPD 和 GRPO 的早期小规模实验有过正向
点估计，但在更严格的确认性实验里都没能胜过 SFT。

完整结论在[最终实验总结](reports/FINAL_EXPERIMENT_SUMMARY_zh.md)。训练记录和原始结果分别
放在 [`reports/`](reports/README.md) 与 [`results/`](results/README.md)。

## 最后选了哪个模型

最终保留 Confirmatory v2 的 SFT seed42：

| checkpoint | dev-audit | 严格准确率 | 格式合规率 | 截断率 |
|---|---:|---:|---:|---:|
| `outputs/confirmatory_v2/sft_seed42/checkpoint-841` | 303/374（81.02%） | 81.02% | 99.47% | 0.53% |

seed42 不是 dev-select 上分数最高的 seed。这里故意不挑 seed44，而是沿用实验前定下的
规范 seed，避免把随机种子搜索包装成模型改进。

当前 OPD 和 GRPO adapter 都不替换这个 SFT checkpoint。OPD 的主要问题是输出失控；
GRPO 的输出正常得多，但三个训练 seed 的准确率都显著低于 SFT。

## 为什么又做了一轮 Confirmatory v2

早期实验适合找方向，不适合下最终结论。当时 SFT 只有一个 seed，validation 看过多次，
OPD/GRPO 的训练预算也小；同一个 `seed` 还同时决定训练样本和优化随机性。GRPO 更麻烦，
旧实现关闭 adapter 后得到的是 Raw Base，并不是训练起点的 SFT policy，因此不能把它当作
正确的 KL reference。

Confirmatory v2 做了这些修正：

- 7,473 条数据固定切成 6,725 train、374 dev-select 和 374 dev-audit；
- SFT 独立训练 seed 42、43、44；
- OPD 每次训练 200 steps / 800 个 on-policy microbatches，GRPO 每次训练
  200 steps / 800 个 rollouts；
- 固定 `data_seed=42`，只改变 training/generation seed；
- GRPO 先合并 SFT，再添加新的 LoRA，`beta=0.04` 对应真正的 SFT reference；
- dev-audit 只打开一次，而且只评测预先指定的 SFT seed42。

### Dev-select 结果

| 方法 | 数值准确率 | 严格准确率 | 格式合规率 | 截断率 | 平均生成 tokens |
|---|---:|---:|---:|---:|---:|
| SFT 三 seed | 83.78% ± 1.47 | 83.69% ± 1.60 | 99.47% ± 0.27 | 0.71% ± 0.15 | 95.48 ± 1.84 |
| OPD 三 seed | 82.80% ± 1.32 | 1.25% ± 0.15 | 2.58% ± 0.15 | 30.93% ± 2.48 | 427.97 ± 22.38 |
| GRPO 三 seed | 78.61% ± 0.53 | 77.90% ± 0.67 | 95.72% ± 0.53 | 4.63% ± 0.67 | 138.49 ± 5.97 |

只看第一列，OPD 的 82.80% 似乎离 SFT 不远。但它的严格准确率只剩 1.25%，近三分之一
的回复撞到 1,024-token 上限。宽松评分还能从这些长回复里抽出数字，模型本身却没有按
要求结束。这个版本不适合继续往 audit 送。

GRPO 没有出现同样严重的格式崩坏，不过三个 seed 相对规范 SFT 分别下降 4.55、5.61、
5.08 个百分点。对应的 McNemar `p` 为 0.00948、0.00191、0.00432，三次都是负向结果。

OPD 和 GRPO 因此都在 audit 前被拒绝。机器可读的汇总、路径和 SHA-256 在
[`confirmatory_v2_progress.json`](results/confirmatory_v2/confirmatory_v2_progress.json)。

## GRPO 后续诊断

确认性协议结束后，我又做了三个单因素实验：把 rollout temperature 从 0.9 降到 0.3，
把 learning rate 从 `5e-6` 降到 `1e-6`，以及从 LoRA targets 中移除 tied `lm_head`。
三次都只看已经使用过的 dev-select，没有碰 audit、GSM8K test 或 SVAMP。

前两个改动没有挽回准确率。移除 `lm_head` 倒是解决了一个实在的工程问题：adapter 从约
454 MiB 缩到 4.17 MiB，也不再保存完整的 `lm_head.base_layer.weight`；但数值准确率仍是
78.61%，和旧 GRPO 没有区别。到这里，这条 GRPO 调参线就停了。

## 早期 test 和 SVAMP 怎么看

下面是 Confirmatory v2 之前的实验。GSM8K 官方 test 与 SVAMP 都已经被使用过，所以这些
数字保留作最终观察，不能再拿来选当前方案。

| 数值准确率 | Raw Base | SFT v7 | OPD 三 seed | GRPO 三 seed |
|---|---:|---:|---:|---:|
| GSM8K test（1,319 题） | 71.80% | 71.65% | 72.91% ± 0.54 | 72.18% ± 0.35 |
| SVAMP（1,000 题） | 85.20% | 81.50% | 81.93% ± 0.32 | 81.20% ± 0.36 |

SFT 在 GSM8K test 上与 Base 基本持平，McNemar `p=0.958659`。它真正改变的是输出行为：
格式合规率从 0% 提高到 98.26%，平均生成长度从 371.74 tokens 降到 102.30。与此同时，
SVAMP 准确率下降了 3.70 个百分点（`p=0.00761528`）。

早期 OPD/GRPO 在 GSM8K 上的小幅收益都没有达到稳定的显著性证据，也没能回到 SVAMP
的 Base 水平。它们说明某些训练运行可能换对一批题，却不足以证明模型整体变强。

## 数据处理

SFT 数据是 `data/gsm8k_sft_clean.json`。清洗脚本删除答案中的
`<<expression=result>>` calculator annotation，保留自然语言推理和 `####` 最终答案。
7,473 条样本共删除 23,716 处标注。

```bash
python3 scripts/prepare_clean_sft_data.py \
  --input data/gsm8k_sft_formal.json \
  --output data/gsm8k_sft_clean.json \
  --overwrite
```

生成 Confirmatory v2 的互斥切分：

```bash
python3 scripts/prepare_confirmatory_splits.py
```

切分索引和数据哈希写在 `data/confirmatory_v2/split_manifest.json`。生成的分区副本只留在
本地，不提交 Git。

## 从哪里开始复现

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

评测规范 SFT 的 dev-select：

```bash
python3 scripts/eval_sft_adapter.py \
  --base-model Qwen/Qwen2.5-Math-1.5B \
  --adapter outputs/confirmatory_v2/sft_seed42/checkpoint-841 \
  --benchmark gsm8k \
  --local-dataset-file data/confirmatory_v2/dev_select.json \
  --local-split-role dev_select \
  --max-new-tokens 1024 \
  --no-stop-after-answer-line \
  --output results/confirmatory_v2/dev/sft_seed42_dev_select_v3.json
```

OPD、GRPO、SFT 合并和 audit 的冻结命令在
[`configs/confirmatory/README.md`](configs/confirmatory/README.md)。正式结果默认不覆盖；
检查环境时请换一个输出目录跑 smoke。

运行测试：

```bash
python3 -m unittest discover -s tests -v
```

## 目录

```text
configs/main/          # 早期 SFT 主线与 v7
configs/confirmatory/  # Confirmatory v2 的 SFT 配置和冻结命令
configs/experimental/  # GRPO v3-v5 单因素实验
data/                  # SFT 数据、数据注册和切分 manifest
scripts/               # 数据准备、训练、评测、配对统计
results/               # JSON 结果与机器汇总
reports/               # 各阶段实验报告
tests/                 # 离线单元测试
outputs/               # 本地 checkpoint，不提交 Git
```

## 评分口径

- 数值准确率：最终数值正确，不要求固定格式。
- 严格准确率：数值正确，并以 `#### <answer>` 结束。
- 格式合规率：存在可解析的 `#### <answer>` 结尾。
- 截断率：回复达到 token 上限，没有自然结束。
- McNemar 检验：比较两个模型在同一批题上的正确性变化。

这些指标需要放在一起看。只报宽松数值准确率，会掩盖“数字能抽出来，但回复没有正常
结束”这一类问题。

## 报告

- [最终实验总结](reports/FINAL_EXPERIMENT_SUMMARY_zh.md)
- [Confirmatory v2 协议与结果](reports/CONFIRMATORY_V2_PROTOCOL_zh.md)
- [SFT 实验报告](reports/SFT_EXPERIMENT_REPORT_zh.md)
- [OPD/GKD 实验报告](reports/OPD_EXPERIMENT_REPORT_zh.md)
- [GRPO 实验报告](reports/GRPO_EXPERIMENT_REPORT_zh.md)
- [SVAMP 泛化报告](reports/SVAMP_EXPERIMENT_REPORT_zh.md)

早期配置、失败消融和调试结果都还在仓库里。它们用于复现实验过程，不替代上面的最终结论。
