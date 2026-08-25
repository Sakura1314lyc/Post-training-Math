# Qwen2.5-Math-1.5B：GSM8K 后训练实验

[简体中文](README.md) | [English](README_en.md)

这个仓库记录了我在一张 8 GiB RTX 5060 Laptop GPU 上完成的数学模型后训练实验。主线从
`Qwen/Qwen2.5-Math-1.5B` 出发，依次做了 LoRA-SFT、On-Policy Distillation
（OPD/GKD）和 GRPO，并补做了一轮多随机种子、冻结 audit 的确认性实验。

想先看结论，可以直接读[最终实验总结](reports/FINAL_EXPERIMENT_SUMMARY_zh.md)。训练细节和
每次实验的原始数字都保留在 [`reports/`](reports/README.md) 和 [`results/`](results/README.md)。

## 结论先说

最后保留的是 **Confirmatory v2 的 SFT seed42**，不是 OPD 或 GRPO：

- `dev_audit`：303/374，数值准确率和严格准确率均为 **81.02%**；
- 格式合规率：**99.47%**；
- 达到 1,024-token 上限：**0.53%**；
- checkpoint：`outputs/confirmatory_v2/sft_seed42/checkpoint-841`。

扩大训练预算后，OPD 的宽松数值准确率接近 SFT，但回复格式和终止行为明显崩坏；GRPO
保住了大部分格式能力，却在三个 seed 上都显著低于 SFT。后续尝试降低采样温度、降低
learning rate 和移除 tied `lm_head`，也没有找回准确率。因此当前证据不支持用 OPD 或
GRPO 替换 SFT。

## Confirmatory v2

早期实验有几个明显问题：SFT 只有一个训练 seed，validation 被反复查看，OPD/GRPO
预算较小，随机种子同时改变训练样本和优化过程，GRPO 的 KL reference 也不是初始 SFT。
Confirmatory v2 专门修正了这些问题：

- 将 7,473 条数据冻结为 6,725 train / 374 dev-select / 374 dev-audit；
- SFT 独立训练 seed 42、43、44，不按 dev-select 挑最好 seed；
- OPD 和 GRPO 每次训练 200 steps / 800 个 on-policy 样本；
- 固定 `data_seed=42`，只改变 training/generation seed；
- GRPO 先合并 SFT，再添加新的 LoRA，使 `beta=0.04` 对应真正的 SFT reference；
- `dev_audit` 只打开一次，只评测预先指定的 SFT seed42。

### Dev-select

| 方法 | 数值准确率 | 严格准确率 | 格式合规率 | 截断率 | 平均生成 tokens |
|---|---:|---:|---:|---:|---:|
| SFT 三 seed | **83.78% ± 1.47** | 83.69% ± 1.60 | **99.47% ± 0.27** | **0.71% ± 0.15** | **95.48 ± 1.84** |
| OPD 三 seed | 82.80% ± 1.32 | 1.25% ± 0.15 | 2.58% ± 0.15 | 30.93% ± 2.48 | 427.97 ± 22.38 |
| GRPO 三 seed | 78.61% ± 0.53 | 77.90% ± 0.67 | 95.72% ± 0.53 | 4.63% ± 0.67 | 138.49 ± 5.97 |

GRPO 三次相对规范 SFT seed42 分别下降 4.55、5.61、5.08 个百分点，McNemar 检验的
`p` 分别为 0.00948、0.00191、0.00432。OPD 和 GRPO 都在进入 audit 前被拒绝；这不是
漏评，而是预先约定的筛选规则。

机器可读的完整汇总和哈希在
[`results/confirmatory_v2/confirmatory_v2_progress.json`](results/confirmatory_v2/confirmatory_v2_progress.json)。

## 早期 test 与跨数据集结果

GSM8K 官方 test 和 SVAMP 在早期实验中已经使用过，下面的数字只作为已完成实验的最终
观察，不再参与 Confirmatory v2 或后续消融的调参。

| 数值准确率 | Raw Base | SFT v7 | OPD 三 seed | GRPO 三 seed |
|---|---:|---:|---:|---:|
| GSM8K test（1,319 题） | 71.80% | 71.65% | **72.91% ± 0.54** | 72.18% ± 0.35 |
| SVAMP（1,000 题） | **85.20%** | 81.50% | 81.93% ± 0.32 | 81.20% ± 0.36 |

SFT 在 GSM8K test 上与 Base 基本持平（McNemar `p=0.958659`），但把格式合规率从 0%
提高到 98.26%，平均生成长度从 371.74 tokens 降到 102.30。代价是 SVAMP 数值准确率
下降 3.70 个百分点（`p=0.00761528`）。早期 OPD/GRPO 在 GSM8K 上的小幅正向点估计
没有形成稳定的显著性证据，也没有恢复到 SVAMP 的 Base 水平。

## 数据处理

SFT 使用 `data/gsm8k_sft_clean.json`。准备脚本删除答案里的
`<<expression=result>>` calculator annotation，但保留自然语言推理和 `####` 最终答案。
7,473 条样本一共删除了 23,716 处标注。

```bash
python3 scripts/prepare_clean_sft_data.py \
  --input data/gsm8k_sft_formal.json \
  --output data/gsm8k_sft_clean.json \
  --overwrite
```

Confirmatory v2 的互斥切分由下面的命令生成：

```bash
python3 scripts/prepare_confirmatory_splits.py
```

切分索引和数据哈希保存在 `data/confirmatory_v2/split_manifest.json`；生成出的分区副本不
提交 Git。

## 复现入口

项目默认在 `sft` Conda 环境中运行：

```bash
cd post-training-math
conda activate sft
```

训练三个 SFT seed：

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

OPD、GRPO、SFT 合并和 audit 的冻结命令见
[`configs/confirmatory/README.md`](configs/confirmatory/README.md)。已有正式结果默认不应
覆盖；如果只是检查环境，先使用独立输出目录运行 smoke。

运行测试：

```bash
python3 -m unittest discover -s tests -v
```

## 项目结构

```text
configs/main/          # 早期 SFT 主线与 v7
configs/confirmatory/  # Confirmatory v2 三 seed SFT 和冻结命令
configs/experimental/  # GRPO v3-v5 单因素诊断协议
data/                  # SFT 数据、数据注册和切分 manifest
scripts/               # 数据准备、训练、评测、配对统计
results/               # 小型 JSON 结果与机器汇总
reports/               # SFT、OPD、GRPO、SVAMP 和最终总结
tests/                 # 离线单元测试
outputs/               # 本地 checkpoint；不提交 Git
```

## 评测口径

- **数值准确率**：最终数值正确，不要求固定格式。
- **严格准确率**：答案正确，并以 `#### <answer>` 结束。
- **格式合规率**：存在可解析的 `#### <answer>` 结尾。
- **截断率**：生成达到 token 上限，没有自然结束。
- **McNemar 检验**：同一批题上的配对比较。

这些指标必须一起看。只看宽松数值准确率，会掩盖 OPD 那种“数字能抽出来，但回复没有正常
结束”的情况。

## 文档

- [最终实验总结](reports/FINAL_EXPERIMENT_SUMMARY_zh.md)
- [Confirmatory v2 协议与结果](reports/CONFIRMATORY_V2_PROTOCOL_zh.md)
- [SFT 实验报告](reports/SFT_EXPERIMENT_REPORT_zh.md)
- [OPD/GKD 实验报告](reports/OPD_EXPERIMENT_REPORT_zh.md)
- [GRPO 实验报告](reports/GRPO_EXPERIMENT_REPORT_zh.md)
- [SVAMP 泛化报告](reports/SVAMP_EXPERIMENT_REPORT_zh.md)

历史配置、失败消融和调试结果没有删除；它们用于复现实验过程，但不替代上面的最终结论。
