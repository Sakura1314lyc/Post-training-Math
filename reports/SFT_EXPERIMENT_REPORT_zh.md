# Qwen2.5-Math-1.5B 在 GSM8K 上的 LoRA-SFT 实验报告

## 1. 实验摘要

本实验以未经指令微调的数学领域基础模型 `Qwen/Qwen2.5-Math-1.5B` 为起点，使用
GSM8K 训练集进行 LoRA-SFT，并在固定 validation split 与官方 test split 上和
Raw Base Model 进行逐题配对比较。

最终版本 v7 在 GSM8K test 的数值准确率为 `945/1319 = 71.65%`，Raw Base 为
`947/1319 = 71.80%`，差值为 `-0.15` 个百分点。逐题配对的精确双侧 McNemar
检验得到 `p=0.958659`，没有证据表明两者的数值准确率存在方向性差异。

SFT 的主要收益体现在输出行为而非最终数值准确率：

- 严格 `#### <answer>` 正确率从 `0.00%` 提升至 `71.65%`；
- 格式遵循率从 `0.00%` 提升至 `98.26%`；
- 达到 `max_new_tokens` 的比例从 `3.11%` 降至 `1.36%`；
- 平均生成长度从 `371.74` token 降至 `102.30` token，减少约 `72.5%`；
- 全量 test 墙钟时间从 `179 分 0 秒` 降至 `64 分 4 秒`，约加速 `2.79` 倍。

因此，本实验的严谨结论是：**v7 SFT 没有提高 GSM8K 数值准确率，但在基本保持
准确率的同时，大幅改善了答案格式、原生终止行为和推理效率。**

## 2. 研究问题

本实验主要回答三个问题：

1. 从数学 Base Model 出发进行 GSM8K SFT，能否提高未见 test 样本上的数值准确率？
2. SFT 是否能让模型稳定生成 GSM8K 的 `#### <answer>` 格式并正确结束生成？
3. 如果准确率没有提高，SFT 是否仍能带来可量化的行为或效率收益？

此前完成的 `Qwen2.5-1.5B-Instruct` continued-SFT 实验保留在仓库中作为历史
消融，但不作为本次主结论，因为 Instruct Model 本身已经经历过后训练。

## 3. 实验设置

### 3.1 模型与训练方法

| 项目 | 设置 |
|---|---|
| 初始模型 | `Qwen/Qwen2.5-Math-1.5B` |
| 方法 | LoRA-SFT（LLaMA-Factory） |
| 最终配置 | `configs/main/qwen25_math_15b_base_lora_sft_v7.yaml` |
| LoRA target | `q_proj,v_proj,lm_head` |
| LoRA rank / alpha | `8 / 16` |
| 学习率 | `2e-5`，cosine scheduler |
| epoch | 1 |
| 有效 batch size | `1 × 8 = 8` |
| 最大序列长度 | 1024 |
| seed | 42 |
| 最终 checkpoint | `checkpoint-888` |

### 3.2 数据

- 来源：Hugging Face `openai/gsm8k`，配置 `main`。
- 训练集：7,473 条。
- validation：从官方 train 按 `test_size=0.05, seed=42` 固定划分，共 374 条。
- 最终测试：官方 test 全部 1,319 条。
- v7 数据：`data/gsm8k_sft_clean.json`。

v7 只删除训练答案中的 `<<expression=result>>` 计算器标注，保留自然语言推理、
样本顺序和最终 `#### <answer>`。脚本共删除 23,716 个标注，并逐条验证清洗前后
最终答案一致。这样做的动机是避免模型学习不必要的中间标记，同时保留可读推理轨迹。

### 3.3 评测协议

Base 与 SFT 使用同一套评测管线：

- Qwen tokenizer 自带 Chat Template；
- 相同 system prompt；
- greedy decoding，`do_sample=False`；
- BF16；
- `max_new_tokens=1024`；
- 同时将模型 EOS 与 `<|im_end|>` 视为终止符；
- 最终实验关闭任务特定的答案行提前停止，即使用原生生成行为；
- 统一评分器版本：`gsm8k_numeric_v3`。

主要指标：

- **数值准确率**：从回复中提取最终数值并归一化比较；
- **严格准确率**：数值正确且回复末尾符合 `#### <answer>`；
- **格式遵循率**：回复末尾存在可解析的 `#### <answer>`；
- **达到长度上限率**：生成使用完 1,024 个新 token；
- **配对转移矩阵与 McNemar 检验**：比较同一道题在两模型上的正确性变化。

软件环境由结果 JSON 自动记录：PyTorch `2.12.1+cu130`、Transformers `5.8.0`、
PEFT `0.18.1`、Datasets `4.0.0`。实验在 NVIDIA GeForce RTX 5060 Laptop GPU
上完成。

## 4. SFT 迭代过程

本实验没有只保留最终成功配置，而是保留了失败版本，以体现“提出问题 → 修改单一
因素 → 小样本诊断 → 完整 validation”的科研闭环。

| 版本 | 关键改动 | 诊断结果 | 判断 |
|---|---|---|---|
| v1 | `target=all`，LR `1e-4` | 完整 validation `82.35%`；无原生 EOS | 对内部表示扰动过强 |
| v2 | LR 降至 `2e-5` | 20 题 checkpoint-250 为 `75%` | 退化减缓但未解决 |
| v3 | target 改为 `q_proj,v_proj,lm_head` | 完整 validation `80.75%`，格式 `98.93%` | 终止正常，但准确率下降 |
| v4 | 只训练 `lm_head` | 10 题中 7 题达到长度上限 | 容量不足，归档 |
| v5 | rank 4 / alpha 8 | 10 题全部达到长度上限 | 终止失败，归档 |
| v6 | rank 8 / alpha 8 | 10 题中 9 题达到长度上限 | 终止失败，归档 |
| v7 | 恢复 v3 LoRA 设置，清除 `<<...>>` 标注 | 10 题全部 EOS；50 题 `88%`；完整 validation `85.29%` | 进入最终 test |

短 smoke test 只用于发现输出退化和筛除明显失败配置，最终判断基于完整 validation。
官方 test 没有参与超参数或 checkpoint 选择。

## 5. 实验结果

### 5.1 Validation

| 模型 | 数值准确率 | 严格准确率 | 格式遵循率 | 达到长度上限 | 原生 EOS |
|---|---:|---:|---:|---:|---:|
| Raw Base | 319/374 (85.29%) | 0/374 (0.00%) | 0/374 (0.00%) | 6/374 (1.60%) | 旧结果未记录汇总 |
| SFT v3 | 302/374 (80.75%) | 302/374 (80.75%) | 370/374 (98.93%) | 3/374 (0.80%) | 371/374 (99.20%) |
| SFT v7 | 319/374 (85.29%) | 319/374 (85.29%) | 372/374 (99.47%) | 2/374 (0.53%) | 372/374 (99.47%) |

Raw Base 与 v7 的配对转移：

| 转移 | 数量 |
|---|---:|
| Base 对、v7 对 | 279 |
| Base 对、v7 错 | 40 |
| Base 错、v7 对 | 40 |
| Base 错、v7 错 | 15 |

两者 validation 准确率完全相同，精确双侧 McNemar `p=1`。v7 相比 v3 则从
`80.75%` 恢复到 `85.29%`；v3 独自答对 16 题，v7 独自答对 33 题，配对
`p=0.0212941`。这支持“清除计算器标注修复了 v3 的退化”，但由于 v7 是在同一
validation 上反复迭代得到的，该结果仍应视为开发集证据，最终结论必须以 test 为准。

### 5.2 官方 Test 最终对比

| 指标 | Raw Base | SFT v7 | 变化 |
|---|---:|---:|---:|
| 数值准确率 | 947/1319 (71.80%) | 945/1319 (71.65%) | -0.15 pp |
| 严格准确率 | 0/1319 (0.00%) | 945/1319 (71.65%) | +71.65 pp |
| 格式遵循率 | 0/1319 (0.00%) | 1296/1319 (98.26%) | +98.26 pp |
| 达到长度上限 | 41/1319 (3.11%) | 18/1319 (1.36%) | -1.75 pp |
| 原生 EOS | 1278/1319 (96.89%) | 1301/1319 (98.64%) | +1.75 pp |
| 平均生成 token | 371.74 | 102.30 | -72.5% |
| 中位生成 token | 339 | 83 | -75.5% |
| P95 生成 token | 648 | 170.1 | -73.7% |
| 全量墙钟时间 | 179m 0s | 64m 4s | 2.79× 加速 |

最终 test 的逐题配对结果：

| 转移 | 数量 |
|---|---:|
| Base 对、SFT 对 | 760 |
| Base 对、SFT 错 | 187 |
| Base 错、SFT 对 | 185 |
| Base 错、SFT 错 | 187 |

Base-only 与 SFT-only 的数量几乎对称（187 对 185），精确双侧 McNemar
`p=0.958659`。因此不能宣称 SFT 提升或显著降低了数值推理准确率。与此同时，
格式、生成长度和运行时间的改善幅度很大，是本实验确定性更强的结果。

## 6. 结果分析

### 6.1 为什么 SFT 没有提高准确率

`Qwen2.5-Math-1.5B` 已经是数学领域 Base Model，在固定 validation 上达到
`85.29%`，留给单一 GSM8K SFT 的提升空间有限。SFT 的交叉熵目标要求模型模仿
单条参考轨迹，并不直接优化答案是否正确；当目标轨迹包含冗余标记或与模型原有解题
方式不一致时，训练还可能覆盖已有能力。v1-v6 的负结果正好展示了这种现象。

### 6.2 v7 实际学到了什么

v7 稳定学会了三件事：生成更短的 GSM8K 风格推理、在末尾输出 `####` 答案、
通过 ChatML 终止符结束生成。它把平均生成长度压缩到 Base 的约 27.5%，所以即使
加载了 LoRA adapter，实际全量评测仍快约 2.79 倍。

这说明 SFT 不只是“提高 benchmark accuracy”的工具，也可以用于**行为塑形、格式
对齐和推理成本控制**。但这些收益必须与数学正确率分开报告。

### 6.3 如何解释统计检验

`p=0.958659` 表示当前配对数据没有显示 Base 与 v7 存在方向性准确率差异；它不等于
“证明两个模型严格等价”。如果要做形式化等效性声明，需要预先规定可接受差异范围并
使用等效性检验或置信区间。本报告只作“准确率基本持平、未检出显著差异”的表述。

## 7. 有效经验与无效尝试

有效：

- 使用真正的 Base Model 作为 post-training 起点；
- Base/SFT 共用 prompt、解码参数、数据顺序和评分器；
- 用完整 validation 选择方案，只在最后运行官方 test；
- 同时报告宽松数值、严格格式、终止方式、长度和逐题转移；
- 清洗训练目标中无必要的专用标记，并验证标签没有改变；
- 先用 10/20/50 题发现明显的生成退化，再进行完整评测。

无效或不足：

- 只看 token-level `eval_loss` 判断数学能力；
- 仅降低学习率而不处理 LoRA 范围和目标数据问题；
- 只训练 `lm_head`，或单独降低 rank/alpha；
- 把 `####` 格式提升直接解释为推理能力提升；
- 根据少量 smoke 样本的百分比作最终结论。

## 8. 局限性

- 只有一个模型规模、一个数据集和一个训练 seed；
- v7 的开发过程多次查看同一 validation，validation 结果可能存在选择偏差；
- test 只用于最终评测是正确做法，但当前仅有一次训练运行，尚未评估训练方差；
- GSM8K 主要评估小学文字题，不能代表更复杂数学推理；
- 数值答案匹配不能完全评价推理过程的忠实性。

## 9. 结论与下一步

SFT 部分已经形成完整、可复现且包含负结果的实验闭环。最终应向导师表述为：

> 在 Qwen2.5-Math-1.5B 上，GSM8K LoRA-SFT v7 没有提升官方 test 数值准确率
>（71.80% → 71.65%，配对 p=0.958659），但将格式遵循率提升到 98.26%，平均
> 生成 token 减少 72.5%，全量评测约加速 2.79 倍。因此它主要提升了输出对齐与
> 效率，而不是基础数学正确率。

后续 OPD/GKD 实验已复用相同 Base、数据划分、评测器和最终 test 纪律完成，详见
[`OPD_EXPERIMENT_REPORT_zh.md`](OPD_EXPERIMENT_REPORT_zh.md)。OPD 在 test 上得到
72.33% 的最高点估计，但相对 SFT 的配对差异不显著。

## 10. 复现入口

- 最终配置：[`configs/main/qwen25_math_15b_base_lora_sft_v7.yaml`](../configs/main/qwen25_math_15b_base_lora_sft_v7.yaml)
- 数据清洗脚本：[`scripts/prepare_clean_sft_data.py`](../scripts/prepare_clean_sft_data.py)
- 最终 Base 结果：[`results/final/test_gsm8k_base_15b_v3.json`](../results/final/test_gsm8k_base_15b_v3.json)
- 最终 SFT 结果：[`results/final/test_gsm8k_sft_v7_15b_ckpt888_native_v3.json`](../results/final/test_gsm8k_sft_v7_15b_ckpt888_native_v3.json)
- 最终配对分析：[`results/final/test_base_sft_v7_ckpt888_transition_analysis.json`](../results/final/test_base_sft_v7_ckpt888_transition_analysis.json)
- validation 配对分析：[`results/dev/sft_v7/base_sft_v7_ckpt888_transition_analysis.json`](../results/dev/sft_v7/base_sft_v7_ckpt888_transition_analysis.json)
- v3/v7 配对分析：[`results/dev/sft_v7/sft_v3_v7_transition_analysis.json`](../results/dev/sft_v7/sft_v3_v7_transition_analysis.json)

训练命令：

```bash
llamafactory-cli train configs/main/qwen25_math_15b_base_lora_sft_v7.yaml
```

最终 SFT 原生生成评测命令：

正式结果文件默认禁止覆盖；重复运行时请更换 `--output`，或在确认后显式加入
`--overwrite`。

```bash
python3 scripts/eval_sft_adapter.py \
  --base-model Qwen/Qwen2.5-Math-1.5B \
  --adapter outputs/qwen25_math_15b_base_lora_sft_v7/checkpoint-888 \
  --eval-split test \
  --max-new-tokens 1024 \
  --no-stop-after-answer-line \
  --output results/final/test_gsm8k_sft_v7_15b_ckpt888_native_v3.json
```

配对分析命令：

```bash
python3 scripts/compare_base_sft.py \
  --base results/final/test_gsm8k_base_15b_v3.json \
  --sft results/final/test_gsm8k_sft_v7_15b_ckpt888_native_v3.json \
  --output results/final/test_base_sft_v7_ckpt888_transition_analysis.json
```
