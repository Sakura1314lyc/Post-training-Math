# Qwen2.5-Math-1.5B 在 GSM8K 上的 GRPO 阶段实验报告

## 1. 当前结论

截至 2026-08-21，本实验已经完成原生 TRL GRPO 实现、1-step smoke、seed 42/43/44
三次 30-step 训练、三次 pilot validation，以及 seed42 pilot test。三次训练均在 8 GiB
RTX 5060 Laptop GPU 上成功，峰值分配显存约 3.33 GiB，LoRA 参数均发生有限更新。

当前结果只能作为阶段证据，不能写成最终 GRPO 对比：今天的 GRPO 评测使用 512-token
上限并启用答案行提前停止，而既有 SFT/OPD 正式协议使用 1,024-token 上限与原生 EOS。
prompt、评分器和样本集合相同，但 generation metadata 不同。为避免错误比较，结果已归档到
`results/grpo/pilot/`；正式同协议 validation/test 留待下一实验日完成。

## 2. 实现与兼容性

当前 LLaMA-Factory checkout 没有 GRPO 训练入口，因此新增
[`scripts/train_grpo.py`](../scripts/train_grpo.py)，直接使用 TRL 0.24 的 `GRPOTrainer`：

- 从 SFT v7 checkpoint-888 加载可训练 LoRA policy；
- 精确复现并排除固定 374 题 validation，训练候选为 7,099 条；
- 主奖励为最终数值正确性（权重 1.0）；
- 辅助奖励为严格 `#### <answer>` 格式（权重 0.1）；
- 使用 Transformers 原生生成，不使用 vLLM；
- 每个 prompt 生成 4 个 completion，采用 DAPO loss 和 group reward scaling；
- 截断 completion 默认从 loss 中屏蔽。

本地环境为 Transformers 5.8.0、TRL 0.24.0、PEFT 0.18.1。为兼容这一组合，脚本处理了
两处上游 API 差异：Transformers 5.x 的私有 optional-package 探测返回 tuple，而 TRL
0.24 假设布尔值；TRL 还访问了 Transformers 5.x 已移除的 `warnings_issued` 字典。
这些兼容逻辑都有离线单元测试。

## 3. 为什么当前固定 `beta=0`

本实验直接继续训练已经存在的 SFT PEFT adapter。TRL 对 PEFT 模型构造 KL reference 时会
通过禁用 adapter 得到参考策略；这里禁用 adapter 后得到的是 Raw Base，而不是“训练开始时
的 SFT policy”。因此非零 `beta` 会把 KL 约束施加到错误参考模型。当前实现显式拒绝
`beta != 0`，将本轮定位为无显式 KL 的小规模 GRPO pilot。若后续需要标准 KL，应先冻结
一份独立的初始 SFT reference，而不是静默引用 Raw Base。

## 4. Smoke 与训练设置

1-step smoke 使用 8 条候选样本、4 generations、128 completion tokens：

- runtime：7.49 秒；
- train loss：−0.1912；
- 数值奖励均值：0.50；格式奖励均值：0.75；
- LoRA 跟踪参数最大绝对变化：`5.00e-06`；
- 峰值分配显存：3.33 GiB。

正式 pilot 固定如下：

| 设置 | 值 |
|---|---:|
| 每次抽取的训练候选 | 256 |
| optimizer steps | 30 |
| generations / prompt | 4 |
| 总 rollout 数 | 120 |
| learning rate | `5e-6` |
| prompt / completion 上限 | 512 / 128 |
| accuracy / format reward 权重 | 1.0 / 0.1 |
| beta | 0 |
| checkpoint | 固定 step 30 |

三次运行的训练状态：

| seed | train loss | runtime | LoRA 参数最大变化 | 峰值显存 |
|---:|---:|---:|---:|---:|
| 42 | −0.0404 | 136.7 s | `1.099e-4` | 3.33 GiB |
| 43 | −0.1080 | 140.8 s | `9.450e-5` | 3.33 GiB |
| 44 | −0.1344 | 140.1 s | `9.720e-5` | 3.33 GiB |

三次均满足参数已更新、指标有限、global step 正确。当前 `--seed` 同时改变 256 条训练样本
抽取与训练随机性，所以它们表示端到端运行波动，不是固定训练数据下的纯优化 seed。

## 5. Pilot Validation

checkpoint-10/20/30 首先在相同的 50 题切片上筛查，三者均为 44/50，格式率 49/50，
无长度截断。基于训练步数与同分结果固定报告 checkpoint-30，之后不再根据 test 改动。

三 seed 的 374 题 pilot validation：

| 指标 | seed42 | seed43 | seed44 | 均值 ± 样本 SD |
|---|---:|---:|---:|---:|
| 数值准确率 | 322/374（86.10%） | 320/374（85.56%） | 318/374（85.03%） | 85.56% ± 0.53 |
| 严格准确率 | 321/374（85.83%） | 320/374（85.56%） | 318/374（85.03%） | 85.47% ± 0.41 |
| 格式遵循率 | 370/374 | 371/374 | 370/374 | 99.02% ± 0.15 |
| 达到 512-token 上限 | 2/374 | 0/374 | 2/374 | 0.36% ± 0.31 |

三个 seed 的正确性只在 19/374 题上出现分歧，平均数值正确数为 320，比 SFT v7 的 319
高 1 题。但由于生成协议不同，这个差异只应视为待复核的方向信号。

## 6. Pilot Test

seed42 在 GSM8K test 上得到：

- 数值准确率：953/1319（72.25%）；
- 严格准确率：953/1319（72.25%）；
- 格式遵循率：1295/1319（98.18%）；
- 达到 512-token 上限：18/1319（1.36%）；
- 评测用时：65m 34s。

若仅观察点估计，它比 SFT v7 的 71.65% 高 0.61 pp，与 OPD seed42 的严格准确率
72.25% 相同。但这两项基线来自 1,024-token 原生 EOS 协议，故这里不能进行正式 McNemar
结论或宣称 GRPO 提升。此前临时生成的跨协议 transition 文件已删除。

## 7. 已发现的实验工程问题

今天的评测命令在 pilot 阶段沿用了 512-token 默认值，随后被错误地与 1,024-token 正式
基线放在一起。虽然 seed42 test 没有触发答案行提前停止，仍有 18 条回复撞到 512-token
上限，不能假设它们扩展到 1,024 token 后结果不变。

为防止再次发生，`compare_base_sft.py` 现在除 evaluator version 和题目集合外，还要求
`benchmark`、`dataset`、`prompt` 与完整 `generation` metadata 一致；协议不同会直接报错。

## 8. 下一步

下一实验日按 [`results/grpo/README.md`](../results/grpo/README.md) 中的固定命令执行：

1. seed42/43/44 全部重新跑 374 题 validation，使用 1,024 token 与原生 EOS；
2. 不再根据结果调参或改 checkpoint；
3. 三个固定 checkpoint 全部跑官方 test；
4. 生成 SFT→GRPO、OPD→GRPO 的同协议配对分析与三 seed 汇总；
5. 若要评测 SVAMP，只能预先固定协议，并将其作为新增泛化结果，而不能回头选择现有
   checkpoint。

## 9. 文件

- 训练脚本：[`scripts/train_grpo.py`](../scripts/train_grpo.py)
- 单元测试：[`tests/test_train_grpo.py`](../tests/test_train_grpo.py)
- 阶段汇总：[`results/grpo/grpo_stage_summary.json`](../results/grpo/grpo_stage_summary.json)
- Pilot 结果：[`results/grpo/pilot/`](../results/grpo/pilot/)
- 复现与明日命令：[`results/grpo/README.md`](../results/grpo/README.md)
