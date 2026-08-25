# Qwen2.5-Math-1.5B 后训练项目最终实验总结

## 1. 最终状态

截至 2026-08-25，本项目预先声明的训练、评测和确认性修正任务均已完成。项目覆盖
LoRA-SFT、On-Policy Distillation（OPD/GKD）和 GRPO，并在 8 GiB RTX 5060 Laptop GPU
上完成可复现训练闭环。

最终模型选择结论是：**保留规范 SFT seed42，不将当前 OPD 或 GRPO adapter 作为能力
改进模型。** OPD 和 GRPO 都提供了有价值的工程与负结果证据，但确认性实验不支持它们
稳定优于 SFT。

Confirmatory v2 已正式结束：封存的 `dev_audit` 只评测一次规范 SFT；OPD/GRPO 因未通过
`dev_select` 门槛而未接触 audit。GSM8K 官方 test 和 SVAMP 已在早期实验中消费，不能继续
用于现有方案调参。确认性实验结束后的 v3--v5 GRPO 消融全部是单独标记的 post-hoc
探索，不改变确认性结论。

## 2. 研究问题

本项目依次回答了以下问题：

1. 清洗后的 GSM8K 推理轨迹能否让 Math Base 学会简洁、稳定的 `#### <answer>` 输出？
2. 在 8 GiB 显存下，是否能完成 1.5B 学生与量化教师同时驻留的 OPD/GKD？
3. OPD 或 GRPO 是否能在保持输出行为的同时稳定提高数值准确率？
4. 早期实验中的单 SFT seed、重复查看 validation、小训练预算、seed 混杂和错误 KL
   reference 是否会改变结论？
5. GRPO 的负结果能否通过降低 temperature、降低 learning rate 或移除 tied `lm_head`
   这一类简单单因素修补解决？

## 3. 数据、模型与统一评测

- 基础模型：`Qwen/Qwen2.5-Math-1.5B`。
- SFT 数据：清洗后的 7,473 条 GSM8K 训练样本，共移除 23,716 处 calculator annotation，
  保留自然语言推理和最终 `####` 答案。
- Confirmatory v2 切分：6,725 train / 374 dev-select / 374 dev-audit，三者互斥。
- LoRA 主配置：`q_proj,v_proj,lm_head`，rank 8，alpha 16，SFT learning rate `2e-5`，
  训练 1 epoch。
- 正式生成：贪心解码、`max_new_tokens=1024`、禁用答案行人为提前停止，观察原生 EOS。
- GSM8K 评分器：`gsm8k_numeric_v3`；SVAMP 评分器：`svamp_numeric_v1`。
- 同题方法比较使用精确双侧 McNemar 检验，同时报告数值准确率、严格 `####` 准确率、
  格式合规率、截断率和生成长度。

## 4. 第一阶段：SFT 与早期后训练实验

### 4.1 SFT v7

早期 v1--v6 暴露了学习率过高、输出不终止、LoRA 容量不足等问题。最终 v7 的关键修正
不是继续调整生成参数，而是删除训练答案中的 calculator annotation，使训练目标与原生
文本生成一致。

在旧固定 validation 上，Raw Base 与 SFT v7 都是 319/374（85.29%），逐题交换为
40/40，McNemar `p=1`。SFT 没有提高这一集合的数值总分，但把格式合规率提高到 99.47%，
平均回复显著缩短。

在 GSM8K 官方 test 上，Raw Base 为 947/1319（71.80%），SFT v7 为 945/1319
（71.65%）；Base-only 187 题、SFT-only 185 题，`p=0.958659`。因此 SFT 的可靠收益是
**格式与效率控制**，不是已证实的 GSM8K 数值能力提升。

### 4.2 早期 OPD 与 GRPO

早期 OPD 使用冻结的 Qwen2.5-Math-1.5B-Instruct 教师，完成三次 50-step pilot；早期
GRPO 从 SFT v7 出发，完成三次 30-step / 120-rollout pilot。这一阶段已经使用 GSM8K
官方 test 和 SVAMP，因此结果只能作为最终观察，不能反向指导后续超参数。

| GSM8K test（1,319 题） | Raw Base | SFT v7 | OPD 均值 ± SD | GRPO 均值 ± SD |
|---|---:|---:|---:|---:|
| 数值准确率 | 71.80% | 71.65% | **72.91% ± 0.54** | 72.18% ± 0.35 |
| 严格准确率 | 0.00% | 71.65% | **72.83% ± 0.54** | 72.10% ± 0.40 |
| 格式合规率 | 0.00% | **98.26%** | 97.68% ± 0.16 | 98.13% ± 0.09 |
| 达到长度上限 | 3.11% | **1.36%** | 1.95% ± 0.16 | 1.42% ± 0.04 |
| 平均生成 tokens | 371.74 | **102.30** | 111.60 ± 1.34 | 102.99 ± 1.06 |

OPD 三次 test 点估计均高于 SFT，但配对 `p` 分别为 0.439440、0.050487、0.117213，
均未低于 0.05；GRPO 三次相对 SFT 的 `p` 分别为 0.291215、0.168978、0.885433。
不能声称两者已经显著提升 GSM8K 能力。

独立 SVAMP 进一步揭示跨数据集代价：

| SVAMP（1,000 题） | Raw Base | SFT v7 | OPD 均值 ± SD | GRPO 均值 ± SD |
|---|---:|---:|---:|---:|
| 数值准确率 | **85.20%** | 81.50% | 81.93% ± 0.32 | 81.20% ± 0.36 |
| 格式合规率 | 0.00% | **98.80%** | 98.47% ± 0.21 | 98.63% ± 0.25 |
| 达到长度上限 | 7.00% | **0.90%** | 1.23% ± 0.21 | 1.10% ± 0.17 |

SFT 相对 Base 下降 3.70 pp，配对 `p=0.00761528`。OPD 只能恢复其中很小一部分，GRPO
没有复现 GSM8K 上的正向趋势。这说明格式专门化可能以跨数据集数值泛化为代价。

## 5. Confirmatory v2：对导师提出问题的系统修正

Confirmatory v2 不追认早期最好结果，而是重新冻结协议：SFT 使用三个训练 seed；拆出互斥
`dev_select/dev_audit`；OPD/GRPO 预算扩大到每次 200 steps / 800 on-policy
microbatches；固定 `data_seed=42`，只改变 training/generation seed；GRPO 先合并 SFT
LoRA，再新建 GRPO LoRA，使关闭新 adapter 后得到真正的 SFT KL reference，并启用
`beta=0.04`。算术一致性奖励只作为 proxy，不被解释为推理忠实性证明。

### 5.1 SFT 多随机种子

| SFT seed | dev-select 数值准确率 | 严格准确率 | 格式合规率 | 截断率 |
|---:|---:|---:|---:|---:|
| 42 | 83.69% | 83.69% | 99.73% | 0.80% |
| 43 | 82.35% | 82.09% | 99.20% | 0.80% |
| 44 | 85.29% | 85.29% | 99.47% | 0.53% |
| 均值 ± SD | **83.78% ± 1.47** | 83.69% ± 1.60 | 99.47% ± 0.27 | 0.71% ± 0.15 |

下游实验没有选择 dev-select 最好的 seed44，而是按预声明规则固定 seed42。这样避免了把
随机种子搜索误当作模型改进。

### 5.2 扩大预算后的 OPD

三次 OPD 的宽松数值准确率为 81.28%、83.69%、83.42%，均值 82.80% ± 1.32 pp，
与规范 SFT 的 83.69% 接近，三次 McNemar `p` 为 0.281、1.000、1.000。

但输出行为发生严重崩坏：严格准确率均值仅 **1.25%**，格式合规率仅 **2.58%**，
截断率 **30.93%**，平均生成 **427.97 tokens**。因此 OPD 在 audit 前被拒绝。宽松
评分能够从冗长或非合规回复中抽取正确数字，不代表模型可部署。

### 5.3 正确 KL reference 下的 GRPO

三次 GRPO 都使用合并后的 SFT seed42 作为有效 reference，完成 200 steps / 800
rollouts。训练期平均 KL 为 `3.027e-4`，说明 KL 路径确实生效；平均 rollout 截断率仍为
44.29%，截断 completion 已从 token loss 中屏蔽。

| 方法 | 数值准确率 | 严格准确率 | 格式合规率 | 截断率 | 平均 tokens |
|---|---:|---:|---:|---:|---:|
| 规范 SFT seed42 | 83.69% | 83.69% | 99.73% | 0.80% | 96.83 |
| GRPO 三 seed 均值 ± SD | **78.61% ± 0.53** | 77.90% ± 0.67 | 95.72% ± 0.53 | 4.63% ± 0.67 | 138.49 ± 5.97 |

GRPO seed42/43/44 相对 SFT 分别下降 4.55、5.61、5.08 pp；精确 McNemar `p` 分别为
0.00948、0.00191、0.00432。三次均为显著负向结果，因此也在 audit 前被拒绝。

### 5.4 一次性 dev-audit

只有预先指定的 SFT seed42 获准查看封存 audit。其结果为 303/374（**81.02%**），严格
准确率同为 81.02%，格式合规率 99.47%，截断率 0.53%，平均 98.89 tokens。audit 比
dev-select 低 2.67 pp；由于分区互斥，独立两比例检验为 `p=0.3375`，差异不显著，audit
准确率的 95% Wilson 区间为 76.73%--84.67%。

最终应把 **81.02%** 作为规范 SFT 在未参与调参数据上的点估计。audit 至此消费完毕，
不能继续用于调参。

## 6. 协议结束后的 GRPO 诊断

Confirmatory v2 后只在已经消费的 `dev_select` 上做单因素诊断，且不查看 audit、GSM8K
test 或 SVAMP：

| 50-step GRPO 方案 | 数值正确率 | 格式合规率 | 截断率 | 对 SFT 的 McNemar p | 判定 |
|---|---:|---:|---:|---:|---|
| 原始 temperature 0.9 / LR `5e-6` | 78.34% | 94.12% | 6.15% | — | 诊断基线 |
| temperature 0.3 | 78.61% | 92.78% | 7.75% | 0.00540 | 负结果 |
| learning rate `1e-6` | 78.61% | 92.78% | 7.75% | 0.00661 | 负结果 |
| targets `q_proj,v_proj` | 78.61% | 94.12% | 6.42% | 0.00540 | 负结果 |

降低 temperature 虽缩短训练 rollout，却造成更低 entropy 和更多零组内奖励方差；降低
learning rate 明显缩小参数变化，但没有保留 SFT 能力。移除 tied `lm_head` 将 adapter
从约 454 MiB 降至 4.17 MiB，并消除完整 `lm_head.base_layer.weight` 保存路径，这是
明确的工程修复；但它相对旧 GRPO 仅净增 1 题，`p=1`，没有能力收益。

因此简单的温度、学习率或 target 修补均不能解释或解决 GRPO 退化。v5 是最后一次允许
复用该 `dev_select` 的 pilot，到此触发停止规则。

## 7. 最终结论

### 可以可靠陈述

- 清洗 SFT 轨迹能让 Math Base 稳定输出短而合规的 `####` 答案，并显著提高生成效率。
- 8 GiB 消费级 GPU 可以完成 1.5B 级 LoRA-SFT、量化教师 OPD 和原生 TRL GRPO。
- 早期 OPD 在 GSM8K test 上出现跨 seed 一致的正向点估计，但没有单次显著证据，也没有
  恢复 SVAMP 上相对 Base 的退化。
- 修正预算、seed 和 KL reference 后，当前 OPD 出现输出行为崩坏，当前 GRPO 则在三个
  seed 上显著降低 dev-select 数值准确率。
- 移除 tied `lm_head` 是有效的 adapter 工程优化，但不是 GRPO 性能退化的主要原因。

### 不能声称

- SFT 提升了 Raw Base 的通用数学数值能力；
- OPD 或 GRPO 已被证明稳定优于 SFT；
- 数值奖励或算术一致性 proxy 能证明自然语言推理过程忠实；
- 事后表现最好的 seed、checkpoint 或 test 点估计是可部署最优模型。

## 8. 最终推荐方案

若目标是当前项目范围内最可靠、格式稳定且有未调参 audit 支持的模型，应使用：

- 规范模型：SFT seed42，`outputs/confirmatory_v2/sft_seed42/checkpoint-841`；
- 训练配置：一个 epoch、learning rate `2e-5`、LoRA rank 8 / alpha 16；
- 报告点估计：dev-audit 81.02%，严格准确率 81.02%，格式合规率 99.47%；
- 不选择 OPD/GRPO adapter 作为当前最终模型。

模型 checkpoint 位于被忽略的 `outputs/`，仓库提交配置、脚本、结果 JSON、配对分析和
SHA-256 清单，不提交大模型权重。

## 9. 尚存边界与未来工作

当前没有漏跑的预声明任务。以下事项属于需要**新数据和新协议**的后续研究，而不是本轮
补实验：

1. 固定 training/generation seed，单独改变 `data_seed`，独立估计训练子集方差；
2. 使用新的、未消费的开发集复现 OPD/GRPO，而不是重用现有 dev-select/audit/test；
3. 引入人工过程标注、程序证明检查器或独立过程监督数据，研究推理忠实性；
4. 重新设计能约束终止、长度和格式的 OPD/GRPO 目标，并预先注册通过门槛；
5. 在新 benchmark 上检验 SFT 的格式收益与跨数据集数值泛化之间的权衡。

在建立这些条件之前，不应继续扫描当前 GRPO 的 temperature、learning rate、target 或
中间 checkpoint。

## 10. 复现与证据入口

- 确认性机器汇总：[`../results/confirmatory_v2/confirmatory_v2_progress.json`](../results/confirmatory_v2/confirmatory_v2_progress.json)
- Confirmatory v2 协议：[`CONFIRMATORY_V2_PROTOCOL_zh.md`](CONFIRMATORY_V2_PROTOCOL_zh.md)
- SFT 报告：[`SFT_EXPERIMENT_REPORT_zh.md`](SFT_EXPERIMENT_REPORT_zh.md)
- OPD 报告：[`OPD_EXPERIMENT_REPORT_zh.md`](OPD_EXPERIMENT_REPORT_zh.md)
- GRPO 报告：[`GRPO_EXPERIMENT_REPORT_zh.md`](GRPO_EXPERIMENT_REPORT_zh.md)
- SVAMP 报告：[`SVAMP_EXPERIMENT_REPORT_zh.md`](SVAMP_EXPERIMENT_REPORT_zh.md)
- GRPO v3/v4/v5 诊断：[`GRPO_V3_EXPLORATORY_LOG_zh.md`](GRPO_V3_EXPLORATORY_LOG_zh.md)、
  [`GRPO_V4_EXPLORATORY_LOG_zh.md`](GRPO_V4_EXPLORATORY_LOG_zh.md)、
  [`GRPO_V5_EXPLORATORY_LOG_zh.md`](GRPO_V5_EXPLORATORY_LOG_zh.md)
