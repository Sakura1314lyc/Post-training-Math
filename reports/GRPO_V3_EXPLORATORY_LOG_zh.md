# GRPO v3 探索性消融日志

## 低温 rollout pilot

本实验在 Confirmatory v2 完成后开展，属于 post-hoc exploratory pilot，不修改已有确认性
结论。它只改变 rollout temperature：从 0.9 降到 0.3；初始策略、训练数据子集、三个
随机种子、学习率、KL、奖励、completion 长度和 50-step 预算保持不变。

预注册假设：较低温度会减少 256-token 截断，并可能缓解原 seed42 在前 50 steps 已出现的
dev-select 退化。主要指标是 dev-select 数值准确率；截断、格式与生成长度是诊断指标。

本 pilot 只使用已经消费的 dev-select。禁止查看或使用 dev-audit、GSM8K test、SVAMP；
单 seed 结果不能作为稳定改进结论。

冻结配置见 `configs/experimental/grpo_v3_temperature03_pilot.json`。

## 运行结果

训练于 2026-08-25 完成，50 steps / 200 rollouts 耗时 343.47 秒，峰值显存
3.59 GiB。跟踪 LoRA 参数的最大绝对变化为 `1.6804e-4`，训练指标有限且 adapter 已正常
保存。run manifest、summary 与 adapter 权重哈希已写入冻结配置。

### 训练期 rollout 诊断

下表比较原 temperature=0.9 seed42 正式 run 的前 50 steps 与本次 temperature=0.3
pilot；其他设置一致。

| 指标（50-step 均值） | temp=0.9 | temp=0.3 | 变化 |
|---|---:|---:|---:|
| completion 截断比例 | 46.00% | 32.50% | -13.50 pp |
| completion 平均长度 | 177.62 | 127.58 | -50.04 tokens |
| 总奖励 | 0.4388 | 0.7265 | +0.2878 |
| 奖励标准差 | 0.3795 | 0.2159 | -0.1636 |
| 零组内奖励方差比例 | 0.00% | 28.00% | +28.00 pp |
| token entropy | 0.3484 | 0.0764 | -0.2720 |

低温确实让训练 rollout 更短、截断更少，采样奖励也更高；但 entropy 大幅下降，且 28%
的 group 没有相对奖励差异。这意味着四个候选更相似，GRPO 可利用的组内排序信号变弱。

### 固定 dev-select 评测

评测使用与 Confirmatory v2 相同的 374 题、贪心解码、1024-token 上限，且不在完整
`####` 行后人为截停。

| 模型 | 数值准确率 | 严格准确率 | 格式合规 | 截断率 | 平均 tokens |
|---|---:|---:|---:|---:|---:|
| SFT seed42 step-0 | 83.69% | 83.69% | 99.73% | 0.80% | 96.83 |
| GRPO temp=0.9 step-50 | 78.34% | 77.01% | 94.12% | 6.15% | 153.31 |
| GRPO temp=0.3 step-50 | 78.61% | 76.47% | 92.78% | 7.75% | 166.75 |

与 SFT step-0 配对比较时，31 题由对变错，12 题由错变对，准确率差为 -5.08 pp，
exact McNemar `p=0.00540`。因此低温 GRPO 相对初始 SFT 的退化具有统计证据。

与 temperature=0.9 step-50 配对比较时，6 题由对变错，7 题由错变对，准确率只增加
0.27 pp，exact McNemar `p=1.0`。两种温度在任务准确率上不可区分；低温版本的严格格式、
截断率和平均生成长度反而略差。

## 结论与边界

本次单因素 pilot 得到负结果：降低 rollout temperature 能改善训练期采样长度和截断统计，
但不能修复 50 steps 时已经出现的 dev-select 能力退化。训练采样分布更“整齐”不等价于
最终贪心策略更可靠。

这只是单 seed、post-hoc 探索结果，不改变 Confirmatory v2 结论，也不能据此选择部署模型。
本轮没有查看 dev-audit、GSM8K test 或 SVAMP。若继续研究，优先方向应是改善 reward 的组内
辨识度或限制策略漂移，而不是继续在同一个 dev-select 上扫描更多 temperature。
