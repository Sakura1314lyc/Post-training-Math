# Confirmatory v2 GRPO checkpoint 轨迹诊断

## 定位

本报告是确认性实验完成后的 **post-hoc exploratory diagnostic**。正式协议预先指定
step-200 为最终 adapter；这里对 seed42 的恢复 checkpoint 进行评测，只用于解释训练动态，
不得改写正式结果、事后选择最好 checkpoint，或声称获得新的确认性性能。

所有 checkpoint 使用同一个 merged SFT seed42 起点、data seed 42、training/generation
seed 42，并在相同的 374 题 `dev_select` 上采用 greedy、1024 max-new-tokens、关闭答案行
提前停止的协议。`dev_audit` 没有用于该分析。

## 轨迹

| Step | 数值准确率 | 严格准确率 | 格式合规率 | 截断率 | 平均生成 tokens | 相对 SFT McNemar p |
|---:|---:|---:|---:|---:|---:|---:|
| 0（SFT） | 83.69% | 83.69% | 99.73% | 0.80% | 96.83 | — |
| 50 | 78.34% | 77.01% | 94.12% | 6.15% | 153.31 | 0.00366 |
| 100 | 77.81% | 76.74% | 94.92% | 5.61% | 147.37 | 0.000941 |
| 150 | 77.81% | 76.20% | 94.12% | 5.88% | 150.24 | 0.000472 |
| 200 | 79.14% | 78.61% | 95.72% | 4.55% | 137.48 | 0.00948 |

step50 已经相对 SFT 净减少 20 个正确答案（32 个退化、12 个改善），说明损伤出现在
最初 50 个 optimizer steps 内。step100/150 更低，step200 又略微恢复，因此曲线不是随
训练步数单调恶化；“只因训练太久”不足以解释结果。

## 诊断解释

更符合现有证据的解释是：随机 rollout 分布、当前数值/格式/算术代理奖励以及确定性评测
目标之间存在早期错配。训练期间约 44% rollout 达到 256-token 上限，虽然这些 completion
已通过 masking 排除 token loss，但有效学习信号被明显稀释；确定性评测同时表现为回答
变长、截断增加和准确率下降。

非零 KL reference 的实现已验证正确，三次正式训练的平均 KL 也很小，但小的 sampled-token
KL 并不自动保证任务准确率不变。上述观察只能定位问题范围，不能单独识别温度、KL 系数、
奖励权重或 completion 长度中的哪一个是根因。

## 下一步边界

若继续研究，应新建独立的开发协议，并一次只改变一个因素，例如更低 rollout temperature、
更强 KL、不同格式奖励或更长 completion 上限。新的选择过程不得使用已经消费的
`dev_audit`、GSM8K test 或 SVAMP；否则只能标记为探索性分析。

精确逐题结果和 SHA-256 见
`results/confirmatory_v2/diagnostics/grpo_seed42_trajectory_summary.json`。
