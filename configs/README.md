# Experiment configurations / 实验配置

## `main/`

Qwen2.5-Math-1.5B 主实验配置。v1-v3 保留实验迭代过程，v7 是最终方案。

| 配置 | 控制变量 | 结果与用途 |
|---|---|---|
| `qwen25_math_15b_base_lora_sft_v1.yaml` | `target=all`，LR `1e-4` | 初始方案；准确率与终止行为退化 |
| `qwen25_math_15b_base_lora_sft_v2.yaml` | LR 降至 `2e-5` | 学习率消融；未解决终止问题 |
| `qwen25_math_15b_base_lora_sft_v3.yaml` | `q_proj,v_proj,lm_head` | 原生 EOS 恢复，但 validation 准确率为 80.75% |
| `qwen25_math_15b_base_lora_sft_v7.yaml` | 使用清洗后的推理轨迹 | 最终方案；validation 85.29%，进入官方 test |

最终 v7 的核心设置：

```yaml
model_name_or_path: Qwen/Qwen2.5-Math-1.5B
dataset: gsm8k_sft_clean
lora_target: q_proj,v_proj,lm_head
lora_rank: 8
lora_alpha: 16
learning_rate: 2.0e-5
num_train_epochs: 1.0
seed: 42
```

## `archive/math_15b/`

在正式训练后未通过原生生成 smoke test 的消融：

- v4：只训练 `lm_head`，10 题中 7 题达到 token 上限；
- v5：rank 4 / alpha 8，10 题全部达到 token 上限；
- v6：rank 8 / alpha 8，10 题中 9 题达到 token 上限。

这些配置不是待运行任务，而是为复现负结果而保留。

## `controls/`

- `qwen25_15b_base_lora_sft_v1.yaml`：通用 `Qwen2.5-1.5B` Base 对照，
  用于区分数学领域预训练收益与 SFT 收益，不属于最终主结论。

## 其他 `archive/`

早期 `Qwen2.5-1.5B-Instruct` continued-SFT 与 0.5B pipeline 实验。

所有配置均假设从项目根目录运行，因此 `dataset_dir: data` 与
`output_dir: outputs/...` 会按仓库相对路径解析。
