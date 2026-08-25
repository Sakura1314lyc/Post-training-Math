# GRPO adapter 结构与漂移诊断

## 背景

Confirmatory v2、v3 低温 pilot 和 v4 低学习率 pilot 均显示：GRPO 在很小的训练预算内就会
显著损害 SFT 的 dev-select 数值能力与格式稳定性。单独降低 rollout temperature 或 learning
rate 都未修复问题，因此在继续实验前检查 adapter 结构和漂移测量是否可靠。

## 现有 adapter 的结构发现

当前 fresh GRPO LoRA 的 target modules 为 `q_proj,v_proj,lm_head`。Qwen2.5-Math-1.5B
设置了 tied word embeddings，PEFT 因此在保存时持续提示 `lm_head` 属于 tied target module，
合并或格式转换可能复杂化。

对 v4 final adapter 的 safetensors 逐项检查得到：

- 文件大小 476,031,728 bytes，即约 453.98 MiB；
- 共 115 个 tensor；
- 除 LoRA A/B 外，还保存了完整的 `base_model.model.lm_head.base_layer.weight`；
- 该完整张量形状为 `[151936, 1536]`，含 233,373,696 个参数；
- `lm_head` 自身的 LoRA A/B 分别含 12,288 和 1,215,488 个参数；
- 28 层 `q_proj,v_proj` LoRA 合计约 1,089,536 个可训练参数。

完整 `lm_head` 并不是可训练 LoRA 参数，却主导了最终 adapter 文件大小和保存/加载路径。
这不能单独证明它导致性能退化，但构成了明确的结构性混杂因素，值得通过只删除
`lm_head` target 的单因素实验隔离。

## 旧漂移指标的不足

此前 `train_grpo.py` 只保存第一个可训练张量
`layers.0.self_attn.q_proj.lora_A` 的最大绝对变化。LoRA B 通常从零初始化，A/B 的梯度和尺度
也不同，因此单个 A 张量不能代表整个 adapter 的更新幅度，更无法判断 `lm_head` 是否主导
漂移。

脚本现已增加全 trainable adapter CPU 快照，并在训练结束后输出：

- 全局 tensor/parameter 数、更新 tensor 数；
- 最大和平均绝对变化、L2 变化、训练前后 L2 范数、相对 L2 变化；
- 按 `lora_A/lora_B` 分组；
- 按 target module 分组；
- 按 target module 与 A/B 联合分组。

旧的单参数字段继续保留，确保历史消费者兼容。额外 CPU 内存约为所有可训练参数的一份
FP32 快照；对现有 231 万参数 adapter 约为 9 MiB，不增加 GPU 常驻显存。

## 下一步边界

下一轮只将 target modules 从 `q_proj,v_proj,lm_head` 改为 `q_proj,v_proj`，恢复已确认的
标准 temperature=0.9 和 learning rate `5e-6`，其他设置不变。首先运行 1-step smoke，验证：

1. 新的全 adapter 漂移统计存在且有限；
2. 保存文件不再含 `lm_head.base_layer.weight`；
3. adapter 大小显著下降；
4. PEFT tied-target 警告消失。

只有 smoke 通过后才运行预注册的 50-step pilot。即使结果改善，也只能作为 post-hoc
探索证据；本轮不得使用 dev-audit、GSM8K test 或 SVAMP。
