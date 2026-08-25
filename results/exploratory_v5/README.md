# Exploratory v5 results

本目录保存最后一次允许复用 `dev_select` 的 GRPO 结构消融结果。实验只移除 fresh GRPO
LoRA targets 中 tied `lm_head`，其余训练与评测设置保持冻结。

## 文件

- `grpo_qv_only_data42_train42_gen42_step50_dev_select_v3.json`：唯一一次 final step-50
  评测，294/374（78.61%），严格正确 287/374，格式合规 352/374，24/374 达到长度上限。
- `sft_vs_grpo_qv_only_step50_transition_analysis.json`：相对 SFT 的配对比较；31 题退化、
  12 题改善，准确率差 -5.08 个百分点，精确双侧 McNemar `p=0.00540157`。
- `grpo_qvlm_vs_qv_only_step50_transition_analysis.json`：相对旧 `q/v/lm_head` GRPO 的
  配对比较；8 题退化、9 题改善，准确率差 +0.27 个百分点，McNemar `p=1`。

## 判定

正确数至少 306、格式合规率至少 97%、长度上限命中率不超过 3% 三项预声明门槛均未
通过。移除 `lm_head` 解决了 adapter 保存结构和体积问题，但没有恢复 SFT 数值能力。
按照冻结协议，不再基于该 `dev_select` 继续进行 GRPO 超参数或结构扫描。
