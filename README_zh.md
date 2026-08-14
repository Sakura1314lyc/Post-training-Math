# GSM8K 大模型后训练实验

[English](README.md) | [简体中文](README_zh.md)

本项目当前的主实验路线为：

> `Qwen/Qwen2.5-Math-1.5B`（数学领域 Base Model）→ GSM8K LoRA-SFT → 配对评测

Qwen 官方将该 checkpoint 定义为数学领域 Base Model，并称其更适合作为后续微调的起点。其[官方模型卡](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B)也提供了 Qwen Chat Template，因此 SFT 前后的模型可以使用完全相同的序列化 Prompt 进行公平比较。

此前完成的 `Qwen2.5-1.5B-Instruct` → continued-SFT 实验不会删除，而是作为附加消融实验保留。该实验说明：继续 SFT 可能提高答案格式遵循能力，但同时降低模型原有的数值推理准确率。

## 实验资源

- 主实验配置：`configs/main/qwen25_math_15b_base_lora_sft_v1.yaml` 至
  `qwen25_math_15b_base_lora_sft_v3.yaml`
- 初始模型：`Qwen/Qwen2.5-Math-1.5B`
- 训练数据：`data/gsm8k_sft_formal.json`
- 通用 Base 对照配置：`configs/controls/qwen25_15b_base_lora_sft_v1.yaml`
- 历史实验配置：`configs/archive/`
- 历史实验分析：`results/archive/instruct_15b/base_sft_transition_analysis_v2.json`

现有 GSM8K SFT 数据可以直接用于 Base Model，不需要重新生成。SFT 数据描述的是输入问题与目标推理轨迹，并不依赖模型初始化来自 Base 还是 Instruct checkpoint。

## 当前主实验结论

Raw Base 在固定 374 道 validation 上取得 `319/374 = 85.29%`。目前完成
完整评测的最佳 v1 adapter 是 checkpoint-100，结果为 `308/374 = 82.35%`，
点估计下降 `2.94` 个百分点。逐题配对中，Base 独自答对 53 题，SFT 独自答对
42 题；精确双侧 McNemar 检验为 `p=0.3049`，因此该下降没有达到统计显著，
不能写成“SFT 显著降低准确率”。

更明确的问题来自模型行为：v1 checkpoint-100 在 374 道题中没有一次原生
EOS，且只有一道题同时满足严格格式与答案正确；v1 后期 checkpoint 进一步退化。
v2 将学习率从 `1e-4` 降至 `2e-5`，在 20 题诊断中只减缓了退化，没有解决
终止和格式异常。这些负结果会完整保留，而不是删除。

下一步 v3 控制实验保持 v2 的数据和学习率不变，只把 LoRA target 从全部内部
投影层改为 `q_proj,v_proj,lm_head`。其目的分别是减少对数学推理能力的扰动，并
直接训练输出头学习 ChatML 终止行为。

## 项目结构

```text
configs/main/       # Qwen2.5-Math 主实验
configs/controls/   # 通用 Qwen2.5 Base 对照实验
configs/archive/    # 早期 Instruct 与 0.5B 实验
data/               # GSM8K SFT 数据与数据集注册信息
scripts/            # 数据准备、评测、重评分与配对分析
results/            # 新的主实验和对照实验结果
results/archive/    # 历史 Instruct continued-SFT 结果
tests/              # 评测器单元测试
outputs/            # 本地模型 checkpoint，不提交到 Git
```

## 标准实验流程

请进入项目目录并激活 `sft` Conda 环境：

```bash
cd /home/sakura/projects/llm/post-training-math
conda activate sft
```

### 1. 对 Raw Base Model 进行 Smoke Test

第一次运行需要下载约 3.1 GB 的 Base Model。

```bash
python3 scripts/eval_sft_adapter.py \
  --base-model Qwen/Qwen2.5-Math-1.5B \
  --eval-split train_validation \
  --num-samples 20 \
  --max-new-tokens 1024 \
  --stop-after-answer-line \
  --output results/dev_math_base_15b_smoke20_tok1024.json
```

该步骤只评测 20 道题，用于确认以下环节可以正常工作：

- 模型与 tokenizer 加载；
- Qwen Chat Template；
- GPU 与 bfloat16 推理；
- GSM8K validation split；
- 答案提取、评分和 JSON 保存。

Smoke Test 只能检查评测管线，不用于得出最终性能结论。

### 2. 从 Raw Base Model 开始 LoRA-SFT

已经完成的 v1 和 v2 作为负结果对照保留。下一轮控制实验运行：

```bash
llamafactory-cli train configs/main/qwen25_math_15b_base_lora_sft_v3.yaml
```

主要训练参数如下：

- Base Model：`Qwen/Qwen2.5-Math-1.5B`
- 数据集：GSM8K train
- 微调方法：LoRA
- LoRA rank：8
- LoRA target：全部适合的线性层
- 有效 batch size：`1 × 8 = 8`
- 学习率：`1e-4`
- 训练轮数：1 epoch
- 随机种子：42
- 验证集比例：5%

训练和评测都使用 Qwen ChatML 格式。评测脚本同时将 Base Model 的
`<|endoftext|>` 和 ChatML 的 `<|im_end|>` 视为生成停止符，还可以在检测到完整
`#### <number>` 答案行后停止。每份结果都会记录生成是由 EOS、答案行还是最大
token 上限终止，避免把数学正确性与终止异常混为一谈。

### 3. 在 Validation Split 上选择 Checkpoint

训练配置使用：

```yaml
val_size: 0.05
seed: 42
```

`train_validation` 会复现 LLaMA-Factory 训练时使用的同一组 374 条 validation 样本。

首先评测完整的 Raw Base validation baseline：

```bash
python3 scripts/eval_sft_adapter.py \
  --base-model Qwen/Qwen2.5-Math-1.5B \
  --eval-split train_validation \
  --max-new-tokens 1024 \
  --stop-after-answer-line \
  --output results/dev_math_base_15b_v3.json
```

然后评测每个 LoRA checkpoint，例如：

```bash
python3 scripts/eval_sft_adapter.py \
  --base-model Qwen/Qwen2.5-Math-1.5B \
  --adapter outputs/qwen25_math_15b_base_lora_sft_v3/checkpoint-CANDIDATE \
  --eval-split train_validation \
  --max-new-tokens 1024 \
  --stop-after-answer-line \
  --output results/dev_math_base_sft_v3_15b_ckptCANDIDATE.json
```

将 `checkpoint-CANDIDATE` 替换为实际 checkpoint。20 题 smoke 只用于诊断；
最终 checkpoint 必须根据完整 validation 数值准确率选择，不能只看 token-level
`eval_loss`。

注意：

- 不要只根据 token-level `eval_loss` 选择模型；
- 不要使用 GSM8K test set 选择 checkpoint；
- test set 应尽量只用于最终评测，避免测试集信息泄漏。

### 4. 进行最终完整 Test 对比

不指定 `--num-samples` 时，脚本会评测 GSM8K test 的全部 1,319 道题。

Raw Base Model：

```bash
python3 scripts/eval_sft_adapter.py \
  --base-model Qwen/Qwen2.5-Math-1.5B \
  --eval-split test \
  --max-new-tokens 1024 \
  --stop-after-answer-line \
  --output results/final_math_base_15b_test.json
```

最佳 SFT checkpoint：

```bash
python3 scripts/eval_sft_adapter.py \
  --base-model Qwen/Qwen2.5-Math-1.5B \
  --adapter outputs/qwen25_math_15b_base_lora_sft_v3/checkpoint-BEST \
  --eval-split test \
  --max-new-tokens 1024 \
  --stop-after-answer-line \
  --output results/final_math_base_sft_15b_test.json
```

其中 `checkpoint-BEST` 应替换为 validation 上选出的最佳 checkpoint。

### 5. 生成 Base 与 SFT 配对分析

```bash
python3 scripts/compare_base_sft.py \
  --base results/final_math_base_15b_test.json \
  --sft results/final_math_base_sft_15b_test.json \
  --output results/final_math_base_sft_analysis.json
```

对比报告包含：

- 宽松数值准确率；
- 严格 `#### <answer>` 准确率；
- Base/SFT 样本级正确性转移矩阵；
- 输出格式遵循率；
- 回复长度统计；
- 精确双侧 McNemar 检验。

宽松准确率用于判断模型最终数值结论是否正确，即使模型没有严格遵循指定格式；严格准确率则同时要求答案正确并符合 `#### <answer>` 格式。分别报告二者，可以避免把“推理能力”和“格式遵循能力”混为一谈。

## 历史 Instruct → Continued-SFT 实验

在已经被反复查看的前 100 道 GSM8K test 样本上：

- 宽松数值准确率：73% → 49%；
- 严格 `####` 准确率：21% → 49%；
- 可解析的 `#### <number>` 格式遵循率：25% → 100%。

该结果不能作为当前主任务的 Base→SFT 结论，因为初始模型是已经经过后训练的 Instruct Model。同时，这 100 道 test 样本已被用于多次比较，应视为探索性数据。

但该实验仍然有价值，它表明：

> Continued SFT 可以让模型更好地模仿目标数据的输出风格，却可能破坏 Instruct Model 已经具备的部分推理能力。

因此该实验及其结果被完整保留，作为附加消融和负结果分析。

## 脚本说明

- `scripts/eval_sft_adapter.py`：使用同一管线评测 Base Model 或 LoRA adapter。
- `scripts/evaluation_utils.py`：统一实现 GSM8K 答案提取、数值归一化和评分。
- `scripts/rescore.py`：使用当前统一评分器重新评分历史 JSON，不覆盖旧文件。
- `scripts/compare_base_sft.py`：对相同题目的 Base/SFT 结果进行严格配对分析。
- `scripts/prepare_formal_sft_data.py`：生成正式 Alpaca 格式 GSM8K SFT 数据。
- `scripts/prepare_sft_data.py`：生成早期 pipeline smoke/full 数据。

## 测试

运行评分器测试：

```bash
python3 -m unittest discover -s tests -v
```

当前测试覆盖：

- GSM8K ground truth 提取；
- 整数和小数归一化；
- `\boxed{}` 答案；
- 宽松与严格准确率的区别；
- 结论句中的答案提取；
- 无答案输出；
- 精确 McNemar 检验。

## Git 与实验管理建议

- 配置、脚本、README 和小型结果文件应提交到 Git；
- `outputs/` 中的模型 checkpoint 默认不提交；
- 每次实验使用不同的 `output_dir` 和结果文件名；
- 不要在没有 `--overwrite` 的情况下覆盖正式评测结果；
- 在实验报告中记录模型、checkpoint、数据划分、seed、Prompt、生成参数和评分器版本。
