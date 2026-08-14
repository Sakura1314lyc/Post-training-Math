# GSM8K Post-training Experiment

The main assignment track is now:

> `Qwen/Qwen2.5-1.5B` (raw pretrained base) → GSM8K LoRA-SFT → paired evaluation

Qwen describes this checkpoint as a pretraining-only base model and explicitly
recommends applying post-training such as SFT or RLHF. The
[official model card](https://huggingface.co/Qwen/Qwen2.5-1.5B) also provides a
Qwen chat template, so the pre-SFT and post-SFT models can use the same
serialized prompt.

The earlier `Qwen2.5-1.5B-Instruct` → continued-SFT experiment is preserved as
an auxiliary ablation. It shows that continued SFT can improve answer-format
following while reducing numerical reasoning accuracy.

## Experiment assets

- Main config: `configs/qwen25_15b_base_lora_sft_v1.yaml`
- Main initial checkpoint: `Qwen/Qwen2.5-1.5B`
- Training data: `data/gsm8k_sft_formal.json`
- Historical Instruct config: `configs/qwen25_15b_lora_sft_v1.yaml`
- Historical analysis: `results/base_sft_transition_analysis_v2.json`

The existing GSM8K SFT data is valid for both tracks and is intentionally
reused. Changing the initial checkpoint does not require regenerating targets.

## Canonical workflow

Run commands from this directory in the `sft` Conda environment:

```bash
cd /home/sakura/projects/llm/post-training-math
conda activate sft
```

### 1. Smoke-test the raw base baseline

The first run downloads the roughly 3.1 GB base checkpoint.

```bash
python3 scripts/eval_sft_adapter.py \
  --base-model Qwen/Qwen2.5-1.5B \
  --eval-split train_validation \
  --num-samples 20 \
  --output results/dev_raw_base_15b_smoke20.json
```

### 2. Train LoRA-SFT from the raw base

```bash
llamafactory-cli train configs/qwen25_15b_base_lora_sft_v1.yaml
```

This uses the Qwen ChatML template for both training and evaluation. The
evaluator stops on both the base model EOS token and `<|im_end|>`, which becomes
important after chat-style SFT.

### 3. Select a checkpoint on held-out validation

The config uses `val_size: 0.05` and `seed: 42`. `train_validation` reproduces
the same deterministic 374-example split used by LLaMA-Factory.

Evaluate the complete raw-base validation baseline:

```bash
python3 scripts/eval_sft_adapter.py \
  --base-model Qwen/Qwen2.5-1.5B \
  --eval-split train_validation \
  --output results/dev_raw_base_15b.json
```

Evaluate each saved adapter checkpoint:

```bash
python3 scripts/eval_sft_adapter.py \
  --base-model Qwen/Qwen2.5-1.5B \
  --adapter outputs/qwen25_15b_base_lora_sft_v1/checkpoint-100 \
  --eval-split train_validation \
  --output results/dev_raw_base_sft_15b_ckpt100.json
```

Repeat for checkpoint-200 onward and select by validation numerical accuracy,
not token-level eval loss. Do not select checkpoints on GSM8K test.

### 4. Run the final full-test comparison once

Omitting `--num-samples` evaluates all 1,319 GSM8K test examples.

```bash
python3 scripts/eval_sft_adapter.py \
  --base-model Qwen/Qwen2.5-1.5B \
  --eval-split test \
  --output results/final_raw_base_15b_test.json

python3 scripts/eval_sft_adapter.py \
  --base-model Qwen/Qwen2.5-1.5B \
  --adapter outputs/qwen25_15b_base_lora_sft_v1/checkpoint-BEST \
  --eval-split test \
  --output results/final_raw_base_sft_15b_test.json
```

### 5. Produce the paired report

```bash
python3 scripts/compare_base_sft.py \
  --base results/final_raw_base_15b_test.json \
  --sft results/final_raw_base_sft_15b_test.json \
  --output results/final_raw_base_sft_analysis.json
```

The report includes relaxed numerical accuracy, strict `####` accuracy, paired
transitions, format compliance, response length, and the exact McNemar test.

## Historical Instruct → continued-SFT result

On the repeatedly inspected first 100 test examples:

- relaxed numerical accuracy: 73% → 49%;
- strict `####` accuracy: 21% → 49%;
- parseable `#### <number>` compliance: 25% → 100%.

This is not the main assignment result. It is retained as evidence that
continued SFT can teach output style while degrading an instruction model's
existing reasoning behavior.

## Tests

```bash
python3 -m unittest discover -s tests -v
```
