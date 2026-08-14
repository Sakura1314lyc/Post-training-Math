import argparse
import json
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation_utils import (
    EVALUATION_VERSION,
    SYSTEM_PROMPT,
    extract_ground_truth,
    follows_answer_format,
    score_response,
)


def package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def resolve_dtype(name: str, device: str) -> torch.dtype:
    if name == "float32" or device == "cpu":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_evaluation_dataset(args: argparse.Namespace):
    source_split = "train" if args.eval_split == "train_validation" else "test"
    dataset = load_dataset(args.dataset_name, args.dataset_config, split=source_split)
    dataset = dataset.add_column("source_index", list(range(len(dataset))))

    if args.eval_split == "train_validation":
        dataset = dataset.train_test_split(
            test_size=args.validation_size,
            seed=args.seed,
        )["test"]

    if args.start_index >= len(dataset):
        raise ValueError(
            f"start-index {args.start_index} is outside the selected split "
            f"(size={len(dataset)})"
        )

    end = len(dataset)
    if args.num_samples is not None:
        end = min(end, args.start_index + args.num_samples)
    return dataset.select(range(args.start_index, end)), source_split


def main():
    parser = argparse.ArgumentParser(description="Evaluate a base or LoRA model on GSM8K.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-name", default="openai/gsm8k")
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument(
        "--eval-split",
        choices=("test", "train_validation"),
        default="test",
        help="Use test, or reproduce LLaMA-Factory's held-out train validation split.",
    )
    parser.add_argument("--validation-size", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--num-samples",
        type=int,
        help="Number of examples. Omit to evaluate the entire selected split.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument("--show-responses", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    if args.num_samples is not None and args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    if args.start_index < 0:
        parser.error("--start-index must be non-negative")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.eval_split == "train_validation" and not 0 < args.validation_size < 1:
        parser.error("--validation-size must be between 0 and 1")
    if args.output.exists() and not args.overwrite:
        parser.error(f"output already exists: {args.output} (pass --overwrite to replace it)")

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = resolve_dtype(args.dtype, device)
    print(f"Device: {device}; dtype: {dtype}")

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype)

    if args.adapter:
        print("Loading LoRA adapter:", args.adapter)
        model = PeftModel.from_pretrained(base_model, args.adapter)
    else:
        print("No adapter. Evaluating base model.")
        model = base_model

    model = model.to(device)
    model.eval()

    print("Loading evaluation data...")
    dataset, source_split = load_evaluation_dataset(args)
    print(f"Evaluation split: {args.eval_split}; samples: {len(dataset)}")

    correct = 0
    strict_correct = 0
    format_compliant = 0
    results = []

    for idx, sample in enumerate(dataset):
        question = sample["question"]
        ground_truth = extract_ground_truth(sample["answer"])
        if ground_truth is None:
            raise ValueError(f"Cannot extract ground truth at source index {sample['source_index']}")

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {"role": "user", "content": question},
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        score = score_response(response, ground_truth)
        compliant = follows_answer_format(response)

        if score["correct"]:
            correct += 1
        if score["strict_correct"]:
            strict_correct += 1
        if compliant:
            format_compliant += 1

        print(
            f"[{idx + 1}/{len(dataset)}] source={sample['source_index']} "
            f"pred={score['predicted_answer_normalized']} gt={score['ground_truth_normalized']} "
            f"correct={score['correct']}"
        )
        if args.show_responses:
            print(response)

        results.append({
            "source_index": sample["source_index"],
            "question": question,
            "response": response,
            "ground_truth": ground_truth,
            **score,
            "format_compliant": compliant,
        })

    accuracy = correct / len(dataset)
    strict_accuracy = strict_correct / len(dataset)
    format_rate = format_compliant / len(dataset)
    payload = {
        "schema_version": 2,
        "evaluation_version": EVALUATION_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.base_model,
        "adapter": args.adapter,
        "dataset": {
            "name": args.dataset_name,
            "config": args.dataset_config,
            "source_split": source_split,
            "evaluation_split": args.eval_split,
            "validation_size": args.validation_size if args.eval_split == "train_validation" else None,
            "seed": args.seed,
            "start_index": args.start_index,
        },
        "prompt": {"system": SYSTEM_PROMPT, "chat_template": "tokenizer.apply_chat_template"},
        "generation": {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "dtype": str(dtype).removeprefix("torch."),
        },
        "environment": {
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
            "peft": package_version("peft"),
            "datasets": package_version("datasets"),
        },
        "num_samples": len(dataset),
        "correct": correct,
        "accuracy": accuracy,
        "strict_correct": strict_correct,
        "strict_accuracy": strict_accuracy,
        "format_compliant": format_compliant,
        "format_compliance_rate": format_rate,
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Final: {correct}/{len(dataset)} = {accuracy:.2%}")
    print(f"Strict #### accuracy: {strict_correct}/{len(dataset)} = {strict_accuracy:.2%}")
    print(f"Format compliance: {format_compliant}/{len(dataset)} = {format_rate:.2%}")
    print("Saved to:", args.output)


if __name__ == "__main__":
    main()
