import argparse
import json
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

from evaluation_benchmarks import (
    BENCHMARKS,
    adapt_benchmark_sample,
    get_benchmark_spec,
    resolve_dataset_identity,
    resolve_source_split,
)
from evaluation_utils import (
    SYSTEM_PROMPT,
    follows_answer_format,
    has_completed_answer_line,
    score_response,
)
from experiment_protocol import sha256_file


class CompletedAnswerLineStoppingCriteria(StoppingCriteria):
    """Stop batch-size-one generation after a completed ``####`` answer line."""

    def __init__(self, tokenizer, prompt_length: int, tail_token_window: int = 256):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.tail_token_window = tail_token_window

    def __call__(self, input_ids, scores, **kwargs) -> torch.BoolTensor:
        decisions = []
        for sequence in input_ids:
            generated_start = max(
                self.prompt_length,
                int(sequence.shape[-1]) - self.tail_token_window,
            )
            generated_tail = self.tokenizer.decode(
                sequence[generated_start:],
                skip_special_tokens=True,
            )
            decisions.append(has_completed_answer_line(generated_tail))
        return torch.tensor(decisions, dtype=torch.bool, device=input_ids.device)


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


def resolve_terminators(tokenizer) -> list[int]:
    """Stop on both the model EOS and Qwen ChatML's assistant terminator."""
    terminators = []
    if tokenizer.eos_token_id is not None:
        terminators.append(tokenizer.eos_token_id)

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id >= 0 and im_end_id not in terminators:
        terminators.append(im_end_id)

    if not terminators:
        raise ValueError("Tokenizer has no usable EOS token")
    return terminators


def load_local_gsm8k_dataset(path: Path) -> Dataset:
    with path.open("r", encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list) or not records:
        raise ValueError("local dataset must be a non-empty JSON list")
    converted = []
    for local_index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"local record {local_index} is not an object")
        question = record.get("instruction")
        answer = record.get("output")
        source_index = record.get("source_index")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"local record {local_index} has no instruction")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"local record {local_index} has no output")
        if not isinstance(source_index, int):
            raise ValueError(f"local record {local_index} has no integer source_index")
        converted.append(
            {
                "question": question,
                "answer": answer,
                "source_index": source_index,
            }
        )
    if len({record["source_index"] for record in converted}) != len(converted):
        raise ValueError("local dataset contains duplicate source_index values")
    return Dataset.from_list(converted)


def load_evaluation_dataset(args: argparse.Namespace, benchmark_spec):
    local_dataset_file = getattr(args, "local_dataset_file", None)
    if local_dataset_file is not None:
        dataset = load_local_gsm8k_dataset(local_dataset_file)
        source_split = f"local:{args.local_split_role}"
    else:
        source_split = resolve_source_split(benchmark_spec, args.eval_split)
        dataset = load_dataset(args.dataset_name, args.dataset_config, split=source_split)
        dataset = dataset.add_column("source_index", list(range(len(dataset))))

    if local_dataset_file is None and args.eval_split == "train_validation":
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
    parser = argparse.ArgumentParser(
        description="Evaluate a base or LoRA model on a numeric reasoning benchmark."
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--benchmark",
        choices=tuple(sorted(BENCHMARKS)),
        default="gsm8k",
        help="Benchmark adapter and dataset defaults (default: gsm8k).",
    )
    parser.add_argument(
        "--dataset-name",
        help="Override the benchmark's default Hugging Face dataset name.",
    )
    parser.add_argument(
        "--dataset-config",
        help="Override the benchmark's default Hugging Face dataset config.",
    )
    parser.add_argument(
        "--eval-split",
        choices=("test", "train_validation"),
        default="test",
        help="Use test, or reproduce LLaMA-Factory's held-out train validation split.",
    )
    parser.add_argument(
        "--local-dataset-file",
        type=Path,
        help=(
            "Evaluate a generated confirmatory JSON partition instead of loading "
            "a Hugging Face split. GSM8K-format records only."
        ),
    )
    parser.add_argument(
        "--local-split-role",
        choices=("dev_select", "dev_audit"),
        help="Required provenance label when --local-dataset-file is used.",
    )
    parser.add_argument("--validation-size", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--num-samples",
        type=int,
        help="Number of examples. Omit to evaluate the entire selected split.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument(
        "--stop-after-answer-line",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop after a completed #### answer line (enabled by default).",
    )
    parser.add_argument("--show-responses", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    benchmark_spec = get_benchmark_spec(args.benchmark)
    if args.local_dataset_file is not None:
        if args.benchmark != "gsm8k":
            parser.error("--local-dataset-file currently supports only --benchmark gsm8k")
        if args.local_split_role is None:
            parser.error("--local-split-role is required with --local-dataset-file")
        if not args.local_dataset_file.is_file():
            parser.error(f"local dataset does not exist: {args.local_dataset_file}")
        if args.eval_split != "test":
            parser.error("local partitions cannot be combined with --eval-split")
    elif args.local_split_role is not None:
        parser.error("--local-split-role requires --local-dataset-file")
    try:
        resolve_source_split(benchmark_spec, args.eval_split)
    except ValueError as error:
        parser.error(str(error))
    args.dataset_name, args.dataset_config = resolve_dataset_identity(
        benchmark_spec,
        args.dataset_name,
        args.dataset_config,
    )
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
    terminators = resolve_terminators(tokenizer)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
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
    dataset, source_split = load_evaluation_dataset(args, benchmark_spec)
    print(
        f"Benchmark: {benchmark_spec.name}; evaluation split: {args.eval_split}; "
        f"samples: {len(dataset)}"
    )

    correct = 0
    strict_correct = 0
    format_compliant = 0
    hit_max_new_tokens_count = 0
    termination_counts = {
        "eos": 0,
        "answer_line": 0,
        "max_new_tokens": 0,
        "other": 0,
    }
    results = []

    for idx, sample in enumerate(dataset):
        try:
            question, ground_truth, sample_metadata = adapt_benchmark_sample(
                sample,
                benchmark_spec.name,
            )
        except ValueError as error:
            raise ValueError(
                f"Cannot adapt {benchmark_spec.name} sample at source index "
                f"{sample['source_index']}: {error}"
            ) from error

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
        prompt_length = int(inputs["input_ids"].shape[1])
        stopping_criteria = StoppingCriteriaList()
        if args.stop_after_answer_line:
            stopping_criteria.append(
                CompletedAnswerLineStoppingCriteria(tokenizer, prompt_length)
            )

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
                eos_token_id=terminators,
                stopping_criteria=stopping_criteria,
            )

        generated_ids = outputs[0][prompt_length:]
        num_generated_tokens = int(generated_ids.shape[-1])
        hit_max_new_tokens = num_generated_tokens >= args.max_new_tokens
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        ended_with_eos = (
            num_generated_tokens > 0 and int(generated_ids[-1].item()) in terminators
        )
        completed_answer_line = has_completed_answer_line(response)
        if ended_with_eos:
            termination_reason = "eos"
        elif args.stop_after_answer_line and completed_answer_line:
            termination_reason = "answer_line"
        elif hit_max_new_tokens:
            termination_reason = "max_new_tokens"
        else:
            termination_reason = "other"

        score = score_response(response, ground_truth)
        compliant = follows_answer_format(response)

        if score["correct"]:
            correct += 1
        if score["strict_correct"]:
            strict_correct += 1
        if compliant:
            format_compliant += 1
        if hit_max_new_tokens:
            hit_max_new_tokens_count += 1
        termination_counts[termination_reason] += 1

        print(
            f"[{idx + 1}/{len(dataset)}] source={sample['source_index']} "
            f"pred={score['predicted_answer_normalized']} gt={score['ground_truth_normalized']} "
            f"correct={score['correct']} tokens={num_generated_tokens} "
            f"stop={termination_reason}"
        )
        if args.show_responses:
            print(response)

        results.append(
            {
                "source_index": sample["source_index"],
                "question": question,
                "response": response,
                "ground_truth": ground_truth,
                "num_generated_tokens": num_generated_tokens,
                "hit_max_new_tokens": hit_max_new_tokens,
                "termination_reason": termination_reason,
                **score,
                "format_compliant": compliant,
                **{
                    key: value
                    for key, value in sample_metadata.items()
                    if value is not None
                },
            }
        )

    accuracy = correct / len(dataset)
    strict_accuracy = strict_correct / len(dataset)
    format_rate = format_compliant / len(dataset)
    payload = {
        "schema_version": 2,
        "evaluation_version": benchmark_spec.evaluation_version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.base_model,
        "adapter": args.adapter,
        "benchmark": benchmark_spec.name,
        "dataset": {
            "name": args.dataset_name,
            "config": args.dataset_config,
            "source_split": source_split,
            "evaluation_split": args.eval_split,
            "local_file": (
                str(args.local_dataset_file)
                if args.local_dataset_file is not None
                else None
            ),
            "local_file_sha256": (
                sha256_file(args.local_dataset_file)
                if args.local_dataset_file is not None
                else None
            ),
            "local_split_role": args.local_split_role,
            "validation_size": args.validation_size if args.eval_split == "train_validation" else None,
            "seed": args.seed,
            "start_index": args.start_index,
        },
        "prompt": {"system": SYSTEM_PROMPT, "chat_template": "tokenizer.apply_chat_template"},
        "generation": {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "dtype": str(dtype).removeprefix("torch."),
            "eos_token_ids": terminators,
            "pad_token_id": pad_token_id,
            "stop_after_completed_answer_line": args.stop_after_answer_line,
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
        "hit_max_new_tokens_count": hit_max_new_tokens_count,
        "hit_max_new_tokens_rate": hit_max_new_tokens_count / len(dataset),
        "termination_counts": termination_counts,
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Final: {correct}/{len(dataset)} = {accuracy:.2%}")
    print(f"Strict #### accuracy: {strict_correct}/{len(dataset)} = {strict_accuracy:.2%}")
    print(f"Format compliance: {format_compliant}/{len(dataset)} = {format_rate:.2%}")
    print(
        f"Hit max new tokens: {hit_max_new_tokens_count}/{len(dataset)} = "
        f"{hit_max_new_tokens_count / len(dataset):.2%}"
    )
    print(
        "Termination: "
        + ", ".join(f"{key}={value}" for key, value in termination_counts.items())
    )
    print("Saved to:", args.output)


if __name__ == "__main__":
    main()
