"""Run memory-conscious on-policy distillation with TRL GKD.

The student is the existing trainable LoRA adapter. The frozen teacher is loaded
in NF4 by default so that both 1.5B models fit on a single 8 GiB GPU. The
LLaMA-Factory validation split is excluded from all training samples.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GKDConfig, GKDTrainer


DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Math-1.5B"
DEFAULT_ADAPTER = Path("outputs/qwen25_math_15b_base_lora_sft_v7/checkpoint-888")
DEFAULT_TEACHER_MODEL = "Qwen/Qwen2.5-Math-1.5B-Instruct"
DEFAULT_DATASET = Path("data/gsm8k_sft_clean.json")
DEFAULT_OUTPUT_DIR = Path("outputs/opd/qwen25_math_15b_gkd_smoke")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small TRL GKD training job for on-policy distillation."
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--teacher-model", default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument(
        "--all-training-samples",
        action="store_true",
        help="Use every sample outside the fixed validation split.",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.05,
        help="Held-out fraction used by the SFT run; these samples are excluded.",
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument(
        "--lmbda",
        type=float,
        default=1.0,
        help="Probability of using student-generated rollouts; 1.0 is fully on-policy.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="Generalized JSD interpolation coefficient in [0, 1].",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--save-steps",
        type=int,
        default=0,
        help="Save a resumable adapter checkpoint every N optimizer steps; 0 disables it.",
    )
    parser.add_argument("--save-total-limit", type=int, default=5)
    parser.add_argument(
        "--save-final-adapter",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--overwrite-output-dir",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--teacher-load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--use-liger-kernel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use fused GKD loss after installing liger-kernel.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_samples <= 0 and not args.all_training_samples:
        raise ValueError("--num-samples must be positive")
    if not 0.0 <= args.validation_size < 1.0:
        raise ValueError("--validation-size must be in [0, 1)")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.max_length <= 0:
        raise ValueError("--max-length must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if not 0.0 <= args.lmbda <= 1.0:
        raise ValueError("--lmbda must be between 0 and 1")
    if not 0.0 <= args.beta <= 1.0:
        raise ValueError("--beta must be between 0 and 1")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be positive")
    if args.save_steps < 0:
        raise ValueError("--save-steps must be non-negative")
    if args.save_total_limit <= 0:
        raise ValueError("--save-total-limit must be positive")
    if not args.adapter.is_dir():
        raise FileNotFoundError(f"adapter directory does not exist: {args.adapter}")
    if not args.dataset.is_file():
        raise FileNotFoundError(f"dataset file does not exist: {args.dataset}")
    if args.resume_from_checkpoint is not None and not args.resume_from_checkpoint.is_dir():
        raise FileNotFoundError(
            f"resume checkpoint does not exist: {args.resume_from_checkpoint}"
        )


def load_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list) or not records:
        raise ValueError("dataset must be a non-empty JSON list")
    return records


def split_source_indices(
    num_records: int, validation_size: float, seed: int
) -> tuple[list[int], list[int]]:
    """Reproduce LLaMA-Factory's non-streaming train/validation split."""
    if num_records <= 0:
        raise ValueError("num_records must be positive")
    if not 0.0 <= validation_size < 1.0:
        raise ValueError("validation_size must be in [0, 1)")
    if validation_size == 0.0:
        return list(range(num_records)), []

    index_dataset = Dataset.from_dict({"source_index": list(range(num_records))})
    split = index_dataset.train_test_split(test_size=validation_size, seed=seed)
    return list(split["train"]["source_index"]), list(split["test"]["source_index"])


def build_conversational_records(
    records: list[dict],
    num_samples: int,
    seed: int,
    candidate_source_indices: list[int] | None = None,
) -> list[dict]:
    """Select records deterministically and convert Alpaca fields to ChatML messages."""
    candidates = (
        list(range(len(records)))
        if candidate_source_indices is None
        else list(candidate_source_indices)
    )
    if any(index < 0 or index >= len(records) for index in candidates):
        raise ValueError("candidate source index is outside the dataset")
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidate source indices contain duplicates")
    if num_samples > len(candidates):
        raise ValueError(
            f"--num-samples {num_samples} exceeds training split size {len(candidates)}"
        )

    selected_indices = random.Random(seed).sample(candidates, num_samples)
    converted = []
    for source_index in selected_indices:
        record = records[source_index]
        instruction = record.get("instruction")
        response = record.get("output")
        system = record.get("system")
        extra_input = record.get("input", "")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"record {source_index} has no valid instruction")
        if not isinstance(response, str) or not response.strip():
            raise ValueError(f"record {source_index} has no valid output")
        if not isinstance(system, str) or not system.strip():
            raise ValueError(f"record {source_index} has no valid system prompt")
        if not isinstance(extra_input, str):
            raise ValueError(f"record {source_index} has a non-string input")

        user_content = instruction.strip()
        if extra_input.strip():
            user_content = f"{user_content}\n\n{extra_input.strip()}"
        converted.append(
            {
                "source_index": source_index,
                "messages": [
                    {"role": "system", "content": system.strip()},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": response.strip()},
                ],
            }
        )
    return converted


def resolve_terminators(tokenizer) -> list[int]:
    """Use both the base EOS and Qwen ChatML's end-of-message token."""
    terminators = []
    if tokenizer.eos_token_id is not None:
        terminators.append(int(tokenizer.eos_token_id))

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id >= 0 and im_end_id not in terminators:
        terminators.append(im_end_id)
    if not terminators:
        raise ValueError("tokenizer has no usable EOS token")
    return terminators


def cuda_memory_gib() -> dict[str, float]:
    return {
        "allocated_gib": torch.cuda.memory_allocated() / 1024**3,
        "reserved_gib": torch.cuda.memory_reserved() / 1024**3,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "total_gib": torch.cuda.get_device_properties(0).total_memory / 1024**3,
    }


def print_cuda_memory(label: str) -> dict[str, float]:
    memory = cuda_memory_gib()
    print(
        f"{label}: allocated={memory['allocated_gib']:.2f} GiB, "
        f"reserved={memory['reserved_gib']:.2f} GiB, "
        f"peak={memory['peak_allocated_gib']:.2f} GiB, "
        f"total={memory['total_gib']:.2f} GiB"
    )
    return memory


def make_teacher_quantization_config(enabled: bool):
    if not enabled:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    temporary_path.replace(path)


def build_gkd_config_kwargs(args: argparse.Namespace) -> dict:
    """Return TRL arguments shared by smoke and pilot runs."""
    return {
        "output_dir": str(args.output_dir),
        "max_steps": args.max_steps,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": "constant",
        "warmup_steps": 0,
        "bf16": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "use_cache": False,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "lmbda": args.lmbda,
        "beta": args.beta,
        "seq_kd": False,
        "use_liger_kernel": args.use_liger_kernel,
        "logging_steps": 1,
        "logging_first_step": True,
        "save_strategy": "steps" if args.save_steps > 0 else "no",
        "save_steps": max(1, args.save_steps),
        "save_total_limit": args.save_total_limit,
        "eval_strategy": "no",
        "report_to": "none",
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": False,
        "seed": args.seed,
        "data_seed": args.seed,
    }


def build_gkd_config(args: argparse.Namespace) -> GKDConfig:
    """Build version-checked TRL arguments shared by smoke and pilot runs."""
    return GKDConfig(**build_gkd_config_kwargs(args))


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this OPD/GKD script")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected GPU does not report bfloat16 support")
    if args.use_liger_kernel:
        try:
            import liger_kernel  # noqa: F401
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "--use-liger-kernel requires the liger-kernel package"
            ) from error

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("Loading and converting OPD data...")
    records = load_records(args.dataset)
    training_source_indices, validation_source_indices = split_source_indices(
        len(records), args.validation_size, args.split_seed
    )
    selected_sample_count = (
        len(training_source_indices) if args.all_training_samples else args.num_samples
    )
    conversational_records = build_conversational_records(
        records,
        selected_sample_count,
        args.seed,
        candidate_source_indices=training_source_indices,
    )
    train_dataset = Dataset.from_list(conversational_records)
    source_indices = list(train_dataset["source_index"])
    validation_source_index_set = set(validation_source_indices)
    if any(index in validation_source_index_set for index in source_indices):
        raise RuntimeError("training data overlaps the fixed validation split")
    print(
        f"Dataset split: train={len(training_source_indices)}, "
        f"validation={len(validation_source_indices)}"
    )
    print("Selected source indices:", source_indices)

    output_contents = (
        {path.name for path in args.output_dir.iterdir()}
        if args.output_dir.exists()
        else set()
    )
    initialization_only = output_contents == {"run_manifest.json"}
    if initialization_only:
        print("Reusing output directory from an initialization-only failed run.")
    if (
        output_contents
        and not initialization_only
        and not args.overwrite_output_dir
        and args.resume_from_checkpoint is None
    ):
        raise FileExistsError(
            f"output directory is not empty: {args.output_dir} "
            "(choose a new directory or pass --overwrite-output-dir)"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "status": "initialized",
        "mode": "on_policy_distillation_gkd",
        "base_model": args.base_model,
        "adapter": str(args.adapter),
        "teacher_model": args.teacher_model,
        "teacher_load_in_4bit": args.teacher_load_in_4bit,
        "dataset": str(args.dataset),
        "dataset_num_records": len(records),
        "validation_size": args.validation_size,
        "split_seed": args.split_seed,
        "training_split_size": len(training_source_indices),
        "validation_split_size": len(validation_source_indices),
        "validation_source_indices": validation_source_indices,
        "selected_training_source_indices": source_indices,
        "num_samples": selected_sample_count,
        "max_steps": args.max_steps,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "learning_rate": args.learning_rate,
        "temperature": args.temperature,
        "lmbda": args.lmbda,
        "beta": args.beta,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "save_final_adapter": args.save_final_adapter,
        "use_liger_kernel": args.use_liger_kernel,
        "seed": args.seed,
        "resume_from_checkpoint": (
            str(args.resume_from_checkpoint)
            if args.resume_from_checkpoint is not None
            else None
        ),
    }
    manifest_path = args.output_dir / "run_manifest.json"
    write_json_atomic(manifest_path, run_manifest)
    print("Saved run manifest to:", manifest_path)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    if tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError("student and teacher tokenizers do not have identical vocabularies")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    terminators = resolve_terminators(tokenizer)
    print("Generation terminators:", terminators)
    del teacher_tokenizer

    print("Loading BF16 student and trainable LoRA adapter...")
    student_base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
    )
    student = PeftModel.from_pretrained(
        student_base,
        args.adapter,
        is_trainable=True,
    )
    student.config.use_cache = False
    student.enable_input_require_grads()
    trainable_parameters = [
        (name, parameter)
        for name, parameter in student.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("the student has no trainable adapter parameters")
    student.print_trainable_parameters()

    tracked_name, tracked_parameter = trainable_parameters[0]
    tracked_before = tracked_parameter.detach().float().cpu().clone()

    print("Loading frozen teacher...")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        quantization_config=make_teacher_quantization_config(
            args.teacher_load_in_4bit
        ),
        device_map={"": 0},
    )
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    memory_after_load = print_cuda_memory("After model loading")

    training_args = build_gkd_config(args)

    trainer = GKDTrainer(
        model=student,
        teacher_model=teacher,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.generation_config.eos_token_id = terminators
    trainer.generation_config.pad_token_id = tokenizer.pad_token_id

    print(
        "Starting OPD/GKD training: "
        f"steps={args.max_steps}, lmbda={args.lmbda}, beta={args.beta}, "
        f"max_new_tokens={args.max_new_tokens}"
    )
    try:
        train_result = trainer.train(
            resume_from_checkpoint=(
                str(args.resume_from_checkpoint)
                if args.resume_from_checkpoint is not None
                else None
            )
        )
    except torch.OutOfMemoryError:
        print(
            "CUDA OOM during the real GKD step. Keep this result: the static load "
            "fits, but the unfused vocabulary loss does not. The next option is "
            "liger-kernel with --use-liger-kernel."
        )
        raise

    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    final_adapter_path = None
    if args.save_final_adapter:
        final_adapter_dir = args.output_dir / "final_adapter"
        trainer.save_model(str(final_adapter_dir))
        tokenizer.save_pretrained(final_adapter_dir)
        final_adapter_path = str(final_adapter_dir)
        print("Saved final adapter to:", final_adapter_dir)
    memory_after_train = print_cuda_memory("After training")

    tracked_after = tracked_parameter.detach().float().cpu()
    parameter_max_abs_delta = float((tracked_after - tracked_before).abs().max().item())
    parameter_updated = parameter_max_abs_delta > 0.0
    metrics_are_finite = all(
        not isinstance(value, float) or math.isfinite(value)
        for value in train_result.metrics.values()
    )

    summary = {
        "status": "success",
        "mode": "on_policy_distillation_gkd",
        "base_model": args.base_model,
        "adapter": str(args.adapter),
        "teacher_model": args.teacher_model,
        "teacher_load_in_4bit": args.teacher_load_in_4bit,
        "dataset": str(args.dataset),
        "dataset_num_records": len(records),
        "validation_size": args.validation_size,
        "split_seed": args.split_seed,
        "training_split_size": len(training_source_indices),
        "validation_split_size": len(validation_source_indices),
        "validation_source_indices": validation_source_indices,
        "source_indices": source_indices,
        "num_samples": selected_sample_count,
        "max_steps": args.max_steps,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "learning_rate": args.learning_rate,
        "temperature": args.temperature,
        "lmbda": args.lmbda,
        "beta": args.beta,
        "eos_token_ids": terminators,
        "use_liger_kernel": args.use_liger_kernel,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "final_adapter": final_adapter_path,
        "tracked_parameter": tracked_name,
        "tracked_parameter_max_abs_delta": parameter_max_abs_delta,
        "tracked_parameter_updated": parameter_updated,
        "metrics_are_finite": metrics_are_finite,
        "metrics": train_result.metrics,
        "memory_after_load": memory_after_load,
        "memory_after_train": memory_after_train,
    }
    summary_path = args.output_dir / "opd_run_summary.json"
    write_json_atomic(summary_path, summary)

    print("Tracked parameter:", tracked_name)
    print(f"Tracked parameter max |delta|: {parameter_max_abs_delta:.8g}")
    print("Parameter updated:", parameter_updated)
    print("Metrics finite:", metrics_are_finite)
    print("Saved run summary to:", summary_path)
    if trainer.state.global_step != args.max_steps:
        raise RuntimeError(
            f"expected {args.max_steps} optimizer steps, got {trainer.state.global_step}"
        )
    if not metrics_are_finite:
        raise RuntimeError("training produced a non-finite metric")


if __name__ == "__main__":
    main()
