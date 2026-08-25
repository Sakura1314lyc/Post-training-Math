"""Run memory-conscious GRPO from either a continued or merged SFT policy.

The confirmatory path loads an SFT-merged model and adds a fresh GRPO LoRA.
TRL can then disable only that fresh adapter to obtain the correct frozen SFT
reference for a non-zero KL penalty. Native Transformers generation is used so
the job remains suitable for a single 8 GiB GPU.
"""

from __future__ import annotations

import argparse
import ast
import math
import random
import re
from fractions import Fraction
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

import trl.import_utils as trl_import_utils


def normalize_trl_availability_flags() -> list[str]:
    """Work around TRL 0.24 with Transformers 5.x availability tuples.

    Transformers 5.x returns ``(available, version)`` from its private package
    probe even when the caller does not request a version. TRL 0.24 stores that
    tuple as a boolean, making unavailable optional packages look truthy and
    breaking GRPO imports. Normalize only those cached private flags; no package
    behavior changes when the flags are already booleans.
    """
    patched = []
    for name, value in vars(trl_import_utils).items():
        if (
            name.startswith("_")
            and name.endswith("_available")
            and isinstance(value, tuple)
        ):
            setattr(trl_import_utils, name, bool(value[0]))
            patched.append(name)
    return patched


PATCHED_TRL_FLAGS = normalize_trl_availability_flags()

from trl import GRPOConfig, GRPOTrainer  # noqa: E402

from evaluation_utils import (  # noqa: E402
    extract_ground_truth,
    follows_answer_format,
    score_response,
)
from experiment_protocol import (  # noqa: E402
    load_split_manifest,
    resolve_seed_args,
    resolved_seed,
    sha256_file,
)
from train_opd_gkd import (  # noqa: E402
    load_records,
    print_cuda_memory,
    resolve_terminators,
    split_source_indices,
    write_json_atomic,
)


DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Math-1.5B"
DEFAULT_ADAPTER = Path("outputs/qwen25_math_15b_base_lora_sft_v7/checkpoint-888")
DEFAULT_DATASET = Path("data/gsm8k_sft_clean.json")
DEFAULT_OUTPUT_DIR = Path("outputs/grpo/qwen25_math_15b_grpo_smoke")
POLICY_INITIALIZATIONS = ("continued_adapter", "merged_sft")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small exact-answer GRPO job from the final SFT adapter."
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument(
        "--policy-initialization",
        choices=POLICY_INITIALIZATIONS,
        default="continued_adapter",
        help=(
            "continued_adapter preserves legacy beta=0 runs; merged_sft adds a "
            "fresh GRPO LoRA to an already merged SFT model and supports KL."
        ),
    )
    parser.add_argument(
        "--merged-sft-model",
        type=Path,
        help="Directory produced by merge_sft_adapter.py (required for merged_sft).",
    )
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,v_proj,lm_head",
        help="Comma-separated target modules for the fresh GRPO LoRA.",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument(
        "--all-training-samples",
        action="store_true",
        help="Use every sample outside the fixed validation split.",
    )
    parser.add_argument("--validation-size", type=float, default=0.05)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        help=(
            "Frozen train/dev-select/dev-audit manifest. When provided, it "
            "replaces --validation-size and --split-seed."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-completion-length", type=int, default=128)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5.0e-6)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0, help="KL coefficient.")
    parser.add_argument("--accuracy-reward-weight", type=float, default=1.0)
    parser.add_argument("--format-reward-weight", type=float, default=0.1)
    parser.add_argument(
        "--arithmetic-consistency-reward-weight",
        type=float,
        default=0.0,
        help=(
            "Optional bonus for internally correct explicit numeric equations. "
            "This is an arithmetic proxy, not a proof of reasoning faithfulness."
        ),
    )
    parser.add_argument(
        "--mask-truncated-completions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--save-steps", type=int, default=0)
    parser.add_argument("--save-total-limit", type=int, default=5)
    parser.add_argument(
        "--save-final-adapter",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument(
        "--seed",
        type=int,
        help="Deprecated shorthand that sets all three independent seeds.",
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        help="Controls which records are sampled from the frozen training partition.",
    )
    parser.add_argument(
        "--training-seed",
        type=int,
        help="Controls trainer sampling and optimization initialization.",
    )
    parser.add_argument(
        "--generation-seed",
        type=int,
        help="Controls stochastic rollout generation.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_samples <= 0 and not args.all_training_samples:
        raise ValueError("--num-samples must be positive")
    if not 0.0 <= args.validation_size < 1.0:
        raise ValueError("--validation-size must be in [0, 1)")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.max_prompt_length <= 0 or args.max_completion_length <= 0:
        raise ValueError("prompt and completion lengths must be positive")
    if args.num_generations < 2:
        raise ValueError("--num-generations must be at least 2 for group-relative rewards")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be positive")
    if args.gradient_accumulation_steps % args.num_generations != 0:
        raise ValueError(
            "gradient accumulation must be divisible by --num-generations "
            "for one-process, batch-size-one GRPO"
        )
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if not 0.0 < args.top_p <= 1.0:
        raise ValueError("--top-p must be in (0, 1]")
    if args.beta < 0.0:
        raise ValueError("--beta must be non-negative")
    policy_initialization = getattr(args, "policy_initialization", "continued_adapter")
    if policy_initialization not in POLICY_INITIALIZATIONS:
        raise ValueError(f"unknown policy initialization: {policy_initialization}")
    if policy_initialization == "continued_adapter" and args.beta != 0.0:
        raise ValueError(
            "--beta must remain 0 for continued SFT-adapter GRPO; a nonzero KL "
            "would incorrectly use Raw Base as the PEFT reference"
        )
    if getattr(args, "lora_rank", 8) <= 0 or getattr(args, "lora_alpha", 16) <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    if policy_initialization == "merged_sft":
        merged_sft_model = getattr(args, "merged_sft_model", None)
        if merged_sft_model is None or not merged_sft_model.is_dir():
            raise FileNotFoundError(
                "--merged-sft-model must be an existing directory for merged_sft"
            )
    if args.accuracy_reward_weight <= 0:
        raise ValueError("--accuracy-reward-weight must be positive")
    if args.format_reward_weight < 0:
        raise ValueError("--format-reward-weight must be non-negative")
    if getattr(args, "arithmetic_consistency_reward_weight", 0.0) < 0:
        raise ValueError("--arithmetic-consistency-reward-weight must be non-negative")
    if args.save_steps < 0:
        raise ValueError("--save-steps must be non-negative")
    if args.save_total_limit <= 0:
        raise ValueError("--save-total-limit must be positive")
    split_manifest = getattr(args, "split_manifest", None)
    if split_manifest is not None and not split_manifest.is_file():
        raise FileNotFoundError(f"split manifest does not exist: {split_manifest}")
    if policy_initialization == "continued_adapter" and not args.adapter.is_dir():
        raise FileNotFoundError(f"adapter directory does not exist: {args.adapter}")
    if not args.dataset.is_file():
        raise FileNotFoundError(f"dataset file does not exist: {args.dataset}")
    if args.resume_from_checkpoint is not None and not args.resume_from_checkpoint.is_dir():
        raise FileNotFoundError(
            f"resume checkpoint does not exist: {args.resume_from_checkpoint}"
        )


def build_grpo_records(
    records: list[dict],
    num_samples: int,
    seed: int,
    candidate_source_indices: list[int] | None = None,
) -> list[dict]:
    """Select prompts deterministically and retain only numeric reward labels."""
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
        ground_truth = extract_ground_truth(response)
        if ground_truth is None:
            raise ValueError(f"record {source_index} has no parseable #### answer")

        user_content = instruction.strip()
        if extra_input.strip():
            user_content = f"{user_content}\n\n{extra_input.strip()}"
        converted.append(
            {
                "source_index": source_index,
                "prompt": [
                    {"role": "system", "content": system.strip()},
                    {"role": "user", "content": user_content},
                ],
                "ground_truth": ground_truth,
            }
        )
    return converted


def completion_to_text(completion: Any) -> str:
    """Normalize TRL standard or conversational completion structures."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        content = completion.get("content")
        return content if isinstance(content, str) else ""
    if isinstance(completion, list):
        for message in reversed(completion):
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
    return ""


def numeric_accuracy_reward(
    completions: list[Any], ground_truth: list[str], **_: Any
) -> list[float]:
    if len(completions) != len(ground_truth):
        raise ValueError("completions and ground_truth have different lengths")
    return [
        float(score_response(completion_to_text(completion), answer)["correct"])
        for completion, answer in zip(completions, ground_truth)
    ]


def strict_format_reward(completions: list[Any], **_: Any) -> list[float]:
    return [
        float(follows_answer_format(completion_to_text(completion)))
        for completion in completions
    ]


EQUATION_PATTERN = re.compile(
    r"(?<![\w.])([+\-]?(?:\d|\()[\d\s().+\-*/]*?)\s*=\s*"
    r"([+\-]?\d+(?:\.\d+)?)"
)


def evaluate_numeric_expression(expression: str) -> Fraction:
    """Safely evaluate a numeric arithmetic expression as an exact fraction."""
    tree = ast.parse(expression.strip(), mode="eval")

    def evaluate(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return Fraction(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
        raise ValueError("unsupported arithmetic expression")

    return evaluate(tree)


def arithmetic_consistency_score(response: str) -> float:
    """Return 1 only when all detected explicit numeric equations are correct."""
    checked = 0
    for match in EQUATION_PATTERN.finditer(response):
        try:
            left = evaluate_numeric_expression(match.group(1))
            right = Fraction(match.group(2))
        except (SyntaxError, ValueError, ZeroDivisionError):
            continue
        checked += 1
        if left != right:
            return 0.0
    return float(checked > 0)


def arithmetic_consistency_reward(completions: list[Any], **_: Any) -> list[float]:
    return [
        arithmetic_consistency_score(completion_to_text(completion))
        for completion in completions
    ]


def build_reward_spec(args: argparse.Namespace) -> tuple[list[Any], list[float]]:
    functions: list[Any] = [numeric_accuracy_reward, strict_format_reward]
    weights = [args.accuracy_reward_weight, args.format_reward_weight]
    arithmetic_weight = getattr(args, "arithmetic_consistency_reward_weight", 0.0)
    if arithmetic_weight > 0:
        functions.append(arithmetic_consistency_reward)
        weights.append(arithmetic_weight)
    return functions, weights


def ensure_trl_model_compatibility(model: Any) -> list[str]:
    """Restore model attributes assumed by TRL 0.24 but removed in Transformers 5.x."""
    patched = []
    try:
        model.warnings_issued
    except AttributeError:
        model.warnings_issued = {}
        patched.append("warnings_issued")
    return patched


def parse_lora_target_modules(value: str) -> list[str]:
    modules = [module.strip() for module in value.split(",") if module.strip()]
    if not modules:
        raise ValueError("--lora-target-modules must contain at least one module")
    if len(set(modules)) != len(modules):
        raise ValueError("--lora-target-modules contains duplicates")
    return modules


def snapshot_trainable_parameters(
    trainable_parameters: list[tuple[str, torch.nn.Parameter]],
) -> dict[str, torch.Tensor]:
    """Copy every trainable tensor to CPU for post-training drift diagnostics."""
    snapshots = {}
    for name, parameter in trainable_parameters:
        if name in snapshots:
            raise ValueError(f"duplicate trainable parameter name: {name}")
        snapshots[name] = parameter.detach().float().cpu().clone()
    if not snapshots:
        raise ValueError("cannot snapshot an empty trainable parameter list")
    return snapshots


def classify_trainable_parameter(name: str) -> tuple[str, str]:
    """Return the target module and LoRA matrix family encoded in a PEFT name."""
    match = re.search(r"\.([^.]+)\.(lora_[AB])(?:\.|$)", name)
    if match is None:
        return "other", "other"
    return match.group(1), match.group(2)


def summarize_trainable_parameter_drift(
    trainable_parameters: list[tuple[str, torch.nn.Parameter]],
    snapshots: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Summarize parameter changes globally and by target/LoRA matrix family."""
    current_names = {name for name, _ in trainable_parameters}
    snapshot_names = set(snapshots)
    if current_names != snapshot_names:
        raise ValueError(
            "trainable parameter names changed during training: "
            f"missing={sorted(snapshot_names - current_names)}, "
            f"unexpected={sorted(current_names - snapshot_names)}"
        )

    accumulators: dict[str, dict[str, float | int]] = {}

    def update(group: str, before: torch.Tensor, after: torch.Tensor) -> None:
        delta = after - before
        absolute_delta = delta.abs()
        maximum = float(absolute_delta.max().item()) if delta.numel() else 0.0
        accumulator = accumulators.setdefault(
            group,
            {
                "tensor_count": 0,
                "parameter_count": 0,
                "updated_tensor_count": 0,
                "absolute_delta_sum": 0.0,
                "squared_delta_sum": 0.0,
                "squared_before_sum": 0.0,
                "squared_after_sum": 0.0,
                "max_abs_delta": 0.0,
            },
        )
        accumulator["tensor_count"] += 1
        accumulator["parameter_count"] += delta.numel()
        accumulator["updated_tensor_count"] += int(maximum > 0.0)
        accumulator["absolute_delta_sum"] += float(absolute_delta.sum().item())
        accumulator["squared_delta_sum"] += float(delta.double().square().sum().item())
        accumulator["squared_before_sum"] += float(
            before.double().square().sum().item()
        )
        accumulator["squared_after_sum"] += float(after.double().square().sum().item())
        accumulator["max_abs_delta"] = max(
            float(accumulator["max_abs_delta"]), maximum
        )

    for name, parameter in trainable_parameters:
        before = snapshots[name]
        after = parameter.detach().float().cpu()
        if before.shape != after.shape:
            raise ValueError(
                f"trainable parameter shape changed for {name}: "
                f"before={tuple(before.shape)}, after={tuple(after.shape)}"
            )
        target, matrix = classify_trainable_parameter(name)
        for group in (
            "all",
            f"matrix:{matrix}",
            f"target:{target}",
            f"target_matrix:{target}.{matrix}",
        ):
            update(group, before, after)

    def finalize(accumulator: dict[str, float | int]) -> dict[str, Any]:
        parameter_count = int(accumulator["parameter_count"])
        before_l2 = math.sqrt(float(accumulator["squared_before_sum"]))
        delta_l2 = math.sqrt(float(accumulator["squared_delta_sum"]))
        return {
            "tensor_count": int(accumulator["tensor_count"]),
            "parameter_count": parameter_count,
            "updated_tensor_count": int(accumulator["updated_tensor_count"]),
            "max_abs_delta": float(accumulator["max_abs_delta"]),
            "mean_abs_delta": (
                float(accumulator["absolute_delta_sum"]) / parameter_count
                if parameter_count
                else 0.0
            ),
            "l2_delta": delta_l2,
            "l2_before": before_l2,
            "l2_after": math.sqrt(float(accumulator["squared_after_sum"])),
            "relative_l2_delta": delta_l2 / before_l2 if before_l2 else None,
        }

    return {
        "all": finalize(accumulators.pop("all")),
        "by_lora_matrix": {
            key.removeprefix("matrix:"): finalize(value)
            for key, value in sorted(accumulators.items())
            if key.startswith("matrix:")
        },
        "by_target_module": {
            key.removeprefix("target:"): finalize(value)
            for key, value in sorted(accumulators.items())
            if key.startswith("target:")
        },
        "by_target_and_matrix": {
            key.removeprefix("target_matrix:"): finalize(value)
            for key, value in sorted(accumulators.items())
            if key.startswith("target_matrix:")
        },
    }


def policy_reference_description(args: argparse.Namespace) -> dict[str, Any]:
    initialization = getattr(args, "policy_initialization", "continued_adapter")
    if initialization == "merged_sft":
        return {
            "policy_initialization": initialization,
            "initial_policy": str(args.merged_sft_model),
            "trainable_adapter": "fresh_grpo_lora",
            "kl_reference": "merged_sft_policy",
            "kl_reference_correct": True,
        }
    return {
        "policy_initialization": initialization,
        "initial_policy": f"{args.base_model} + {args.adapter}",
        "trainable_adapter": "continued_sft_lora",
        "kl_reference": None,
        "kl_reference_correct": args.beta == 0.0,
    }


def load_policy_and_tokenizer(args: argparse.Namespace):
    """Load the legacy policy or build the confirmatory merged-SFT policy."""
    if args.policy_initialization == "merged_sft":
        model_source = str(args.merged_sft_model)
        tokenizer = AutoTokenizer.from_pretrained(model_source)
        merged_policy = AutoModelForCausalLM.from_pretrained(
            model_source,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": 0},
        )
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            bias="none",
            target_modules=parse_lora_target_modules(args.lora_target_modules),
        )
        return get_peft_model(merged_policy, lora_config), tokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
    )
    return PeftModel.from_pretrained(
        base_model,
        args.adapter,
        is_trainable=True,
    ), tokenizer


def build_grpo_config_kwargs(args: argparse.Namespace) -> dict:
    _, reward_weights = build_reward_spec(args)
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
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "num_generations": args.num_generations,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "beta": args.beta,
        "reward_weights": reward_weights,
        "scale_rewards": "group",
        "loss_type": "dapo",
        "mask_truncated_completions": args.mask_truncated_completions,
        "remove_unused_columns": False,
        "use_vllm": False,
        "logging_steps": 1,
        "logging_first_step": True,
        "save_strategy": "steps" if args.save_steps > 0 else "no",
        "save_steps": max(1, args.save_steps),
        "save_total_limit": args.save_total_limit,
        "eval_strategy": "no",
        "report_to": "none",
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": False,
        "seed": resolved_seed(args, "training_seed"),
        "data_seed": resolved_seed(args, "training_seed"),
        "disable_dropout": True,
    }


def build_grpo_config(args: argparse.Namespace) -> GRPOConfig:
    return GRPOConfig(**build_grpo_config_kwargs(args))


def main() -> None:
    args = parse_args()
    resolve_seed_args(args)
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this GRPO script")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected GPU does not report bfloat16 support")

    random.seed(args.training_seed)
    torch.manual_seed(args.training_seed)
    torch.cuda.manual_seed_all(args.training_seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("Loading and converting GRPO data...")
    records = load_records(args.dataset)
    if args.split_manifest is not None:
        partitions = load_split_manifest(args.split_manifest, args.dataset, len(records))
        training_source_indices = partitions["train"]
        validation_source_indices = partitions["dev_select"]
        audit_source_indices = partitions["dev_audit"]
    else:
        training_source_indices, validation_source_indices = split_source_indices(
            len(records), args.validation_size, args.split_seed
        )
        audit_source_indices = []
    selected_sample_count = (
        len(training_source_indices) if args.all_training_samples else args.num_samples
    )
    grpo_records = build_grpo_records(
        records,
        selected_sample_count,
        args.data_seed,
        candidate_source_indices=training_source_indices,
    )
    train_dataset = Dataset.from_list(grpo_records)
    source_indices = list(train_dataset["source_index"])
    if set(source_indices) & set(validation_source_indices):
        raise RuntimeError("training data overlaps the fixed validation split")
    if set(source_indices) & set(audit_source_indices):
        raise RuntimeError("training data overlaps the frozen audit split")
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
    if output_contents and not initialization_only and args.resume_from_checkpoint is None:
        raise FileExistsError(
            f"output directory is not empty: {args.output_dir} (choose a new directory)"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reference_description = policy_reference_description(args)
    reward_functions, reward_weights = build_reward_spec(args)
    manifest = {
        "status": "initialized",
        "mode": "grpo_exact_numeric_reward",
        "base_model": args.base_model,
        "initial_adapter": (
            str(args.adapter)
            if args.policy_initialization == "continued_adapter"
            else None
        ),
        "merged_sft_model": (
            str(args.merged_sft_model) if args.merged_sft_model is not None else None
        ),
        **reference_description,
        "dataset": str(args.dataset),
        "dataset_num_records": len(records),
        "validation_size": args.validation_size,
        "split_seed": args.split_seed,
        "split_manifest": str(args.split_manifest) if args.split_manifest else None,
        "split_manifest_sha256": (
            sha256_file(args.split_manifest) if args.split_manifest else None
        ),
        "training_split_size": len(training_source_indices),
        "validation_split_size": len(validation_source_indices),
        "validation_source_indices": validation_source_indices,
        "audit_split_size": len(audit_source_indices),
        "audit_source_indices": audit_source_indices,
        "selected_training_source_indices": source_indices,
        "num_samples": selected_sample_count,
        "max_steps": args.max_steps,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "num_generations": args.num_generations,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "beta": args.beta,
        "accuracy_reward_weight": args.accuracy_reward_weight,
        "format_reward_weight": args.format_reward_weight,
        "arithmetic_consistency_reward_weight": (
            args.arithmetic_consistency_reward_weight
        ),
        "arithmetic_consistency_reward_scope": (
            "explicit_numeric_equations_only; not a faithfulness guarantee"
        ),
        "reward_functions": [function.__name__ for function in reward_functions],
        "reward_weights": reward_weights,
        "mask_truncated_completions": args.mask_truncated_completions,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "save_final_adapter": args.save_final_adapter,
        "fresh_lora_rank": (
            args.lora_rank if args.policy_initialization == "merged_sft" else None
        ),
        "fresh_lora_alpha": (
            args.lora_alpha if args.policy_initialization == "merged_sft" else None
        ),
        "fresh_lora_target_modules": (
            parse_lora_target_modules(args.lora_target_modules)
            if args.policy_initialization == "merged_sft"
            else None
        ),
        "legacy_seed": args.seed,
        "data_seed": args.data_seed,
        "training_seed": args.training_seed,
        "generation_seed": args.generation_seed,
        "trainer_data_order_seed": args.training_seed,
        "trl_transformers_compat_flags_patched": PATCHED_TRL_FLAGS,
        "resume_from_checkpoint": (
            str(args.resume_from_checkpoint)
            if args.resume_from_checkpoint is not None
            else None
        ),
    }
    manifest_path = args.output_dir / "run_manifest.json"
    write_json_atomic(manifest_path, manifest)
    print("Saved run manifest to:", manifest_path)

    print("Loading policy and tokenizer...")
    policy, tokenizer = load_policy_and_tokenizer(args)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    terminators = resolve_terminators(tokenizer)
    print("Generation terminators:", terminators)

    policy.config.use_cache = False
    policy.enable_input_require_grads()
    patched_model_attributes = ensure_trl_model_compatibility(policy)
    if patched_model_attributes:
        print("Patched TRL model attributes:", patched_model_attributes)
    trainable_parameters = [
        (name, parameter)
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("the policy has no trainable adapter parameters")
    policy.print_trainable_parameters()
    trainable_parameter_snapshots = snapshot_trainable_parameters(trainable_parameters)
    tracked_name, tracked_parameter = trainable_parameters[0]
    tracked_before = trainable_parameter_snapshots[tracked_name]
    memory_after_load = print_cuda_memory("After model loading")

    trainer = GRPOTrainer(
        model=policy,
        reward_funcs=reward_functions,
        args=build_grpo_config(args),
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.generation_config.eos_token_id = terminators
    trainer.generation_config.pad_token_id = tokenizer.pad_token_id

    # Trainer construction consumes the optimization seed. Reset only rollout
    # RNG state here; the data subset is already frozen by data_seed.
    random.seed(args.generation_seed)
    torch.manual_seed(args.generation_seed)
    torch.cuda.manual_seed_all(args.generation_seed)

    print(
        "Starting GRPO training: "
        f"steps={args.max_steps}, samples={selected_sample_count}, "
        f"generations={args.num_generations}, "
        f"max_completion_length={args.max_completion_length}"
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
            "CUDA OOM during GRPO. Retry a fresh output directory with "
            "--num-generations 2 --gradient-accumulation-steps 2 and, if "
            "needed, --max-completion-length 96."
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
    trainable_parameter_drift = summarize_trainable_parameter_drift(
        trainable_parameters,
        trainable_parameter_snapshots,
    )
    print(
        "All trainable parameters: "
        f"updated tensors={trainable_parameter_drift['all']['updated_tensor_count']}/"
        f"{trainable_parameter_drift['all']['tensor_count']}, "
        f"max |delta|={trainable_parameter_drift['all']['max_abs_delta']:.8g}, "
        f"L2 delta={trainable_parameter_drift['all']['l2_delta']:.8g}"
    )
    metrics_are_finite = all(
        not isinstance(value, float) or math.isfinite(value)
        for value in train_result.metrics.values()
    )
    summary = {
        **manifest,
        "status": "success",
        "eos_token_ids": terminators,
        "final_adapter": final_adapter_path,
        "tracked_parameter": tracked_name,
        "tracked_parameter_max_abs_delta": parameter_max_abs_delta,
        "tracked_parameter_updated": parameter_updated,
        "trainable_parameter_drift": trainable_parameter_drift,
        "metrics_are_finite": metrics_are_finite,
        "metrics": train_result.metrics,
        "trainer_log_history": trainer.state.log_history,
        "memory_after_load": memory_after_load,
        "memory_after_train": memory_after_train,
        "trl_transformers_model_attributes_patched": patched_model_attributes,
    }
    summary_path = args.output_dir / "grpo_run_summary.json"
    write_json_atomic(summary_path, summary)

    print("Tracked parameter:", tracked_name)
    print(f"Tracked parameter max |delta|: {parameter_max_abs_delta:.8g}")
    print("Parameter updated:", parameter_updated)
    if not parameter_updated:
        print(
            "Note: a one-step smoke may have identical rewards within its group, "
            "which correctly produces zero GRPO advantage and no parameter update."
        )
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
