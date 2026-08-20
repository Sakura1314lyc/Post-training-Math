"""Benchmark-specific dataset defaults and sample adapters for numeric evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from evaluation_utils import extract_ground_truth


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    dataset_name: str
    dataset_config: str
    evaluation_version: str
    supported_eval_splits: tuple[str, ...]


BENCHMARKS = {
    "gsm8k": BenchmarkSpec(
        name="gsm8k",
        dataset_name="openai/gsm8k",
        dataset_config="main",
        evaluation_version="gsm8k_numeric_v3",
        supported_eval_splits=("test", "train_validation"),
    ),
    "svamp": BenchmarkSpec(
        name="svamp",
        dataset_name="MU-NLPC/Calc-svamp",
        dataset_config="default",
        evaluation_version="svamp_numeric_v1",
        supported_eval_splits=("test",),
    ),
}


def get_benchmark_spec(name: str) -> BenchmarkSpec:
    try:
        return BENCHMARKS[name]
    except KeyError as error:
        choices = ", ".join(sorted(BENCHMARKS))
        raise ValueError(f"unknown benchmark {name!r}; expected one of: {choices}") from error


def resolve_dataset_identity(
    spec: BenchmarkSpec,
    dataset_name: str | None,
    dataset_config: str | None,
) -> tuple[str, str]:
    """Apply benchmark defaults while preserving explicit CLI overrides."""
    return dataset_name or spec.dataset_name, dataset_config or spec.dataset_config


def resolve_source_split(spec: BenchmarkSpec, eval_split: str) -> str:
    if eval_split not in spec.supported_eval_splits:
        supported = ", ".join(spec.supported_eval_splits)
        raise ValueError(
            f"benchmark {spec.name!r} does not support eval split {eval_split!r}; "
            f"supported: {supported}"
        )
    return "train" if eval_split == "train_validation" else "test"


def adapt_benchmark_sample(
    sample: Mapping[str, object],
    benchmark: str,
) -> tuple[str, str, dict[str, object]]:
    """Return prompt text, numeric ground truth, and non-answer metadata."""
    spec = get_benchmark_spec(benchmark)

    if spec.name == "gsm8k":
        question = _required_text(sample, "question", benchmark)
        answer = _required_text(sample, "answer", benchmark)
        ground_truth = extract_ground_truth(answer)
        if ground_truth is None:
            raise ValueError("gsm8k sample does not contain a parseable #### answer")
        return question, ground_truth, {}

    question = _required_text(sample, "question", benchmark)
    ground_truth = _required_text(sample, "result", benchmark)
    metadata = {
        "sample_id": sample.get("id"),
        "problem_type": sample.get("problem_type"),
    }
    return question, ground_truth, metadata


def _required_text(
    sample: Mapping[str, object],
    field: str,
    benchmark: str,
) -> str:
    value = sample.get(field)
    if value is None or not str(value).strip():
        raise ValueError(f"{benchmark} sample has no valid {field!r} field")
    return str(value).strip()
