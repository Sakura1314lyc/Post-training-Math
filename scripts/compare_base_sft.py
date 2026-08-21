"""Paired comparison of two numeric benchmark evaluation files."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from evaluation_utils import (
    SUPPORTED_EVALUATION_VERSIONS,
    exact_mcnemar_p_value,
    follows_answer_format,
)


TRANSITION_KEYS = (
    "base_correct_sft_correct",
    "base_correct_sft_wrong",
    "base_wrong_sft_correct",
    "base_wrong_sft_wrong",
)

PROTOCOL_KEYS = (
    "benchmark",
    "dataset",
    "prompt",
    "generation",
)


def load_payload(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload.get("results"), list):
        raise ValueError(f"{path} does not contain a results list")
    return payload


def validate_comparison_protocol(base_payload: dict, sft_payload: dict) -> None:
    """Reject paired comparisons produced by different evaluation protocols."""
    base_version = base_payload.get("evaluation_version")
    sft_version = sft_payload.get("evaluation_version")
    if base_version != sft_version:
        raise ValueError(
            "Base and SFT were scored by different evaluators: "
            f"base={base_version!r}, sft={sft_version!r}"
        )

    mismatched = [
        key
        for key in PROTOCOL_KEYS
        if base_payload.get(key) != sft_payload.get(key)
    ]
    if mismatched:
        raise ValueError(
            "Base and SFT use different evaluation protocols for: "
            + ", ".join(mismatched)
        )


def index_by_question(results: list[dict], label: str) -> dict[str, dict]:
    indexed = {}
    duplicates = []
    for item in results:
        question = item.get("question")
        if not question:
            raise ValueError(f"{label} contains a result without a question")
        if question in indexed:
            duplicates.append(question)
        indexed[question] = item
    if duplicates:
        raise ValueError(f"{label} contains {len(duplicates)} duplicate questions")
    return indexed


def response_stats(results: list[dict]) -> dict:
    lengths = [len(item.get("response", "")) for item in results]
    compliant = sum(follows_answer_format(item.get("response", "")) for item in results)
    strict_values = [item.get("strict_correct") for item in results]
    has_strict_scores = all(value is not None for value in strict_values)
    return {
        "num_samples": len(results),
        "mean_response_characters": statistics.fmean(lengths) if lengths else 0.0,
        "median_response_characters": statistics.median(lengths) if lengths else 0.0,
        "format_compliant": compliant,
        "format_compliance_rate": compliant / len(results) if results else 0.0,
        "strict_correct": sum(bool(value) for value in strict_values) if has_strict_scores else None,
        "strict_accuracy": (
            sum(bool(value) for value in strict_values) / len(results)
            if has_strict_scores and results
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare paired numeric benchmark results."
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/base_sft_transition_analysis.json"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        parser.error(f"output already exists: {args.output} (pass --overwrite to replace it)")

    base_payload = load_payload(args.base)
    sft_payload = load_payload(args.sft)
    validate_comparison_protocol(base_payload, sft_payload)
    base_version = base_payload.get("evaluation_version")
    sft_version = sft_payload.get("evaluation_version")
    if base_version not in SUPPORTED_EVALUATION_VERSIONS:
        print(
            f"Warning: comparing unrecognized evaluator version {base_version!r}; "
            "verify that both files use the intended scoring protocol."
        )

    base_results = base_payload["results"]
    sft_results = sft_payload["results"]
    base_map = index_by_question(base_results, "base")
    sft_map = index_by_question(sft_results, "sft")

    base_questions = set(base_map)
    sft_questions = set(sft_map)
    if base_questions != sft_questions:
        raise ValueError(
            "Base and SFT result sets differ: "
            f"base_only={len(base_questions - sft_questions)}, "
            f"sft_only={len(sft_questions - base_questions)}"
        )

    transitions = Counter()
    details = {key: [] for key in TRANSITION_KEYS}
    for index, question in enumerate(base_map, 1):
        base = base_map[question]
        sft = sft_map[question]
        base_correct = bool(base.get("correct"))
        sft_correct = bool(sft.get("correct"))

        if base_correct and sft_correct:
            key = "base_correct_sft_correct"
        elif base_correct:
            key = "base_correct_sft_wrong"
        elif sft_correct:
            key = "base_wrong_sft_correct"
        else:
            key = "base_wrong_sft_wrong"
        transitions[key] += 1
        details[key].append(
            {
                "index": index,
                "source_index": base.get("source_index"),
                "question": question,
                "ground_truth": base.get("ground_truth"),
                "base_pred": base.get("predicted_answer"),
                "sft_pred": sft.get("predicted_answer"),
                "base_response": base.get("response", ""),
                "sft_response": sft.get("response", ""),
            }
        )

    num_samples = len(base_map)
    if num_samples == 0:
        raise ValueError("No matched samples")

    base_only = transitions["base_correct_sft_wrong"]
    sft_only = transitions["base_wrong_sft_correct"]
    both_correct = transitions["base_correct_sft_correct"]
    base_correct = both_correct + base_only
    sft_correct = both_correct + sft_only
    base_accuracy = base_correct / num_samples
    sft_accuracy = sft_correct / num_samples

    summary = {
        "num_matched": num_samples,
        "base_correct": base_correct,
        "base_accuracy": base_accuracy,
        "sft_correct": sft_correct,
        "sft_accuracy": sft_accuracy,
        "accuracy_delta": sft_accuracy - base_accuracy,
        "transitions": {key: transitions[key] for key in TRANSITION_KEYS},
        "mcnemar_exact_two_sided_p": exact_mcnemar_p_value(base_only, sft_only),
        "base_response_stats": response_stats(base_results),
        "sft_response_stats": response_stats(sft_results),
    }
    output = {
        "evaluation_version_expected": base_version,
        "base_source": str(args.base),
        "sft_source": str(args.sft),
        "base_evaluation_version": base_version,
        "sft_evaluation_version": sft_version,
        "comparison_protocol": {
            key: base_payload.get(key) for key in PROTOCOL_KEYS
        },
        "summary": summary,
        "details": details,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Matched: {num_samples}")
    print(f"Base: {base_correct}/{num_samples} = {base_accuracy:.2%}")
    print(f"SFT : {sft_correct}/{num_samples} = {sft_accuracy:.2%}")
    print(f"Delta: {sft_accuracy - base_accuracy:+.2%}")
    print(f"Base correct -> SFT wrong: {base_only}")
    print(f"Base wrong -> SFT correct: {sft_only}")
    print(f"Exact McNemar p: {summary['mcnemar_exact_two_sided_p']:.6g}")
    print("Saved to:", args.output)


if __name__ == "__main__":
    main()
