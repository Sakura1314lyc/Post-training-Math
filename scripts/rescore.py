"""Rescore legacy result files with the canonical GSM8K evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation_utils import EVALUATION_VERSION, follows_answer_format, score_response


def rescore_file(input_path: Path, suffix: str, overwrite: bool) -> None:
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data.get("results"), list):
        print(f"[SKIP] {input_path}: no results list")
        return

    output_path = input_path.with_name(input_path.stem + suffix + input_path.suffix)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it")

    old_correct = data.get("correct")
    new_correct = 0
    strict_correct = 0
    format_compliant = 0
    changed_predictions = 0
    fixed_false_negatives = 0
    newly_wrong = 0
    new_results = []

    for item in data["results"]:
        old_prediction = item.get("predicted_answer")
        old_item_correct = item.get("correct")
        score = score_response(item.get("response", ""), item.get("ground_truth"))
        compliant = follows_answer_format(item.get("response", ""))

        if score["correct"]:
            new_correct += 1
        if score["strict_correct"]:
            strict_correct += 1
        if compliant:
            format_compliant += 1
        if old_prediction != score["predicted_answer"]:
            changed_predictions += 1
        if old_item_correct is False and score["correct"]:
            fixed_false_negatives += 1
        if old_item_correct is True and not score["correct"]:
            newly_wrong += 1

        new_item = dict(item)
        new_item.update(
            {
                "old_predicted_answer": old_prediction,
                "old_correct": old_item_correct,
                **score,
                "format_compliant": compliant,
            }
        )
        new_results.append(new_item)

    num_samples = len(new_results)
    new_data = dict(data)
    new_data.update(
        {
            "evaluation_version": EVALUATION_VERSION,
            "old_correct": old_correct,
            "old_accuracy": data.get("accuracy"),
            "correct": new_correct,
            "accuracy": new_correct / num_samples if num_samples else 0.0,
            "strict_correct": strict_correct,
            "strict_accuracy": strict_correct / num_samples if num_samples else 0.0,
            "format_compliant": format_compliant,
            "format_compliance_rate": format_compliant / num_samples if num_samples else 0.0,
            "results": new_results,
        }
    )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(f"{input_path} -> {output_path}")
    print(f"Old correct: {old_correct}; new correct: {new_correct}/{num_samples}")
    print(
        "Changed predictions: "
        f"{changed_predictions}; fixed false negatives: {fixed_false_negatives}; "
        f"newly wrong: {newly_wrong}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescore legacy GSM8K result JSON files.")
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--suffix", default="_rescored_v1")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for path in args.files:
        if not path.exists():
            print(f"[NOT FOUND] {path}")
            continue
        rescore_file(path, args.suffix, args.overwrite)


if __name__ == "__main__":
    main()
