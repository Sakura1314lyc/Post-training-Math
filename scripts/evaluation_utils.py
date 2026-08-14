"""Shared GSM8K answer extraction and scoring utilities."""

from __future__ import annotations

import math
import re
from decimal import Decimal, DecimalException


EVALUATION_VERSION = "gsm8k_numeric_v3"

SYSTEM_PROMPT = (
    "You are a helpful math assistant. "
    "Solve the problem step by step, but keep the reasoning concise. "
    "At the end, output the final numerical answer in the exact format: "
    "#### <answer>"
)

NUMBER = r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"

_ANSWER_MARKER_PATTERN = re.compile(rf"####\s*\$?\s*({NUMBER})", re.IGNORECASE)
_STRICT_ANSWER_PATTERN = re.compile(
    rf"(?:^|\n)\s*####\s*\$?\s*({NUMBER})\s*\Z",
    re.IGNORECASE,
)
_COMPLETED_ANSWER_LINE_PATTERN = re.compile(
    rf"(?:^|\n)\s*####\s*\$?\s*{NUMBER}[^\n]*\n",
    re.IGNORECASE,
)

_PREDICTION_PATTERNS = (
    _ANSWER_MARKER_PATTERN,
    re.compile(rf"\\boxed\{{\s*\$?\s*({NUMBER})\s*\}}", re.IGNORECASE),
    re.compile(
        rf"(?:final numerical answer|final answer|answer)"
        rf"\s*(?:is|:|=)\s*\$?\s*({NUMBER})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?:therefore|thus|hence)\b[^.\n]{{0,300}}?"
        rf"(?:needs?\s+to\s+pay|will\s+pay|will\s+be|pays?|spends?|saves?|"
        rf"earns?|makes?|raises?|costs?|has|have|gets?|equals?|is|are|was|were)"
        rf"\s*(?::|=)?\s*(?:approximately\s+|about\s+|exactly\s+)?"
        rf"\*{{0,2}}\s*\$?\s*({NUMBER})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?:^|\n)\s*so\b[^\n]{{0,250}}?"
        rf"(?:needs?\s+to\s+pay|will\s+pay|will\s+be|pays?|spends?|saves?|"
        rf"earns?|makes?|raises?|costs?|has|have|gets?|equals?|is|are|was|were)"
        rf"\s*(?::|=)?\s*(?:approximately\s+|about\s+|exactly\s+)?"
        rf"\*{{0,2}}\s*\$?\s*({NUMBER})",
        re.IGNORECASE,
    ),
)


def extract_ground_truth(answer: str) -> str | None:
    """Extract the official GSM8K answer after the final ``####`` marker."""
    if "####" not in answer:
        return None

    matches = re.findall(NUMBER, answer.split("####")[-1].replace(",", ""))
    return matches[-1] if matches else None


def extract_predicted_answer(text: str) -> str | None:
    """Extract a numeric prediction, preferring explicit final-answer markers."""
    text = text.replace(",", "")

    for pattern in _PREDICTION_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return matches[-1]

    numbers = re.findall(NUMBER, text)
    return numbers[-1] if numbers else None


def extract_strict_answer(text: str) -> str | None:
    """Extract an answer only when the requested ``####`` format is present."""
    matches = _STRICT_ANSWER_PATTERN.findall(text.replace(",", ""))
    return matches[-1] if matches else None


def normalize_number(value: str | None) -> str | None:
    """Normalize equivalent decimal strings, e.g. ``60.0`` and ``60``."""
    if value is None:
        return None

    cleaned = str(value).strip().replace(",", "").replace("$", "").replace("%", "")

    try:
        number = Decimal(cleaned)

        if not number.is_finite():
            return None

        if number == number.to_integral():
            return str(number.quantize(Decimal("1")))

        return format(number.normalize(), "f").rstrip("0").rstrip(".")
    except DecimalException:
        # Malformed or excessively large model outputs should be scored as
        # ordinary wrong predictions instead of aborting the whole evaluation.
        return cleaned


def score_response(response: str, ground_truth: str) -> dict[str, str | bool | None]:
    predicted = extract_predicted_answer(response)
    strict_predicted = extract_strict_answer(response)
    predicted_normalized = normalize_number(predicted)
    strict_predicted_normalized = normalize_number(strict_predicted)
    ground_truth_normalized = normalize_number(ground_truth)
    correct = (
        predicted_normalized is not None
        and ground_truth_normalized is not None
        and predicted_normalized == ground_truth_normalized
    )
    strict_correct = (
        strict_predicted_normalized is not None
        and ground_truth_normalized is not None
        and strict_predicted_normalized == ground_truth_normalized
    )
    return {
        "predicted_answer": predicted,
        "predicted_answer_normalized": predicted_normalized,
        "strict_predicted_answer": strict_predicted,
        "strict_predicted_answer_normalized": strict_predicted_normalized,
        "ground_truth_normalized": ground_truth_normalized,
        "correct": correct,
        "strict_correct": strict_correct,
    }


def follows_answer_format(text: str) -> bool:
    """Return whether the response contains a parseable ``#### <number>`` answer."""
    return bool(_STRICT_ANSWER_PATTERN.search(text.replace(",", "")))


def has_completed_answer_line(text: str) -> bool:
    """Return whether a complete ``#### <number>`` line ended with a newline."""
    return bool(_COMPLETED_ANSWER_LINE_PATTERN.search(text.replace(",", "")))


def exact_mcnemar_p_value(base_only_correct: int, sft_only_correct: int) -> float:
    """Two-sided exact McNemar/binomial test without a scipy dependency."""
    discordant = base_only_correct + sft_only_correct
    if discordant == 0:
        return 1.0

    smaller = min(base_only_correct, sft_only_correct)
    lower_tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2**discordant)
    return min(1.0, 2 * lower_tail)
