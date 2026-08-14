import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluation_utils import (  # noqa: E402
    exact_mcnemar_p_value,
    extract_ground_truth,
    follows_answer_format,
    has_completed_answer_line,
    score_response,
)


class EvaluationUtilsTest(unittest.TestCase):
    def test_ground_truth(self):
        self.assertEqual(extract_ground_truth("reasoning\n#### 1,200"), "1200")

    def test_decimal_normalization(self):
        score = score_response("work\n#### 60.0", "60")
        self.assertTrue(score["correct"])
        self.assertTrue(score["strict_correct"])

    def test_boxed_answer_is_relaxed_not_strict(self):
        score = score_response(r"The result is \boxed{3.5}", "3.5")
        self.assertTrue(score["correct"])
        self.assertFalse(score["strict_correct"])

    def test_conclusion_ignores_trailing_problem_number(self):
        score = score_response(
            "Therefore, Terry spends $75.00 on yogurt over 30 days.",
            "75",
        )
        self.assertTrue(score["correct"])
        self.assertFalse(score["strict_correct"])

    def test_missing_prediction(self):
        score = score_response("I cannot solve this problem.", "7")
        self.assertFalse(score["correct"])
        self.assertIsNone(score["predicted_answer"])

    def test_huge_number_does_not_crash(self):
        score = score_response("work\n#### 1e999999", "3")
        self.assertFalse(score["correct"])
        self.assertEqual(score["predicted_answer_normalized"], "1e999999")

    def test_completed_answer_line_requires_line_end(self):
        self.assertFalse(has_completed_answer_line("work\n#### 3"))
        self.assertTrue(has_completed_answer_line("work\n#### 3.3 ALOG\n"))
        self.assertFalse(has_completed_answer_line("use #### 3 as an example\n"))

    def test_trailing_garbage_is_relaxed_not_strict(self):
        score = score_response("work\n#### 3 ALOG\n", "3")
        self.assertTrue(score["correct"])
        self.assertFalse(score["strict_correct"])
        self.assertFalse(follows_answer_format("work\n#### 3 ALOG\n"))

    def test_exact_mcnemar(self):
        self.assertAlmostEqual(
            exact_mcnemar_p_value(29, 5),
            3.8558151572942734e-05,
        )


if __name__ == "__main__":
    unittest.main()
