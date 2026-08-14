import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluation_utils import (  # noqa: E402
    exact_mcnemar_p_value,
    extract_ground_truth,
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

    def test_exact_mcnemar(self):
        self.assertAlmostEqual(
            exact_mcnemar_p_value(29, 5),
            3.8558151572942734e-05,
        )


if __name__ == "__main__":
    unittest.main()
