import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluation_benchmarks import (  # noqa: E402
    adapt_benchmark_sample,
    get_benchmark_spec,
    resolve_dataset_identity,
    resolve_source_split,
)


class EvaluationBenchmarksTest(unittest.TestCase):
    def test_gsm8k_defaults_and_splits(self):
        spec = get_benchmark_spec("gsm8k")
        self.assertEqual(
            resolve_dataset_identity(spec, None, None),
            ("openai/gsm8k", "main"),
        )
        self.assertEqual(resolve_source_split(spec, "test"), "test")
        self.assertEqual(resolve_source_split(spec, "train_validation"), "train")

    def test_svamp_defaults_and_split(self):
        spec = get_benchmark_spec("svamp")
        self.assertEqual(
            resolve_dataset_identity(spec, None, None),
            ("MU-NLPC/Calc-svamp", "default"),
        )
        self.assertEqual(resolve_source_split(spec, "test"), "test")
        with self.assertRaisesRegex(ValueError, "does not support"):
            resolve_source_split(spec, "train_validation")

    def test_explicit_dataset_override_is_preserved(self):
        spec = get_benchmark_spec("svamp")
        self.assertEqual(
            resolve_dataset_identity(spec, "local/svamp", "custom"),
            ("local/svamp", "custom"),
        )

    def test_adapt_gsm8k_sample(self):
        question, ground_truth, metadata = adapt_benchmark_sample(
            {"question": "How many?", "answer": "Reasoning\n#### 1,200"},
            "gsm8k",
        )
        self.assertEqual(question, "How many?")
        self.assertEqual(ground_truth, "1200")
        self.assertEqual(metadata, {})

    def test_adapt_svamp_sample(self):
        question, ground_truth, metadata = adapt_benchmark_sample(
            {
                "id": "chal-123",
                "question": "A shop had 9 pens and sold 4. How many remain?",
                "result": "5",
                "problem_type": "Subtraction",
            },
            "svamp",
        )
        self.assertEqual(
            question,
            "A shop had 9 pens and sold 4. How many remain?",
        )
        self.assertEqual(ground_truth, "5")
        self.assertEqual(
            metadata,
            {"sample_id": "chal-123", "problem_type": "Subtraction"},
        )

    def test_missing_required_field_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "result"):
            adapt_benchmark_sample({"question": "How many?"}, "svamp")

    def test_unknown_benchmark_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown benchmark"):
            get_benchmark_spec("unknown")


if __name__ == "__main__":
    unittest.main()
