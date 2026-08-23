import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from eval_sft_adapter import load_local_gsm8k_dataset  # noqa: E402


class EvalSftAdapterLocalDatasetTest(unittest.TestCase):
    def test_loads_generated_confirmatory_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "dev_audit.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "instruction": "What is 1+1?",
                            "output": "1+1=2\n#### 2",
                            "source_index": 17,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            dataset = load_local_gsm8k_dataset(path)
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset[0]["question"], "What is 1+1?")
            self.assertEqual(dataset[0]["answer"], "1+1=2\n#### 2")
            self.assertEqual(dataset[0]["source_index"], 17)

    def test_rejects_missing_source_index(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid.json"
            path.write_text(
                json.dumps([{"instruction": "Question", "output": "#### 1"}]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "source_index"):
                load_local_gsm8k_dataset(path)


if __name__ == "__main__":
    unittest.main()
