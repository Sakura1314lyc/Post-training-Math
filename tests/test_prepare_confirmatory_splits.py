import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prepare_confirmatory_splits import build_partitions  # noqa: E402
from train_opd_gkd import split_source_indices  # noqa: E402


class PrepareConfirmatorySplitsTest(unittest.TestCase):
    def test_partitions_are_reproducible_complete_and_disjoint(self):
        first = build_partitions(100, 0.1, 42, 0.1, 20260823)
        second = build_partitions(100, 0.1, 42, 0.1, 20260823)
        self.assertEqual(first, second)
        self.assertEqual({name: len(value) for name, value in first.items()}, {
            "train": 80,
            "dev_select": 10,
            "dev_audit": 10,
        })
        self.assertEqual(
            set(first["train"]) | set(first["dev_select"]) | set(first["dev_audit"]),
            set(range(100)),
        )
        self.assertTrue(set(first["train"]).isdisjoint(first["dev_select"]))
        self.assertTrue(set(first["train"]).isdisjoint(first["dev_audit"]))
        self.assertTrue(set(first["dev_select"]).isdisjoint(first["dev_audit"]))

    def test_selection_partition_preserves_legacy_v7_validation(self):
        partitions = build_partitions(7473, 0.05, 42, 0.05, 20260823)
        _, legacy_validation = split_source_indices(7473, 0.05, 42)
        self.assertEqual(partitions["dev_select"], legacy_validation)


if __name__ == "__main__":
    unittest.main()
