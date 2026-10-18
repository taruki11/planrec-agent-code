import importlib
import tempfile
import unittest
from pathlib import Path

import torch


class E6FeatureIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = None
        try:
            self.module = importlib.import_module("scripts.prepare_e6_feature_v2_indices")
        except ModuleNotFoundError:
            pass

    def test_selector_is_seeded_sorted_unique_and_in_range(self) -> None:
        self.assertIsNotNone(self.module)
        first = self.module.select_fixed_user_indices(split_size=12, count=5, seed=42)
        second = self.module.select_fixed_user_indices(split_size=12, count=5, seed=42)
        self.assertEqual(first.dtype, torch.long)
        self.assertEqual(first.tolist(), second.tolist())
        self.assertEqual(first.tolist(), sorted(first.tolist()))
        self.assertEqual(len(first.unique()), 5)
        self.assertTrue(torch.all((first >= 0) & (first < 12)))

    def test_selector_rejects_count_larger_than_split(self) -> None:
        self.assertIsNotNone(self.module)
        with self.assertRaisesRegex(ValueError, "exceeds split size"):
            self.module.select_fixed_user_indices(split_size=3, count=4, seed=42)

    def test_writer_records_dataset_and_tensor_hashes(self) -> None:
        self.assertIsNotNone(self.module)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "datasets.pt"
            torch.save({"val_x": torch.zeros(6, 3), "test_x": torch.zeros(7, 3)}, dataset)
            manifest = self.module.write_fixed_user_indices(dataset, root / "indices", count=4, seed=42)
            self.assertEqual(manifest["dataset_sha256"], self.module.sha256(dataset))
            self.assertEqual(manifest["splits"]["val"]["count"], 4)
            self.assertEqual(manifest["splits"]["test"]["count"], 4)
            self.assertTrue((root / "indices" / "val_seed42_n4.pt").exists())
            self.assertTrue((root / "indices" / "test_seed42_n4.pt").exists())

    def test_tensor_hash_is_stable_across_separate_pt_artifacts(self) -> None:
        self.assertIsNotNone(self.module)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = root / "first.pt", root / "second.pt"
            indices = torch.tensor([1, 4, 9], dtype=torch.long)
            torch.save(indices, first)
            torch.save(indices, second)
            self.assertEqual(
                self.module.tensor_sha256(torch.load(first, weights_only=True)),
                self.module.tensor_sha256(torch.load(second, weights_only=True)),
            )
