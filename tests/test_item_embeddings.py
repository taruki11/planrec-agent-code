import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_item_embeddings import load_item_texts, mean_pool


class ItemEmbeddingTests(unittest.TestCase):
    def test_text_ids_are_contiguous_and_pooling_ignores_padding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "items.json"
            path.write_text(json.dumps({"1": "first", "2": "second"}), encoding="utf-8")
            self.assertEqual(load_item_texts(path), {1: "first", 2: "second"})
        hidden = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [99.0, 99.0]]])
        mask = torch.tensor([[1, 1, 0]])
        self.assertTrue(torch.equal(mean_pool(hidden, mask), torch.tensor([[2.0, 4.0]])))


if __name__ == "__main__":
    unittest.main()
