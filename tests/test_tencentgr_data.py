import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from planrec.data.tencentgr import prepare_tencentgr_leave_one_out_splits
from scripts.build_tencentgr_e6_emb81 import sanitize_and_normalize_catalog_embeddings


class TencentGRDataPipelineTests(unittest.TestCase):
    def test_leave_one_out_inputs_use_right_padding(self) -> None:
        sequences = [[10, 20, 30, 40, 50]]
        train_ds, val_ds, test_ds = prepare_tencentgr_leave_one_out_splits(
            sequences,
            max_seq_len=5,
            min_seq_len=3,
        )

        self.assertEqual(test_ds[0][0].tolist(), [10, 20, 30, 40, 0])
        self.assertEqual(val_ds[0][0].tolist(), [10, 20, 30, 0, 0])
        self.assertEqual(train_ds[0][0].tolist(), [10, 0, 0, 0, 0])

    def test_catalog_embedding_sanitation_zeros_invalid_rows_then_normalizes(self) -> None:
        matrix = torch.tensor(
            [
                [9.0, 9.0, 9.0],
                [3.0, 4.0, 0.0],
                [float("nan"), 1.0, 2.0],
                [1.0, float("inf"), 2.0],
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 2.0],
            ]
        )

        output, counts = sanitize_and_normalize_catalog_embeddings(matrix)

        self.assertEqual(counts["nonfinite_input_rows_zeroed"], 2)
        self.assertEqual(counts["nonfinite_input_values_zeroed"], 2)
        self.assertTrue(torch.isfinite(output).all())
        torch.testing.assert_close(output[0], torch.zeros(3), atol=0.0, rtol=0.0)
        torch.testing.assert_close(output[2], torch.zeros(3), atol=0.0, rtol=0.0)
        torch.testing.assert_close(output[3], torch.zeros(3), atol=0.0, rtol=0.0)
        torch.testing.assert_close(output[4], torch.zeros(3), atol=0.0, rtol=0.0)
        torch.testing.assert_close(output[1].norm(), torch.tensor(1.0))
        torch.testing.assert_close(output[5].norm(), torch.tensor(1.0))


if __name__ == "__main__":
    unittest.main()
