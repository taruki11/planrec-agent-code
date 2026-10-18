import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from planrec.config import SASRecConfig
from planrec.evaluation import ranking_metrics
from planrec.models.sasrec import SASRec
from planrec.uncertainty import normalized_entropy, top2_margin


class SASRecBaselineTests(unittest.TestCase):
    def test_logits_shape(self) -> None:
        config = SASRecConfig(num_items=20, max_seq_len=5, hidden_size=16, num_layers=1, num_heads=2, dropout=0.0)
        model = SASRec(config)
        sequence = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]])
        self.assertEqual(model(sequence).shape, (2, 21))

    def test_one_training_step_is_finite(self) -> None:
        torch.manual_seed(0)
        model = SASRec(SASRecConfig(num_items=12, max_seq_len=4, hidden_size=8, num_layers=1, num_heads=2, dropout=0.0))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sequence = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 0]])
        target = torch.tensor([4, 7])
        loss = torch.nn.functional.cross_entropy(model(sequence), target)
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))

    def test_different_right_padded_histories_change_logits(self) -> None:
        torch.manual_seed(7)
        model = SASRec(
            SASRecConfig(
                num_items=20,
                max_seq_len=5,
                hidden_size=16,
                num_layers=1,
                num_heads=2,
                dropout=0.0,
            )
        ).eval()
        histories = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 0, 0]])
        with torch.no_grad():
            logits = model(histories)
        self.assertFalse(torch.allclose(logits[0], logits[1]))

    def test_metrics_for_known_ranks(self) -> None:
        logits = torch.tensor([[0.0, 5.0, 1.0], [0.0, 1.0, 5.0]])
        metrics = ranking_metrics(logits, torch.tensor([1, 1]), k=2)
        self.assertAlmostEqual(metrics["hit_rate@2"], 1.0)
        self.assertAlmostEqual(metrics["mrr@2"], 0.75)

    def test_uncertainty_boundaries(self) -> None:
        uniform = torch.zeros(1, 4)
        confident = torch.tensor([[20.0, 0.0, 0.0, 0.0]])
        self.assertAlmostEqual(normalized_entropy(uniform).item(), 1.0, places=6)
        self.assertLess(normalized_entropy(confident).item(), 1e-5)
        self.assertGreater(top2_margin(confident).item(), 0.99)

    def test_invalid_head_dimension_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SASRecConfig(num_items=10, hidden_size=10, num_heads=3)


if __name__ == "__main__":
    unittest.main()
