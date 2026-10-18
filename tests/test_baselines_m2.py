import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from planrec.benchmark import count_parameters, measure_inference_latency
from planrec.config import GRU4RecConfig
from planrec.models.gru4rec import GRU4Rec
from planrec.models.popularity import Popularity


class M2BaselinesTests(unittest.TestCase):
    def test_popularity_model(self) -> None:
        model = Popularity(num_items=10)
        # Target sequence with high frequency for item 5
        targets = torch.tensor([5, 5, 5, 5, 2, 2, 1])
        model.fit_frequencies(targets)

        seq = torch.tensor([[1, 2, 3]])
        logits = model(seq)

        self.assertEqual(logits.shape, (1, 11))
        # Item 5 score should be higher than item 1
        self.assertGreater(logits[0, 5].item(), logits[0, 1].item())

    def test_gru4rec_forward_and_training(self) -> None:
        config = GRU4RecConfig(num_items=15, max_seq_len=6, hidden_size=16, num_layers=1)
        model = GRU4Rec(config)
        seq = torch.tensor([[1, 2, 3, 0, 0, 0], [4, 5, 6, 7, 0, 0]])
        logits = model(seq)

        self.assertEqual(logits.shape, (2, 16))

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        target = torch.tensor([4, 8])
        loss = torch.nn.functional.cross_entropy(logits, target)
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))

    def test_parameter_counting_and_latency(self) -> None:
        config = GRU4RecConfig(num_items=20, max_seq_len=5, hidden_size=16)
        model = GRU4Rec(config)
        param_count = count_parameters(model)
        self.assertGreater(param_count, 0)

        seq = torch.tensor([[1, 2, 3, 0, 0]])
        latency = measure_inference_latency(model, seq, torch.device("cpu"), warmup_runs=2, num_runs=5)
        self.assertGreater(latency, 0.0)


if __name__ == "__main__":
    unittest.main()
