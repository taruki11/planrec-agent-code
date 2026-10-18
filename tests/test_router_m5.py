import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from planrec.config import SASRecConfig
from planrec.models.sasrec import SASRec
from planrec.router.calibrated_router import CalibratedRouter, EntropyThresholdRouter
from planrec.router.features import extract_router_features
from planrec.router.metrics import compute_brier_score, compute_expected_calibration_error, compute_nll


class M5RouterTests(unittest.TestCase):
    def test_ece_and_calibration_metrics(self) -> None:
        probs = torch.tensor([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]])
        labels = torch.tensor([0, 1])

        ece = compute_expected_calibration_error(probs, labels)
        brier = compute_brier_score(probs, labels)
        nll = compute_nll(probs, labels)

        self.assertGreaterEqual(ece, 0.0)
        self.assertGreaterEqual(brier, 0.0)
        self.assertGreaterEqual(nll, 0.0)

    def test_feature_extraction_and_router_training(self) -> None:
        config = SASRecConfig(num_items=10, max_seq_len=5, hidden_size=16)
        model = SASRec(config)
        seq = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 0]])

        features = extract_router_features(model, seq, max_seq_len=5)
        self.assertEqual(features.shape, (2, 5))

        router = CalibratedRouter(input_dim=5, hidden_dim=16, num_depths=3)
        probs = router(features)
        self.assertEqual(probs.shape, (2, 3))

        # Temperature fitting test
        labels = torch.tensor([0, 1])
        opt_t = router.fit_temperature(features, labels)
        self.assertGreater(opt_t, 0.0)

    def test_item_norm_uses_last_item_embedding(self) -> None:
        config = SASRecConfig(num_items=10, max_seq_len=5, hidden_size=16)
        model = SASRec(config)
        with torch.no_grad():
            model.item_embedding.weight.zero_()
            model.item_embedding.weight[3].fill_(2.0)
        features = extract_router_features(model, torch.tensor([[3, 0, 0, 0, 0]]), max_seq_len=5)
        self.assertAlmostEqual(features[0, 3].item(), 2.0, places=5)


if __name__ == "__main__":
    unittest.main()
