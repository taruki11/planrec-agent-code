import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from planrec.config import SASRecConfig
from planrec.models.semantic_planner import SemanticPlannerModel
from planrec.planner.schema import (
    PlanSchema,
    compute_plan_item_consistency,
    validate_plan_structure,
)
from planrec.tokenizer.semantic_tokenizer import SemanticTokenizer


class M4PlannerTests(unittest.TestCase):
    def test_plan_schema_validation(self) -> None:
        self.assertTrue(validate_plan_structure([], depth=0))
        self.assertTrue(validate_plan_structure([2], depth=1))
        self.assertTrue(validate_plan_structure([2, 4], depth=2))
        self.assertFalse(validate_plan_structure([2], depth=2))

    def test_plan_consistency_uses_actual_tokenizer_cluster(self) -> None:
        item_to_code = {10: (1, 7), 11: (2, 3), 12: (5, 1)}
        predicted = [(1,), (2,), (5,)]
        self.assertEqual(
            compute_plan_item_consistency(predicted, [10, 11, 12], item_to_code),
            1.0,
        )

    def test_semantic_planner_depths(self) -> None:
        num_items = 12
        config = SASRecConfig(num_items=num_items, max_seq_len=6, hidden_size=16, num_layers=1)
        item_embeddings = torch.randn(num_items + 1, 16)
        tokenizer = SemanticTokenizer(num_items=num_items, num_clusters_c1=2)
        _ = tokenizer.fit_from_item_embeddings(item_embeddings)

        planner = SemanticPlannerModel(config, tokenizer, PlanSchema(depth=2))
        seq = torch.tensor([[1, 2, 3, 0, 0, 0], [4, 5, 6, 7, 0, 0]])

        # Test depth 0 (Direct)
        logits0, plans0 = planner.forward_with_depth(seq, depth=0)
        self.assertEqual(logits0.shape, (2, 13))
        self.assertEqual(len(plans0[0]), 0)

        # Test depth 1 (Plan-1)
        logits1, plans1 = planner.forward_with_depth(seq, depth=1)
        self.assertEqual(logits1.shape, (2, 13))
        self.assertEqual(len(plans1[0]), 1)

        # Test depth 2 (Plan-2)
        logits2, plans2 = planner.forward_with_depth(seq, depth=2)
        self.assertEqual(logits2.shape, (2, 13))
        self.assertEqual(len(plans2[0]), 2)

        # Test corrupted plan
        logits_corrupt, plans_corrupt = planner.forward_with_depth(seq, depth=2, corrupt_plan=True)
        self.assertEqual(logits_corrupt.shape, (2, 13))
        self.assertEqual(len(plans_corrupt[0]), 2)


if __name__ == "__main__":
    unittest.main()
