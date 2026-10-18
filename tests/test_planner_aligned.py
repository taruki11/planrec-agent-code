import unittest

import torch

from planrec.config import SASRecConfig
from planrec.models.semantic_planner import SemanticPlannerModel
from planrec.planner.schema import PlanSchema
from planrec.tokenizer.semantic_tokenizer import SemanticTokenizer


def make(num_items=40, num_clusters=4, aligned=False):
    cfg = SASRecConfig(num_items=num_items, hidden_size=16, num_heads=2, num_layers=1)
    tok = SemanticTokenizer(num_items=num_items, num_clusters_c1=num_clusters)
    emb = torch.randn(num_items + 1, cfg.hidden_size)
    tok.fit_from_item_embeddings(emb, seed=42)
    schema = PlanSchema(depth=2, num_intent_clusters=num_clusters,
                        num_interest_clusters=num_clusters)
    return cfg, tok, schema, SemanticPlannerModel(cfg, tok, schema, aligned_head=aligned)


class TestPlannerAlignedHead(unittest.TestCase):
    def test_aligned_forward_shapes(self):
        cfg, tok, schema, m = make(aligned=True)
        x = torch.randint(1, cfg.num_items + 1, (4, cfg.max_seq_len))
        for d in (0, 1, 2):
            logits, plans = m.forward_with_depth(x, depth=d)
            self.assertEqual(logits.shape, (4, cfg.num_items + 1))
            self.assertEqual(len(plans), 4)

    def test_aligned_d0_is_sasrec_shaped(self):
        # d0 with zero plan tokens: fused == user_rep and the output is the
        # tied item-embedding matmul (SASRec-shaped, no separate item_head)
        cfg, tok, schema, m = make(aligned=True)
        self.assertIsNone(m.item_head)
        self.assertFalse(m.z_proj.bias is not None)  # no bias leak at depth 0
        x = torch.randint(1, cfg.num_items + 1, (2, cfg.max_seq_len))
        logits, _ = m.forward_with_depth(x, depth=0)
        self.assertFalse(torch.isnan(logits).any())
        self.assertFalse(torch.isinf(logits).any())

    def test_legacy_preserved(self):
        cfg, tok, schema, m = make(aligned=False)
        self.assertEqual(m.item_head.in_features, cfg.hidden_size * 3)
        x = torch.randint(1, cfg.num_items + 1, (2, cfg.max_seq_len))
        logits, _ = m.forward_with_depth(x, depth=2, corrupt_plan=True, corrupt_seed=42)
        self.assertEqual(logits.shape, (2, cfg.num_items + 1))


if __name__ == "__main__":
    unittest.main()
