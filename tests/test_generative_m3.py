import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from planrec.config import SASRecConfig
from planrec.models.direct_generator import DirectGenerativeRecommender
from planrec.tokenizer.semantic_tokenizer import SemanticTokenizer


class M3GenerativeTests(unittest.TestCase):
    def test_semantic_tokenizer_fit_and_trie(self) -> None:
        num_items = 12
        torch.manual_seed(3)
        item_embeddings = torch.randn(num_items + 1, 16)
        tokenizer = SemanticTokenizer(num_items=num_items, num_clusters_c1=3)
        stats = tokenizer.fit_from_item_embeddings(item_embeddings)

        self.assertEqual(stats["num_items"], 12)
        self.assertEqual(stats["algorithm"], "hierarchical_kmeans_residual_order_v1")
        self.assertEqual(len(tokenizer.item_to_code), 12)
        self.assertEqual(tokenizer.max_c2, max(c[1] for c in tokenizer.valid_codes))
        self.assertTrue(tokenizer.trie.is_valid(*tokenizer.item_to_code[1]))

        repeat = SemanticTokenizer(num_items=num_items, num_clusters_c1=3)
        repeat.fit_from_item_embeddings(item_embeddings)
        self.assertEqual(tokenizer.item_to_code, repeat.item_to_code)

    def test_direct_generative_recommender_forward(self) -> None:
        num_items = 10
        config = SASRecConfig(num_items=num_items, max_seq_len=6, hidden_size=16, num_layers=1)

        item_embeddings = torch.randn(num_items + 1, 16)
        tokenizer = SemanticTokenizer(num_items=num_items, num_clusters_c1=2)
        _ = tokenizer.fit_from_item_embeddings(item_embeddings)

        model = DirectGenerativeRecommender(config, tokenizer)
        seq = torch.tensor([[1, 2, 3, 0, 0, 0], [4, 5, 6, 7, 0, 0]])
        logits = model(seq)

        self.assertEqual(logits.shape, (2, 11))
        self.assertTrue(torch.isfinite(logits).all())

    def test_code_distribution_masks_invalid_trie_tokens(self) -> None:
        num_items = 10
        config = SASRecConfig(num_items=num_items, max_seq_len=6, hidden_size=16, num_layers=1)
        tokenizer = SemanticTokenizer(num_items=num_items, num_clusters_c1=2)
        tokenizer.fit_from_item_embeddings(torch.randn(num_items + 1, 16), seed=9)
        model = DirectGenerativeRecommender(config, tokenizer).eval()
        with torch.no_grad():
            _, c2_probs = model.code_distribution(torch.tensor([[1, 2, 3, 0, 0, 0]]))
        for c1 in range(1, model.max_c1 + 1):
            valid = set(tokenizer.trie.get_valid_second_tokens(c1))
            for c2 in range(1, model.max_c2 + 1):
                if c2 not in valid:
                    self.assertLess(c2_probs[0, c1, c2].item(), 1e-6)


if __name__ == "__main__":
    unittest.main()
