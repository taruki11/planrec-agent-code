import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from planrec.genrec.constraints import TokenSequenceTrie
from planrec.genrec.corpus import GenRecCorpusBuilder, write_jsonl
from planrec.genrec.rewards import recommendation_reward
from planrec.genrec.semantic_ids import ResidualKMeansSemanticTokenizer


class GenRecMainlineTests(unittest.TestCase):
    def setUp(self) -> None:
        embeddings = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.9, 0.1],
                [0.0, 0.0, 1.0],
                [0.1, 0.0, 0.9],
            ]
        )
        self.semantic_ids = ResidualKMeansSemanticTokenizer(num_items=6, num_levels=2, codebook_size=2)
        self.stats = self.semantic_ids.fit(embeddings, seed=7)

    def test_semantic_ids_are_unique_deterministic_and_round_trip(self) -> None:
        self.assertEqual(len(set(self.semantic_ids.item_to_code.values())), 6)
        for item_id in range(1, 7):
            self.assertEqual(self.semantic_ids.item_from_tokens(self.semantic_ids.tokens_for_item(item_id)), item_id)

        duplicate = ResidualKMeansSemanticTokenizer(num_items=6, num_levels=2, codebook_size=2)
        embeddings = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0],
             [0.0, 0.9, 0.1], [0.0, 0.0, 1.0], [0.1, 0.0, 0.9]]
        )
        duplicate.fit(embeddings, seed=7)
        self.assertEqual(self.semantic_ids.item_to_code, duplicate.item_to_code)

    def test_save_load_preserves_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "semantic_ids.json"
            self.semantic_ids.save(path)
            loaded = ResidualKMeansSemanticTokenizer.load(path)
            self.assertEqual(loaded.item_to_code, self.semantic_ids.item_to_code)
            self.assertEqual(loaded.added_tokens(), self.semantic_ids.added_tokens())

    def test_corpus_has_real_autoregressive_target_and_hard_negative(self) -> None:
        builder = GenRecCorpusBuilder(self.semantic_ids, max_history=3)
        record = builder.sft_record([0, 1, 2, 3], 4)
        self.assertTrue(str(record["prompt"]).endswith("<rec_target>"))
        self.assertIn("<sid_l0_", str(record["completion"]))

        preference = builder.preference_record([1, 2, 3], 4)
        self.assertNotEqual(preference["target_item"], preference["rejected_item"])
        self.assertNotIn(preference["rejected_item"], [1, 2, 3])

    def test_trie_rejects_invalid_prefix(self) -> None:
        trie = TokenSequenceTrie.from_sequences([([10, 11, 12], 1), ([10, 13, 14], 2)])
        self.assertEqual(set(trie.allowed_next([10])), {11, 13})
        self.assertEqual(trie.decode([10, 11, 12]), 1)
        self.assertIsNone(trie.decode([10, 99]))

    def test_reward_orders_exact_valid_and_invalid(self) -> None:
        target = 1
        exact = " ".join(self.semantic_ids.completion_tokens_for_item(target))
        other = " ".join(self.semantic_ids.completion_tokens_for_item(2))
        self.assertGreater(
            recommendation_reward(exact, target, self.semantic_ids),
            recommendation_reward(other, target, self.semantic_ids),
        )
        self.assertGreater(
            recommendation_reward(other, target, self.semantic_ids),
            recommendation_reward("not-a-semantic-id", target, self.semantic_ids),
        )

    def test_jsonl_writer_preserves_prompt_completion_schema(self) -> None:
        builder = GenRecCorpusBuilder(self.semantic_ids)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sft.jsonl"
            count = write_jsonl(path, [builder.sft_record([1, 2], 3)])
            row = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(count, 1)
            self.assertEqual(set(row), {"prompt", "completion", "target_item"})

    def test_e6_sid_builder_uses_explicit_embedding_and_output_paths(self) -> None:
        """A rebuilt finite E6 embedding must not overwrite historical SID assets."""
        embeddings = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.9, 0.1],
                [0.0, 0.0, 1.0],
                [0.1, 0.0, 0.9],
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processed = root / "processed-without-default-assets"
            item_embeddings = root / "finite_embeddings.pt"
            output = root / "semantic_ids_finite_v1.json"
            torch.save(embeddings, item_embeddings)

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "build_tencentgr_e6_sid.py"),
                    "--processed",
                    str(processed),
                    "--item-embeddings",
                    str(item_embeddings),
                    "--output",
                    str(output),
                    "--num-levels",
                    "2",
                    "--codebook-size",
                    "2",
                    "--seed",
                    "7",
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(output.is_file())
            self.assertFalse((processed / "semantic_ids.json").exists())
            built = ResidualKMeansSemanticTokenizer.load(output)
            self.assertEqual(len(set(built.item_to_code.values())), 6)
            for item_id in range(1, 7):
                self.assertEqual(built.item_from_tokens(built.tokens_for_item(item_id)), item_id)


if __name__ == "__main__":
    unittest.main()
