import types
import unittest
import tempfile
from pathlib import Path

import torch

from planrec.models.effect_dual import EffectDualConfig, EffectDualRecommender
import scripts.train_effect_dual as runner
from scripts.train_effect_dual import validate_resume_feature_lineage


class EffectDualTests(unittest.TestCase):
    def test_validation_schedule_runs_on_interval_or_final_epoch_and_rejects_zero_interval(self) -> None:
        self.assertFalse(runner.should_validate_epoch(epoch=1, total_epochs=5, interval=5))
        self.assertTrue(runner.should_validate_epoch(epoch=5, total_epochs=5, interval=7))
        self.assertTrue(runner.should_validate_epoch(epoch=5, total_epochs=10, interval=5))
        with self.assertRaisesRegex(ValueError, "positive"):
            runner.should_validate_epoch(epoch=1, total_epochs=5, interval=0)

    def test_selection_metric_uses_requested_ndcg_and_rejects_unknown_metric(self) -> None:
        validation = {
            "direct": {"ndcg@10": 0.125},
            "hybrid": {"ndcg@10": 0.375},
        }
        self.assertEqual(runner.select_validation_score(validation, "direct_ndcg"), 0.125)
        self.assertEqual(runner.select_validation_score(validation, "hybrid_ndcg"), 0.375)
        with self.assertRaisesRegex(ValueError, "selection metric"):
            runner.select_validation_score(validation, "sid_ndcg")

    def test_fixed_indices_select_exact_nonprefix_rows(self) -> None:
        x = torch.tensor([[10], [11], [12], [13], [14]])
        y = torch.tensor([20, 21, 22, 23, 24])
        selected_x, selected_y, metadata = runner.select_evaluation_rows(
            x, y, indices=torch.tensor([4, 1]), split_name="val"
        )
        self.assertEqual(selected_x.tolist(), [[14], [11]])
        self.assertEqual(selected_y.tolist(), [24, 21])
        self.assertEqual(metadata["count"], 2)

    def test_fixed_indices_reject_duplicate_and_out_of_range_values(self) -> None:
        x = torch.zeros(3, 2)
        y = torch.zeros(3, dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "unique"):
            runner.select_evaluation_rows(x, y, indices=torch.tensor([1, 1]), split_name="val")
        with self.assertRaisesRegex(ValueError, "out of range"):
            runner.select_evaluation_rows(x, y, indices=torch.tensor([3]), split_name="val")

    def test_required_fixed_indices_reject_missing_path(self) -> None:
        args = types.SimpleNamespace(
            require_fixed_user_indices=True,
            val_user_indices=None,
            test_user_indices=None,
        )
        with self.assertRaisesRegex(ValueError, "--val-user-indices"):
            runner.validate_fixed_index_arguments(args)

    def test_fixed_index_hash_file_is_canonical_and_requires_both_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            selections = {
                "val": {"path": "C:/indices/val.pt", "count": 4096, "file_sha256": "a" * 64, "tensor_sha256": "c" * 64},
                "test": {"path": "C:/indices/test.pt", "count": 4096, "file_sha256": "b" * 64, "tensor_sha256": "d" * 64},
            }
            path = runner.write_fixed_user_indices_sha256(output, selections)
            self.assertEqual(
                path.read_text(encoding="utf-8"),
                f"tensor_sha256  {'c' * 64}  C:/indices/val.pt\n"
                f"tensor_sha256  {'d' * 64}  C:/indices/test.pt\n",
            )
            self.assertIsNone(runner.write_fixed_user_indices_sha256(output, {"val": None, "test": None}))
            with self.assertRaisesRegex(ValueError, "both"):
                runner.write_fixed_user_indices_sha256(output, {"val": selections["val"], "test": None})

    def test_selected_index_metadata_has_file_and_canonical_tensor_hashes(self) -> None:
        x = torch.tensor([[10], [11], [12]])
        y = torch.tensor([20, 21, 22])
        with tempfile.TemporaryDirectory() as temporary:
            index_path = Path(temporary) / "indices.pt"
            indices = torch.tensor([2, 0], dtype=torch.long)
            torch.save(indices, index_path)
            _, _, metadata = runner.select_evaluation_rows(
                x, y, indices=indices, split_name="val", source_path=index_path
            )
            self.assertEqual(metadata["file_sha256"], runner.sha256(index_path))
            self.assertEqual(metadata["tensor_sha256"], runner.tensor_sha256(indices))

    def test_per_user_metrics_average_to_existing_ranking_semantics(self) -> None:
        logits = torch.tensor([[0.1, 0.8, 0.2], [0.9, 0.1, 0.8]])
        targets = torch.tensor([1, 2])
        values = runner.per_user_ranking_metrics(logits, targets, k=2)
        self.assertEqual(values["hit_rate@2"], [1.0, 1.0])
        self.assertEqual(values["mrr@2"], [1.0, 0.5])
        self.assertAlmostEqual(sum(values["ndcg@2"]) / 2, (1.0 + 1.0 / torch.log2(torch.tensor(3.0)).item()) / 2)

    def test_evaluation_labels_short_catalog_topk_metrics_as_at_10(self) -> None:
        output = runner.evaluate(
            self.model,
            torch.tensor([[1, 2, 0, 0, 0]]),
            torch.tensor([3]),
            self.codes,
            batch_size=1,
            catalog_chunk_size=2,
            hybrid_sid_weight=1.0,
            device=torch.device("cpu"),
        )
        self.assertEqual(set(output["direct"]), {"hit_rate@10", "mrr@10", "ndcg@10", "per_user"})

    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = EffectDualRecommender(
            EffectDualConfig(
                num_items=6,
                max_seq_len=5,
                sid_vocab_sizes=(3, 3, 2),
                hidden_size=16,
                num_layers=2,
                dropout=0.0,
            )
        )
        self.codes = torch.tensor(
            [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0]],
            dtype=torch.long,
        )

    def test_joint_loss_and_catalog_scores_are_finite(self) -> None:
        sequence = torch.tensor([[1, 2, 3, 0, 0], [2, 4, 5, 6, 0]])
        targets = torch.tensor([4, 1])
        losses = self.model.loss(sequence, targets, self.codes[targets])
        self.assertTrue(torch.isfinite(losses["loss"]))
        losses["loss"].backward()
        with torch.no_grad():
            states = self.model.encode(sequence)
            scores = self.model.catalog_sid_scores(states, self.codes[1:])
        self.assertEqual(scores.shape, (2, 6))
        self.assertTrue(torch.isfinite(scores).all())

    def test_catalog_sid_scores_matches_naive_formula_and_shares_prefix_states(self) -> None:
        torch.manual_seed(23)
        model = EffectDualRecommender(
            EffectDualConfig(
                num_items=12,
                max_seq_len=5,
                sid_vocab_sizes=(4, 4, 4, 16),
                hidden_size=8,
                num_layers=1,
                dropout=0.0,
            )
        )
        user_state = torch.randn(3, 8)
        candidate_codes = torch.randint(0, 4, (12, 4), dtype=torch.long)
        candidate_codes[:, 3] = torch.arange(12)
        candidate_codes[1, :3] = candidate_codes[0, :3]
        candidate_codes[2, :2] = candidate_codes[0, :2]

        # Independent copy of the pre-cache candidate-expanded autoregressive formula.
        batch_size, candidate_count = user_state.shape[0], candidate_codes.shape[0]
        naive_state = user_state[:, None, :].expand(-1, candidate_count, -1).reshape(-1, 8)
        naive_codes = candidate_codes[None, :, :].expand(batch_size, -1, -1).reshape(-1, 4)
        naive_scores = naive_state.new_zeros(batch_size * candidate_count)
        for level, head in enumerate(model.sid_heads):
            naive_log_prob = torch.log_softmax(head(naive_state), dim=-1)
            naive_scores += naive_log_prob.gather(1, naive_codes[:, level : level + 1]).squeeze(1)
            if level < len(model.sid_cells):
                naive_state = model.sid_cells[level](
                    model.sid_embeddings[level](naive_codes[:, level]), naive_state
                )
        naive_scores = naive_scores.view(batch_size, candidate_count)

        head_input_rows: list[int] = []
        hooks = [
            head.register_forward_hook(lambda _module, inputs, _output: head_input_rows.append(inputs[0].size(0)))
            for head in model.sid_heads
        ]
        try:
            cached_scores = model.catalog_sid_scores(user_state, candidate_codes)
        finally:
            for hook in hooks:
                hook.remove()

        torch.testing.assert_close(cached_scores, naive_scores, atol=1e-7, rtol=1e-7)
        expected_rows = [batch_size] + [
            batch_size * torch.unique(candidate_codes[:, :level], dim=0).size(0)
            for level in range(1, candidate_codes.size(1))
        ]
        self.assertEqual(head_input_rows, expected_rows)

    def test_evaluate_passes_full_sid_catalog_once_per_batch(self) -> None:
        calls: list[torch.Tensor] = []
        original = self.model.catalog_sid_scores

        def record_candidates(user_state: torch.Tensor, candidate_codes: torch.Tensor) -> torch.Tensor:
            calls.append(candidate_codes.detach().cpu().clone())
            return original(user_state, candidate_codes)

        self.model.catalog_sid_scores = record_candidates  # type: ignore[method-assign]
        runner.evaluate(
            self.model,
            torch.tensor([[1, 2, 0, 0, 0], [2, 3, 0, 0, 0], [3, 4, 0, 0, 0]]),
            torch.tensor([3, 4, 5]),
            self.codes,
            batch_size=2,
            catalog_chunk_size=2,
            hybrid_sid_weight=1.0,
            device=torch.device("cpu"),
        )
        self.assertEqual(len(calls), 2)
        for candidates in calls:
            self.assertTrue(torch.equal(candidates, self.codes[1:]))

    def test_extra_right_padding_does_not_change_state(self) -> None:
        self.model.eval()
        short = torch.tensor([[1, 2, 3]])
        padded = torch.tensor([[1, 2, 3, 0, 0]])
        with torch.no_grad():
            short_state = self.model.encode(short)
            padded_state = self.model.encode(padded)
        torch.testing.assert_close(short_state, padded_state, atol=1e-6, rtol=1e-6)

    def test_invalid_sid_shape_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.model.catalog_sid_scores(torch.zeros(2, 16), torch.zeros(6, 2, dtype=torch.long))

    def _feature_model(
        self,
        *,
        feature_mode: str,
        dense: torch.Tensor | None = None,
        sparse: torch.Tensor | None = None,
    ) -> EffectDualRecommender:
        return EffectDualRecommender(
            EffectDualConfig(
                num_items=6,
                max_seq_len=5,
                sid_vocab_sizes=(3, 3, 2),
                hidden_size=4,
                num_layers=1,
                dropout=0.0,
                dense_feature_dim=4,
                sparse_field_cardinalities=(4, 5),
                feature_mode=feature_mode,
            ),
            item_dense_features=dense,
            item_sparse_features=sparse,
        )

    def test_id_mode_is_finite_without_feature_tensors(self) -> None:
        model = self._feature_model(feature_mode="id")
        sequence = torch.tensor([[1, 2, 3, 0, 0]])
        targets = torch.tensor([4])
        codes = torch.tensor([[1, 0, 0]])
        losses = model.loss(sequence, targets, codes)
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertTrue(torch.isfinite(model.direct_logits(model.encode(sequence))).all())

    def test_dense_features_reject_nonfinite_catalog_values(self) -> None:
        dense = torch.zeros(7, 4)
        dense[2, 0] = float("nan")
        with self.assertRaises(ValueError):
            self._feature_model(feature_mode="dense", dense=dense)

    def test_enabled_dense_row_changes_representation_and_direct_score(self) -> None:
        dense = torch.zeros(7, 4)
        dense[2, 0] = 1.0
        altered = dense.clone()
        altered[2, 1] = 2.0
        torch.manual_seed(17)
        baseline = self._feature_model(feature_mode="dense", dense=dense)
        torch.manual_seed(17)
        changed = self._feature_model(feature_mode="dense", dense=altered)
        with torch.no_grad():
            baseline.dense_gate.fill_(1.0)
            changed.dense_gate.fill_(1.0)
            state = torch.tensor([[0.7, -0.4, 0.2, 0.9]])
            baseline_representation = baseline.item_representation(torch.tensor([2]))
            changed_representation = changed.item_representation(torch.tensor([2]))
            baseline_score = baseline.direct_logits(state)[:, 2]
            changed_score = changed.direct_logits(state)[:, 2]
        self.assertFalse(torch.allclose(baseline_representation, changed_representation))
        self.assertFalse(torch.allclose(baseline_score, changed_score))

    def test_enabled_sparse_row_changes_representation_and_direct_score(self) -> None:
        sparse = torch.zeros(7, 2, dtype=torch.long)
        sparse[2] = torch.tensor([1, 1])
        altered = sparse.clone()
        altered[2] = torch.tensor([2, 3])
        torch.manual_seed(23)
        baseline = self._feature_model(feature_mode="sparse", sparse=sparse)
        torch.manual_seed(23)
        changed = self._feature_model(feature_mode="sparse", sparse=altered)
        with torch.no_grad():
            baseline.sparse_gate.fill_(1.0)
            changed.sparse_gate.fill_(1.0)
            state = torch.tensor([[0.7, -0.4, 0.2, 0.9]])
            baseline_representation = baseline.item_representation(torch.tensor([2]))
            changed_representation = changed.item_representation(torch.tensor([2]))
            baseline_score = baseline.direct_logits(state)[:, 2]
            changed_score = changed.direct_logits(state)[:, 2]
        self.assertFalse(torch.allclose(baseline_representation, changed_representation))
        self.assertFalse(torch.allclose(baseline_score, changed_score))

    def test_feature_padding_row_is_exactly_zero_and_right_padding_is_invariant(self) -> None:
        dense = torch.randn(7, 4)
        sparse = torch.randint(1, 4, (7, 2))
        model = self._feature_model(feature_mode="dense_sparse", dense=dense, sparse=sparse)
        model.eval()
        with torch.no_grad():
            padding = model.item_representation(torch.tensor([0]))
            short_state = model.encode(torch.tensor([[1, 2, 3]]))
            padded_state = model.encode(torch.tensor([[1, 2, 3, 0, 0]]))
        torch.testing.assert_close(padding, torch.zeros_like(padding), atol=0.0, rtol=0.0)
        torch.testing.assert_close(short_state, padded_state, atol=1e-6, rtol=1e-6)

    def test_chunked_direct_catalog_scores_are_finite_and_shape_correct(self) -> None:
        dense = torch.randn(7, 4)
        sparse = torch.randint(1, 4, (7, 2))
        model = self._feature_model(feature_mode="dense_sparse", dense=dense, sparse=sparse)
        states = model.encode(torch.tensor([[1, 2, 3, 0, 0], [2, 4, 5, 6, 0]]))
        scores = model.catalog_direct_scores(states, catalog_chunk_size=2)
        self.assertEqual(scores.shape, (2, 7))
        self.assertTrue(torch.isfinite(scores).all())

    def test_resume_requires_exact_feature_lineage(self) -> None:
        dense_metadata = {
            "feature_mode": "dense",
            "item_dense_features_sha256": "dense-sha",
            "item_sparse_features_sha256": None,
        }
        validate_resume_feature_lineage(dense_metadata, dense_metadata)
        with self.assertRaisesRegex(ValueError, "clean output directory"):
            validate_resume_feature_lineage(None, {
                "feature_mode": "id",
                "item_dense_features_sha256": None,
                "item_sparse_features_sha256": None,
            })
        with self.assertRaisesRegex(ValueError, "clean output directory"):
            validate_resume_feature_lineage(
                {**dense_metadata, "item_dense_features_sha256": "other-sha"},
                dense_metadata,
            )


if __name__ == "__main__":
    unittest.main()
