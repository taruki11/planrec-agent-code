import unittest

import torch
import torch.nn.functional as F

from planrec.models.effect_dual import EffectDualConfig, EffectDualRecommender
import scripts.train_effect_dual as runner


class GenV2DistillationTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(31)
        self.codes = torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
            ],
            dtype=torch.long,
        )
        self.sequence = torch.tensor([[1, 2, 3, 0, 0], [2, 4, 5, 6, 0]])
        self.targets = torch.tensor([4, 1])

    def _model(self, *, distill_weight: float) -> EffectDualRecommender:
        return EffectDualRecommender(
            EffectDualConfig(
                num_items=6,
                max_seq_len=5,
                sid_vocab_sizes=(3, 3, 2),
                hidden_size=8,
                num_layers=1,
                dropout=0.0,
                distill_weight=distill_weight,
                distill_temperature=2.0,
            )
        ).eval()

    def test_distillation_is_full_catalog_kl_from_detached_direct_teacher(self) -> None:
        model = self._model(distill_weight=1.0)
        with torch.no_grad():
            state = model.encode(self.sequence)
            teacher_scores = model.direct_logits(state)[:, 1:]
            sid_scores = model.catalog_sid_scores(state, self.codes[1:])
            expected = F.kl_div(
                F.log_softmax(sid_scores / 2.0, dim=1),
                F.softmax(teacher_scores.detach() / 2.0, dim=1),
                reduction="batchmean",
            ) * 4.0

        losses = model.loss(
            self.sequence,
            self.targets,
            self.codes[self.targets],
            catalog_codes=self.codes[1:],
        )

        torch.testing.assert_close(losses["distill_loss"], expected, atol=1e-7, rtol=1e-7)
        torch.testing.assert_close(
            losses["loss"],
            losses["direct_loss"] + model.config.sid_loss_weight * losses["sid_loss"] + expected,
            atol=1e-7,
            rtol=1e-7,
        )

    def test_zero_distillation_keeps_joint_loss_and_reports_zero_component(self) -> None:
        model = self._model(distill_weight=0.0)
        losses = model.loss(
            self.sequence,
            self.targets,
            self.codes[self.targets],
            catalog_codes=self.codes[1:],
        )

        self.assertEqual(losses["distill_loss"].item(), 0.0)
        torch.testing.assert_close(
            losses["loss"],
            losses["direct_loss"] + model.config.sid_loss_weight * losses["sid_loss"],
            atol=0.0,
            rtol=0.0,
        )

    def test_resume_rejects_model_config_with_different_distillation_settings(self) -> None:
        legacy = {"distill_weight": 0.0, "distill_temperature": 2.0, "hidden_size": 64}
        runner.validate_resume_model_config(legacy, legacy)
        with self.assertRaisesRegex(ValueError, "model config"):
            runner.validate_resume_model_config(
                legacy,
                {"distill_weight": 1.0, "distill_temperature": 2.0, "hidden_size": 64},
            )


if __name__ == "__main__":
    unittest.main()
