from __future__ import annotations

import unittest

import torch

from src.models.hf_commitment_control_model import (
    compute_boundary_propagation_loss_replace_margin_stopgrad_early,
    compute_conditionally_masked_answer_loss,
    compute_boundary_propagation_loss,
    compute_conditional_propagation_loss,
)


class HFConditionalPropagationLossTests(unittest.TestCase):
    def test_preserve_and_replace_terms_follow_control_label(self) -> None:
        answer_logits = torch.log(
            torch.tensor(
                [
                    [0.8, 0.15, 0.05],
                    [0.7, 0.1, 0.2],
                ],
                dtype=torch.float32,
            )
        )
        payload = compute_conditional_propagation_loss(
            answer_logits=answer_logits,
            control_labels=torch.tensor([0, 2], dtype=torch.long),
            answer_labels=torch.tensor([0, 2], dtype=torch.long),
            early_answer_labels=torch.tensor([0, 0], dtype=torch.long),
            control_to_idx={"preserve": 0, "weaken": 1, "replace": 2},
            lambda_pres=0.2,
            lambda_rep=0.08,
            beta_preserve_margin=0.1,
            preserve_margin_m=0.5,
            beta_replace_margin=0.2,
            margin_m=0.5,
        )

        expected_preserve_ce = -torch.log(torch.tensor(0.8)).item()
        expected_preserve_margin = max(0.0, 0.5 - (torch.log(torch.tensor(0.8)) - torch.log(torch.tensor(0.15))).item())
        expected_preserve_total = expected_preserve_ce + 0.1 * expected_preserve_margin
        self.assertAlmostEqual(payload["preserve_ce_loss"].item(), expected_preserve_ce, places=5)
        self.assertAlmostEqual(payload["preserve_margin_loss"].item(), expected_preserve_margin, places=5)
        self.assertAlmostEqual(payload["preserve_propagation_loss"].item(), expected_preserve_total, places=5)
        self.assertAlmostEqual(payload["replace_alignment_loss"].item(), -torch.log(torch.tensor(0.2)).item(), places=5)

        expected_margin = max(0.0, 0.5 - (torch.log(torch.tensor(0.2)) - torch.log(torch.tensor(0.7))).item())
        self.assertAlmostEqual(payload["replace_margin_loss"].item(), expected_margin, places=5)

        expected_replace_total = -torch.log(torch.tensor(0.2)).item() + 0.2 * expected_margin
        self.assertAlmostEqual(payload["replace_propagation_loss"].item(), expected_replace_total, places=5)
        expected_total = 0.2 * expected_preserve_total + 0.08 * expected_replace_total
        self.assertAlmostEqual(payload["propagation_loss"].item(), expected_total, places=5)

    def test_weaken_rows_stay_neutral(self) -> None:
        payload = compute_conditional_propagation_loss(
            answer_logits=torch.zeros((1, 3), dtype=torch.float32),
            control_labels=torch.tensor([1], dtype=torch.long),
            answer_labels=torch.tensor([2], dtype=torch.long),
            early_answer_labels=torch.tensor([0], dtype=torch.long),
            control_to_idx={"preserve": 0, "weaken": 1, "replace": 2},
            lambda_pres=0.2,
            lambda_rep=0.08,
            beta_preserve_margin=0.1,
            preserve_margin_m=0.5,
            beta_replace_margin=0.2,
            margin_m=0.5,
        )

        self.assertEqual(payload["propagation_loss"].item(), 0.0)
        self.assertEqual(payload["preserve_propagation_loss"].item(), 0.0)
        self.assertEqual(payload["preserve_ce_loss"].item(), 0.0)
        self.assertEqual(payload["preserve_margin_loss"].item(), 0.0)
        self.assertEqual(payload["replace_propagation_loss"].item(), 0.0)
        self.assertEqual(payload["replace_alignment_loss"].item(), 0.0)
        self.assertEqual(payload["replace_margin_loss"].item(), 0.0)


class HFBoundaryPropagationLossTests(unittest.TestCase):
    def test_boundary_terms_use_log_prob_scores(self) -> None:
        answer_logits = torch.log(
            torch.tensor(
                [
                    [0.8, 0.15, 0.05],
                    [0.7, 0.1, 0.2],
                ],
                dtype=torch.float32,
            )
        )
        payload = compute_boundary_propagation_loss(
            answer_logits=answer_logits,
            control_labels=torch.tensor([0, 2], dtype=torch.long),
            answer_labels=torch.tensor([0, 2], dtype=torch.long),
            early_answer_labels=torch.tensor([0, 0], dtype=torch.long),
            control_to_idx={"preserve": 0, "weaken": 1, "replace": 2},
            lambda_pres=0.2,
            lambda_rep=0.08,
            beta_pres=0.1,
            beta_rep=0.05,
            m_pres=0.3,
            m_rep=0.5,
        )

        preserve_s_early = torch.log(torch.tensor(0.8)).item()
        preserve_max_non_early = torch.log(torch.tensor(0.15)).item()
        expected_preserve_ce = -preserve_s_early
        expected_preserve_margin = max(0.0, 0.3 - (preserve_s_early - preserve_max_non_early))
        expected_preserve_total = expected_preserve_ce + 0.1 * expected_preserve_margin

        replace_s_gold = torch.log(torch.tensor(0.2)).item()
        replace_s_early = torch.log(torch.tensor(0.7)).item()
        expected_replace_ce = -replace_s_gold
        expected_replace_margin = max(0.0, 0.5 - (replace_s_gold - replace_s_early))
        expected_replace_total = expected_replace_ce + 0.05 * expected_replace_margin
        expected_total = 0.2 * expected_preserve_total + 0.08 * expected_replace_total

        self.assertAlmostEqual(payload["preserve_ce_loss"].item(), expected_preserve_ce, places=5)
        self.assertAlmostEqual(payload["preserve_margin_loss"].item(), expected_preserve_margin, places=5)
        self.assertAlmostEqual(payload["preserve_propagation_loss"].item(), expected_preserve_total, places=5)
        self.assertAlmostEqual(payload["replace_alignment_loss"].item(), expected_replace_ce, places=5)
        self.assertAlmostEqual(payload["replace_margin_loss"].item(), expected_replace_margin, places=5)
        self.assertAlmostEqual(payload["replace_propagation_loss"].item(), expected_replace_total, places=5)
        self.assertAlmostEqual(payload["propagation_loss"].item(), expected_total, places=5)

    def test_boundary_weaken_rows_stay_neutral(self) -> None:
        payload = compute_boundary_propagation_loss(
            answer_logits=torch.zeros((1, 3), dtype=torch.float32),
            control_labels=torch.tensor([1], dtype=torch.long),
            answer_labels=torch.tensor([2], dtype=torch.long),
            early_answer_labels=torch.tensor([0], dtype=torch.long),
            control_to_idx={"preserve": 0, "weaken": 1, "replace": 2},
            lambda_pres=0.2,
            lambda_rep=0.08,
            beta_pres=0.1,
            beta_rep=0.05,
            m_pres=0.3,
            m_rep=0.5,
        )

        self.assertEqual(payload["propagation_loss"].item(), 0.0)
        self.assertEqual(payload["preserve_propagation_loss"].item(), 0.0)
        self.assertEqual(payload["preserve_ce_loss"].item(), 0.0)
        self.assertEqual(payload["preserve_margin_loss"].item(), 0.0)
        self.assertEqual(payload["replace_propagation_loss"].item(), 0.0)
        self.assertEqual(payload["replace_alignment_loss"].item(), 0.0)
        self.assertEqual(payload["replace_margin_loss"].item(), 0.0)


class HFConditionallyMaskedAnswerLossTests(unittest.TestCase):
    def test_conditional_targets_follow_control_label(self) -> None:
        answer_logits = torch.log(
            torch.tensor(
                [
                    [0.1, 0.8, 0.1],
                    [0.1, 0.2, 0.7],
                    [0.1, 0.7, 0.2],
                ],
                dtype=torch.float32,
            )
        )
        payload = compute_conditionally_masked_answer_loss(
            answer_logits=answer_logits,
            control_labels=torch.tensor([0, 2, 1], dtype=torch.long),
            answer_labels=torch.tensor([2, 2, 1], dtype=torch.long),
            early_answer_labels=torch.tensor([1, 0, 0], dtype=torch.long),
            control_to_idx={"preserve": 0, "weaken": 1, "replace": 2},
        )

        expected_preserve = -torch.log(torch.tensor(0.8)).item()
        expected_replace = -torch.log(torch.tensor(0.7)).item()
        expected_weaken = -torch.log(torch.tensor(0.7)).item()

        self.assertAlmostEqual(payload["answer_loss_preserve"].item(), expected_preserve, places=5)
        self.assertAlmostEqual(payload["answer_loss_replace"].item(), expected_replace, places=5)
        self.assertAlmostEqual(payload["answer_loss_weaken"].item(), expected_weaken, places=5)

    def test_non_empty_group_means_are_averaged(self) -> None:
        answer_logits = torch.log(
            torch.tensor(
                [
                    [0.1, 0.8, 0.1],
                    [0.2, 0.3, 0.5],
                    [0.1, 0.2, 0.7],
                    [0.6, 0.2, 0.2],
                ],
                dtype=torch.float32,
            )
        )
        payload = compute_conditionally_masked_answer_loss(
            answer_logits=answer_logits,
            control_labels=torch.tensor([0, 0, 2, 2], dtype=torch.long),
            answer_labels=torch.tensor([2, 2, 2, 0], dtype=torch.long),
            early_answer_labels=torch.tensor([1, 2, 0, 1], dtype=torch.long),
            control_to_idx={"preserve": 0, "weaken": 1, "replace": 2},
        )

        preserve_group = (
            -torch.log(torch.tensor(0.8)).item()
            + -torch.log(torch.tensor(0.5)).item()
        ) / 2.0
        replace_group = (
            -torch.log(torch.tensor(0.7)).item()
            + -torch.log(torch.tensor(0.6)).item()
        ) / 2.0
        expected_total = (preserve_group + replace_group) / 2.0

        self.assertAlmostEqual(payload["answer_loss_preserve"].item(), preserve_group, places=5)
        self.assertAlmostEqual(payload["answer_loss_replace"].item(), replace_group, places=5)
        self.assertEqual(payload["answer_loss_weaken"].item(), 0.0)
        self.assertAlmostEqual(payload["answer_loss"].item(), expected_total, places=5)


class HFBoundaryStopgradReplaceMarginTests(unittest.TestCase):
    def test_stopgrad_variant_matches_forward_values(self) -> None:
        answer_logits = torch.log(
            torch.tensor(
                [
                    [0.8, 0.15, 0.05],
                    [0.7, 0.1, 0.2],
                ],
                dtype=torch.float32,
            )
        )
        base = compute_boundary_propagation_loss(
            answer_logits=answer_logits,
            control_labels=torch.tensor([0, 2], dtype=torch.long),
            answer_labels=torch.tensor([0, 2], dtype=torch.long),
            early_answer_labels=torch.tensor([0, 0], dtype=torch.long),
            control_to_idx={"preserve": 0, "weaken": 1, "replace": 2},
            lambda_pres=0.2,
            lambda_rep=0.08,
            beta_pres=0.1,
            beta_rep=0.05,
            m_pres=0.3,
            m_rep=0.5,
        )
        stopgrad = compute_boundary_propagation_loss_replace_margin_stopgrad_early(
            answer_logits=answer_logits,
            control_labels=torch.tensor([0, 2], dtype=torch.long),
            answer_labels=torch.tensor([0, 2], dtype=torch.long),
            early_answer_labels=torch.tensor([0, 0], dtype=torch.long),
            control_to_idx={"preserve": 0, "weaken": 1, "replace": 2},
            lambda_pres=0.2,
            lambda_rep=0.08,
            beta_pres=0.1,
            beta_rep=0.05,
            m_pres=0.3,
            m_rep=0.5,
        )

        self.assertAlmostEqual(stopgrad["propagation_loss"].item(), base["propagation_loss"].item(), places=5)
        self.assertAlmostEqual(stopgrad["replace_propagation_loss"].item(), base["replace_propagation_loss"].item(), places=5)
        self.assertAlmostEqual(stopgrad["replace_margin_loss"].item(), base["replace_margin_loss"].item(), places=5)


if __name__ == "__main__":
    unittest.main()
