from __future__ import annotations

import unittest

import torch

from src.models.hf_commitment_control_model import compute_conditional_propagation_loss


class HFConditionalPropagationLossTests(unittest.TestCase):
    def test_preserve_and_replace_terms_follow_control_label(self) -> None:
        answer_logits = torch.log(
            torch.tensor(
                [
                    [0.8, 0.1, 0.1],
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
            beta_replace_margin=0.2,
            margin_m=0.5,
        )

        self.assertAlmostEqual(payload["preserve_propagation_loss"].item(), -torch.log(torch.tensor(0.8)).item(), places=5)
        self.assertAlmostEqual(payload["replace_alignment_loss"].item(), -torch.log(torch.tensor(0.2)).item(), places=5)

        expected_margin = max(0.0, 0.5 - (torch.log(torch.tensor(0.2)) - torch.log(torch.tensor(0.7))).item())
        self.assertAlmostEqual(payload["replace_margin_loss"].item(), expected_margin, places=5)

        expected_replace_total = -torch.log(torch.tensor(0.2)).item() + 0.2 * expected_margin
        expected_total = 0.5 * (-torch.log(torch.tensor(0.8)).item() + expected_replace_total)
        self.assertAlmostEqual(payload["propagation_loss"].item(), expected_total, places=5)

    def test_weaken_rows_stay_neutral(self) -> None:
        payload = compute_conditional_propagation_loss(
            answer_logits=torch.zeros((1, 3), dtype=torch.float32),
            control_labels=torch.tensor([1], dtype=torch.long),
            answer_labels=torch.tensor([2], dtype=torch.long),
            early_answer_labels=torch.tensor([0], dtype=torch.long),
            control_to_idx={"preserve": 0, "weaken": 1, "replace": 2},
            beta_replace_margin=0.2,
            margin_m=0.5,
        )

        self.assertEqual(payload["propagation_loss"].item(), 0.0)
        self.assertEqual(payload["preserve_propagation_loss"].item(), 0.0)
        self.assertEqual(payload["replace_alignment_loss"].item(), 0.0)
        self.assertEqual(payload["replace_margin_loss"].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
