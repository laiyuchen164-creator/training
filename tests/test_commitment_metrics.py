from __future__ import annotations

import unittest

from src.eval.commitment_metrics import compute_commitment_metrics


class CommitmentMetricsTests(unittest.TestCase):
    def test_metrics_capture_persistence_and_takeover(self) -> None:
        examples = [
            {
                "control_label": "preserve",
                "final_answer_label": "a",
                "early_commitment_label": "a",
                "source_type": "assistant_inferred",
            },
            {
                "control_label": "replace",
                "final_answer_label": "c",
                "early_commitment_label": "a",
                "source_type": "assistant_inferred",
            },
        ]
        predictions = [
            {
                "predicted_control_decision": "preserve",
                "predicted_final_answer": "a",
            },
            {
                "predicted_control_decision": "replace",
                "predicted_final_answer": "a",
            },
        ]

        metrics = compute_commitment_metrics(examples, predictions)

        self.assertEqual(metrics["control_decision_accuracy"], 1.0)
        self.assertEqual(metrics["final_answer_accuracy"], 0.5)
        self.assertEqual(metrics["joint_accuracy"], 0.5)
        self.assertEqual(metrics["early_commitment_persistence"], 1.0)
        self.assertEqual(metrics["late_evidence_takeover"], 0.0)
        self.assertEqual(metrics["assistant_assumption_survival"], 1.0)
        self.assertEqual(metrics["correction_uptake"], 0.0)


if __name__ == "__main__":
    unittest.main()
