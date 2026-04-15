from __future__ import annotations

import unittest

from src.commitment_data import build_belief_r_commitment_control_records


def _record(pair_id: str, condition: str, initial_label: str, final_label: str) -> dict:
    return {
        "dataset": "belief_r",
        "pair_id": pair_id,
        "modus": "ponens",
        "relation_type": "If-Event-Then-MentalState",
        "answer_options": {
            "a": "A",
            "b": "B",
            "c": "C",
        },
        "initial_premises": ["If alpha, then beta", "alpha"],
        "update_premises": ["If gamma, then beta"],
        "revised_query": "query",
        "gold_initial_label": initial_label,
        "gold_final_label": final_label,
        "example_id": f"{pair_id}::{condition}",
        "condition": condition,
    }


class CommitmentDataTests(unittest.TestCase):
    def test_build_records_is_deterministic_and_binary_labeled(self) -> None:
        rows = []
        for index in range(6):
            pair_id = f"pair-{index}"
            initial = "a"
            final = "a" if index % 2 == 0 else "c"
            rows.append(_record(pair_id, "full_info", initial, final))
            rows.append(
                _record(
                    pair_id,
                    "incremental_no_overturn" if final == initial else "incremental_overturn_reasoning",
                    initial,
                    final,
                )
            )

        split_records_a, stats_a = build_belief_r_commitment_control_records(rows, seed=11)
        split_records_b, stats_b = build_belief_r_commitment_control_records(rows, seed=11)

        self.assertEqual(stats_a, stats_b)
        self.assertEqual(split_records_a, split_records_b)

        all_rows = split_records_a["train"] + split_records_a["dev"] + split_records_a["test"]
        self.assertTrue(all(row["control_label"] in {"preserve", "replace"} for row in all_rows))
        self.assertTrue(all(row["metadata"]["split"] in {"train", "dev", "test"} for row in all_rows))
        self.assertEqual(stats_a["total_pairs"], 6)
        self.assertEqual(stats_a["overall_control_counts"]["preserve"], 6)
        self.assertEqual(stats_a["overall_control_counts"]["replace"], 6)

    def test_pair_key_separates_shared_pair_id_across_modus(self) -> None:
        rows = [
            {
                **_record("shared", "full_info", "a", "a"),
                "modus": "ponens",
            },
            {
                **_record("shared", "incremental_no_overturn", "a", "a"),
                "modus": "ponens",
            },
            {
                **_record("shared", "full_info", "a", "c"),
                "modus": "tollens",
            },
            {
                **_record("shared", "incremental_overturn_reasoning", "a", "c"),
                "modus": "tollens",
            },
        ]
        _, stats = build_belief_r_commitment_control_records(rows, seed=5)
        self.assertEqual(stats["total_pairs"], 2)


if __name__ == "__main__":
    unittest.main()
