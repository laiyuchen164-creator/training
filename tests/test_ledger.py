from __future__ import annotations

import unittest

from src.ledger import BeliefLedger


class LedgerTest(unittest.TestCase):
    def test_add_and_confirm(self) -> None:
        ledger = BeliefLedger()
        belief = ledger.add_belief(
            content="John learns something new.",
            belief_type="intermediate_conclusion",
            status="tentative",
            source="assistant_inferred",
            turn_id=1,
            confidence=0.8,
        )
        ledger.confirm_belief(belief.belief_id, turn_id=2)
        self.assertEqual(ledger.get(belief.belief_id).status, "confirmed")

    def test_revise_transition(self) -> None:
        ledger = BeliefLedger()
        belief = ledger.add_belief(
            content="John learns something new.",
            belief_type="intermediate_conclusion",
            status="confirmed",
            source="assistant_inferred",
            turn_id=1,
            confidence=0.8,
        )
        revised = ledger.revise_belief(
            belief.belief_id,
            new_content="John may or may not learn something new.",
            turn_id=2,
            source="assistant_inferred",
            confidence=0.7,
            metadata={"label": "c"},
        )
        self.assertEqual(ledger.get(belief.belief_id).status, "corrected")
        self.assertEqual(revised.parent_or_conflict_target, belief.belief_id)
        self.assertEqual(revised.status, "confirmed")

    def test_deprecate_transition(self) -> None:
        ledger = BeliefLedger()
        belief = ledger.add_belief(
            content="John learns something new.",
            belief_type="intermediate_conclusion",
            status="tentative",
            source="assistant_inferred",
            turn_id=1,
            confidence=0.8,
        )
        ledger.deprecate_belief(belief.belief_id, turn_id=2)
        self.assertEqual(ledger.get(belief.belief_id).status, "deprecated")

    def test_unresolved_transition(self) -> None:
        ledger = BeliefLedger()
        belief = ledger.add_belief(
            content="John learns something new.",
            belief_type="intermediate_conclusion",
            status="confirmed",
            source="assistant_inferred",
            turn_id=1,
            confidence=0.8,
        )
        ledger.mark_unresolved(belief.belief_id, turn_id=2)
        self.assertEqual(ledger.get(belief.belief_id).status, "unresolved")


if __name__ == "__main__":
    unittest.main()
