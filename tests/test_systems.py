from __future__ import annotations

import unittest

from src.systems import _apply_source_revision_gate


class SourceRevisionGateTest(unittest.TestCase):
    def test_preserve_relations_keep_prior_label(self) -> None:
        final_label, gated = _apply_source_revision_gate(
            initial_prediction="a",
            model_prediction="c",
            relation_to_prior="elaborate",
        )
        self.assertEqual(final_label, "a")
        self.assertTrue(gated)

    def test_aggressive_relations_allow_revision(self) -> None:
        final_label, gated = _apply_source_revision_gate(
            initial_prediction="a",
            model_prediction="c",
            relation_to_prior="replace",
        )
        self.assertEqual(final_label, "c")
        self.assertFalse(gated)


if __name__ == "__main__":
    unittest.main()
