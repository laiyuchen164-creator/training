from __future__ import annotations

import unittest

from src.llm_client import parse_label_from_text, parse_payload_from_text


class LLMClientParsingTest(unittest.TestCase):
    def test_parse_json_label(self) -> None:
        label, mode = parse_label_from_text('{"label":"b","rationale":"x"}')
        self.assertEqual(label, "b")
        self.assertEqual(mode, "json")

    def test_parse_tagged_label(self) -> None:
        label, mode = parse_label_from_text("Final answer: c")
        self.assertEqual(label, "c")
        self.assertEqual(mode, "tagged")

    def test_parse_missing_label(self) -> None:
        label, mode = parse_label_from_text("no clear answer")
        self.assertIsNone(label)
        self.assertEqual(mode, "unparsed")

    def test_parse_structured_payload(self) -> None:
        payload, mode = parse_payload_from_text(
            '{"label":"c","premise_role":"extra_requirement","relation_to_prior":"replace"}'
        )
        self.assertEqual(mode, "json")
        self.assertIsNotNone(payload)
        self.assertEqual(payload["premise_role"], "extra_requirement")


if __name__ == "__main__":
    unittest.main()
