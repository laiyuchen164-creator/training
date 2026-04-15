from __future__ import annotations

import unittest

from src.prompting import full_info_user_prompt, system_prompt, turn1_user_prompt


class PromptingSelectionTest(unittest.TestCase):
    def test_generic_logic_prompts_support_variable_premises(self) -> None:
        example = {
            "prompt_family": "generic_logic_revision",
            "initial_premises": ["P1", "P2", "P3", "P4"],
            "revised_premises": ["Q1", "Q2", "Q3", "Q4", "Q5"],
            "initial_query": "Initial query",
            "revised_query": "Revised query",
            "answer_options": {"a": "A", "b": "B", "c": "C"},
        }

        self.assertIn("explicit context edits", system_prompt(example))
        self.assertIn("Q5", full_info_user_prompt(example))
        self.assertIn("P4", turn1_user_prompt(example))


if __name__ == "__main__":
    unittest.main()
