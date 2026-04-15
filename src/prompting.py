from __future__ import annotations

from functools import lru_cache
from pathlib import Path


PROMPTS_DIR = Path("prompts")


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def _join_lines(lines: list[str]) -> str:
    return "\n".join(f"- {line}" for line in lines)


def _prompt_family(example: dict) -> str:
    return example.get("prompt_family", "belief_revision_suppression")


def _belief_r_source_revision_prompt_name(example: dict) -> str:
    version = example.get("belief_r_source_revision_prompt_version", "v2")
    mapping = {
        "v2": "llm_followup_source_revision_belief_r_v2.txt",
        "v3": "llm_followup_source_revision_belief_r_v3.txt",
        "v4": "llm_followup_source_revision_belief_r_v4.txt",
    }
    return mapping.get(version, mapping["v2"])


def system_prompt(example: dict) -> str:
    template_name = {
        "belief_revision_suppression": "llm_answer_system_belief_r_v2.txt",
        "generic_logic_revision": "llm_answer_system_generic_logic_v1.txt",
    }[_prompt_family(example)]
    return load_prompt(template_name)


def full_info_user_prompt(example: dict) -> str:
    if _prompt_family(example) == "generic_logic_revision":
        return load_prompt("llm_full_info_generic_logic_v1.txt").format(
            premises="\n".join(example["revised_premises"]),
            query=example["revised_query"],
            **example["answer_options"],
        )
    return load_prompt("llm_full_info_belief_r_v2.txt").format(
        premise_1=example["revised_premises"][0],
        premise_2=example["revised_premises"][1],
        premise_3=example["revised_premises"][2],
        query=example["revised_query"],
        **example["answer_options"],
    )


def turn1_user_prompt(example: dict) -> str:
    template_name = {
        "belief_revision_suppression": "llm_turn1_belief_r_v2.txt",
        "generic_logic_revision": "llm_turn1_generic_logic_v1.txt",
    }[_prompt_family(example)]
    return load_prompt(template_name).format(
        premises="\n".join(example["initial_premises"]),
        query=example["initial_query"],
        **example["answer_options"],
    )


def followup_user_prompt(
    example: dict,
    system_name: str,
    *,
    previous_answer: str,
    previous_label: str,
) -> str:
    updates = "\n".join(example["update_premises"])
    common = {
        "updates": updates,
        "query": example["initial_query"],
        "previous_answer": previous_answer,
        "commitment": example["answer_options"][previous_label],
        "initial_premises": _join_lines(example["initial_premises"]),
        "initial_evidence": _join_lines(example["initial_premises"]),
        "summary": (
            "Initial premises:\n"
            f"{_join_lines(example['initial_premises'])}\n"
            f"Current answer guess: {example['answer_options'][previous_label]}"
        ),
        **example["answer_options"],
    }
    prompt_family = _prompt_family(example)
    template_name = {
        "belief_revision_suppression": {
            "raw_history": "llm_followup_raw_history_belief_r_v2.txt",
            "running_summary": "llm_followup_running_summary_belief_r_v2.txt",
            "structured_no_source": "llm_followup_structured_no_source_belief_r_v2.txt",
            "source_no_revision": "llm_followup_source_no_revision_belief_r_v2.txt",
            "source_revision": _belief_r_source_revision_prompt_name(example),
        },
        "generic_logic_revision": {
            "raw_history": "llm_followup_raw_history_v1.txt",
            "running_summary": "llm_followup_running_summary_v1.txt",
            "structured_no_source": "llm_followup_structured_no_source_v1.txt",
            "source_no_revision": "llm_followup_source_no_revision_generic_logic_v1.txt",
            "source_revision": "llm_followup_source_revision_generic_logic_v1.txt",
        },
    }[prompt_family][system_name]
    return load_prompt(template_name).format(**common)
