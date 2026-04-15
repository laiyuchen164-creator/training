from __future__ import annotations

import re
from dataclasses import dataclass


SYSTEM_THRESHOLDS = {
    "raw_history": 0.20,
    "running_summary": 0.15,
    "structured_no_source": 0.12,
    "source_no_revision": 0.12,
    "source_revision": 0.08,
}

FULL_INFO_THRESHOLD = 0.06


@dataclass
class RevisionSignal:
    overlap: float
    threshold: float
    should_revise: bool
    new_antecedent: str
    original_consequent: str


def _extract_conditional_parts(premise: str) -> tuple[str, str]:
    match = re.match(r"If (.*), then (.*)", premise)
    if not match:
        return premise.lower(), ""
    return match.group(1).strip().lower(), match.group(2).strip().lower()


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z']+", text.lower()))


def initial_label_for_modus(modus: str) -> str:
    if modus == "ponens":
        return "a"
    if modus == "tollens":
        return "b"
    if modus == "generic_true":
        return "a"
    if modus == "generic_false":
        return "b"
    if modus == "generic_uncertain":
        return "c"
    raise ValueError(f"Unsupported modus: {modus}")


def build_revision_signal(example: dict, system_name: str) -> RevisionSignal:
    threshold = (
        FULL_INFO_THRESHOLD
        if example["condition"] == "full_info"
        else SYSTEM_THRESHOLDS[system_name]
    )
    _, original_consequent = _extract_conditional_parts(example["initial_premises"][0])
    new_antecedent, _ = _extract_conditional_parts(example["revised_premises"][-1])
    antecedent_tokens = _tokens(new_antecedent)
    consequent_tokens = _tokens(original_consequent)
    numerator = len(antecedent_tokens & consequent_tokens)
    denominator = max(1, len(antecedent_tokens | consequent_tokens))
    overlap = numerator / denominator
    if len(antecedent_tokens) >= 12:
        overlap += 0.02
    return RevisionSignal(
        overlap=overlap,
        threshold=threshold,
        should_revise=overlap >= threshold,
        new_antecedent=new_antecedent,
        original_consequent=original_consequent,
    )


def predict_final_label(example: dict, system_name: str) -> tuple[str, RevisionSignal]:
    initial_label = initial_label_for_modus(example["modus"])
    if example.get("dataset") == "reviseqa_incremental":
        signal = RevisionSignal(
            overlap=0.0,
            threshold=1.0,
            should_revise=False,
            new_antecedent="",
            original_consequent="",
        )
        return initial_label, signal
    signal = build_revision_signal(example, system_name)
    return ("c" if signal.should_revise else initial_label), signal
