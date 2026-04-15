from __future__ import annotations

from dataclasses import asdict, dataclass

from src.heuristics import RevisionSignal, initial_label_for_modus, predict_final_label
from src.llm_client import ChatAPIClient
from src.ledger import BeliefLedger
from src.prompting import (
    followup_user_prompt,
    full_info_user_prompt,
    system_prompt,
    turn1_user_prompt,
)
from src.utils import estimate_tokens

ALTERNATIVE_ROLES = {"alternative_pathway"}
REVISION_ROLES = {"extra_requirement", "contradiction"}
AGGRESSIVE_RELATIONS = {"contradict", "replace"}
PRESERVE_RELATIONS = {"confirm", "elaborate", "unrelated"}


@dataclass
class RunRecord:
    example_id: str
    system: str
    condition: str
    gold_initial_label: str
    gold_final_label: str
    initial_prediction: str | None
    final_prediction: str
    prompt_tokens_approx: int
    api_calls: int
    revision_signal: dict
    trace: dict


def _build_trace_payload(
    *,
    example: dict,
    system_name: str,
    ledger: BeliefLedger,
    turns: list[dict],
    signal: dict,
) -> dict:
    return {
        "example_id": example["example_id"],
        "system": system_name,
        "condition": example["condition"],
        "pair_id": example["pair_id"],
        "modus": example["modus"],
        "turns": turns,
        "final_snapshot": ledger.snapshot(),
        "revision_signal": signal,
    }


def _normalize_tag(value: object, allowed: set[str], default: str) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace("/", "_")
    if normalized == "n/a":
        normalized = "n_a"
    return normalized if normalized in allowed else default


def _premise_role_from_response(response: LLMResponse) -> str:
    return _normalize_tag(
        (response.parsed_payload or {}).get("premise_role"),
        {"alternative_pathway", "extra_requirement", "contradiction", "unclear"},
        "unclear",
    )


def _relation_to_prior_from_response(
    *,
    example: dict,
    response: LLMResponse,
    initial_prediction: str,
    model_prediction: str,
) -> str:
    parsed_relation = _normalize_tag(
        (response.parsed_payload or {}).get("relation_to_prior"),
        {"confirm", "elaborate", "contradict", "replace", "unrelated", "unclear", "n_a"},
        "unclear",
    )
    if parsed_relation not in {"unclear", "n_a"}:
        return parsed_relation

    premise_role = _premise_role_from_response(response)
    if example.get("prompt_family") == "belief_revision_suppression":
        if premise_role in ALTERNATIVE_ROLES:
            return "elaborate"
        if premise_role in REVISION_ROLES:
            return "replace"

    if model_prediction == initial_prediction:
        return "confirm"
    return "replace"


def _apply_source_revision_gate(
    *,
    initial_prediction: str,
    model_prediction: str,
    relation_to_prior: str,
) -> tuple[str, bool]:
    if relation_to_prior in PRESERVE_RELATIONS:
        return initial_prediction, model_prediction != initial_prediction
    return model_prediction, False


def _run_heuristic_system_on_example(example: dict, system_name: str) -> RunRecord:
    ledger = BeliefLedger()
    turns = []

    if example["condition"] == "full_info":
        for premise in example["revised_premises"]:
            ledger.add_belief(
                content=premise,
                belief_type="premise",
                status="confirmed",
                source="user_explicit",
                turn_id=1,
                confidence=1.0,
            )
        final_prediction, signal = predict_final_label(example, system_name)
        ledger.add_belief(
            content=example["answer_options"][final_prediction],
            belief_type="intermediate_conclusion",
            status="confirmed",
            source="assistant_inferred",
            turn_id=1,
            confidence=0.7,
            metadata={"label": final_prediction},
        )
        turns.append(
            {
                "turn_id": 1,
                "prediction": final_prediction,
                "ledger": ledger.snapshot(),
            }
        )
        trace = _build_trace_payload(
            example=example,
            system_name=system_name,
            ledger=ledger,
            turns=turns,
            signal=asdict(signal),
        )
        return RunRecord(
            example_id=example["example_id"],
            system=system_name,
            condition=example["condition"],
            gold_initial_label=example["gold_initial_label"],
            gold_final_label=example["gold_final_label"],
            initial_prediction=None,
            final_prediction=final_prediction,
            prompt_tokens_approx=estimate_tokens(
                *example["revised_premises"],
                example["revised_query"],
            ),
            api_calls=1,
            revision_signal=asdict(signal),
            trace=trace,
        )

    initial_label = initial_label_for_modus(example["modus"])
    for premise in example["initial_premises"]:
        ledger.add_belief(
            content=premise,
            belief_type="premise",
            status="confirmed",
            source="user_explicit",
            turn_id=1,
            confidence=1.0,
        )

    initial_status = "tentative" if system_name == "source_revision" else "confirmed"
    initial_commitment = ledger.add_belief(
        content=example["answer_options"][initial_label],
        belief_type="intermediate_conclusion",
        status=initial_status,
        source="assistant_inferred",
        turn_id=1,
        confidence=0.8,
        metadata={"label": initial_label},
    )
    turns.append(
        {
            "turn_id": 1,
            "prediction": initial_label,
            "ledger": ledger.snapshot(),
        }
    )

    for premise in example["update_premises"]:
        ledger.add_belief(
            content=premise,
            belief_type="premise",
            status="confirmed",
            source="user_explicit",
            turn_id=2,
            confidence=1.0,
        )

    final_prediction, signal = predict_final_label(example, system_name)

    if final_prediction == "c":
        if system_name == "source_revision":
            ledger.deprecate_belief(initial_commitment.belief_id, turn_id=2)
            ledger.add_belief(
                content=example["answer_options"]["c"],
                belief_type="intermediate_conclusion",
                status="confirmed",
                source="assistant_inferred",
                turn_id=2,
                confidence=0.7,
                parent_or_conflict_target=initial_commitment.belief_id,
                metadata={"label": "c"},
            )
        elif system_name in {"structured_no_source", "source_no_revision"}:
            ledger.mark_unresolved(initial_commitment.belief_id, turn_id=2)
            ledger.add_belief(
                content=example["answer_options"]["c"],
                belief_type="intermediate_conclusion",
                status="confirmed",
                source="assistant_inferred",
                turn_id=2,
                confidence=0.7,
                parent_or_conflict_target=initial_commitment.belief_id,
                metadata={"label": "c"},
            )
        else:
            ledger.revise_belief(
                initial_commitment.belief_id,
                new_content=example["answer_options"]["c"],
                turn_id=2,
                source="assistant_inferred",
                confidence=0.7,
                metadata={"label": "c"},
            )
    else:
        if system_name == "source_revision":
            ledger.confirm_belief(initial_commitment.belief_id, turn_id=2)

    turns.append(
        {
            "turn_id": 2,
            "prediction": final_prediction,
            "ledger": ledger.snapshot(),
        }
    )

    trace = _build_trace_payload(
        example=example,
        system_name=system_name,
        ledger=ledger,
        turns=turns,
        signal=asdict(signal),
    )
    return RunRecord(
        example_id=example["example_id"],
        system=system_name,
        condition=example["condition"],
        gold_initial_label=example["gold_initial_label"],
        gold_final_label=example["gold_final_label"],
        initial_prediction=initial_label,
        final_prediction=final_prediction,
        prompt_tokens_approx=estimate_tokens(
            *example["initial_premises"],
            *example["update_premises"],
            example["initial_query"],
        ),
        api_calls=2,
        revision_signal=asdict(signal),
        trace=trace,
    )


def _build_turn_messages(
    example: dict,
    system_name: str,
    *,
    full_info: bool = False,
    previous_answer: str | None = None,
    previous_label: str | None = None,
) -> list[dict]:
    messages = [{"role": "system", "content": system_prompt(example)}]
    if previous_answer is None or previous_label is None:
        content = full_info_user_prompt(example) if full_info else turn1_user_prompt(example)
        messages.append({"role": "user", "content": content})
        return messages

    if system_name == "raw_history":
        messages.append({"role": "user", "content": turn1_user_prompt(example)})
        messages.append({"role": "assistant", "content": previous_answer})
        messages.append(
            {
                "role": "user",
                "content": followup_user_prompt(
                    example,
                    system_name,
                    previous_answer=previous_answer,
                    previous_label=previous_label,
                ),
            }
        )
        return messages

    messages.append(
        {
            "role": "user",
            "content": followup_user_prompt(
                example,
                system_name,
                previous_answer=previous_answer,
                previous_label=previous_label,
            ),
        }
    )
    return messages


def _llm_label_or_fallback(example: dict, system_name: str, response_label: str | None) -> str:
    if response_label in {"a", "b", "c"}:
        return response_label
    heuristic_label, _ = predict_final_label(example, system_name)
    return heuristic_label


def _run_llm_system_on_example(
    example: dict,
    system_name: str,
    backend_config: dict,
) -> RunRecord:
    client = ChatAPIClient(backend_config)
    ledger = BeliefLedger()
    turns = []
    prompt_tokens_total = 0
    api_calls = 0

    if example["condition"] == "full_info":
        for premise in example["revised_premises"]:
            ledger.add_belief(
                content=premise,
                belief_type="premise",
                status="confirmed",
                source="user_explicit",
                turn_id=1,
                confidence=1.0,
            )
        full_example = dict(example)
        full_example["initial_premises"] = example["revised_premises"]
        full_example["initial_query"] = example["revised_query"]
        messages = _build_turn_messages(full_example, system_name, full_info=True)
        response = client.complete(messages)
        final_prediction = _llm_label_or_fallback(full_example, system_name, response.label)
        premise_role = _premise_role_from_response(response)
        prompt_tokens_total += response.usage.get("prompt_tokens", 0) or estimate_tokens(
            full_example["revised_query"], *full_example["revised_premises"]
        )
        api_calls += 1
        ledger.add_belief(
            content=example["answer_options"][final_prediction],
            belief_type="intermediate_conclusion",
            status="confirmed",
            source="assistant_inferred",
            turn_id=1,
            confidence=0.7,
            metadata={"label": final_prediction},
        )
        signal = {
            "backend": "llm",
            "provider": backend_config["provider"],
            "model": response.model,
            "parse_mode": response.parse_mode,
            "label_from_model": response.label,
            "premise_role": premise_role,
        }
        turns.append(
            {
                "turn_id": 1,
                "prediction": final_prediction,
                "raw_output": response.content,
                "parsed_payload": response.parsed_payload,
                "usage": response.usage,
                "messages": messages,
                "ledger": ledger.snapshot(),
            }
        )
        trace = _build_trace_payload(
            example=example,
            system_name=system_name,
            ledger=ledger,
            turns=turns,
            signal=signal,
        )
        return RunRecord(
            example_id=example["example_id"],
            system=system_name,
            condition=example["condition"],
            gold_initial_label=example["gold_initial_label"],
            gold_final_label=example["gold_final_label"],
            initial_prediction=None,
            final_prediction=final_prediction,
            prompt_tokens_approx=prompt_tokens_total,
            api_calls=api_calls,
            revision_signal=signal,
            trace=trace,
        )

    for premise in example["initial_premises"]:
        ledger.add_belief(
            content=premise,
            belief_type="premise",
            status="confirmed",
            source="user_explicit",
            turn_id=1,
            confidence=1.0,
        )

    turn1_messages = _build_turn_messages(example, system_name)
    turn1_response = client.complete(turn1_messages)
    initial_prediction = _llm_label_or_fallback(example, system_name, turn1_response.label)
    prompt_tokens_total += turn1_response.usage.get("prompt_tokens", 0) or estimate_tokens(
        example["initial_query"], *example["initial_premises"]
    )
    api_calls += 1

    initial_status = "tentative" if system_name == "source_revision" else "confirmed"
    initial_commitment = ledger.add_belief(
        content=example["answer_options"][initial_prediction],
        belief_type="intermediate_conclusion",
        status=initial_status,
            source="assistant_inferred",
            turn_id=1,
            confidence=0.8,
            metadata={
                "label": initial_prediction,
                "relation_to_prior": "n_a",
            },
        )
    turns.append(
        {
            "turn_id": 1,
            "prediction": initial_prediction,
            "raw_output": turn1_response.content,
            "parsed_payload": turn1_response.parsed_payload,
            "usage": turn1_response.usage,
            "messages": turn1_messages,
            "ledger": ledger.snapshot(),
        }
    )

    for premise in example["update_premises"]:
        ledger.add_belief(
            content=premise,
            belief_type="premise",
            status="confirmed",
            source="user_explicit",
            turn_id=2,
            confidence=1.0,
        )

    turn2_messages = _build_turn_messages(
        example,
        system_name,
        previous_answer=turn1_response.content,
        previous_label=initial_prediction,
    )
    turn2_response = client.complete(turn2_messages)
    model_final_prediction = _llm_label_or_fallback(example, system_name, turn2_response.label)
    relation_to_prior = _relation_to_prior_from_response(
        example=example,
        response=turn2_response,
        initial_prediction=initial_prediction,
        model_prediction=model_final_prediction,
    )
    premise_role = _premise_role_from_response(turn2_response)
    final_prediction = model_final_prediction
    gate_preserved_prior = False
    if system_name == "source_revision":
        final_prediction, gate_preserved_prior = _apply_source_revision_gate(
            initial_prediction=initial_prediction,
            model_prediction=model_final_prediction,
            relation_to_prior=relation_to_prior,
        )
    prompt_tokens_total += turn2_response.usage.get("prompt_tokens", 0) or estimate_tokens(
        example["initial_query"],
        *example["initial_premises"],
        *example["update_premises"],
    )
    api_calls += 1

    if final_prediction != initial_prediction:
        if system_name == "source_revision" and relation_to_prior in AGGRESSIVE_RELATIONS:
            ledger.deprecate_belief(initial_commitment.belief_id, turn_id=2)
        elif system_name == "structured_no_source":
            ledger.mark_unresolved(initial_commitment.belief_id, turn_id=2)
        elif system_name == "source_no_revision":
            ledger.revise_belief(
                initial_commitment.belief_id,
                new_content=example["answer_options"][final_prediction],
                turn_id=2,
                source="assistant_inferred",
                confidence=0.7,
                metadata={"label": final_prediction},
            )
        else:
            ledger.revise_belief(
                initial_commitment.belief_id,
                new_content=example["answer_options"][final_prediction],
                turn_id=2,
                source="assistant_inferred",
                confidence=0.7,
                metadata={"label": final_prediction},
            )
        if system_name in {"source_revision", "structured_no_source"}:
            ledger.add_belief(
                content=example["answer_options"][final_prediction],
                belief_type="intermediate_conclusion",
                status="confirmed",
                source="assistant_inferred",
                turn_id=2,
                confidence=0.7,
                parent_or_conflict_target=initial_commitment.belief_id,
                metadata={
                    "label": final_prediction,
                    "relation_to_prior": relation_to_prior,
                },
            )
    elif system_name == "source_revision":
        ledger.confirm_belief(initial_commitment.belief_id, turn_id=2)

    signal = {
        "backend": "llm",
        "provider": backend_config["provider"],
        "turn1_model": turn1_response.model,
        "turn1_parse_mode": turn1_response.parse_mode,
        "turn1_label_from_model": turn1_response.label,
        "turn2_model": turn2_response.model,
        "turn2_parse_mode": turn2_response.parse_mode,
        "turn2_label_from_model": turn2_response.label,
        "turn2_model_prediction": model_final_prediction,
        "relation_to_prior": relation_to_prior,
        "premise_role": premise_role,
        "gate_preserved_prior": gate_preserved_prior,
        "changed_answer": final_prediction != initial_prediction,
    }
    turns.append(
        {
            "turn_id": 2,
            "prediction": final_prediction,
            "raw_output": turn2_response.content,
            "parsed_payload": turn2_response.parsed_payload,
            "relation_to_prior": relation_to_prior,
            "premise_role": premise_role,
            "model_prediction": model_final_prediction,
            "gate_preserved_prior": gate_preserved_prior,
            "usage": turn2_response.usage,
            "messages": turn2_messages,
            "ledger": ledger.snapshot(),
        }
    )

    trace = _build_trace_payload(
        example=example,
        system_name=system_name,
        ledger=ledger,
        turns=turns,
        signal=signal,
    )
    return RunRecord(
        example_id=example["example_id"],
        system=system_name,
        condition=example["condition"],
        gold_initial_label=example["gold_initial_label"],
        gold_final_label=example["gold_final_label"],
        initial_prediction=initial_prediction,
        final_prediction=final_prediction,
        prompt_tokens_approx=prompt_tokens_total,
        api_calls=api_calls,
        revision_signal=signal,
        trace=trace,
    )


def run_system_on_example(
    example: dict,
    system_name: str,
    backend_config: dict | None = None,
) -> RunRecord:
    if backend_config and backend_config.get("type") == "llm":
        return _run_llm_system_on_example(example, system_name, backend_config)
    return _run_heuristic_system_on_example(example, system_name)
