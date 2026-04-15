from __future__ import annotations

import ast
import csv
import io
import json
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from src.utils import write_json, write_jsonl

BELIEF_R_URLS = {
    "basic_time_t.csv": (
        "https://raw.githubusercontent.com/HLTCHKUST/belief-revision/main/"
        "dataset/belief_r/basic_time_t.csv"
    ),
    "queries_time_t1.csv": (
        "https://raw.githubusercontent.com/HLTCHKUST/belief-revision/main/"
        "dataset/belief_r/queries_time_t1.csv"
    ),
}

ATOMIC_URL = "https://homes.cs.washington.edu/~msap/atomic/data/atomic_data.tgz"
REVISEQA_ZIP_URL = "https://codeload.github.com/ChadiHelwe/reviseqa/zip/refs/heads/main"


def _download(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read().decode("utf-8")


def _download_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def download_belief_r(raw_dir: Path, *, refresh: bool = False) -> dict[str, Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    downloaded = {}
    for filename, url in BELIEF_R_URLS.items():
        destination = raw_dir / filename
        if refresh or not destination.exists():
            destination.write_text(_download(url), encoding="utf-8")
        downloaded[filename] = destination
    return downloaded


def download_atomic(raw_dir: Path, *, refresh: bool = False) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / "atomic_data.tgz"
    if refresh or not destination.exists():
        destination.write_bytes(_download_bytes(ATOMIC_URL))
    return destination


def download_reviseqa(raw_dir: Path, *, refresh: bool = False) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / "reviseqa_main.zip"
    if refresh or not destination.exists():
        destination.write_bytes(_download_bytes(REVISEQA_ZIP_URL))
    return destination


def _read_csv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))


def _read_atomic_csv_from_archive(archive_path: Path, member_name: str) -> list[dict[str, str]]:
    with tarfile.open(archive_path, mode="r:gz") as archive:
        member = archive.extractfile(member_name)
        if member is None:
            raise FileNotFoundError(f"Missing member in archive: {member_name}")
        text = member.read().decode("utf-8", errors="replace")
    return list(csv.DictReader(io.StringIO(text)))


def _pair_key(row: dict[str, str]) -> tuple[str, str]:
    return row["dataset_id"], row["modus"]


def _split_question(question: str) -> tuple[list[str], str]:
    premises = []
    question_lines = []
    seen_break = False
    for raw_line in question.splitlines():
        line = raw_line.strip()
        if not line and not seen_break:
            seen_break = True
            continue
        if not seen_break:
            premises.append(line)
        elif line:
            question_lines.append(line)
    return premises, "\n".join(question_lines)


def _answer_options(row: dict[str, str]) -> dict[str, str]:
    return {"a": row["a"], "b": row["b"], "c": row["c"]}


def _clean_atomic_text(text: str) -> bool:
    value = text.strip()
    if not value:
        return False
    if value.lower() == "none":
        return False
    if "___" in value or "PersonZ" in value:
        return False
    if len(value) > 120:
        return False
    return True


def _parse_atomic_list(raw_value: str) -> list[str]:
    if not raw_value:
        return []
    try:
        parsed = ast.literal_eval(raw_value)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [
        str(item).strip()
        for item in parsed
        if isinstance(item, str) and _clean_atomic_text(str(item))
    ]


@dataclass
class AtomicExampleSeed:
    event: str
    effect: str
    need: str
    alt_event: str


def _normalize_atomic_event(event: str) -> str:
    value = event.strip()
    if not value.endswith("."):
        value += "."
    return value


def _normalize_atomic_outcome(effect: str) -> str:
    effect = effect.strip().rstrip(".")
    if effect.lower().startswith("personx "):
        proposition = effect
    elif effect.lower().startswith("persony "):
        proposition = effect
    else:
        proposition = f"PersonX {effect}"
    if not proposition.endswith("."):
        proposition += "."
    return proposition


def _atomic_question(outcome: str) -> str:
    return (
        "Given the premises above, what necessarily follows about this outcome?\n"
        f'Outcome: "{outcome}"\n'
        "(a) The outcome definitely happens.\n"
        "(b) The outcome definitely does not happen.\n"
        "(c) The outcome may or may not happen."
    )


def _atomic_answer_options() -> dict[str, str]:
    return {
        "a": "The outcome definitely happens.",
        "b": "The outcome definitely does not happen.",
        "c": "The outcome may or may not happen.",
    }


def _reviseqa_answer_options() -> dict[str, str]:
    return {
        "a": "The conclusion sentence as written is definitely true.",
        "b": "The negation of the conclusion sentence is definitely true.",
        "c": "It cannot be determined whether the conclusion sentence is true or false.",
    }


def _reviseqa_query(conclusion: str) -> str:
    return (
        "Given the premises above, what is the status of the conclusion?\n"
        f'Conclusion: "{conclusion.strip()}"\n'
        "(a) The conclusion sentence as written is definitely true.\n"
        "(b) The negation of the conclusion sentence is definitely true.\n"
        "(c) It cannot be determined whether the conclusion sentence is true or false."
    )


def _map_reviseqa_answer(answer: str) -> str:
    normalized = answer.strip().lower()
    mapping = {
        "true": "a",
        "false": "b",
        "uncertain": "c",
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported ReviseQA answer label: {answer}")
    return mapping[normalized]


def _reviseqa_modus(label: str) -> str:
    return {
        "a": "generic_true",
        "b": "generic_false",
        "c": "generic_uncertain",
    }[label]


def _clean_lines(lines: list[str]) -> list[str]:
    return [line.strip() for line in lines if isinstance(line, str) and line.strip()]


def _render_reviseqa_updates(edit: dict) -> list[str]:
    updates = [
        (
            f"Context edit #{edit.get('edit_number', '?')} "
            f"({str(edit.get('modification_type', 'unknown')).strip() or 'unknown'}): "
            "apply the following changes to the earlier context."
        )
    ]
    edits_made = edit.get("edits_made", {}) or {}

    for item in edits_made.get("removed_facts", []):
        updates.append(f"Remove this earlier fact from the context: {item['nl']}")
    for item in edits_made.get("removed_rules", []):
        updates.append(f"Remove this earlier rule from the context: {item['nl']}")
    for item in edits_made.get("added_facts", []):
        updates.append(f"Add this new fact to the context: {item['nl']}")
    for item in edits_made.get("added_rules", []):
        updates.append(f"Add this new rule to the context: {item['nl']}")
    return updates


def _build_atomic_seed_rows(rows: list[dict[str, str]]) -> list[AtomicExampleSeed]:
    effect_to_events: dict[str, list[str]] = {}
    event_to_needs: dict[str, list[str]] = {}
    event_to_effects: dict[str, list[str]] = {}

    for row in rows:
        event = row["event"].strip()
        if not _clean_atomic_text(event):
            continue
        effects = _parse_atomic_list(row["xEffect"])
        needs = _parse_atomic_list(row["xNeed"])
        if effects:
            event_to_effects[event] = effects
        if needs:
            event_to_needs[event] = needs
        for effect in effects:
            effect_to_events.setdefault(effect.lower(), [])
            if event not in effect_to_events[effect.lower()]:
                effect_to_events[effect.lower()].append(event)

    seeds: list[AtomicExampleSeed] = []
    for event, effects in sorted(event_to_effects.items()):
        if event not in event_to_needs:
            continue
        need = event_to_needs[event][0]
        for effect in effects:
            alternatives = [
                alt_event
                for alt_event in effect_to_events.get(effect.lower(), [])
                if alt_event != event
            ]
            if not alternatives:
                continue
            seeds.append(
                AtomicExampleSeed(
                    event=event,
                    effect=effect,
                    need=need,
                    alt_event=alternatives[0],
                )
            )
            break
    return seeds


def transform_belief_r(
    raw_dir: Path,
    processed_path: Path,
    stats_path: Path,
    *,
    refresh: bool = False,
) -> dict:
    downloads = download_belief_r(raw_dir, refresh=refresh)
    basic_rows = _read_csv(downloads["basic_time_t.csv"])
    revised_rows = _read_csv(downloads["queries_time_t1.csv"])
    basic_map = {_pair_key(row): row for row in basic_rows}

    records = []
    stats = {
        "source": "Belief-R strong paired subset",
        "paired_examples": 0,
        "full_info": 0,
        "incremental_no_overturn": 0,
        "incremental_overturn_reasoning": 0,
        "skipped_unpaired": 0,
    }

    for revised_row in revised_rows:
        pair_key = _pair_key(revised_row)
        if pair_key not in basic_map:
            stats["skipped_unpaired"] += 1
            continue

        basic_row = basic_map[pair_key]
        initial_premises, initial_query = _split_question(basic_row["questions"])
        revised_premises, revised_query = _split_question(revised_row["questions"])
        update_premises = revised_premises[len(initial_premises) :]

        final_condition = (
            "incremental_overturn_reasoning"
            if basic_row["ground_truth"] != revised_row["ground_truth"]
            else "incremental_no_overturn"
        )

        example_stub = {
            "dataset": "belief_r",
            "pair_id": revised_row["dataset_id"],
            "modus": revised_row["modus"],
            "relation_type": revised_row["types_of_relation"],
            "answer_options": _answer_options(revised_row),
            "initial_premises": initial_premises,
            "revised_premises": revised_premises,
            "update_premises": update_premises,
            "initial_query": initial_query,
            "revised_query": revised_query,
            "gold_initial_label": basic_row["ground_truth"],
            "gold_final_label": revised_row["ground_truth"],
        }

        records.append(
            {
                **example_stub,
                "example_id": (
                    f"belief_r::{revised_row['dataset_id']}::{revised_row['modus']}::full_info"
                ),
                "condition": "full_info",
            }
        )
        records.append(
            {
                **example_stub,
                "example_id": (
                    f"belief_r::{revised_row['dataset_id']}::{revised_row['modus']}::incremental"
                ),
                "condition": final_condition,
            }
        )
        stats["paired_examples"] += 1
        stats["full_info"] += 1
        stats[final_condition] += 1

    write_jsonl(processed_path, records)
    write_json(stats_path, stats)
    return stats


def transform_atomic_explicit_revision(
    raw_dir: Path,
    processed_path: Path,
    stats_path: Path,
    *,
    refresh: bool = False,
    max_seed_examples: int = 600,
) -> dict:
    archive_path = download_atomic(raw_dir, refresh=refresh)
    rows = _read_atomic_csv_from_archive(archive_path, "v4_atomic_trn.csv")
    seeds = _build_atomic_seed_rows(rows)[:max_seed_examples]

    records = []
    stats = {
        "source": "ATOMIC explicit revision synthetic transform",
        "seed_examples": len(seeds),
        "full_info": 0,
        "incremental_no_overturn": 0,
        "incremental_overturn_reasoning": 0,
    }

    for index, seed in enumerate(seeds):
        event_sentence = _normalize_atomic_event(seed.event)
        alt_event_sentence = _normalize_atomic_event(seed.alt_event)
        outcome_sentence = _normalize_atomic_outcome(seed.effect)
        question = _atomic_question(outcome_sentence)
        answer_options = _atomic_answer_options()

        premise_1 = f'If the event "{event_sentence}" happens, then the outcome "{outcome_sentence}" happens.'
        premise_2 = f'The event "{event_sentence}" happens.'
        maintain_premise = (
            f'An alternative rule is also true: if the event "{alt_event_sentence}" happens, '
            f'then the outcome "{outcome_sentence}" happens.'
        )
        update_premise = (
            f'Correction to the earlier rule: if the event "{event_sentence}" happens and '
            f'this extra condition holds: "{seed.need}", then the outcome "{outcome_sentence}" happens.'
        )

        base_stub = {
            "dataset": "atomic_explicit_revision",
            "pair_id": f"atomic_seed_{index}",
            "modus": "ponens",
            "relation_type": "xEffect",
            "answer_options": answer_options,
            "initial_premises": [premise_1, premise_2],
            "initial_query": question,
            "gold_initial_label": "a",
            "metadata": {
                "source_event": seed.event,
                "alternative_event": seed.alt_event,
                "effect": seed.effect,
                "need": seed.need,
            },
        }

        records.append(
            {
                **base_stub,
                "example_id": f"atomic_explicit::{index}::maintain::full_info",
                "condition": "full_info",
                "revised_premises": [premise_1, premise_2, maintain_premise],
                "update_premises": [maintain_premise],
                "revised_query": question,
                "gold_final_label": "a",
            }
        )
        records.append(
            {
                **base_stub,
                "example_id": f"atomic_explicit::{index}::maintain::incremental",
                "condition": "incremental_no_overturn",
                "revised_premises": [premise_1, premise_2, maintain_premise],
                "update_premises": [maintain_premise],
                "revised_query": question,
                "gold_final_label": "a",
            }
        )
        records.append(
            {
                **base_stub,
                "example_id": f"atomic_explicit::{index}::update::full_info",
                "condition": "full_info",
                "revised_premises": [premise_1, premise_2, update_premise],
                "update_premises": [update_premise],
                "revised_query": question,
                "gold_final_label": "c",
            }
        )
        records.append(
            {
                **base_stub,
                "example_id": f"atomic_explicit::{index}::update::incremental",
                "condition": "incremental_overturn_reasoning",
                "revised_premises": [premise_1, premise_2, update_premise],
                "update_premises": [update_premise],
                "revised_query": question,
                "gold_final_label": "c",
            }
        )
        stats["full_info"] += 2
        stats["incremental_no_overturn"] += 1
        stats["incremental_overturn_reasoning"] += 1

    write_jsonl(processed_path, records)
    write_json(stats_path, stats)
    return stats


def transform_reviseqa_incremental(
    raw_dir: Path,
    processed_path: Path,
    stats_path: Path,
    *,
    refresh: bool = False,
    max_original_examples: int | None = None,
    max_edits_per_example: int | None = None,
) -> dict:
    archive_path = download_reviseqa(raw_dir, refresh=refresh)

    with zipfile.ZipFile(archive_path) as archive:
        names = sorted(
            (
                name
                for name in archive.namelist()
                if name.startswith("reviseqa-main/reviseqa_data/nl/ex_")
                and name.endswith(".json")
            ),
            key=lambda value: int(Path(value).stem.split("_")[1]),
        )

        if max_original_examples is not None:
            names = names[:max_original_examples]

        records = []
        stats = {
            "source": "ReviseQA official natural-language edits",
            "original_examples": 0,
            "edits_total": 0,
            "full_info": 0,
            "incremental_no_overturn": 0,
            "incremental_overturn_reasoning": 0,
            "modification_types": {},
            "answer_transitions": {},
        }

        for member_name in names:
            payload = json.loads(archive.read(member_name).decode("utf-8"))
            original_answer = str(payload["answer"])
            gold_initial_label = _map_reviseqa_answer(original_answer)
            initial_premises = _clean_lines(payload["original_context"])
            query = _reviseqa_query(payload["conclusion"])
            answer_options = _reviseqa_answer_options()
            example_stem = Path(member_name).stem

            edits = payload.get("edits", [])
            if max_edits_per_example is not None:
                edits = edits[:max_edits_per_example]

            if not edits:
                continue

            stats["original_examples"] += 1

            for edit in edits:
                revised_premises = _clean_lines(edit["edited_natural_language_context"])
                update_premises = _render_reviseqa_updates(edit)
                final_answer = str(edit["answer"])
                gold_final_label = _map_reviseqa_answer(final_answer)
                modification_type = (
                    str(edit.get("modification_type", "unknown")).strip() or "unknown"
                )
                transition_key = f"{original_answer}->{final_answer}"
                condition = (
                    "incremental_overturn_reasoning"
                    if gold_initial_label != gold_final_label
                    else "incremental_no_overturn"
                )
                pair_id = f"{example_stem}::edit_{edit['edit_number']}"

                example_stub = {
                    "dataset": "reviseqa_incremental",
                    "prompt_family": "generic_logic_revision",
                    "pair_id": pair_id,
                    "modus": _reviseqa_modus(gold_initial_label),
                    "relation_type": modification_type,
                    "answer_options": answer_options,
                    "initial_premises": initial_premises,
                    "revised_premises": revised_premises,
                    "update_premises": update_premises,
                    "initial_query": query,
                    "revised_query": query,
                    "gold_initial_label": gold_initial_label,
                    "gold_final_label": gold_final_label,
                    "metadata": {
                        "modification_type": modification_type,
                        "edit_number": edit["edit_number"],
                        "original_answer": original_answer,
                        "final_answer": final_answer,
                        "removed_facts": len(edit.get("edits_made", {}).get("removed_facts", [])),
                        "removed_rules": len(edit.get("edits_made", {}).get("removed_rules", [])),
                        "added_facts": len(edit.get("edits_made", {}).get("added_facts", [])),
                        "added_rules": len(edit.get("edits_made", {}).get("added_rules", [])),
                    },
                }

                records.append(
                    {
                        **example_stub,
                        "example_id": f"reviseqa::{pair_id}::full_info",
                        "condition": "full_info",
                    }
                )
                records.append(
                    {
                        **example_stub,
                        "example_id": f"reviseqa::{pair_id}::incremental",
                        "condition": condition,
                    }
                )

                stats["edits_total"] += 1
                stats["full_info"] += 1
                stats[condition] += 1
                stats["modification_types"][modification_type] = (
                    stats["modification_types"].get(modification_type, 0) + 1
                )
                stats["answer_transitions"][transition_key] = (
                    stats["answer_transitions"].get(transition_key, 0) + 1
                )

    write_jsonl(processed_path, records)
    write_json(stats_path, stats)
    return stats
