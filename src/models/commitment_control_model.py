from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


TOKEN_PATTERN = re.compile(r"[a-z0-9_']+")


def _extract_conditional_parts(text: str) -> tuple[str, str]:
    lowered = text.strip().lower()
    if lowered.startswith("if ") and ", then " in lowered:
        antecedent, consequent = lowered[3:].split(", then ", maxsplit=1)
        return antecedent.strip(), consequent.strip()
    return lowered, ""


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _bucketize(value: float) -> str:
    if value >= 0.8:
        return "very_high"
    if value >= 0.5:
        return "high"
    if value >= 0.25:
        return "medium"
    if value > 0.0:
        return "low"
    return "zero"


@dataclass
class TrainingState:
    epoch: int
    train_loss: float
    dev_control_accuracy: float
    dev_answer_accuracy: float
    dev_joint_accuracy: float


class CommitmentControlModel:
    def __init__(
        self,
        *,
        control_labels: list[str],
        answer_labels: list[str],
        feature_dim: int = 4096,
        seed: int = 7,
    ) -> None:
        self.control_labels = control_labels
        self.answer_labels = answer_labels
        self.feature_dim = feature_dim
        self.seed = seed
        self.control_to_idx = {label: index for index, label in enumerate(control_labels)}
        self.answer_to_idx = {label: index for index, label in enumerate(answer_labels)}
        scale = 0.01
        rng = np.random.default_rng(seed)
        self.control_weights = rng.normal(
            0.0, scale, size=(len(control_labels), feature_dim)
        ).astype(np.float32)
        self.answer_weights = rng.normal(
            0.0, scale, size=(len(answer_labels), feature_dim)
        ).astype(np.float32)
        self.control_bias = np.zeros(len(control_labels), dtype=np.float32)
        self.answer_bias = np.zeros(len(answer_labels), dtype=np.float32)

    def _hash_index(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="little") % self.feature_dim

    def _feature_tokens(self, example: dict) -> list[str]:
        tokens: list[str] = []
        for field in (
            "early_context",
            "early_commitment_text",
            "late_evidence",
            "question",
        ):
            for token in _tokenize(example[field]):
                tokens.append(f"{field}:{token}")

        tokens.extend(
            [
                f"source:{example['source_type']}",
                f"condition:{example['condition']}",
                f"early_label:{example['early_commitment_label']}",
                f"modus:{example['task_metadata']['modus']}",
                f"relation:{example['task_metadata']['relation_type'].lower()}",
            ]
        )

        early_rule = example["early_context"].splitlines()[0].split(": ", maxsplit=1)[-1]
        late_rule = example["late_evidence"].splitlines()[0].split(": ", maxsplit=1)[-1]
        early_ant, early_cons = _extract_conditional_parts(early_rule)
        late_ant, late_cons = _extract_conditional_parts(late_rule)
        early_ant_tokens = set(_tokenize(early_ant))
        late_ant_tokens = set(_tokenize(late_ant))
        early_cons_tokens = set(_tokenize(early_cons))
        late_cons_tokens = set(_tokenize(late_cons))

        ant_overlap = _jaccard(early_ant_tokens, late_ant_tokens)
        cons_overlap = _jaccard(early_cons_tokens, late_cons_tokens)
        tokens.extend(
            [
                f"ant_overlap:{_bucketize(ant_overlap)}",
                f"cons_overlap:{_bucketize(cons_overlap)}",
                f"late_has_and:{'and' in late_ant_tokens}",
                f"early_subset_late:{early_ant_tokens.issubset(late_ant_tokens) if early_ant_tokens else False}",
                f"late_subset_early:{late_ant_tokens.issubset(early_ant_tokens) if late_ant_tokens else False}",
                f"same_consequent:{early_cons_tokens == late_cons_tokens and bool(early_cons_tokens)}",
                f"late_token_delta:{max(0, len(late_ant_tokens) - len(early_ant_tokens))}",
            ]
        )
        return tokens

    def encode_examples(self, examples: list[dict]) -> np.ndarray:
        matrix = np.zeros((len(examples), self.feature_dim), dtype=np.float32)
        for row_index, example in enumerate(examples):
            counts: dict[int, float] = {}
            for token in self._feature_tokens(example):
                feature_index = self._hash_index(token)
                counts[feature_index] = counts.get(feature_index, 0.0) + 1.0
            norm = math.sqrt(sum(value * value for value in counts.values())) or 1.0
            for feature_index, value in counts.items():
                matrix[row_index, feature_index] = value / norm
        return matrix

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def _forward(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ctrl_logits = features @ self.control_weights.T + self.control_bias
        ans_logits = features @ self.answer_weights.T + self.answer_bias
        return ctrl_logits, ans_logits

    @staticmethod
    def _weighted_ce(
        probs: np.ndarray,
        gold: np.ndarray,
        class_weights: np.ndarray,
    ) -> float:
        eps = 1e-9
        sample_weights = class_weights[gold]
        losses = -np.log(probs[np.arange(len(gold)), gold] + eps) * sample_weights
        return float(np.mean(losses))

    def fit(
        self,
        train_examples: list[dict],
        dev_examples: list[dict],
        *,
        num_epochs: int,
        learning_rate: float,
        batch_size: int,
        l2: float,
        answer_loss_weight: float,
    ) -> list[TrainingState]:
        train_features = self.encode_examples(train_examples)
        dev_features = self.encode_examples(dev_examples)
        train_ctrl = np.array(
            [self.control_to_idx[example["control_label"]] for example in train_examples],
            dtype=np.int64,
        )
        train_ans = np.array(
            [self.answer_to_idx[example["final_answer_label"]] for example in train_examples],
            dtype=np.int64,
        )
        dev_ctrl = np.array(
            [self.control_to_idx[example["control_label"]] for example in dev_examples],
            dtype=np.int64,
        )
        dev_ans = np.array(
            [self.answer_to_idx[example["final_answer_label"]] for example in dev_examples],
            dtype=np.int64,
        )

        ctrl_counts = np.bincount(train_ctrl, minlength=len(self.control_labels))
        ans_counts = np.bincount(train_ans, minlength=len(self.answer_labels))
        ctrl_weights = np.zeros(len(self.control_labels), dtype=np.float32)
        ans_weights = np.zeros(len(self.answer_labels), dtype=np.float32)
        for index, count in enumerate(ctrl_counts):
            ctrl_weights[index] = (
                len(train_examples) / (len(self.control_labels) * count) if count else 0.0
            )
        for index, count in enumerate(ans_counts):
            ans_weights[index] = (
                len(train_examples) / (len(self.answer_labels) * count) if count else 0.0
            )

        rng = np.random.default_rng(self.seed)
        best_score = -1.0
        best_snapshot = None
        history: list[TrainingState] = []

        for epoch in range(1, num_epochs + 1):
            indices = np.arange(len(train_examples))
            rng.shuffle(indices)
            epoch_loss = 0.0

            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                x_batch = train_features[batch_indices]
                ctrl_gold = train_ctrl[batch_indices]
                ans_gold = train_ans[batch_indices]

                ctrl_logits, ans_logits = self._forward(x_batch)
                ctrl_probs = self._softmax(ctrl_logits)
                ans_probs = self._softmax(ans_logits)

                ctrl_loss = self._weighted_ce(ctrl_probs, ctrl_gold, ctrl_weights)
                ans_loss = self._weighted_ce(ans_probs, ans_gold, ans_weights)
                loss = ctrl_loss + answer_loss_weight * ans_loss
                epoch_loss += loss * len(batch_indices)

                ctrl_grad = ctrl_probs
                ctrl_grad[np.arange(len(batch_indices)), ctrl_gold] -= 1.0
                ctrl_grad *= ctrl_weights[ctrl_gold][:, None] / len(batch_indices)

                ans_grad = ans_probs
                ans_grad[np.arange(len(batch_indices)), ans_gold] -= 1.0
                ans_grad *= ans_weights[ans_gold][:, None] / len(batch_indices)
                ans_grad *= answer_loss_weight

                grad_ctrl_w = ctrl_grad.T @ x_batch + l2 * self.control_weights
                grad_ctrl_b = np.sum(ctrl_grad, axis=0)
                grad_ans_w = ans_grad.T @ x_batch + l2 * self.answer_weights
                grad_ans_b = np.sum(ans_grad, axis=0)

                self.control_weights -= learning_rate * grad_ctrl_w.astype(np.float32)
                self.control_bias -= learning_rate * grad_ctrl_b.astype(np.float32)
                self.answer_weights -= learning_rate * grad_ans_w.astype(np.float32)
                self.answer_bias -= learning_rate * grad_ans_b.astype(np.float32)

            dev_predictions = self.predict(dev_examples, features=dev_features)
            dev_control_accuracy = float(
                np.mean(
                    [
                        prediction["predicted_control_decision"] == example["control_label"]
                        for prediction, example in zip(dev_predictions, dev_examples)
                    ]
                )
            )
            dev_answer_accuracy = float(
                np.mean(
                    [
                        prediction["predicted_final_answer"] == example["final_answer_label"]
                        for prediction, example in zip(dev_predictions, dev_examples)
                    ]
                )
            )
            dev_joint_accuracy = float(
                np.mean(
                    [
                        (
                            prediction["predicted_control_decision"] == example["control_label"]
                            and prediction["predicted_final_answer"] == example["final_answer_label"]
                        )
                        for prediction, example in zip(dev_predictions, dev_examples)
                    ]
                )
            )

            history.append(
                TrainingState(
                    epoch=epoch,
                    train_loss=epoch_loss / max(1, len(train_examples)),
                    dev_control_accuracy=dev_control_accuracy,
                    dev_answer_accuracy=dev_answer_accuracy,
                    dev_joint_accuracy=dev_joint_accuracy,
                )
            )

            score = 0.5 * (dev_control_accuracy + dev_answer_accuracy)
            if score > best_score:
                best_score = score
                best_snapshot = (
                    self.control_weights.copy(),
                    self.control_bias.copy(),
                    self.answer_weights.copy(),
                    self.answer_bias.copy(),
                )

        if best_snapshot is not None:
            (
                self.control_weights,
                self.control_bias,
                self.answer_weights,
                self.answer_bias,
            ) = best_snapshot
        return history

    def predict(self, examples: list[dict], *, features: np.ndarray | None = None) -> list[dict]:
        feature_matrix = features if features is not None else self.encode_examples(examples)
        ctrl_logits, ans_logits = self._forward(feature_matrix)
        ctrl_probs = self._softmax(ctrl_logits)
        ans_probs = self._softmax(ans_logits)
        ctrl_pred = np.argmax(ctrl_probs, axis=1)
        ans_pred = np.argmax(ans_probs, axis=1)
        rows = []
        for index, example in enumerate(examples):
            rows.append(
                {
                    "example_id": example["example_id"],
                    "condition": example["condition"],
                    "gold_control_decision": example["control_label"],
                    "gold_final_answer": example["final_answer_label"],
                    "predicted_control_decision": self.control_labels[int(ctrl_pred[index])],
                    "predicted_final_answer": self.answer_labels[int(ans_pred[index])],
                    "control_confidence": round(float(ctrl_probs[index, ctrl_pred[index]]), 4),
                    "answer_confidence": round(float(ans_probs[index, ans_pred[index]]), 4),
                    "output_json": {
                        "control_decision": self.control_labels[int(ctrl_pred[index])],
                        "final_answer": self.answer_labels[int(ans_pred[index])],
                    },
                }
            )
        return rows

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            control_weights=self.control_weights,
            control_bias=self.control_bias,
            answer_weights=self.answer_weights,
            answer_bias=self.answer_bias,
        )
        metadata = {
            "control_labels": self.control_labels,
            "answer_labels": self.answer_labels,
            "feature_dim": self.feature_dim,
            "seed": self.seed,
        }
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CommitmentControlModel":
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        model = cls(
            control_labels=metadata["control_labels"],
            answer_labels=metadata["answer_labels"],
            feature_dim=metadata["feature_dim"],
            seed=metadata["seed"],
        )
        payload = np.load(path)
        model.control_weights = payload["control_weights"]
        model.control_bias = payload["control_bias"]
        model.answer_weights = payload["answer_weights"]
        model.answer_bias = payload["answer_bias"]
        return model
