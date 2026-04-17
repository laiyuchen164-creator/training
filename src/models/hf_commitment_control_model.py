from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:  # pragma: no cover - exercised in lightweight test envs
    LoraConfig = None
    TaskType = None
    get_peft_model = None

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:  # pragma: no cover - exercised in lightweight test envs
    AutoModel = None
    AutoTokenizer = None


def _infer_target_modules(model_name: str) -> list[str]:
    lowered = model_name.lower()
    if "qwen" in lowered or "llama" in lowered or "mistral" in lowered:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    return ["query", "key", "value"]


@dataclass
class HFModelBundle:
    model: nn.Module
    tokenizer: object


def compute_conditional_propagation_loss(
    *,
    answer_logits: torch.Tensor,
    control_labels: torch.Tensor,
    answer_labels: torch.Tensor,
    early_answer_labels: torch.Tensor,
    control_to_idx: dict[str, int],
    beta_replace_margin: float,
    margin_m: float,
) -> dict[str, torch.Tensor]:
    log_answer_probs = F.log_softmax(answer_logits, dim=-1)
    zero = torch.zeros((), device=answer_logits.device, dtype=answer_logits.dtype)

    preserve_idx = control_to_idx["preserve"]
    replace_idx = control_to_idx.get("replace")
    preserve_mask = control_labels == preserve_idx
    replace_mask = control_labels == replace_idx if replace_idx is not None else torch.zeros_like(control_labels, dtype=torch.bool)

    # Preserve-side propagation consistency: keep the final answer aligned with
    # the early implied answer when the gold control decision is preserve.
    preserve_loss = zero
    if torch.any(preserve_mask):
        preserve_log_probs = log_answer_probs[preserve_mask]
        preserve_targets = early_answer_labels[preserve_mask].unsqueeze(1)
        preserve_loss = -preserve_log_probs.gather(dim=1, index=preserve_targets).squeeze(1).mean()

    # Replace-side propagation consistency keeps replace examples tied to the
    # updated gold answer while leaving weaken neutral.
    replace_alignment_loss = zero
    replace_margin_loss = zero
    if torch.any(replace_mask):
        replace_log_probs = log_answer_probs[replace_mask]
        replace_targets = answer_labels[replace_mask].unsqueeze(1)
        early_targets = early_answer_labels[replace_mask].unsqueeze(1)
        gold_log_probs = replace_log_probs.gather(dim=1, index=replace_targets).squeeze(1)
        early_log_probs = replace_log_probs.gather(dim=1, index=early_targets).squeeze(1)
        replace_alignment_loss = -gold_log_probs.mean()

        # Anti-early margin term: on replace examples, require the gold final
        # answer to outrank the early implied answer by at least margin_m.
        margin_gap = gold_log_probs - early_log_probs
        replace_margin_loss = torch.relu(margin_m - margin_gap).mean()

    active_terms = []
    if torch.any(preserve_mask):
        active_terms.append(preserve_loss)
    if torch.any(replace_mask):
        active_terms.append(replace_alignment_loss + beta_replace_margin * replace_margin_loss)
    propagation_loss = torch.stack(active_terms).mean() if active_terms else zero
    return {
        "propagation_loss": propagation_loss,
        "preserve_propagation_loss": preserve_loss,
        "replace_alignment_loss": replace_alignment_loss,
        "replace_margin_loss": replace_margin_loss,
    }


class HFCommitmentControlModel(nn.Module):
    def __init__(
        self,
        *,
        model_name: str,
        control_label_count: int,
        answer_label_count: int,
        control_to_idx: dict[str, int],
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        use_bf16: bool = True,
    ) -> None:
        super().__init__()
        if AutoModel is None or LoraConfig is None or TaskType is None or get_peft_model is None:
            raise ImportError(
                "HFCommitmentControlModel requires optional training dependencies "
                "'transformers' and 'peft' to be installed."
            )
        dtype = torch.bfloat16 if use_bf16 and torch.cuda.is_available() else torch.float16
        self.backbone = AutoModel.from_pretrained(
            model_name,
            torch_dtype=dtype if torch.cuda.is_available() else torch.float32,
        )
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=_infer_target_modules(model_name),
            bias="none",
        )
        self.backbone = get_peft_model(self.backbone, peft_config)
        hidden_size = self.backbone.config.hidden_size
        self.control_to_idx = control_to_idx
        self.control_head = nn.Linear(hidden_size, control_label_count)
        self.answer_head = nn.Linear(hidden_size, answer_label_count)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        control_labels: torch.Tensor | None = None,
        answer_labels: torch.Tensor | None = None,
        early_answer_labels: torch.Tensor | None = None,
        control_class_weights: torch.Tensor | None = None,
        answer_class_weights: torch.Tensor | None = None,
        answer_loss_weight: float = 1.0,
        consistency_loss_weight: float = 0.0,
        lambda_prop: float = 0.0,
        beta_replace_margin: float = 0.0,
        margin_m: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        pooled = hidden_states[
            torch.arange(hidden_states.size(0), device=hidden_states.device),
            attention_mask.sum(dim=1) - 1,
        ]
        pooled_for_heads = pooled.to(self.control_head.weight.dtype)
        control_logits = self.control_head(pooled_for_heads)
        answer_logits = self.answer_head(pooled_for_heads)
        payload = {
            "control_logits": control_logits,
            "answer_logits": answer_logits,
        }
        if control_labels is not None and answer_labels is not None:
            ctrl_loss_fct = nn.CrossEntropyLoss(weight=control_class_weights)
            ans_loss_fct = nn.CrossEntropyLoss(weight=answer_class_weights)
            ctrl_loss = ctrl_loss_fct(control_logits, control_labels)
            ans_loss = ans_loss_fct(answer_logits, answer_labels)
            total_loss = ctrl_loss + answer_loss_weight * ans_loss
            consistency_loss = torch.zeros((), device=control_logits.device, dtype=control_logits.dtype)
            propagation_loss = torch.zeros((), device=control_logits.device, dtype=control_logits.dtype)
            preserve_propagation_loss = propagation_loss
            replace_alignment_loss = propagation_loss
            replace_margin_loss = propagation_loss
            if consistency_loss_weight > 0.0 and early_answer_labels is not None:
                control_probs = torch.softmax(control_logits, dim=-1)
                answer_probs = torch.softmax(answer_logits, dim=-1)
                preserve_idx = self.control_to_idx["preserve"]
                preserve_answer_prob = answer_probs.gather(
                    dim=1,
                    index=early_answer_labels.unsqueeze(1),
                ).squeeze(1)
                consistency_loss = torch.mean((preserve_answer_prob - control_probs[:, preserve_idx]) ** 2)
                total_loss = total_loss + consistency_loss_weight * consistency_loss
            if lambda_prop > 0.0 and early_answer_labels is not None:
                propagation_payload = compute_conditional_propagation_loss(
                    answer_logits=answer_logits,
                    control_labels=control_labels,
                    answer_labels=answer_labels,
                    early_answer_labels=early_answer_labels,
                    control_to_idx=self.control_to_idx,
                    beta_replace_margin=beta_replace_margin,
                    margin_m=margin_m,
                )
                propagation_loss = propagation_payload["propagation_loss"]
                preserve_propagation_loss = propagation_payload["preserve_propagation_loss"]
                replace_alignment_loss = propagation_payload["replace_alignment_loss"]
                replace_margin_loss = propagation_payload["replace_margin_loss"]
                total_loss = total_loss + lambda_prop * propagation_loss
            payload["loss"] = total_loss
            payload["control_loss"] = ctrl_loss
            payload["answer_loss"] = ans_loss
            payload["consistency_loss"] = consistency_loss
            payload["propagation_loss"] = propagation_loss
            payload["preserve_propagation_loss"] = preserve_propagation_loss
            payload["replace_alignment_loss"] = replace_alignment_loss
            payload["replace_margin_loss"] = replace_margin_loss
        return payload

    @classmethod
    def build_bundle(
        cls,
        *,
        model_name: str,
        control_label_count: int,
        answer_label_count: int,
        control_to_idx: dict[str, int],
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        use_bf16: bool,
    ) -> HFModelBundle:
        if AutoTokenizer is None:
            raise ImportError(
                "HFCommitmentControlModel.build_bundle requires the optional "
                "'transformers' dependency to be installed."
            )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = cls(
            model_name=model_name,
            control_label_count=control_label_count,
            answer_label_count=answer_label_count,
            control_to_idx=control_to_idx,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_bf16=use_bf16,
        )
        return HFModelBundle(model=model, tokenizer=tokenizer)
