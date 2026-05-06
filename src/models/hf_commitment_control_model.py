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
    lambda_pres: float,
    lambda_rep: float,
    beta_preserve_margin: float,
    preserve_margin_m: float,
    beta_replace_margin: float,
    margin_m: float,
) -> dict[str, torch.Tensor]:
    log_answer_probs = F.log_softmax(answer_logits, dim=-1)
    zero = torch.zeros((), device=answer_logits.device, dtype=answer_logits.dtype)

    preserve_idx = control_to_idx["preserve"]
    replace_idx = control_to_idx.get("replace")
    preserve_mask = control_labels == preserve_idx
    replace_mask = control_labels == replace_idx if replace_idx is not None else torch.zeros_like(control_labels, dtype=torch.bool)

    # Preserve-side propagation consistency: combine direct CE-to-early
    # supervision with a smaller margin term so preserve cases are anchored
    # without relying on the margin alone.
    preserve_loss = zero
    preserve_ce_loss = zero
    preserve_margin_loss = zero
    if torch.any(preserve_mask):
        preserve_log_probs = log_answer_probs[preserve_mask]
        preserve_targets = early_answer_labels[preserve_mask]
        preserve_target_log_probs = preserve_log_probs.gather(dim=1, index=preserve_targets.unsqueeze(1)).squeeze(1)
        preserve_ce_loss = -preserve_target_log_probs.mean()
        alternative_log_probs = preserve_log_probs.masked_fill(
            F.one_hot(preserve_targets, num_classes=preserve_log_probs.size(1)).bool(),
            float("-inf"),
        )
        strongest_alternative_log_probs = alternative_log_probs.max(dim=1).values
        preserve_margin_gap = preserve_target_log_probs - strongest_alternative_log_probs
        preserve_margin_loss = torch.relu(preserve_margin_m - preserve_margin_gap).mean()
        preserve_loss = preserve_ce_loss + beta_preserve_margin * preserve_margin_loss

    # Replace-side propagation consistency: on replace examples, align the
    # final answer with the updated gold answer while leaving weaken neutral.
    replace_alignment_loss = zero
    replace_margin_loss = zero
    replace_loss = zero
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
        replace_loss = replace_alignment_loss + beta_replace_margin * replace_margin_loss

    # Keep preserve and replace in separate pools so replace-heavy batches do
    # not dominate the propagation signal through pooled averaging.
    propagation_loss = lambda_pres * preserve_loss + lambda_rep * replace_loss
    return {
        "propagation_loss": propagation_loss,
        "preserve_propagation_loss": preserve_loss,
        "preserve_ce_loss": preserve_ce_loss,
        "preserve_margin_loss": preserve_margin_loss,
        "replace_propagation_loss": replace_loss,
        "replace_alignment_loss": replace_alignment_loss,
        "replace_margin_loss": replace_margin_loss,
    }


def compute_boundary_propagation_loss(
    *,
    answer_logits: torch.Tensor,
    control_labels: torch.Tensor,
    answer_labels: torch.Tensor,
    early_answer_labels: torch.Tensor,
    control_to_idx: dict[str, int],
    lambda_pres: float,
    lambda_rep: float,
    beta_pres: float,
    beta_rep: float,
    m_pres: float,
    m_rep: float,
) -> dict[str, torch.Tensor]:
    log_answer_probs = F.log_softmax(answer_logits, dim=-1)
    zero = torch.zeros((), device=answer_logits.device, dtype=answer_logits.dtype)

    preserve_idx = control_to_idx["preserve"]
    replace_idx = control_to_idx.get("replace")
    preserve_mask = control_labels == preserve_idx
    replace_mask = replace_idx is not None and control_labels == replace_idx
    if isinstance(replace_mask, bool):
        replace_mask = torch.zeros_like(control_labels, dtype=torch.bool)

    preserve_loss = zero
    preserve_ce_loss = zero
    preserve_margin_loss = zero
    if torch.any(preserve_mask):
        preserve_log_probs = log_answer_probs[preserve_mask]
        preserve_targets = early_answer_labels[preserve_mask]
        s_early = preserve_log_probs.gather(
            dim=1,
            index=preserve_targets.unsqueeze(1),
        ).squeeze(1)
        preserve_ce_loss = -s_early.mean()
        non_early_log_probs = preserve_log_probs.masked_fill(
            F.one_hot(preserve_targets, num_classes=preserve_log_probs.size(1)).bool(),
            float("-inf"),
        )
        max_non_early_score = non_early_log_probs.max(dim=1).values
        preserve_gap = s_early - max_non_early_score
        preserve_margin_loss = torch.relu(m_pres - preserve_gap).mean()
        preserve_loss = preserve_ce_loss + beta_pres * preserve_margin_loss

    replace_loss = zero
    replace_ce_loss = zero
    replace_margin_loss = zero
    if torch.any(replace_mask):
        replace_log_probs = log_answer_probs[replace_mask]
        replace_targets = answer_labels[replace_mask]
        early_targets = early_answer_labels[replace_mask]
        s_gold = replace_log_probs.gather(
            dim=1,
            index=replace_targets.unsqueeze(1),
        ).squeeze(1)
        s_early = replace_log_probs.gather(
            dim=1,
            index=early_targets.unsqueeze(1),
        ).squeeze(1)
        replace_ce_loss = -s_gold.mean()
        replace_gap = s_gold - s_early
        replace_margin_loss = torch.relu(m_rep - replace_gap).mean()
        replace_loss = replace_ce_loss + beta_rep * replace_margin_loss

    propagation_loss = lambda_pres * preserve_loss + lambda_rep * replace_loss
    return {
        "propagation_loss": propagation_loss,
        "preserve_propagation_loss": preserve_loss,
        "preserve_ce_loss": preserve_ce_loss,
        "preserve_margin_loss": preserve_margin_loss,
        "replace_propagation_loss": replace_loss,
        "replace_alignment_loss": replace_ce_loss,
        "replace_margin_loss": replace_margin_loss,
    }


def compute_boundary_propagation_loss_replace_margin_stopgrad_early(
    *,
    answer_logits: torch.Tensor,
    control_labels: torch.Tensor,
    answer_labels: torch.Tensor,
    early_answer_labels: torch.Tensor,
    control_to_idx: dict[str, int],
    lambda_pres: float,
    lambda_rep: float,
    beta_pres: float,
    beta_rep: float,
    m_pres: float,
    m_rep: float,
) -> dict[str, torch.Tensor]:
    log_answer_probs = F.log_softmax(answer_logits, dim=-1)
    zero = torch.zeros((), device=answer_logits.device, dtype=answer_logits.dtype)

    preserve_idx = control_to_idx["preserve"]
    replace_idx = control_to_idx.get("replace")
    preserve_mask = control_labels == preserve_idx
    replace_mask = replace_idx is not None and control_labels == replace_idx
    if isinstance(replace_mask, bool):
        replace_mask = torch.zeros_like(control_labels, dtype=torch.bool)

    preserve_loss = zero
    preserve_ce_loss = zero
    preserve_margin_loss = zero
    if torch.any(preserve_mask):
        preserve_log_probs = log_answer_probs[preserve_mask]
        preserve_targets = early_answer_labels[preserve_mask]
        s_early = preserve_log_probs.gather(
            dim=1,
            index=preserve_targets.unsqueeze(1),
        ).squeeze(1)
        preserve_ce_loss = -s_early.mean()
        non_early_log_probs = preserve_log_probs.masked_fill(
            F.one_hot(preserve_targets, num_classes=preserve_log_probs.size(1)).bool(),
            float("-inf"),
        )
        max_non_early_score = non_early_log_probs.max(dim=1).values
        preserve_gap = s_early - max_non_early_score
        preserve_margin_loss = torch.relu(m_pres - preserve_gap).mean()
        preserve_loss = preserve_ce_loss + beta_pres * preserve_margin_loss

    replace_loss = zero
    replace_ce_loss = zero
    replace_margin_loss = zero
    if torch.any(replace_mask):
        replace_log_probs = log_answer_probs[replace_mask]
        replace_targets = answer_labels[replace_mask]
        early_targets = early_answer_labels[replace_mask]
        s_gold = replace_log_probs.gather(
            dim=1,
            index=replace_targets.unsqueeze(1),
        ).squeeze(1)
        s_early = replace_log_probs.gather(
            dim=1,
            index=early_targets.unsqueeze(1),
        ).squeeze(1)
        replace_ce_loss = -s_gold.mean()
        replace_gap = s_gold - s_early.detach()
        replace_margin_loss = torch.relu(m_rep - replace_gap).mean()
        replace_loss = replace_ce_loss + beta_rep * replace_margin_loss

    propagation_loss = lambda_pres * preserve_loss + lambda_rep * replace_loss
    return {
        "propagation_loss": propagation_loss,
        "preserve_propagation_loss": preserve_loss,
        "preserve_ce_loss": preserve_ce_loss,
        "preserve_margin_loss": preserve_margin_loss,
        "replace_propagation_loss": replace_loss,
        "replace_alignment_loss": replace_ce_loss,
        "replace_margin_loss": replace_margin_loss,
    }


def compute_conditionally_masked_answer_loss(
    *,
    answer_logits: torch.Tensor,
    control_labels: torch.Tensor,
    answer_labels: torch.Tensor,
    early_answer_labels: torch.Tensor,
    control_to_idx: dict[str, int],
    answer_class_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    per_example_loss = F.cross_entropy(
        answer_logits,
        answer_labels,
        weight=answer_class_weights,
        reduction="none",
    )
    preserve_idx = control_to_idx["preserve"]
    replace_idx = control_to_idx.get("replace")
    weaken_idx = control_to_idx.get("weaken")

    zero = torch.zeros((), device=answer_logits.device, dtype=answer_logits.dtype)
    group_losses: list[torch.Tensor] = []

    preserve_loss = zero
    preserve_mask = control_labels == preserve_idx
    if torch.any(preserve_mask):
        preserve_targets = early_answer_labels[preserve_mask]
        preserve_logits = answer_logits[preserve_mask]
        preserve_weights = (
            answer_class_weights
            if answer_class_weights is None
            else answer_class_weights.to(answer_logits.device)
        )
        preserve_loss = F.cross_entropy(
            preserve_logits,
            preserve_targets,
            weight=preserve_weights,
            reduction="mean",
        )
        group_losses.append(preserve_loss)

    replace_loss = zero
    if replace_idx is not None:
        replace_mask = control_labels == replace_idx
        if torch.any(replace_mask):
            replace_loss = per_example_loss[replace_mask].mean()
            group_losses.append(replace_loss)

    weaken_loss = zero
    if weaken_idx is not None:
        weaken_mask = control_labels == weaken_idx
        if torch.any(weaken_mask):
            weaken_loss = per_example_loss[weaken_mask].mean()
            group_losses.append(weaken_loss)

    if group_losses:
        answer_loss = torch.stack(group_losses).mean()
    else:
        answer_loss = zero

    return {
        "answer_loss": answer_loss,
        "answer_loss_preserve": preserve_loss,
        "answer_loss_replace": replace_loss,
        "answer_loss_weaken": weaken_loss,
    }


def compute_gated_propagation_loss(
    *,
    control_logits: torch.Tensor,
    answer_logits: torch.Tensor,
    answer_labels: torch.Tensor,
    early_answer_labels: torch.Tensor,
    control_to_idx: dict[str, int],
    gated_lambda_pres: float,
    gated_lambda_rep: float,
    gated_beta_replace_margin: float,
    gated_margin_m: float,
) -> dict[str, torch.Tensor]:
    log_answer_probs = F.log_softmax(answer_logits, dim=-1)
    control_probs = torch.softmax(control_logits.detach(), dim=-1)
    preserve_idx = control_to_idx["preserve"]
    replace_idx = control_to_idx.get("replace")
    preserve_gate = control_probs[:, preserve_idx]
    replace_gate = (
        control_probs[:, replace_idx]
        if replace_idx is not None
        else torch.zeros_like(preserve_gate)
    )

    early_log_probs = log_answer_probs.gather(
        dim=1,
        index=early_answer_labels.unsqueeze(1),
    ).squeeze(1)
    gold_log_probs = log_answer_probs.gather(
        dim=1,
        index=answer_labels.unsqueeze(1),
    ).squeeze(1)

    preserve_loss = -(preserve_gate * early_log_probs).mean()
    replace_alignment_loss = -(replace_gate * gold_log_probs).mean()
    replace_margin_loss = torch.zeros(
        (),
        device=answer_logits.device,
        dtype=answer_logits.dtype,
    )
    if gated_beta_replace_margin > 0.0:
        margin_gap = gold_log_probs - early_log_probs
        replace_margin_loss = (
            replace_gate * torch.relu(gated_margin_m - margin_gap)
        ).mean()
    gated_loss = (
        gated_lambda_pres * preserve_loss
        + gated_lambda_rep * replace_alignment_loss
        + gated_lambda_rep * gated_beta_replace_margin * replace_margin_loss
    )
    return {
        "gated_propagation_loss": gated_loss,
        "gated_preserve_loss": preserve_loss,
        "gated_replace_loss": replace_alignment_loss,
        "gated_replace_margin_loss": replace_margin_loss,
        "gated_preserve_gate_mean": preserve_gate.mean(),
        "gated_replace_gate_mean": replace_gate.mean(),
    }


def compute_gold_gated_preserve_loss(
    *,
    answer_logits: torch.Tensor,
    control_labels: torch.Tensor,
    early_answer_labels: torch.Tensor,
    control_to_idx: dict[str, int],
    gold_preserve_lambda: float,
) -> dict[str, torch.Tensor]:
    preserve_idx = control_to_idx["preserve"]
    preserve_mask = control_labels == preserve_idx
    zero = torch.zeros((), device=answer_logits.device, dtype=answer_logits.dtype)
    preserve_loss = zero
    preserve_count = preserve_mask.sum().to(answer_logits.dtype)
    if torch.any(preserve_mask):
        log_answer_probs = F.log_softmax(answer_logits[preserve_mask], dim=-1)
        preserve_targets = early_answer_labels[preserve_mask]
        target_log_probs = log_answer_probs.gather(
            dim=1,
            index=preserve_targets.unsqueeze(1),
        ).squeeze(1)
        preserve_loss = -target_log_probs.mean()
    return {
        "gold_gated_preserve_loss": gold_preserve_lambda * preserve_loss,
        "gold_gated_preserve_raw_loss": preserve_loss,
        "gold_gated_preserve_count": preserve_count,
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
        answer_loss_variant: str = "global_gold_ce",
        answer_loss_weight: float = 1.0,
        consistency_loss_weight: float = 0.0,
        propagation_variant: str = "legacy_conditional",
        lambda_prop: float = 0.0,
        lambda_pres: float = 0.0,
        lambda_rep: float = 0.0,
        beta_preserve_margin: float = 0.0,
        preserve_margin_m: float = 0.0,
        beta_replace_margin: float = 0.0,
        margin_m: float = 0.0,
        beta_pres: float = 0.0,
        beta_rep: float = 0.0,
        m_pres: float = 0.0,
        m_rep: float = 0.0,
        gated_lambda_pres: float = 0.0,
        gated_lambda_rep: float = 0.0,
        gated_beta_replace_margin: float = 0.0,
        gated_margin_m: float = 0.0,
        gold_preserve_lambda: float = 0.0,
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
            ctrl_loss = ctrl_loss_fct(control_logits, control_labels)
            answer_loss_preserve = torch.zeros((), device=control_logits.device, dtype=control_logits.dtype)
            answer_loss_replace = answer_loss_preserve
            answer_loss_weaken = answer_loss_preserve
            if answer_loss_variant == "conditional_masked_v1" and early_answer_labels is not None:
                answer_loss_payload = compute_conditionally_masked_answer_loss(
                    answer_logits=answer_logits,
                    control_labels=control_labels,
                    answer_labels=answer_labels,
                    early_answer_labels=early_answer_labels,
                    control_to_idx=self.control_to_idx,
                    answer_class_weights=answer_class_weights,
                )
                ans_loss = answer_loss_payload["answer_loss"]
                answer_loss_preserve = answer_loss_payload["answer_loss_preserve"]
                answer_loss_replace = answer_loss_payload["answer_loss_replace"]
                answer_loss_weaken = answer_loss_payload["answer_loss_weaken"]
            else:
                ans_loss_fct = nn.CrossEntropyLoss(weight=answer_class_weights)
                ans_loss = ans_loss_fct(answer_logits, answer_labels)
            total_loss = ctrl_loss + answer_loss_weight * ans_loss
            consistency_loss = torch.zeros((), device=control_logits.device, dtype=control_logits.dtype)
            propagation_loss = torch.zeros((), device=control_logits.device, dtype=control_logits.dtype)
            preserve_propagation_loss = propagation_loss
            preserve_ce_loss = propagation_loss
            preserve_margin_loss = propagation_loss
            replace_propagation_loss = propagation_loss
            replace_alignment_loss = propagation_loss
            replace_margin_loss = propagation_loss
            gated_propagation_loss = propagation_loss
            gated_preserve_loss = propagation_loss
            gated_replace_loss = propagation_loss
            gated_replace_margin_loss = propagation_loss
            gated_preserve_gate_mean = propagation_loss
            gated_replace_gate_mean = propagation_loss
            gold_gated_preserve_loss = propagation_loss
            gold_gated_preserve_raw_loss = propagation_loss
            gold_gated_preserve_count = propagation_loss
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
            effective_lambda_pres = lambda_pres if lambda_pres > 0.0 else lambda_prop
            effective_lambda_rep = lambda_rep if lambda_rep > 0.0 else lambda_prop
            if (effective_lambda_pres > 0.0 or effective_lambda_rep > 0.0) and early_answer_labels is not None:
                if propagation_variant == "boundary_objective_v1":
                    propagation_payload = compute_boundary_propagation_loss(
                        answer_logits=answer_logits,
                        control_labels=control_labels,
                        answer_labels=answer_labels,
                        early_answer_labels=early_answer_labels,
                        control_to_idx=self.control_to_idx,
                        lambda_pres=effective_lambda_pres,
                        lambda_rep=effective_lambda_rep,
                        beta_pres=beta_pres,
                        beta_rep=beta_rep,
                        m_pres=m_pres,
                        m_rep=m_rep,
                    )
                elif propagation_variant == "boundary_objective_v5_replace_margin_stopgrad_early_v1":
                    propagation_payload = compute_boundary_propagation_loss_replace_margin_stopgrad_early(
                        answer_logits=answer_logits,
                        control_labels=control_labels,
                        answer_labels=answer_labels,
                        early_answer_labels=early_answer_labels,
                        control_to_idx=self.control_to_idx,
                        lambda_pres=effective_lambda_pres,
                        lambda_rep=effective_lambda_rep,
                        beta_pres=beta_pres,
                        beta_rep=beta_rep,
                        m_pres=m_pres,
                        m_rep=m_rep,
                    )
                else:
                    propagation_payload = compute_conditional_propagation_loss(
                        answer_logits=answer_logits,
                        control_labels=control_labels,
                        answer_labels=answer_labels,
                        early_answer_labels=early_answer_labels,
                        control_to_idx=self.control_to_idx,
                        lambda_pres=effective_lambda_pres,
                        lambda_rep=effective_lambda_rep,
                        beta_preserve_margin=beta_preserve_margin,
                        preserve_margin_m=preserve_margin_m,
                        beta_replace_margin=beta_replace_margin,
                        margin_m=margin_m,
                    )
                propagation_loss = propagation_payload["propagation_loss"]
                preserve_propagation_loss = propagation_payload["preserve_propagation_loss"]
                preserve_ce_loss = propagation_payload["preserve_ce_loss"]
                preserve_margin_loss = propagation_payload["preserve_margin_loss"]
                replace_propagation_loss = propagation_payload["replace_propagation_loss"]
                replace_alignment_loss = propagation_payload["replace_alignment_loss"]
                replace_margin_loss = propagation_payload["replace_margin_loss"]
                total_loss = total_loss + propagation_loss
            if (
                (gated_lambda_pres > 0.0 or gated_lambda_rep > 0.0)
                and early_answer_labels is not None
            ):
                gated_payload = compute_gated_propagation_loss(
                    control_logits=control_logits,
                    answer_logits=answer_logits,
                    answer_labels=answer_labels,
                    early_answer_labels=early_answer_labels,
                    control_to_idx=self.control_to_idx,
                    gated_lambda_pres=gated_lambda_pres,
                    gated_lambda_rep=gated_lambda_rep,
                    gated_beta_replace_margin=gated_beta_replace_margin,
                    gated_margin_m=gated_margin_m,
                )
                gated_propagation_loss = gated_payload["gated_propagation_loss"]
                gated_preserve_loss = gated_payload["gated_preserve_loss"]
                gated_replace_loss = gated_payload["gated_replace_loss"]
                gated_replace_margin_loss = gated_payload["gated_replace_margin_loss"]
                gated_preserve_gate_mean = gated_payload["gated_preserve_gate_mean"]
                gated_replace_gate_mean = gated_payload["gated_replace_gate_mean"]
                total_loss = total_loss + gated_propagation_loss
            if gold_preserve_lambda > 0.0 and early_answer_labels is not None:
                gold_preserve_payload = compute_gold_gated_preserve_loss(
                    answer_logits=answer_logits,
                    control_labels=control_labels,
                    early_answer_labels=early_answer_labels,
                    control_to_idx=self.control_to_idx,
                    gold_preserve_lambda=gold_preserve_lambda,
                )
                gold_gated_preserve_loss = gold_preserve_payload["gold_gated_preserve_loss"]
                gold_gated_preserve_raw_loss = gold_preserve_payload["gold_gated_preserve_raw_loss"]
                gold_gated_preserve_count = gold_preserve_payload["gold_gated_preserve_count"]
                total_loss = total_loss + gold_gated_preserve_loss
            payload["loss"] = total_loss
            payload["control_loss"] = ctrl_loss
            payload["answer_loss"] = ans_loss
            payload["answer_loss_preserve"] = answer_loss_preserve
            payload["answer_loss_replace"] = answer_loss_replace
            payload["answer_loss_weaken"] = answer_loss_weaken
            payload["consistency_loss"] = consistency_loss
            payload["propagation_loss"] = propagation_loss
            payload["preserve_propagation_loss"] = preserve_propagation_loss
            payload["preserve_ce_loss"] = preserve_ce_loss
            payload["preserve_margin_loss"] = preserve_margin_loss
            payload["replace_propagation_loss"] = replace_propagation_loss
            payload["replace_alignment_loss"] = replace_alignment_loss
            payload["replace_margin_loss"] = replace_margin_loss
            payload["gated_propagation_loss"] = gated_propagation_loss
            payload["gated_preserve_loss"] = gated_preserve_loss
            payload["gated_replace_loss"] = gated_replace_loss
            payload["gated_replace_margin_loss"] = gated_replace_margin_loss
            payload["gated_preserve_gate_mean"] = gated_preserve_gate_mean
            payload["gated_replace_gate_mean"] = gated_replace_gate_mean
            payload["gold_gated_preserve_loss"] = gold_gated_preserve_loss
            payload["gold_gated_preserve_raw_loss"] = gold_gated_preserve_raw_loss
            payload["gold_gated_preserve_count"] = gold_gated_preserve_count
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
