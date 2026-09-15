"""Pure PyTorch losses used by preference-distillation training.

This module contains the reusable mathematical core recovered from the
temporary research scripts. It deliberately has no dependency on TRL or the
Transformers trainer stack.
"""

from __future__ import annotations

from itertools import permutations
from math import isfinite, log
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F


def compress_probabilities(
    logits: torch.Tensor,
    *,
    top_k: int = 50,
) -> list[dict[str, Any]]:
    """Compress token logits into top-k probabilities and one residual-mass bin."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape [tokens, vocabulary]")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    count = min(top_k, logits.shape[-1])
    probabilities = logits.softmax(dim=-1)
    values, indices = probabilities.topk(count, dim=-1)
    residual = (1.0 - values.sum(dim=-1)).clamp_min(0.0)
    return [
        {
            "indices": token_indices.tolist(),
            "values": token_values.tolist(),
            "remaining_probs_sum": float(token_residual),
        }
        for token_indices, token_values, token_residual in zip(indices, values, residual)
    ]


def compressed_log_ratio_advantages(
    teacher: Sequence[Mapping[str, Any]],
    reference: Sequence[Mapping[str, Any]],
    *,
    epsilon: float = 1e-8,
) -> list[dict[str, list[float] | list[int]]]:
    """Build sparse teacher-reference log-probability margins for ADPA."""

    if len(teacher) != len(reference):
        raise ValueError("Teacher and reference token sequences must have equal lengths")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    output = []
    for teacher_item, reference_item in zip(teacher, reference):
        teacher_map = dict(zip(teacher_item.get("indices", ()), teacher_item.get("values", ())))
        reference_map = dict(
            zip(reference_item.get("indices", ()), reference_item.get("values", ()))
        )
        indices = sorted(set(teacher_map) | set(reference_map))
        margins = [
            log(max(float(teacher_map.get(index, epsilon)), epsilon))
            - log(max(float(reference_map.get(index, epsilon)), epsilon))
            for index in indices
        ]
        order = sorted(range(len(indices)), key=margins.__getitem__, reverse=True)
        output.append(
            {
                "indices": [int(indices[index]) for index in order],
                "values": [margins[index] for index in order],
            }
        )
    return output


def _score_matrix(scores: torch.Tensor, name: str) -> torch.Tensor:
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError(f"{name} must have shape [batch, responses] with at least two responses")
    if not torch.isfinite(scores).all():
        raise ValueError(f"{name} contains a non-finite value")
    return scores


def plackett_luce_log_probability(
    scores: torch.Tensor,
    rankings: torch.Tensor,
) -> torch.Tensor:
    """Return the log probability of each ranking under Plackett-Luce scores."""

    _score_matrix(scores, "scores")
    if rankings.shape != scores.shape:
        raise ValueError("rankings must have the same shape as scores")
    ordered = scores.gather(1, rankings.long())
    denominators = torch.logcumsumexp(ordered.flip(-1), dim=-1).flip(-1)
    return (ordered - denominators).sum(dim=-1)


def vpd_loss(
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    *,
    beta: float = 1.0,
) -> torch.Tensor:
    """Vanilla preference distillation via teacher-induced ranking likelihood."""

    _score_matrix(student_scores, "student_scores")
    _score_matrix(teacher_scores, "teacher_scores")
    if student_scores.shape != teacher_scores.shape:
        raise ValueError("student_scores and teacher_scores must have the same shape")
    rankings = teacher_scores.argsort(dim=-1, descending=True)
    return -plackett_luce_log_probability(beta * student_scores, rankings)


def ppd_loss(
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    *,
    beta: float = 1.0,
    exact_ranking_limit: int = 8,
) -> torch.Tensor:
    """Probabilistic preference distillation using exact ranking-distribution JSD."""

    _score_matrix(student_scores, "student_scores")
    _score_matrix(teacher_scores, "teacher_scores")
    if student_scores.shape != teacher_scores.shape:
        raise ValueError("student_scores and teacher_scores must have the same shape")
    batch_size, response_count = student_scores.shape
    if response_count > exact_ranking_limit:
        raise ValueError(
            f"Exact PPD enumerates rankings and supports at most {exact_ranking_limit} "
            f"responses; received {response_count}"
        )

    ranking_table = torch.tensor(
        tuple(permutations(range(response_count))),
        device=student_scores.device,
        dtype=torch.long,
    )
    ranking_count = ranking_table.shape[0]
    expanded_rankings = ranking_table.unsqueeze(0).expand(batch_size, -1, -1)

    def distribution(scores: torch.Tensor) -> torch.Tensor:
        flat_scores = scores[:, None, :].expand(-1, ranking_count, -1).reshape(-1, response_count)
        flat_rankings = expanded_rankings.reshape(-1, response_count)
        log_probs = plackett_luce_log_probability(flat_scores, flat_rankings)
        log_probs = log_probs.reshape(batch_size, ranking_count)
        return log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)

    student_log_probs = distribution(beta * student_scores)
    teacher_log_probs = distribution(beta * teacher_scores)
    mixture_log_probs = torch.logaddexp(student_log_probs, teacher_log_probs) - torch.log(
        student_scores.new_tensor(2.0)
    )
    student_kl = (
        student_log_probs.exp() * (student_log_probs - mixture_log_probs)
    ).sum(dim=-1)
    teacher_kl = (
        teacher_log_probs.exp() * (teacher_log_probs - mixture_log_probs)
    ).sum(dim=-1)
    return 0.5 * (student_kl + teacher_kl)


def causal_token_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
    average: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return sequence scores, aligned token log-probabilities, and their mask."""

    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels must have shapes [B,T,V] and [B,T]")
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:].clone()
    mask = shifted_labels.ne(ignore_index)
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    token_logps = shifted_logits.log_softmax(-1).gather(
        dim=-1, index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    token_logps = token_logps.masked_fill(~mask, 0.0)
    token_counts = mask.sum(dim=-1).clamp_min(1)
    sequence_logps = token_logps.sum(dim=-1)
    if average:
        sequence_logps = sequence_logps / token_counts
    return sequence_logps, token_logps, mask


def _compressed_projection(
    student_probs: torch.Tensor,
    compressed: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    """Project a full student distribution onto teacher top-k bins plus a residual bin."""

    raw_indices = tuple(int(index) for index in compressed.get("indices", ()))
    raw_values = tuple(float(value) for value in compressed.get("values", ()))
    if len(raw_indices) != len(raw_values):
        raise ValueError("Compressed distribution indices and values must have equal length")
    if any(not isfinite(value) or value < 0 for value in raw_values):
        raise ValueError("Compressed probabilities must be finite and non-negative")

    merged: dict[int, float] = {}
    residual = float(compressed.get("remaining_probs_sum", 0.0))
    if not isfinite(residual) or residual < -1e-6:
        raise ValueError("Compressed residual probability must be finite and non-negative")
    for index, value in zip(raw_indices, raw_values):
        if 0 <= index < student_probs.shape[-1]:
            merged[index] = merged.get(index, 0.0) + value
        else:
            residual += value

    indices = tuple(merged)
    if indices:
        index_tensor = torch.tensor(indices, device=student_probs.device, dtype=torch.long)
        selected = student_probs.index_select(0, index_tensor)
    else:
        selected = student_probs.new_empty((0,))
    student_residual = (1.0 - selected.sum()).clamp_min(0.0)
    projected_student = torch.cat((selected, student_residual.unsqueeze(0)))
    projected_teacher = student_probs.new_tensor([merged[index] for index in indices] + [residual])
    projected_student = projected_student.clamp_min(1e-8)
    projected_teacher = projected_teacher.clamp_min(0.0)
    projected_student = projected_student / projected_student.sum()
    teacher_total = projected_teacher.sum()
    if teacher_total <= 0:
        raise ValueError("Compressed teacher distribution has no positive probability mass")
    projected_teacher = projected_teacher / teacher_total
    return projected_student, projected_teacher, indices


def compressed_forward_kl(
    student_probs: torch.Tensor,
    labels: torch.Tensor,
    compressed_batch: Sequence[Sequence[Mapping[str, Any]]],
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute token-mean KL(teacher || student) from compressed teacher probabilities."""

    losses: list[torch.Tensor] = []
    for row, compressed_row in enumerate(compressed_batch):
        positions = labels[row].ne(ignore_index).nonzero(as_tuple=False).flatten()
        if len(compressed_row) != len(positions):
            raise ValueError(
                "Compressed teacher-token count does not match non-masked label count"
            )
        for position, compressed in zip(positions.tolist(), compressed_row):
            student, teacher, _ = _compressed_projection(student_probs[row, position], compressed)
            losses.append((teacher * (teacher.clamp_min(1e-8).log() - student.log())).sum())
    if not losses:
        raise ValueError("No teacher-token probabilities align with non-masked labels")
    return torch.stack(losses).mean()


def advantage_expectation_loss(
    student_probs: torch.Tensor,
    labels: torch.Tensor,
    advantages: Sequence[Sequence[Mapping[str, Any]]],
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Maximize the expected teacher-reference token advantage used by ADPA."""

    values: list[torch.Tensor] = []
    for row, advantage_row in enumerate(advantages):
        positions = labels[row].ne(ignore_index).nonzero(as_tuple=False).flatten()
        if len(advantage_row) != len(positions):
            raise ValueError("ADPA token-advantage count does not match label count")
        for position, item in zip(positions.tolist(), advantage_row):
            indices = torch.tensor(
                item.get("indices", ()), device=student_probs.device, dtype=torch.long
            )
            margins = student_probs.new_tensor(item.get("values", ()))
            valid = indices.ge(0) & indices.lt(student_probs.shape[-1])
            if valid.any():
                values.append((student_probs[row, position, indices[valid]] * margins[valid]).sum())
    if not values:
        raise ValueError("No ADPA advantages align with non-masked labels")
    return -torch.stack(values).mean()


def _parent_sums(
    token_logps: torch.Tensor,
    parent_lists: Sequence[Sequence[int]],
) -> torch.Tensor:
    values: list[torch.Tensor] = []
    for parents in parent_lists:
        indices = torch.tensor(parents, device=token_logps.device, dtype=torch.long) - 1
        indices = indices[(indices >= 0) & (indices < token_logps.shape[0])]
        value = (
            token_logps.index_select(0, indices).sum()
            if indices.numel()
            else token_logps.new_zeros(())
        )
        values.append(value)
    return torch.stack(values) if values else token_logps.new_empty((0,))


def ctpd_loss(
    policy_chosen_token_logps: torch.Tensor,
    policy_rejected_token_logps: torch.Tensor,
    reference_chosen_token_logps: torch.Tensor,
    reference_rejected_token_logps: torch.Tensor,
    chosen_parent_lists: Sequence[Sequence[Sequence[int]]],
    rejected_parent_lists: Sequence[Sequence[Sequence[int]]],
    chosen_weights: torch.Tensor,
    rejected_weights: torch.Tensor,
    *,
    beta: float = 0.5,
) -> torch.Tensor:
    """Compute weighted parent-token KD-TIS-DPO losses for CTPD."""

    losses: list[torch.Tensor] = []
    for row in range(policy_chosen_token_logps.shape[0]):
        chosen_policy = _parent_sums(policy_chosen_token_logps[row], chosen_parent_lists[row])
        chosen_reference = _parent_sums(reference_chosen_token_logps[row], chosen_parent_lists[row])
        rejected_policy = _parent_sums(policy_rejected_token_logps[row], rejected_parent_lists[row])
        rejected_reference = _parent_sums(
            reference_rejected_token_logps[row], rejected_parent_lists[row]
        )
        chosen_count = chosen_policy.numel()
        rejected_count = rejected_policy.numel()
        if chosen_weights.shape[1] < chosen_count or rejected_weights.shape[1] < rejected_count:
            raise ValueError("CTPD weight count does not match its parent-token groups")
        chosen_margin = (
            (chosen_policy - chosen_reference) * chosen_weights[row, :chosen_count]
        ).sum()
        rejected_margin = (
            (rejected_policy - rejected_reference) * rejected_weights[row, :rejected_count]
        ).sum()
        losses.append(-F.logsigmoid(beta * (chosen_margin - rejected_margin)))
    if not losses:
        raise ValueError("CTPD received an empty batch")
    return torch.stack(losses)


def _tvkd_sequence_scores(
    student_probs: torch.Tensor,
    labels: torch.Tensor,
    teacher_batch: Sequence[Sequence[Mapping[str, Any]]],
    *,
    value_function: str,
    value_discount: float,
    value_temperature: float,
    student_value_weight: float,
    teacher_value_weight: float,
    ignore_index: int = -100,
) -> torch.Tensor:
    scores: list[torch.Tensor] = []
    for row, teacher_row in enumerate(teacher_batch):
        positions = labels[row].ne(ignore_index).nonzero(as_tuple=False).flatten().tolist()
        if len(teacher_row) != len(positions):
            raise ValueError("TVKD teacher-token count does not match label count")
        actions: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        for position, compressed in zip(positions, teacher_row):
            student, teacher, indices = _compressed_projection(
                student_probs[row, position], compressed
            )
            student_log = student.log()
            teacher_log = teacher.clamp_min(1e-8).log()
            if value_function == "entropy":
                values.append(-(student * student_log).sum())
            elif value_function == "soft":
                combined = (
                    student_value_weight * student_log + teacher_value_weight * teacher_log
                ) / value_temperature
                values.append(value_temperature * torch.logsumexp(combined, dim=-1))
            else:
                raise ValueError(f"Unknown TVKD value function: {value_function}")
            label = int(labels[row, position])
            action_index = indices.index(label) if label in indices else len(indices)
            actions.append(student_log[action_index])
        if not actions:
            raise ValueError("TVKD teacher probabilities do not align with response labels")
        action_tensor = torch.stack(actions)
        value_tensor = torch.stack(values)
        next_values = torch.cat((value_tensor[1:], value_tensor.new_zeros(1)))
        scores.append((action_tensor - value_discount * next_values).sum())
    return torch.stack(scores)


def tvkd_loss(
    chosen_student_probs: torch.Tensor,
    rejected_student_probs: torch.Tensor,
    chosen_labels: torch.Tensor,
    rejected_labels: torch.Tensor,
    teacher_chosen: Sequence[Sequence[Mapping[str, Any]]],
    teacher_rejected: Sequence[Sequence[Mapping[str, Any]]],
    *,
    value_function: str = "entropy",
    value_discount: float = 1.0,
    value_temperature: float = 1.0,
    student_value_weight: float = 0.0,
    teacher_value_weight: float = 0.7,
) -> torch.Tensor:
    """Compute the value-based preference distillation objective used by TVKD."""

    chosen_scores = _tvkd_sequence_scores(
        chosen_student_probs,
        chosen_labels,
        teacher_chosen,
        value_function=value_function,
        value_discount=value_discount,
        value_temperature=value_temperature,
        student_value_weight=student_value_weight,
        teacher_value_weight=teacher_value_weight,
    )
    rejected_scores = _tvkd_sequence_scores(
        rejected_student_probs,
        rejected_labels,
        teacher_rejected,
        value_function=value_function,
        value_discount=value_discount,
        value_temperature=value_temperature,
        student_value_weight=student_value_weight,
        teacher_value_weight=teacher_value_weight,
    )
    return -F.logsigmoid(chosen_scores - rejected_scores)
