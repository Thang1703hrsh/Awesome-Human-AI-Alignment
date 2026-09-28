"""Reward / cost model losses (``safe-rlhf/safe_rlhf/values/{reward,cost}/trainer.py``)."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

ScoreLossType = Literal["sequence-wise", "token-wise"]


def _spans(ids_a, mask_a, ids_b, mask_b, i):
    if torch.equal(ids_a[i], ids_b[i]):
        raise ValueError("The two answers are the same!")
    end = max(mask_a[i].nonzero()[-1].item(), mask_b[i].nonzero()[-1].item())
    diverge = (ids_a[i] != ids_b[i]).nonzero()[0].item()
    return slice(diverge, end + 1)


def reward_model_loss(
    scores: torch.Tensor,  # (2B, L) better rows first
    end_scores: torch.Tensor,  # (2B,)
    better_input_ids: torch.Tensor,
    better_attention_mask: torch.Tensor,
    worse_input_ids: torch.Tensor,
    worse_attention_mask: torch.Tensor,
    loss_type: ScoreLossType = "sequence-wise",
    regularization: float = 0.0,
) -> dict[str, torch.Tensor]:
    higher_rewards, lower_rewards = scores.chunk(2, dim=0)
    higher_end, lower_end = end_scores.chunk(2, dim=0)
    if loss_type == "token-wise":
        losses = []
        for i in range(better_input_ids.size(0)):
            s = _spans(
                better_input_ids,
                better_attention_mask,
                worse_input_ids,
                worse_attention_mask,
                i,
            )
            high, low = higher_rewards[i, s], lower_rewards[i, s]
            loss_i = -F.logsigmoid(high - low).mean()
            if regularization > 0.0:
                loss_i = (
                    loss_i + regularization * torch.stack([low, high]).square().mean()
                )
            losses.append(loss_i)
        loss = torch.stack(losses).mean()
    elif loss_type == "sequence-wise":
        loss = -F.logsigmoid(higher_end - lower_end).mean()
        if regularization > 0.0:
            loss = (
                loss
                + regularization * torch.stack([lower_end, higher_end]).square().mean()
            )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    return {
        "loss": loss,
        "higher_end_reward": higher_end,
        "lower_end_reward": lower_end,
        "accuracy": (higher_end > lower_end).float().mean(),
    }


def cost_model_loss(
    scores: torch.Tensor,  # (2B, L) safer rows first
    end_scores: torch.Tensor,  # (2B,)
    safer_input_ids: torch.Tensor,
    safer_attention_mask: torch.Tensor,
    safer_safety_sign: torch.Tensor,
    unsafer_input_ids: torch.Tensor,
    unsafer_attention_mask: torch.Tensor,
    unsafer_safety_sign: torch.Tensor,
    loss_type: ScoreLossType = "sequence-wise",
    regularization: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Safety sign is +1 for safe, −1 for unsafe; cost sign is its negation."""
    lower_costs, higher_costs = scores.chunk(2, dim=0)
    lower_end, higher_end = end_scores.chunk(2, dim=0)
    lower_sign, higher_sign = -safer_safety_sign, -unsafer_safety_sign
    if loss_type == "token-wise":
        losses = []
        for i in range(safer_input_ids.size(0)):
            s = _spans(
                safer_input_ids,
                safer_attention_mask,
                unsafer_input_ids,
                unsafer_attention_mask,
                i,
            )
            lo, hi = lower_costs[i, s], higher_costs[i, s]
            loss_i = (
                -F.logsigmoid(hi - lo).mean()
                - F.logsigmoid(lower_sign[i] * lo).mean()
                - F.logsigmoid(higher_sign[i] * hi).mean()
            )
            if regularization > 0.0:
                loss_i = loss_i + regularization * torch.stack([lo, hi]).square().mean()
            losses.append(loss_i)
        loss = torch.stack(losses).mean()
    elif loss_type == "sequence-wise":
        loss = (
            -F.logsigmoid(higher_end - lower_end)
            - F.logsigmoid(lower_sign * lower_end)
            - F.logsigmoid(higher_sign * higher_end)
        ).mean()
        if regularization > 0.0:
            loss = (
                loss
                + regularization * torch.stack([lower_end, higher_end]).square().mean()
            )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    accuracy_sign = (
        torch.stack([lower_sign * lower_end > 0.0, higher_sign * higher_end > 0.0])
        .float()
        .mean()
    )
    return {
        "loss": loss,
        "higher_end_cost": higher_end,
        "lower_end_cost": lower_end,
        "accuracy": (higher_end > lower_end).float().mean(),
        "accuracy_sign": accuracy_sign,
    }

