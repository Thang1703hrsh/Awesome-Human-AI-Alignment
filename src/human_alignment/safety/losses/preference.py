"""Pairwise preference losses on sequence log-probs.

All functions take per-example sequence log-probs of shape (B,) and return a ``PairwiseLossOutput`` with
per-example ``losses`` (B,) and detached implicit rewards ``beta * (logp - ref_logp)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F


@dataclass
class PairwiseLossOutput:
    losses: torch.Tensor
    chosen_rewards: torch.Tensor
    rejected_rewards: torch.Tensor


def _implicit_rewards(beta, pw, pl, rw, rl):
    return beta * (pw - rw).detach(), beta * (pl - rl).detach()


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
) -> PairwiseLossOutput:
    """DPO (safe-rlhf ``dpo/trainer.py:170``; TRL sigmoid ``dpo_loss``)."""
    h = (policy_chosen_logps - ref_chosen_logps) - (
        policy_rejected_logps - ref_rejected_logps
    )
    losses = -F.logsigmoid(beta * h)
    return PairwiseLossOutput(
        losses,
        *_implicit_rewards(
            beta,
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
        ),
    )


def safedpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    chosen_unsafe: torch.Tensor,
    rejected_unsafe: torch.Tensor,
    beta: float,
    delta: float,
) -> PairwiseLossOutput:
    """SafeDPO Eq. 12 (arXiv 2505.20065): ``-logσ(β·h − (h̃_l − h̃_w)·Δ)``.

    Must be applied to pairs already passed through ``safedpo_transform`` (T(D)); ``*_unsafe`` are 1 for unsafe.
    """
    h = (policy_chosen_logps - ref_chosen_logps) - (
        policy_rejected_logps - ref_rejected_logps
    )
    margin = (rejected_unsafe.to(h.dtype) - chosen_unsafe.to(h.dtype)) * delta
    losses = -F.logsigmoid(beta * h - margin)
    return PairwiseLossOutput(
        losses,
        *_implicit_rewards(
            beta,
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
        ),
    )


BSOGenerator = Literal["sba", "ba", "logistic", "kliep", "lsif"]


def bso_per_sample(
    log_r: torch.Tensor, generator: BSOGenerator, lam: float = 0.2, s: float = 4.0
) -> torch.Tensor:
    """``ℓ_h(R) = h'(R)·R − h(R) − h'(1/R)`` (BSO Eq. 14) in closed form, evaluated from ``log R``.

    Closed forms (derived from the generators in BSO Appendix D):
      logistic: log(1 + R)                                  (Eq. 16)
      kliep:    R − 1 + log R                               (h = R log R − R + 1)
      lsif:     R² − 2/R + 1                                (h = (R − 1)²)
      ba_λ:     [λR^{1+λ} − (1+λ)R^{−λ} + 1] / λ            (h = (R^{1+λ} − R)/λ)
      sba_λ:    [λR^{1+λ} − (1+λ)R^{−λ} + 1] / (sλ(λ+1))    (h = (R^{1+λ} − R)/(sλ(λ+1)), Eq. 23)
    """
    if generator == "logistic":
        return F.softplus(log_r)
    if generator == "kliep":
        return torch.exp(log_r) - 1 + log_r
    if generator == "lsif":
        return torch.exp(2 * log_r) - 2 * torch.exp(-log_r) + 1
    if generator in ("ba", "sba"):
        if lam <= 0:
            raise ValueError("lambda must be > 0")
        num = (
            lam * torch.exp((1 + lam) * log_r) - (1 + lam) * torch.exp(-lam * log_r) + 1
        )
        denom = lam if generator == "ba" else s * lam * (lam + 1)
        return num / denom
    raise ValueError(f"Unknown BSO generator {generator!r}")


def bso_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    chosen_unsafe: torch.Tensor,
    rejected_unsafe: torch.Tensor,
    beta: float,
    safety_penalty: float = 30.0,
    generator: BSOGenerator = "sba",
    lam: float = 0.2,
    s: float = 4.0,
    penalty_inside_beta: bool = False,
) -> PairwiseLossOutput:
    """BSO (arXiv 2605.12339) Eq. 12/14/20. ``log R_θ = −(β·h + C·(s_w − s_l))``.

    ``penalty_inside_beta=True`` uses ``−β·(h + C·Δs)`` instead: the paper writes C outside β (Eq. 12, 18), which
    with the reported C=30 gives R≈e^30 for safe-winner pairs; the flag exists only because no reference code
    was released to settle this. Defaults follow the paper text (C=30, λ=0.2, s=4, SBA).
    """
    h = (policy_chosen_logps - ref_chosen_logps) - (
        policy_rejected_logps - ref_rejected_logps
    )
    ds = chosen_unsafe.to(h.dtype) - rejected_unsafe.to(h.dtype)
    if penalty_inside_beta:
        log_r = -beta * (h + safety_penalty * ds)
    else:
        log_r = -(beta * h + safety_penalty * ds)
    losses = bso_per_sample(log_r, generator, lam=lam, s=s)
    return PairwiseLossOutput(
        losses,
        *_implicit_rewards(
            beta,
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
        ),
    )


def bfpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    chosen_safe: torch.Tensor,
    rejected_safe: torch.Tensor,
    beta: float,
    b1: float = 3.0,
    alpha: float = 0.5,
) -> PairwiseLossOutput:
    """BFPO (``bfpo/src/alignment/trainer/bfpo.py:125-137``); ``*_safe`` are 1 for SAFE (opposite of SafeDPO)."""
    b3 = 1 / (b1 - 1)
    logits = (policy_chosen_logps - policy_rejected_logps) - (
        ref_chosen_logps - ref_rejected_logps
    )
    bfpo_factor = b1 * b3 * chosen_safe - b3 * rejected_safe
    safe_factor = bfpo_factor - alpha
    losses = (logits - 1 / beta * safe_factor) ** 2
    return PairwiseLossOutput(
        losses,
        *_implicit_rewards(
            beta,
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
        ),
    )


def modpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    chosen_margin_reward: torch.Tensor,
    rejected_margin_reward: torch.Tensor,
    w: torch.Tensor,
    beta: float,
    loss_type: Literal["sigmoid", "hinge"] = "sigmoid",
) -> PairwiseLossOutput:
    """MODPO (``modpo/src/trainer/modpo_trainer.py:151-160``). Margin rewards have shape (B, n-1)."""
    w = torch.as_tensor(
        w, dtype=policy_chosen_logps.dtype, device=policy_chosen_logps.device
    )
    chosen_rewards = (1 / w[0]) * (
        beta * (policy_chosen_logps - ref_chosen_logps) - chosen_margin_reward @ w[1:]
    )
    rejected_rewards = (1 / w[0]) * (
        beta * (policy_rejected_logps - ref_rejected_logps)
        - rejected_margin_reward @ w[1:]
    )
    logits = chosen_rewards - rejected_rewards
    if loss_type == "sigmoid":
        losses = -F.logsigmoid(logits)
    elif loss_type == "hinge":
        losses = torch.relu(1 - logits)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    return PairwiseLossOutput(
        losses, chosen_rewards.detach(), rejected_rewards.detach()
    )


def midpo_expert_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    chosen_score: torch.Tensor,
    rejected_score: torch.Tensor,
    beta: float,
    expert: Literal["safety", "helpfulness"],
) -> PairwiseLossOutput:
    """MidPO expert losses (``mdpo_safety_expert/trainer.py:236-247``, ``mdpo_helpfulness_expert/trainer.py:231-246``).

    safety:      -logσ(β·h + min(0, R(y_w) − R(y_l)))
    helpfulness: -logσ(β·h − max(0, R(y_w) − R(y_l)))
    The score model is under ``no_grad`` in the reference, so scores are detached here.
    """
    h = (policy_chosen_logps - ref_chosen_logps) - (
        policy_rejected_logps - ref_rejected_logps
    )
    diff = (chosen_score - rejected_score).detach().to(h.dtype)
    if expert == "safety":
        logits = beta * h + torch.clamp(diff, max=0)
    elif expert == "helpfulness":
        logits = beta * h - torch.clamp(diff, min=0)
    else:
        raise ValueError(f"Unknown MidPO expert {expert!r}")
    losses = -F.logsigmoid(logits)
    return PairwiseLossOutput(
        losses,
        *_implicit_rewards(
            beta,
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
        ),
    )


def kto_loss(
    policy_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    desirable: torch.Tensor,
    kl: torch.Tensor,
    beta: float,
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """KTO as in TRL KTOTrainer (used by SACPO). ``kl`` is the detached, clamped (>=0) batch KL estimate.

    Returns (per-example losses in batch order, implicit rewards).
    """
    logratios = policy_logps - ref_logps
    desirable = desirable.bool()
    chosen_losses = 1 - torch.sigmoid(beta * (logratios - kl))
    rejected_losses = 1 - torch.sigmoid(beta * (kl - logratios))
    losses = torch.where(
        desirable,
        desirable_weight * chosen_losses,
        undesirable_weight * rejected_losses,
    )
    return losses, beta * logratios.detach()


def kto_kl_estimate(
    policy_kl_logps: torch.Tensor, ref_kl_logps: torch.Tensor
) -> torch.Tensor:
    return (policy_kl_logps - ref_kl_logps).mean().clamp(min=0).detach()


__all__ = [
    "PairwiseLossOutput",
    "dpo_loss",
    "safedpo_loss",
    "bso_loss",
    "bso_per_sample",
    "bfpo_loss",
    "modpo_loss",
    "midpo_expert_loss",
    "kto_loss",
    "kto_kl_estimate",
]

