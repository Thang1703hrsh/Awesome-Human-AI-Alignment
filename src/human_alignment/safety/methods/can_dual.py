"""CAN offline dual solver (arXiv 2405.19544; ``CAN/safe_rlhf/trainers/model_based.py:7-64``).

Minimises the convex dual  D(λ) = β·E_x log E_{y~π_ref} exp((r + λ(g − b))/β)  over λ ≥ 0 by projected gradient
descent, where r = helpfulness score, g = safety score (= −cost), b = threshold, β = ``kl_coeff``.
Inputs are (num_prompts, num_samples_per_prompt) arrays of scores of samples drawn from π_ref.

Reproduced as in the reference: default lr is 1 (and 2·lr when given), per-restart decay ``lr /= 2**loop``
compounds, convergence = |λ − last 5 iterates| < err after ≥10 iterations, at most 5 restarts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import softmax


@dataclass
class DualResult:
    lam_star: float | None
    converged: bool
    lam_trajectory: list = field(default_factory=list)
    objective_trajectory: list = field(default_factory=list)
    constraint_trajectory: list = field(default_factory=list)
    helpfulness_trajectory: list = field(default_factory=list)
    safety_trajectory: list = field(default_factory=list)


def _log_mean_exp(logits: np.ndarray) -> np.ndarray:
    m = logits.max(axis=1)
    return np.log(np.mean(np.exp(logits - m.reshape(-1, 1)), axis=1)) + m


def solve_dual(
    helpfulness_scores: np.ndarray,
    safety_scores: np.ndarray,
    threshold: float,
    kl_coeff: float,
    lam_init: float = 1.0,
    lr: float | None = None,
    num_iters: int = 200,
    err: float = 1e-5,
    max_loops: int = 5,
) -> DualResult:
    help_s = np.asarray(helpfulness_scores, dtype=np.float64)
    safe_s = np.asarray(safety_scores, dtype=np.float64)
    if help_s.shape != safe_s.shape or help_s.ndim != 2:
        raise ValueError(
            "scores must be (num_prompts, num_samples) arrays of equal shape"
        )
    lr = 1.0 if lr is None else 2 * lr
    num_loops = 0
    while True:
        lam = lam_init
        lr = lr / 2**num_loops
        res = DualResult(lam_star=None, converged=False)
        for idx_iter in range(num_iters):
            logits = (help_s + (safe_s - threshold) * lam) / kl_coeff
            probs = softmax(logits, axis=1)
            gradient = np.sum(probs * (safe_s - threshold), axis=1).mean()
            lam = np.maximum(lam - lr * gradient, 0)
            res.lam_trajectory.append(lam)
            res.objective_trajectory.append(
                kl_coeff * np.mean(_log_mean_exp(logits)) - lam * gradient
            )
            res.constraint_trajectory.append(gradient)
            res.helpfulness_trajectory.append(np.sum(probs * help_s, axis=1).mean())
            res.safety_trajectory.append(np.sum(probs * safe_s, axis=1).mean())
            if (
                idx_iter >= 10
                and np.abs(lam - np.array(res.lam_trajectory[-1:-6:-1])).max() < err
            ):
                res.converged = True
                res.lam_star = float(lam)
                return res
        num_loops += 1
        if num_loops >= max_loops:
            return res

