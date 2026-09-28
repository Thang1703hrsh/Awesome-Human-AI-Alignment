"""Construction of (prompt, chosen, rejected) pairs from PKU-SafeRLHF-style rows.

A raw row has the HF schema of ``PKU-Alignment/PKU-SafeRLHF[-30K]``: ``prompt, response_0, response_1,
is_response_0_safe, is_response_1_safe, better_response_id, safer_response_id``.

A pair is a dict ``{prompt, chosen, rejected, chosen_safe, rejected_safe}`` (``*_safe`` are bools, True = safe).
"""

from __future__ import annotations

from typing import Iterable, Iterator, Literal

import torch

Pair = dict
PairSelection = Literal["better", "safer", "safe_better", "safedpo"]


def select_pair(row: dict, by: Literal["better", "safer"]) -> Pair:
    """``better``: safe-rlhf ``better`` / SACPO helpfulness / MODPO ``better`` / BFPO / CAN prefer_safe=False.
    ``safer``: safe-rlhf ``safer`` / SACPO safety / MODPO ``safer`` / MidPO safety expert / CAN prefer_safe=True."""
    idx = int(row[f"{by}_response_id"])
    other = 1 - idx
    return {
        "prompt": row["prompt"],
        "chosen": row[f"response_{idx}"],
        "rejected": row[f"response_{other}"],
        "chosen_safe": bool(row[f"is_response_{idx}_safe"]),
        "rejected_safe": bool(row[f"is_response_{other}_safe"]),
    }


def safedpo_transform(pair: Pair) -> Pair | None:
    """T(D) of SafeDPO §3.2 / BSO §3.5: keep if chosen safe; swap if chosen unsafe & rejected safe; drop if both unsafe."""
    if pair["chosen_safe"]:
        return pair
    if pair["rejected_safe"]:
        return {
            **pair,
            "chosen": pair["rejected"],
            "rejected": pair["chosen"],
            "chosen_safe": True,
            "rejected_safe": False,
        }
    return None


def build_pairs(rows: Iterable[dict], selection: PairSelection) -> Iterator[Pair]:
    """Pair selections used by the method list.

    better       DPO-Helpful, SACPO stage helpful, MODPO primary, BFPO, MidPO helpfulness expert
    safer        DPO-Harmless, SACPO stage safety, MODPO margin model, MidPO safety expert
    safe_better  DPO-SafeBetter: ``better`` pairs, dropping those whose preferred response is unsafe
                 (SafeDPO paper §5.1 definition)
    safedpo      SafeDPO / BSO: ``better`` pairs passed through T(D)
    """
    for row in rows:
        if selection in ("better", "safer"):
            yield select_pair(row, selection)
        elif selection == "safe_better":
            pair = select_pair(row, "better")
            if pair["chosen_safe"]:
                yield pair
        elif selection == "safedpo":
            pair = safedpo_transform(select_pair(row, "better"))
            if pair is not None:
                yield pair
        else:
            raise ValueError(f"Unknown pair selection {selection!r}")


def bernoulli_relabel(
    rows: list[dict],
    logit_1_minus_0: torch.Tensor,
    generator: torch.Generator | None = None,
) -> list[Pair]:
    """CAN relabeling (``dpo.py:233-241``, ``dpo_alg2.py:249-261``): index ~ Bernoulli(σ(w_1 − w_0)); index==1 →
    response_1 is chosen. Logits are float32 as in the reference so seeded draws coincide."""
    probs = torch.sigmoid(logit_1_minus_0)
    idx = torch.bernoulli(probs, generator=generator).long().reshape(-1).tolist()
    pairs = []
    for row, i in zip(rows, idx):
        pairs.append(
            {
                "prompt": row["prompt"],
                "chosen": row[f"response_{i}"],
                "rejected": row[f"response_{1 - i}"],
                "chosen_safe": bool(row.get(f"is_response_{i}_safe", True)),
                "rejected_safe": bool(row.get(f"is_response_{1 - i}_safe", True)),
            }
        )
    return pairs


def mocan_logits(reward_0, reward_1, cost_0, cost_1, lam: float) -> torch.Tensor:
    """MoCAN (``CAN/.../cdpo/dpo.py:223-235``): w_i = r_i + λ·(−c_i); returns w_1 − w_0."""
    reward_0, reward_1, cost_0, cost_1 = (
        torch.as_tensor(t).float() for t in (reward_0, reward_1, cost_0, cost_1)
    )
    return (reward_1 + lam * -cost_1) - (reward_0 + lam * -cost_0)


def pecan_logits(
    help_0, help_1, safe_0, safe_1, ref_0, ref_1, lam: float
) -> torch.Tensor:
    """PeCAN (``CAN/.../cdpo/dpo_alg2.py:245-253``): raw sequence log-ratios of helpful/safe DPO policies, no β."""
    help_0, help_1, safe_0, safe_1, ref_0, ref_1 = (
        torch.as_tensor(t).float()
        for t in (help_0, help_1, safe_0, safe_1, ref_0, ref_1)
    )
    kl_r_0, kl_g_0 = help_0 - ref_0, safe_0 - ref_0
    kl_r_1, kl_g_1 = help_1 - ref_1, safe_1 - ref_1
    return (kl_r_1 - kl_r_0) + (kl_g_1 - kl_g_0) * lam

