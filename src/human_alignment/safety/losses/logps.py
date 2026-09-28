"""Sequence log-probability utilities.

Two conventions exist in the reference code:

* ``masked`` (TRL / MODPO / BFPO / SACPO / CAN / CPO): labels = input_ids with prompt and padding set to
  ``ignore_index``; the sequence log-prob is ``sum_t log p(label_{t+1} | x_{<=t})`` over non-ignored labels.
* ``safe_rlhf`` (safe-rlhf DPO, MidPO): per-token log-probs ``lp[t] = log p(x_{t+1} | x_{<=t})`` are summed over
  ``lp[diverge : end + 1]`` where ``diverge`` is the first index at which the better/worse ``input_ids`` differ and
  ``end`` the last attended index. This skips the first diverging token and includes one position past EOS
  (``safe_rlhf/algorithms/dpo/trainer.py:151-166``); reproduced as-is.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def gather_log_probabilities(
    logits: torch.Tensor, labels: torch.LongTensor
) -> torch.Tensor:
    # bf16 log_softmax over a 32k vocab loses ~2 significant digits; upcast half precision only.
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    log_probs = F.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def token_log_probs(logits: torch.Tensor, input_ids: torch.LongTensor) -> torch.Tensor:
    """``lp[:, t] = log p(input_ids[:, t+1])``; shape (B, L-1)."""
    return gather_log_probabilities(logits[:, :-1], input_ids[:, 1:])


def masked_sequence_logps(
    logits: torch.Tensor,
    labels: torch.LongTensor,
    average: bool = False,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    if logits.shape[:-1] != labels.shape:
        raise ValueError(
            f"logits {tuple(logits.shape)} and labels {tuple(labels.shape)} mismatch"
        )
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1]
    mask = labels != ignore_index
    labels[~mask] = 0
    per_token = gather_log_probabilities(logits, labels)
    total = (per_token * mask).sum(-1)
    if average:
        return total / mask.sum(-1)
    return total


def safe_rlhf_pair_spans(
    better_input_ids: torch.LongTensor,
    better_attention_mask: torch.Tensor,
    worse_input_ids: torch.LongTensor,
    worse_attention_mask: torch.Tensor,
) -> tuple[list[slice], list[slice]]:
    better_slices, worse_slices = [], []
    for i in range(better_input_ids.size(0)):
        if torch.equal(better_input_ids[i], worse_input_ids[i]):
            raise ValueError("The better and worse answers are the same!")
        better_end = better_attention_mask[i].nonzero()[-1].item()
        worse_end = worse_attention_mask[i].nonzero()[-1].item()
        diverge = (better_input_ids[i] != worse_input_ids[i]).nonzero()[0].item()
        if not (0 <= diverge <= better_end and diverge <= worse_end):
            raise ValueError("diverge index is out of range!")
        better_slices.append(slice(diverge, better_end + 1))
        worse_slices.append(slice(diverge, worse_end + 1))
    return better_slices, worse_slices


def sum_over_spans(token_lp: torch.Tensor, spans: list[slice]) -> torch.Tensor:
    return torch.stack([token_lp[i, s].sum(-1) for i, s in enumerate(spans)])

