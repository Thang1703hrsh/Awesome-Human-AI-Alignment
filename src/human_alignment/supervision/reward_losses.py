"""Differentiable feedback objectives; torch is imported only on invocation."""

from difflib import SequenceMatcher


def edit_masks(original_tokens, corrected_tokens):
    """Mark changed token positions on BOTH sides, including insertions/deletions.

    Tokenize responses without prompt/special tokens first. This is deterministic
    sequence alignment, not a claim to recover human span annotations.
    """
    original = [False] * len(original_tokens)
    corrected = [False] * len(corrected_tokens)
    for tag, i, j, k, l in SequenceMatcher(
        a=original_tokens, b=corrected_tokens, autojunk=False
    ).get_opcodes():
        if tag != "equal":
            original[i:j] = [True] * (j - i)
            corrected[k:l] = [True] * (l - k)
    return tuple(original), tuple(corrected)


def _aggregate(rewards, mask, reduction):
    import torch

    if rewards.ndim != 2 or mask.shape != rewards.shape:
        raise ValueError("Rewards and mask must have shape [batch, sequence]")
    if not torch.isfinite(rewards).all() or not ((mask == 0) | (mask == 1)).all():
        raise ValueError("Rewards must be finite and mask must be binary")
    lengths = mask.sum(-1)
    if (lengths <= 0).any() or rewards.shape[0] == 0:
        raise ValueError("Empty batches or sequences are unsupported")
    total = (rewards * mask).sum(-1)
    return total / lengths if reduction == "mean" else total


def preference_reward_loss(left_rewards, right_rewards, left_mask, right_mask,
                           preference_probability=None, *, reduction="sum"):
    """Bradley-Terry cross entropy with soft labels (0.5 denotes a tie).

    Sum rewards for Christiano trajectory segments; mean token rewards for
    Xu et al. (2024), Eq. 1--3. Default target prefers the left response.
    """
    import torch
    import torch.nn.functional as F

    if reduction not in ("sum", "mean"):
        raise ValueError("reduction must be sum or mean")
    left = _aggregate(left_rewards, left_mask, reduction)
    right = _aggregate(right_rewards, right_mask, reduction)
    if left.shape != right.shape:
        raise ValueError("Left and right batch sizes differ")
    target = torch.ones_like(left) if preference_probability is None else torch.as_tensor(
        preference_probability, dtype=left.dtype, device=left.device)
    if target.shape != left.shape or not torch.isfinite(target).all() or (
        (target < 0) | (target > 1)
    ).any():
        raise ValueError("Preference probabilities must have shape [batch] and lie in [0, 1]")
    return F.binary_cross_entropy_with_logits(left - right, target)


def fine_grained_reward_loss(chosen_rewards, rejected_rewards, chosen_mask, rejected_mask):
    """Xu et al. exact sequence-mean objective, without the equal-length approximation."""
    return preference_reward_loss(chosen_rewards, rejected_rewards, chosen_mask,
                                  rejected_mask, reduction="mean")
