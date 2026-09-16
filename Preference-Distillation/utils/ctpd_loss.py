from typing import Sequence

import torch
import torch.nn.functional as F


def _parent_logps(token_logps: torch.Tensor, parent_lists: Sequence[Sequence[int]]) -> torch.Tensor:
    """Aggregate student token log-probabilities into teacher parent-token units."""
    values = []
    for parents in parent_lists:
        if not parents:
            values.append(token_logps.new_zeros(()))
            continue
        indices = torch.as_tensor(parents, device=token_logps.device, dtype=torch.long) - 1
        indices = indices[(indices >= 0) & (indices < token_logps.shape[0])]
        values.append(token_logps[indices].sum() if indices.numel() else token_logps.new_zeros(()))
    return torch.stack(values) if values else token_logps.new_empty((0,))


def ctpd_loss(
    policy_chosen_token_logps: torch.Tensor,
    policy_rejected_token_logps: torch.Tensor,
    reference_chosen_token_logps: torch.Tensor,
    reference_rejected_token_logps: torch.Tensor,
    chosen_parent_lists,
    rejected_parent_lists,
    chosen_weights: torch.Tensor,
    rejected_weights: torch.Tensor,
    beta: float = 0.5,
):
    """CTPD preference loss: weighted parent-token KD-TIS-DPO."""
    chosen_margin = []
    rejected_margin = []
    for i in range(policy_chosen_token_logps.shape[0]):
        chosen_parent_policy = _parent_logps(policy_chosen_token_logps[i], chosen_parent_lists[i])
        chosen_parent_reference = _parent_logps(reference_chosen_token_logps[i], chosen_parent_lists[i])
        rejected_parent_policy = _parent_logps(policy_rejected_token_logps[i], rejected_parent_lists[i])
        rejected_parent_reference = _parent_logps(reference_rejected_token_logps[i], rejected_parent_lists[i])
        chosen_margin.append(((chosen_parent_policy - chosen_parent_reference) * chosen_weights[i, :chosen_parent_policy.numel()]).sum())
        rejected_margin.append(((rejected_parent_policy - rejected_parent_reference) * rejected_weights[i, :rejected_parent_policy.numel()]).sum())

    chosen_margin = torch.stack(chosen_margin)
    rejected_margin = torch.stack(rejected_margin)
    losses = -F.logsigmoid(beta * (chosen_margin - rejected_margin))
    chosen_rewards = beta * chosen_margin.detach()
    rejected_rewards = beta * rejected_margin.detach()
    return losses, chosen_rewards, rejected_rewards
