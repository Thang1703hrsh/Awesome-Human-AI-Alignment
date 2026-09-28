"""PPO math for the two RLHF baselines.

Section A: safe-rlhf (SafeRLHF / PPO-Lagrangian), ``safe-rlhf/safe_rlhf/{trainers/rl_trainer.py,
algorithms/ppo_lag/trainer.py}``. Token axis is the full sequence minus one (``lp[:, t]`` predicts token t+1);
losses use ``[:, start:]`` with ``start = prompt_len - 1``; masked_mean averages per sequence, then over batch.

Section B: TRL 0.8.0 ``PPOTrainer`` (MORLHF in RiC ``ppo/morlhf.py``). masked_mean is a global token mean;
advantages are whitened; value/policy clipping with vf_coef; adaptive KL controller.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


# ============================================================ A. safe-rlhf PPO / PPO-Lagrangian
def masked_mean_per_seq(
    x: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    if mask is None:
        return x.mean()
    return ((x * mask).sum(dim=-1) / mask.sum(dim=-1)).mean()


def end_indices(mask: torch.Tensor) -> torch.Tensor:
    return torch.cat([m.nonzero()[-1] for m in mask])


def kl_shaped_reward(
    reward, log_probs, ref_log_probs, sequence_mask, kl_coeff, clip_range_score
):
    """PPO (``ppo/trainer.py:add_kl_divergence_regularization``): -β·KL per token, score added at the end token."""
    end = end_indices(sequence_mask)
    kl_rewards = -kl_coeff * (log_probs - ref_log_probs)
    rewards = torch.scatter_add(
        kl_rewards,
        dim=-1,
        index=end.unsqueeze(-1),
        src=reward.to(kl_rewards.dtype).unsqueeze(-1),
    )
    return torch.clamp(rewards, min=-clip_range_score, max=clip_range_score)


def kl_shaped_reward_and_cost(
    reward, cost, log_probs, ref_log_probs, sequence_mask, kl_coeff, clip_range_score
):
    """PPO-Lag: reward stream gets −β·KL, cost stream gets +β·KL (``ppo_lag/trainer.py:260-285``)."""
    end = end_indices(sequence_mask)
    kl_rewards = -kl_coeff * (log_probs - ref_log_probs)
    rewards = torch.scatter_add(
        kl_rewards,
        dim=-1,
        index=end.unsqueeze(-1),
        src=reward.to(kl_rewards.dtype).unsqueeze(-1),
    )
    costs = torch.scatter_add(
        -kl_rewards,
        dim=-1,
        index=end.unsqueeze(-1),
        src=cost.to(kl_rewards.dtype).unsqueeze(-1),
    )
    return (
        torch.clamp(rewards, min=-clip_range_score, max=clip_range_score),
        torch.clamp(costs, min=-clip_range_score, max=clip_range_score),
    )


def gae_safe_rlhf(values, rewards, sequence_mask, start, gamma=1.0, gae_lambda=0.95):
    """``RLTrainer.get_advantages_and_returns``; returns (advantages[:, start:], returns[:, start:])."""
    last = 0.0
    rev = []
    values = values * sequence_mask
    rewards = rewards * sequence_mask
    length = rewards.size(-1)
    for t in reversed(range(start, length)):
        next_values = values[:, t + 1] if t < length - 1 else 0.0
        delta = rewards[:, t] + gamma * next_values - values[:, t]
        last = delta + gamma * gae_lambda * last
        rev.append(last)
    advantages = torch.stack(rev[::-1], dim=1)
    returns = advantages + values[:, start:]
    return advantages.detach(), returns


def ppo_actor_loss(log_probs, old_log_probs, advantages, mask, clip_range_ratio):
    ratios = torch.exp(log_probs - old_log_probs)
    surrogate = torch.minimum(
        advantages * ratios,
        advantages
        * torch.clamp(ratios, 1.0 - clip_range_ratio, 1.0 + clip_range_ratio),
    )
    return -masked_mean_per_seq(surrogate, mask)


def ppo_lag_actor_loss(
    log_probs,
    old_log_probs,
    reward_advantages,
    cost_advantages,
    mask,
    multiplier,
    clip_range_ratio,
):
    """``PPOLagTrainer.actor_loss_fn``: A = (A_r − λ·A_c)/(1+λ); λ enters as a python float (``.item()``)."""
    advantages = (reward_advantages - multiplier * cost_advantages) / (1.0 + multiplier)
    return ppo_actor_loss(log_probs, old_log_probs, advantages, mask, clip_range_ratio)


def critic_loss(values, old_values, returns, mask, clip_range_value):
    values_clipped = torch.clamp(
        values, old_values - clip_range_value, old_values + clip_range_value
    )
    vf1 = torch.square(values - returns)
    vf2 = torch.square(values_clipped - returns)
    return 0.5 * masked_mean_per_seq(torch.maximum(vf1, vf2), mask)


class LagrangeMultiplier:
    """λ = exp(log_λ), SGD on ``−(J_c − d)·λ``, clamp log_λ ≤ log(λ_max) (``ppo_lag/trainer.py:43-53, 312-326``)."""

    def __init__(
        self,
        lambda_init: float,
        lambda_lr: float,
        lambda_max: float | None,
        threshold: float,
        update_delay_steps: int = 0,
        device=None,
    ):
        self.log_lambda = torch.nn.Parameter(
            torch.tensor(np.log(lambda_init), device=device), requires_grad=True
        )
        self.log_lambda_max = np.log(lambda_max) if lambda_max else None
        self.optimizer = torch.optim.SGD([self.log_lambda], lr=lambda_lr)
        self.threshold = threshold
        self.update_delay_steps = update_delay_steps

    @property
    def value(self) -> float:
        return self.log_lambda.exp().item()

    def update(self, episode_cost: torch.Tensor, global_step: int) -> None:
        if global_step < self.update_delay_steps:
            return
        lambda_loss = -(episode_cost - self.threshold) * self.log_lambda.exp()
        self.optimizer.zero_grad()
        lambda_loss.backward()
        self.optimizer.step()
        if self.log_lambda_max is not None:
            with torch.no_grad():
                self.log_lambda.clamp_(max=self.log_lambda_max)

    def state_dict(self):
        return {
            "log_lambda": self.log_lambda.detach().cpu(),
            "optimizer": self.optimizer.state_dict(),
        }


# ============================================================ B. TRL 0.8.0 PPO (MORLHF)
def trl_masked_mean(values, mask, axis=None):
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    return (values * mask).sum() / mask.sum()


def trl_masked_var(values, mask, unbiased=True):
    mean = trl_masked_mean(values, mask)
    centered = values - mean
    variance = trl_masked_mean(centered**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError("The sum of the mask is zero.")
        if mask_sum == 1:
            raise ValueError(
                "The sum of the mask is one, which can cause a division by zero."
            )
        variance = variance * (mask_sum / (mask_sum - 1))
    return variance


def trl_masked_whiten(values, mask, shift_mean=True):
    mean, var = trl_masked_mean(values, mask), trl_masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def trl_compute_rewards(scores, logprobs, ref_logprobs, masks, kl_coef):
    """``PPOTrainer.compute_rewards`` with ``kl_penalty='kl'``."""
    rewards, non_score, kls = [], [], []
    for score, lp, ref_lp, mask in zip(scores, logprobs, ref_logprobs, masks):
        kl = lp - ref_lp
        kls.append(kl)
        nsr = -kl_coef * kl
        non_score.append(nsr)
        reward = nsr.clone()
        reward[mask.nonzero()[-1]] += score
        rewards.append(reward)
    return torch.stack(rewards), torch.stack(non_score), torch.stack(kls)


def trl_compute_advantages(
    values, rewards, mask, gamma=1.0, lam=0.95, whiten_rewards=False
):
    last = 0
    rev = []
    gen_len = rewards.shape[-1]
    values = values * mask
    rewards = rewards * mask
    if whiten_rewards:
        rewards = trl_masked_whiten(rewards, mask, shift_mean=False)
    for t in reversed(range(gen_len)):
        nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        last = delta + gamma * lam * last
        rev.append(last)
    advantages = torch.stack(rev[::-1]).transpose(0, 1)
    returns = advantages + values
    advantages = trl_masked_whiten(advantages, mask).detach()
    return values, advantages, returns


@dataclass
class TRLPPOLoss:
    pg_loss: torch.Tensor
    vf_loss: torch.Tensor  # already multiplied by vf_coef
    policykl: torch.Tensor
    skipped: bool


def trl_ppo_loss(
    old_logprobs,
    values,
    vpreds,
    logprobs,
    mask,
    advantages,
    returns,
    cliprange=0.2,
    cliprange_value=0.2,
    vf_coef=0.1,
    ratio_threshold=10.0,
) -> TRLPPOLoss:
    vpredclipped = torch.max(
        torch.min(vpreds, values + cliprange_value), values - cliprange_value
    )
    vf_loss = 0.5 * trl_masked_mean(
        torch.max((vpreds - returns) ** 2, (vpredclipped - returns) ** 2), mask
    )
    ratio = torch.exp(logprobs - old_logprobs)
    pg1 = -advantages * ratio
    pg2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    pg_loss = trl_masked_mean(torch.max(pg1, pg2), mask)
    skipped = trl_masked_mean(ratio, mask).item() > ratio_threshold
    if skipped:
        pg_loss = pg_loss * 0.0
        vf_loss = vf_loss * 0.0
    policykl = trl_masked_mean(old_logprobs - logprobs, mask)
    return TRLPPOLoss(pg_loss, vf_coef * vf_loss, policykl.detach(), skipped)


class AdaptiveKLController:
    """Ziegler et al. 2019; TRL 0.8.0 ``trainer/utils.py:41-56``."""

    def __init__(self, init_kl_coef: float, target: float, horizon: float):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current: float, n_steps: int) -> None:
        proportional_error = np.clip(current / self.target - 1, -0.2, 0.2)
        self.value *= 1 + proportional_error * n_steps / self.horizon

