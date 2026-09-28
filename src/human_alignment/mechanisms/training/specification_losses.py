"""Differentiable reference objectives; no implicit model downloads or trainers.

Torch is imported only when an objective is called. See the specification guide
for missing paper-specific models, replay management and sampling requirements.
"""
import math


def _torch(*values):
    import torch
    for value in values:
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise TypeError("Expected floating-point tensors")
        if not value.numel() or not torch.isfinite(value).all():
            raise ValueError("Expected nonempty finite tensors")
    return torch


class VPLObjective:
    """Negative ELBO for sampled latent-conditioned Bradley-Terry rewards."""

    def loss(self, reward_margins, posterior_mean, posterior_logvar, *, kl_weight=1.):
        torch = _torch(reward_margins, posterior_mean, posterior_logvar)
        if (reward_margins.ndim != 2 or posterior_mean.ndim != 2
                or posterior_mean.shape != posterior_logvar.shape
                or reward_margins.shape[1] != posterior_mean.shape[0]):
            raise ValueError("Use margins [samples,batch] and posterior [batch,latent]")
        if not math.isfinite(kl_weight) or kl_weight < 0:
            raise ValueError("kl_weight must be finite and nonnegative")
        kl = .5 * (posterior_mean.square() + posterior_logvar.exp()
                   - 1 - posterior_logvar).sum(-1)
        return torch.nn.functional.softplus(-reward_margins).mean() + kl_weight * kl.mean()


class DistributionalPreferenceLearning:
    """Hidden-context (aleatoric) Gaussian and categorical preference likelihoods."""

    def gaussian_loss(self, chosen_mean, rejected_mean, chosen_variance, rejected_variance):
        torch = _torch(chosen_mean, rejected_mean, chosen_variance, rejected_variance)
        if any(v.shape != chosen_mean.shape for v in
               (rejected_mean, chosen_variance, rejected_variance)):
            raise ValueError("All shapes must match")
        if (chosen_variance <= 0).any() or (rejected_variance <= 0).any():
            raise ValueError("Variances must be positive (not standard deviations)")
        z = (chosen_mean-rejected_mean) / (chosen_variance+rejected_variance).sqrt()
        return -torch.special.log_ndtr(z).mean()

    def categorical_loss(self, chosen_logits, rejected_logits):
        torch = _torch(chosen_logits, rejected_logits)
        if (chosen_logits.shape != rejected_logits.shape or chosen_logits.ndim < 2
                or chosen_logits.shape[-1] < 2):
            raise ValueError("Matching [batch,...,bins] logits with at least two bins required")
        n = chosen_logits.shape[-1]
        index = torch.arange(n, device=chosen_logits.device)
        weight = (index[:, None] > index[None, :]).to(chosen_logits.dtype)
        weight = weight + .5 * torch.eye(n, device=weight.device, dtype=weight.dtype)
        joint = (chosen_logits.log_softmax(-1).unsqueeze(-1)
                 + rejected_logits.log_softmax(-1).unsqueeze(-2) + weight.log())
        return -torch.logsumexp(joint.flatten(-2), -1).mean()


class COPRObjective:
    """COPR set-distribution fitting and normalized replay Lagrangian kernels."""

    def fit_loss(self, policy_logps, target_logps):
        _torch(policy_logps, target_logps)
        if policy_logps.shape != target_logps.shape or policy_logps.ndim != 2:
            raise ValueError("Matching [batch,ranked_responses] tensors required")
        if policy_logps.shape[-1] < 2:
            raise ValueError("At least two ranked responses required")
        return (policy_logps.log_softmax(-1) - target_logps.detach().log_softmax(-1)).square().mean()

    def loss(self, current_loss, replay_errors, thresholds, log_multipliers):
        _torch(current_loss, replay_errors, thresholds, log_multipliers)
        if (current_loss.numel() != 1 or replay_errors.ndim != 1
                or replay_errors.shape != thresholds.shape
                or replay_errors.shape != log_multipliers.shape):
            raise ValueError("Scalar current loss and matching replay vectors required")
        if (thresholds < 0).any():
            raise ValueError("Replay thresholds cannot be negative")
        weights = log_multipliers.detach().exp()
        return (current_loss + (weights*(replay_errors-thresholds.detach())).sum()) / (1+weights.sum())

    def dual_update(self, log_multipliers, violations, *, learning_rate=.01):
        _torch(log_multipliers, violations)
        if log_multipliers.shape != violations.shape or not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Matching tensors and positive learning rate required")
        return (log_multipliers + learning_rate*log_multipliers.exp()*violations).detach()


class KLDPOObjective:
    """Fixed-temperature, uniform-minibatch KL-DRO envelope gradient."""

    def __init__(self, temperature=1.):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self.temperature = temperature

    def loss(self, per_example_dpo_losses):
        _torch(per_example_dpo_losses)
        if per_example_dpo_losses.ndim != 1:
            raise ValueError("Supply unreduced DPO loss per example")
        weights = (per_example_dpo_losses.detach()/self.temperature).softmax(0)
        return (weights*per_example_dpo_losses).sum()


class WDPOObjective:
    """First-order Wasserstein penalty on continuous INPUT embedding gradients.

    Inputs must have batch as dimension zero and no cross-example coupling.
    Pass BOTH chosen/rejected embeddings if losses depend on both branches.
    """

    def __init__(self, radius=.1):
        if not math.isfinite(radius) or radius < 0:
            raise ValueError("radius must be finite and nonnegative")
        self.radius = radius

    def loss(self, per_example_dpo_losses, input_embeddings):
        embeddings = tuple(input_embeddings)
        torch = _torch(per_example_dpo_losses, *embeddings)
        if per_example_dpo_losses.ndim != 1 or not embeddings:
            raise ValueError("Supply per-example losses and input embedding tensors")
        if any(e.ndim < 2 or e.shape[0] != len(per_example_dpo_losses)
               or not e.requires_grad for e in embeddings):
            raise ValueError("Embeddings require gradients and matching batch dimension")
        if self.radius == 0:
            return per_example_dpo_losses.mean()
        grads = torch.autograd.grad(per_example_dpo_losses.sum(), embeddings, create_graph=True)
        # vector_norm defines a finite zero subgradient at a zero gradient vector.
        flat = torch.cat([g.reshape(-1) for g in grads])
        penalty = torch.linalg.vector_norm(flat) / math.sqrt(len(per_example_dpo_losses))
        return per_example_dpo_losses.mean() + self.radius*penalty
