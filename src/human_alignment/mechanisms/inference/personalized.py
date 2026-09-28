"""PAD decoding and FPPS activation transformations, with lazy torch imports."""

import math
from dataclasses import dataclass


@dataclass
class PAD:
    beta: float = 1.0
    top_k: int = 10

    def __post_init__(self):
        if not math.isfinite(self.beta) or self.beta < 0 or type(self.top_k) is not int or self.top_k < 1:
            raise ValueError("PAD requires finite beta >= 0 and positive top_k")

    def next_token(self, base_logits, personalized_rewards):
        """Eq. 13: top-k by BASE logits, then argmax(logit + beta * PersRM reward).

        Rewards must already be the preference-conditioned w_p^T phi(s,a),
        not raw feature vectors. Returns token indices for every leading batch.
        """
        import torch

        if base_logits.shape != personalized_rewards.shape or base_logits.ndim < 1:
            raise ValueError("Base logits and scalar personalized rewards must have matching shapes")
        if base_logits.shape[-1] < self.top_k or not torch.isfinite(base_logits).all() or \
                not torch.isfinite(personalized_rewards).all():
            raise ValueError("Finite scores and top_k <= vocabulary size required")
        values, ids = base_logits.topk(self.top_k, dim=-1)
        scores = values + self.beta * personalized_rewards.gather(-1, ids)
        return ids.gather(-1, scores.argmax(-1, keepdim=True)).squeeze(-1)


@dataclass
class FPPS:
    variant: str = "mixed"
    threshold: float = 0.5
    intensity: float = 1.0

    def __post_init__(self):
        if self.variant not in {"hard", "soft", "mixed"} or not 0 < self.threshold < 1:
            raise ValueError("FPPS requires hard/soft/mixed and threshold in (0,1)")
        if not math.isfinite(self.intensity) or self.intensity <= 0:
            raise ValueError("Steering intensity must be finite and positive")

    def steer(self, hidden, personalization_shift, factual_direction, risk):
        """Eq. 5--9; risk comes from a separately trained factuality prober.

        Supports [batch, hidden] states; vectors must match or have [hidden] shape.
        This transform does not install model-specific hooks or train the probe.
        """
        import torch

        if hidden.ndim != 2 or not hidden.numel():
            raise ValueError("hidden must have nonempty [batch, hidden] shape")
        for vector in (personalization_shift, factual_direction):
            if vector.shape not in (hidden.shape, hidden.shape[-1:]):
                raise ValueError("Steering vector shape mismatch")
            if not torch.isfinite(vector).all():
                raise ValueError("Steering vectors must be finite")
        risk = torch.as_tensor(risk, dtype=hidden.dtype, device=hidden.device)
        if risk.shape != hidden.shape[:1] or not torch.isfinite(risk).all() or \
                ((risk < 0) | (risk > 1)).any() or not torch.isfinite(hidden).all():
            raise ValueError("Risk must have shape [batch] with values in [0,1]")
        high = (risk >= self.threshold).unsqueeze(-1)
        hard = torch.where(high, hidden-personalization_shift, hidden)
        soft = hidden + self.intensity * (risk-0.5).unsqueeze(-1) * factual_direction
        if self.variant == "hard":
            return hard
        if self.variant == "soft":
            return soft
        return torch.where(high, hidden-personalization_shift, soft)
