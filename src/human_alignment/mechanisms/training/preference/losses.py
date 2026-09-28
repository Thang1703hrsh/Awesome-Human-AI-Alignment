"""Preference objectives with explicit causal masks and reference gradients.

Source equations and implementation decisions: docs/PREFERENCE_OPTIMIZATION.md.
Tensors use (batch, predicted token) axes unless otherwise specified.
"""

import torch
import torch.nn.functional as F


def token_statistics(logits, reference_logits, labels):
    """Return sampled log ratios, forward/reverse KL, and response mask.

    Labels are unshifted; -100 masks prompts/padding. The reference is frozen.
    """
    if logits.shape != reference_logits.shape or logits.shape[:-1] != labels.shape:
        raise ValueError("Policy/reference logits and labels must have matching shapes")
    mask = labels[:, 1:].ne(-100)
    if not mask.any(-1).all():
        raise ValueError("Every sequence must have at least one predicted response token")
    targets = labels[:, 1:].masked_fill(~mask, 0)
    policy = logits[:, :-1].float().log_softmax(-1)
    reference = reference_logits[:, :-1].detach().float().log_softmax(-1)
    delta = policy - reference
    ratios = delta.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    forward_kl = -(reference.exp() * delta).sum(-1)
    reverse_kl = (policy.exp() * delta).sum(-1)
    return tuple(x.masked_fill(~mask, 0) for x in (ratios, forward_kl, reverse_kl)) + (mask,)


def bregman_loss(log_ratio, generator="sba", lam=0.2, scale=4.0, clip=30.0):
    """BPO ratio risk; log_ratio is rejected minus chosen (scaled by beta).

    Additive constants are omitted. LSIF follows h'(R)R-h(R)-h'(1/R),
    correcting the R instead of 1/R typo in the released TBPO LSIF branch.
    """
    if lam <= 0 or scale <= 0 or clip <= 0:
        raise ValueError("lam, scale, and clip must be positive")
    z = log_ratio.clamp(-clip, clip)
    if generator == "logistic":
        return F.softplus(z)
    if generator == "kliep":
        return z.exp() + z
    if generator == "lsif":
        return (2 * z).exp() - 2 * (-z).exp()
    if generator in {"ba", "sba"}:
        risk = ((1 + lam) * z).exp() - (1 + lam) / lam * (-lam * z).exp()
        return risk if generator == "ba" else risk / (scale * (1 + lam))
    raise ValueError(f"Unknown generator: {generator}")


def tdpo_loss(chosen, rejected, chosen_kl, rejected_kl, beta=0.1, alpha=0.5, tdpo2=True):
    correction = (alpha * (rejected_kl - chosen_kl.detach()) if tdpo2
                  else rejected_kl - chosen_kl)
    return F.softplus(-beta * (chosen - rejected - correction))


def pack_responses(values, mask):
    """Left-align response positions without losing the autograd graph."""
    counts = mask.sum(-1)
    length = int(counts.max())
    packed = torch.nn.utils.rnn.pad_sequence(
        [row[m] for row, m in zip(values, mask)], batch_first=True
    )
    valid = torch.arange(length, device=mask.device)[None, :] < counts[:, None]
    return packed, valid


def rank_weights(scores, mask, low=0.7, high=1.3):
    """Released TIS-DPO rank transform, independently per response."""
    weights = torch.zeros_like(scores)
    for row in range(len(scores)):
        values = scores[row, mask[row]].detach()
        if not torch.isfinite(values).all():
            raise ValueError("TIS-DPO scores must be finite")
        ranks = values.argsort(stable=True).argsort().to(scores.dtype)
        weights[row, mask[row]] = (torch.ones_like(values) if len(values) == 1 else
                                  low + (high - low) * ranks / (len(values) - 1))
    return weights


def importance_weights(scores, mask, mix=0.8, sigma_div=4.0):
    """Response-only Gaussian/gradient mixture normalized to mean weight one."""
    weights = torch.zeros_like(scores)
    for row in range(len(scores)):
        s = scores[row, mask[row]].detach().float().clamp_min(0)
        n = s.numel()
        if not n:
            raise ValueError("Attribution requires response tokens")
        positions = torch.arange(n, device=s.device, dtype=s.dtype)
        prior = torch.exp(-0.5 * ((positions - (n - 1) / 2) / max(1., n / sigma_div)) ** 2)
        prior = prior / prior.sum()
        combined = mix * s / s.sum() + (1 - mix) * prior if s.sum() > 0 else prior
        weights[row, mask[row]] = (n * combined / combined.sum()).to(weights.dtype)
    return weights


def triplet_loss(anchor, positive, negative, anchor_mask, positive_mask, negative_mask, margin=0.1):
    packed = [pack_responses(v, m) for v, m in
              ((anchor, anchor_mask), (positive, positive_mask), (negative, negative_mask))]
    length = max(v.shape[1] for v, _ in packed)
    values = [F.pad(v, (0, length - v.shape[1])) for v, _ in packed]
    masks = [F.pad(m, (0, length - m.shape[1])) for _, m in packed]
    near = ((values[0] - values[1]).square() * (masks[0] & masks[1])).sum(-1)
    far = ((values[0] - values[2]).square() * (masks[0] & masks[2])).sum(-1)
    return F.relu(near - far + margin)


def preference_loss(method, logits, reference_logits, labels, config, *, weights=None, baselines=None):
    """Per-pair losses; batch is all chosen sequences followed by all rejected."""
    ratio, fkl, rkl, mask = token_statistics(logits, reference_logits, labels)
    if len(ratio) % 2:
        raise ValueError("Expected equally sized chosen and rejected batches")
    n = len(ratio) // 2
    if method in {"tis_dpo", "ti_dpo"}:
        if weights is None or weights.shape != labels.shape:
            raise ValueError(f"{method} requires weights aligned to unshifted labels")
        w = weights[:, 1:].detach().masked_fill(~mask, 0)
        if not torch.isfinite(w).all() or (w < 0).any() or not (w.sum(-1) > 0).all():
            raise ValueError("Weights must be finite, nonnegative, with positive response mass")
    else:
        w = mask
    margins = (ratio * w).sum(-1)
    if method in {"tdpo", "ti_dpo"}:
        kl = fkl.sum(-1)
        return tdpo_loss(margins[:n], margins[n:], kl[:n], kl[n:],
                         config.beta, config.alpha, config.tdpo2)
    if method == "tis_dpo":
        kl = (rkl * w).sum(-1)
        margin = margins[:n] - margins[n:]
        if config.tis_token_level:
            margin = margin - config.alpha * (kl[:n] - kl[n:])
        return F.softplus(-config.beta * margin)
    if method == "bpo":
        z = config.beta * (margins[n:] - margins[:n])
        paired_mask = None
    elif method in {"tbpo_q", "tbpo_a"}:
        if method == "tbpo_q":
            if baselines is None or baselines.shape != labels.shape:
                raise ValueError("TBPO-Q requires one baseline per causal state")
            extra = baselines[:, :-1]
        else:
            extra = fkl
        chosen, cmask = pack_responses(ratio[:n], mask[:n])
        rejected, rmask = pack_responses(ratio[n:], mask[n:])
        bc, _ = pack_responses(extra[:n], mask[:n])
        br, _ = pack_responses(extra[n:], mask[n:])
        length = min(chosen.shape[1], rejected.shape[1])
        paired_mask = cmask[:, :length] & rmask[:, :length]
        offset = br[:, :length] - bc[:, :length]
        if method == "tbpo_q":
            offset = offset - (offset * paired_mask).sum(-1, keepdim=True) / paired_mask.sum(-1, keepdim=True)
            offset = offset.clamp(-config.baseline_clip, config.baseline_clip)
        z = config.beta * (rejected[:, :length] - chosen[:, :length] + offset)
        z = z.masked_fill(~paired_mask, 0)
    else:
        raise ValueError(f"Unknown native preference method: {method}")
    args = (config.generator, config.bregman_lambda, config.bregman_scale, config.log_ratio_clip)
    loss = bregman_loss(z, *args)
    if config.label_smoothing:
        loss = (1 - config.label_smoothing) * loss + config.label_smoothing * bregman_loss(-z, *args)
    return loss if paired_mask is None else (loss * paired_mask).sum(-1) / paired_mask.sum(-1)
