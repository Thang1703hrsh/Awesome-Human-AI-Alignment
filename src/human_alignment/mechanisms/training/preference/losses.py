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


def dp3o_loss(chosen, rejected, reward_diff=None, beta=0.1, alpha=0.2):
    """DP3O (Zhang et al., 2026): DPO with distilled preference probabilities.

    ``chosen``/``rejected`` are summed policy-minus-reference log ratios. The soft
    target is ``sigmoid(reward_diff)`` from a learned preference (reward) model, as
    in the released ``optimistic_dpo_loss``; ``alpha`` weights the hard-label DPO term.
    """
    logits = beta * (chosen - rejected)
    hard = -F.logsigmoid(logits)
    if alpha == 1.0:
        return hard
    if reward_diff is None or reward_diff.shape != chosen.shape:
        raise ValueError("DP3O with dp3o_alpha < 1 requires one reward difference per pair")
    target = torch.sigmoid(reward_diff.to(logits.dtype))
    soft = -F.logsigmoid(logits) * target - F.logsigmoid(-logits) * (1 - target)
    return alpha * hard + (1 - alpha) * soft


def sigmoid_dpo_loss(chosen, rejected, beta=0.1, label_smoothing=0.0):
    """TRL's ``sigmoid`` DPO loss on summed policy-minus-reference log ratios."""
    logits = beta * (chosen - rejected)
    return -(1 - label_smoothing) * F.logsigmoid(logits) - label_smoothing * F.logsigmoid(-logits)


def beta_dpo_loss(chosen, rejected, state, beta=0.1, a=0.6, filter_ratio=0.2, momentum=0.9):
    """beta-DPO (Wu et al., 2024): batch-calibrated beta with beta-guided filtering.

    Follows ``preference_loss``/``update_and_sync_tensor_mean`` in the released
    trainer: the running reward-gap mean and std (``state``, initialized to zero)
    are updated with this batch first; pairs are then sampled without replacement
    with Gaussian weights around the running mean, keeping ``1 - filter_ratio``.
    The returned per-pair losses are rescaled so their mean is the masked mean.
    """
    gap = (chosen - rejected).detach()
    if gap.numel() < 2:
        raise ValueError("beta-DPO needs at least two pairs per micro-batch for its gap statistics")
    if "gap_mean" not in state:
        state["gap_mean"] = gap.new_zeros(())
        state["gap_std"] = gap.new_zeros(())
    state["gap_mean"] = momentum * state["gap_mean"] + (1 - momentum) * gap.mean()
    state["gap_std"] = momentum * state["gap_std"] + (1 - momentum) * gap.std()
    mean, std = state["gap_mean"], state["gap_std"]
    # With policy == reference (the first step) every gap and the std are zero; the
    # released 0/0 weights are NaN, so use their limit exp(0) = 1 (uniform sampling).
    z = (gap - mean) / std if float(std) > 0 else torch.zeros_like(gap)
    weights = torch.exp(-0.5 * z.pow(2))
    if not torch.isfinite(weights).all() or float(weights.sum()) <= 0:
        # Every weight underflowed (gaps far outside the running std); sample uniformly.
        weights = torch.ones_like(gap)
    kept = int(gap.numel() * (1 - filter_ratio))
    if kept < 1:
        raise ValueError("beta_dpo_filter_ratio leaves no pair in this micro-batch")
    index = torch.multinomial(weights, kept, replacement=False)
    mask = torch.zeros_like(gap)
    mask[index] = 1
    beta_used = (beta * (1 + a * (gap[index].mean() - mean))).clamp(min=1e-3).detach()
    state["beta_used"] = float(beta_used)
    losses = -F.logsigmoid(beta_used * (chosen - rejected))
    return losses * mask * (gap.numel() / mask.sum())


def _sequence_logps(logits, targets, mask):
    return (logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1) * mask).sum(-1)


def epsilon_dpo_steps(logits, reference_logits, labels, epsilon):
    """Per-pair KL-step direction of epsilon-DPO (Lee et al., 2025): +1, -1, or 0.

    Perturbing policy logits toward ``(1 + eps) * policy - eps * ref`` and
    ``(1 - eps) * policy + eps * ref`` and checking how the chosen-minus-rejected
    log ratio moves, as in the released ``concatenated_forward``.
    """
    mask = labels[:, 1:].ne(-100)
    targets = labels[:, 1:].masked_fill(~mask, 0)
    policy = logits[:, :-1].detach().float()
    reference = reference_logits[:, :-1].detach().float()
    n = len(logits) // 2
    values = []
    for candidate in (policy, (1 + epsilon) * policy - epsilon * reference,
                      (1 - epsilon) * policy + epsilon * reference):
        logps = _sequence_logps(candidate, targets, mask)
        values.append(logps[:n] - logps[n:])
    base, positive, negative = values
    up = (positive > base) & (base > negative)
    down = (negative > base) & (base > positive)
    return up.long() - down.long()


def _linear_quantile(values, q):
    """``torch.quantile(values, q, dim=-1, keepdim=True)`` without its input-size limit."""
    count = values.shape[-1]
    position = q * (count - 1)
    low, high = int(position // 1), min(int(position // 1) + 1, count - 1)
    low_value = values.kthvalue(low + 1, dim=-1, keepdim=True).values
    high_value = values.kthvalue(high + 1, dim=-1, keepdim=True).values
    return low_value + (high_value - low_value) * (position - low)


def ra_dpo_risk_ratio(logits, reference_logits, labels, confidence=0.95, split_vocabulary=True):
    """Ra-DPO (Zhang et al., 2025) sequential risk ratio: summed per-token CVaR.

    Per position, ``d = log p_ref - log p_policy`` over the vocabulary; entries
    above the ``1 - confidence`` quantile are weighted by ``p_ref`` and summed
    (released ``calculate_cvar``). With ``split_vocabulary`` (the released
    ``is_split_risk_ratio: true``) the quantile is taken separately over the two
    halves of the vocabulary axis, exactly as the reference slices ``dim=2``.
    """
    mask = labels[:, 1:].ne(-100)
    policy = logits[:, :-1].float().log_softmax(-1)
    reference = reference_logits[:, :-1].detach().float().log_softmax(-1)
    probabilities = reference.exp()
    difference = reference - policy
    middle = difference.shape[-1] // 2
    parts = ((slice(0, middle), slice(middle, None)) if split_vocabulary else (slice(None),))
    risk = 0
    for part in parts:
        values = difference[..., part]
        threshold = _linear_quantile(values.detach(), 1 - confidence)
        risk = risk + (probabilities[..., part] * values * values.gt(threshold)).sum(-1)
    return (risk * mask).sum(-1)


def adpo_segments(ratio, mask, granularity="adaptive", segments=256, static_k=32, reduce="sum"):
    """ADPO (Oi et al., 2026) segment log ratios, as the released ``_to_segments``.

    ``adaptive`` splits each response's valid tokens into ``segments`` contiguous
    parts (the first ``N % m`` one token longer); ``static`` cuts the padded row
    into chunks of ``static_k`` positions. ``reduce`` sums or averages per segment.
    """
    if granularity == "static":
        width = ratio.shape[1]
        count = -(-width // static_k)
        pad = count * static_k - width
        values = F.pad(ratio * mask, (0, pad)).view(len(ratio), count, static_k)
        valid = F.pad(mask, (0, pad)).view(len(ratio), count, static_k)
        sums = values.sum(-1)
        if reduce == "sum":
            return sums
        counts = valid.sum(-1)
        return torch.where(counts > 0, sums / counts.clamp_min(1), torch.zeros_like(sums))
    if granularity != "adaptive":
        raise ValueError(f"Unknown ADPO granularity: {granularity}")
    rows = []
    for row, row_mask in zip(ratio, mask):
        values = row[row_mask]
        total = values.numel()
        sizes = torch.full((segments,), total // segments, dtype=torch.long, device=row.device)
        sizes[: total % segments] += 1
        ids = torch.repeat_interleave(torch.arange(segments, device=row.device), sizes)
        sums = values.new_zeros(segments).index_add(0, ids, values)
        if reduce == "mean":
            sums = torch.where(sizes > 0, sums / sizes.clamp_min(1), torch.zeros_like(sums))
        rows.append(sums)
    return torch.stack(rows)


def preference_loss(method, logits, reference_logits, labels, config, *, weights=None, baselines=None,
                    reward_diff=None, state=None):
    """Per-pair losses; batch is all chosen sequences followed by all rejected."""
    ratio, fkl, rkl, mask = token_statistics(logits, reference_logits, labels)
    if len(ratio) % 2:
        raise ValueError("Expected equally sized chosen and rejected batches")
    n = len(ratio) // 2
    state = {} if state is None else state
    if method == "dp3o":
        margins = ratio.sum(-1)
        return dp3o_loss(margins[:n], margins[n:], reward_diff, config.beta, config.dp3o_alpha)
    if method == "logits_sam":
        margins = ratio.sum(-1)
        return sigmoid_dpo_loss(margins[:n], margins[n:], config.beta, config.label_smoothing)
    if method == "beta_dpo":
        margins = ratio.sum(-1)
        return beta_dpo_loss(margins[:n], margins[n:], state, config.beta, config.beta_dpo_a,
                             config.beta_dpo_filter_ratio, config.beta_dpo_momentum)
    if method == "epsilon_dpo":
        margins = ratio.sum(-1)
        steps = epsilon_dpo_steps(logits, reference_logits, labels, config.epsilon)
        state["last_steps"] = steps
        beta = state.get("beta", config.beta) / (1 + config.epsilon * steps)
        return -F.logsigmoid(beta * (margins[:n] - margins[n:]))
    if method == "ra_dpo":
        margins = ratio.sum(-1)
        risk = ra_dpo_risk_ratio(logits, reference_logits, labels, config.ra_dpo_confidence,
                                 config.ra_dpo_split_vocabulary)
        chosen_risk, rejected_risk = risk[:n], risk[n:]
        if config.ra_dpo_variant == 2:
            penalty = config.ra_dpo_alpha * (rejected_risk - chosen_risk.detach())
        else:
            penalty = rejected_risk - chosen_risk
        return -F.logsigmoid(config.beta * (margins[:n] - margins[n:] - penalty))
    if method == "adpo":
        if config.adpo_granularity == "pass":
            # Per-token comparison over the padded row. The released per-token array
            # has one more column than the predicted positions (its rolled column 0 is
            # always zero), so that zero-margin term is kept in the mean.
            segments = F.pad(ratio * mask, (1, 0))
        else:
            segments = adpo_segments(ratio, mask, config.adpo_granularity, config.adpo_segments,
                                     config.adpo_static_k, config.adpo_segment_reduce)
        # The released loss averages the Bradley-Terry term over every segment pair.
        return sigmoid_dpo_loss(segments[:n], segments[n:], config.beta, config.label_smoothing).mean(-1)
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
