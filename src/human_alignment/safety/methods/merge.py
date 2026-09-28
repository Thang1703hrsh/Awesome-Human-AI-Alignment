"""P-SACPO linear weight merging (``sacpo/config/merge/linear_*.yaml``, mergekit ``merge_method: linear``).

mergekit's linear merge with default ``normalize: true``: θ = Σ_i w_i θ_i / Σ_i w_i. SACPO's P-SACPO configs merge
the stage-1 helpfulness DPO model (weight 1−q) with the stage-2 SACPO model (weight q), q ∈ {0.25, 0.5, 0.75};
the ``naive_*`` configs merge independently trained helpful and safety models instead.
We accumulate in float32 and cast once to the output dtype (mergekit computes in the config dtype, bf16).
"""

from __future__ import annotations

from pathlib import Path

import torch


def merge_state_dicts(
    state_dicts: list[dict[str, torch.Tensor]],
    weights: list[float],
    normalize: bool = True,
    dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    if len(state_dicts) != len(weights) or not state_dicts:
        raise ValueError("need one weight per state dict")
    keys = state_dicts[0].keys()
    for sd in state_dicts[1:]:
        if sd.keys() != keys:
            raise ValueError("state dicts have different keys")
    total = sum(weights)
    merged = {}
    for k in keys:
        acc = sum(w * sd[k].float() for w, sd in zip(weights, state_dicts))
        if normalize:
            acc = acc / total
        merged[k] = acc.to(dtype or state_dicts[0][k].dtype)
    return merged


def merge_linear_checkpoints(
    model_paths: list[str],
    weights: list[float],
    output_dir: str,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    models = [
        AutoModelForCausalLM.from_pretrained(
            p, torch_dtype=dtype, low_cpu_mem_usage=True
        )
        for p in model_paths
    ]
    merged = merge_state_dicts([m.state_dict() for m in models], weights, dtype=dtype)
    base = models[0]
    base.load_state_dict(merged)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    base.save_pretrained(output_dir)
    AutoTokenizer.from_pretrained(model_paths[0]).save_pretrained(output_dir)

