"""Reproducibility manifest and Hugging Face resource downloader.

The manifest deliberately distinguishes *inputs* from checkpoints produced by an
earlier pipeline stage.  For example, MODPO's margin adapter and SACPO's stage-1
model are outputs, so downloading them would silently change the experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ResourceKind = Literal["model", "dataset"]


@dataclass(frozen=True)
class Resource:
    repo_id: str
    kind: ResourceKind
    purpose: str
    revision: str | None = None
    gated: bool = False

    @property
    def local_name(self) -> str:
        return self.repo_id.replace("/", "--")


BASE_MODEL = Resource(
    "PKU-Alignment/alpaca-7b-reproduced", "model", "shared SFT policy"
)
REWARD_MODEL = Resource(
    "PKU-Alignment/beaver-7b-v1.0-reward", "model", "helpfulness reward model"
)
COST_MODEL = Resource("PKU-Alignment/beaver-7b-v1.0-cost", "model", "safety cost model")
UNIFIED_REWARD = Resource(
    "PKU-Alignment/beaver-7b-unified-reward", "model", "MidPO helpfulness scorer"
)
UNIFIED_COST = Resource(
    "PKU-Alignment/beaver-7b-unified-cost", "model", "MidPO safety scorer"
)
PKU = Resource(
    "PKU-Alignment/PKU-SafeRLHF", "dataset", "SafeRLHF train/evaluation preferences"
)
PKU30K = Resource(
    "PKU-Alignment/PKU-SafeRLHF-30K", "dataset", "30K train/test preference benchmark"
)
PKU10K = Resource(
    "PKU-Alignment/PKU-SafeRLHF-10K", "dataset", "MODPO margin preferences"
)


PROFILES: dict[str, tuple[Resource, ...]] = {
    "sft": (
        Resource("huggyllama/llama-7b", "model", "pretrained Llama base", gated=True),
        Resource("tatsu-lab/alpaca", "dataset", "instruction SFT data"),
    ),
    "dpo": (BASE_MODEL, PKU30K),
    "reward_cost": (BASE_MODEL, PKU),
    "saferlhf": (
        BASE_MODEL,
        REWARD_MODEL,
        COST_MODEL,
        PKU,
        Resource("tatsu-lab/alpaca", "dataset", "PTX data"),
    ),
    "ppo": (
        BASE_MODEL,
        REWARD_MODEL,
        PKU,
        Resource("tatsu-lab/alpaca", "dataset", "PTX data"),
    ),
    "morlhf": (BASE_MODEL, REWARD_MODEL, COST_MODEL, PKU),
    "sacpo": (BASE_MODEL, PKU30K),
    "can": (BASE_MODEL, REWARD_MODEL, COST_MODEL, PKU30K),
    "modpo": (BASE_MODEL, PKU10K),
    "cpo": (
        BASE_MODEL,
        Resource("openbmb/UltraSafety", "dataset", "CPO safety responses and ratings"),
        Resource("openbmb/UltraFeedback", "dataset", "CPSFT controllable SFT data"),
    ),
    "bfpo": (
        BASE_MODEL,
        PKU,
        Resource(
            "HuggingFaceH4/ultrafeedback_binarized",
            "dataset",
            "helpfulness replay buffer",
        ),
    ),
    "midpo": (BASE_MODEL, UNIFIED_REWARD, UNIFIED_COST, PKU),
    "safedpo": (BASE_MODEL, PKU30K),
    "bso": (BASE_MODEL, PKU30K),
    "evaluation": (
        REWARD_MODEL,
        COST_MODEL,
        PKU30K,
        Resource(
            "PKU-Alignment/BeaverTails-Evaluation",
            "dataset",
            "SafeRLHF held-out safety prompts",
        ),
        Resource("LibrAI/do-not-answer", "dataset", "MidPO safety benchmark"),
        Resource(
            "allenai/wildguardmix",
            "dataset",
            "MidPO WildGuardTest benchmark",
            gated=True,
        ),
        Resource(
            "walledai/XSTest",
            "dataset",
            "original 250 benign + 200 unsafe over-refusal prompts",
            gated=True,
        ),
    ),
}

ALIASES = {
    "dpo_helpful": "dpo",
    "dpo_harmless": "dpo",
    "dpo_safebetter": "dpo",
    "reward_model": "reward_cost",
    "cost_model": "reward_cost",
    "p_sacpo": "sacpo",
    "sacpo_dpo": "sacpo",
    "sacpo_kto": "sacpo",
    "can_dual": "can",
    "mocan": "can",
    "pecan": "can",
    "modpo_margin": "modpo",
    "cdpo": "cpo",
    "cpsft": "cpo",
    "midpo_safety_expert": "midpo",
    "midpo_helpfulness_expert": "midpo",
    "midpo_router": "midpo",
}


def profile_resources(profile: str, include_evaluation: bool = False) -> list[Resource]:
    """Resolve a profile and remove duplicate repositories while preserving order."""
    profile = ALIASES.get(profile, profile)
    if profile == "all":
        names = [n for n in PROFILES if n != "evaluation"]
    elif profile in PROFILES:
        names = [profile]
    else:
        raise KeyError(
            f"unknown resource profile {profile!r}; known: {sorted(PROFILES)}"
        )
    if include_evaluation and "evaluation" not in names:
        names.append("evaluation")
    out: list[Resource] = []
    seen: set[tuple[str, str, str | None]] = set()
    for name in names:
        for resource in PROFILES[name]:
            key = (resource.kind, resource.repo_id, resource.revision)
            if key not in seen:
                seen.add(key)
                out.append(resource)
    return out


def download_resources(
    profile: str,
    root: str | Path = "downloads",
    include_evaluation: bool = False,
    token: str | None = None,
    dry_run: bool = False,
) -> list[Path]:
    """Download an immutable local snapshot for every external input in ``profile``.

    Hugging Face's cache is still used, while ``local_dir`` gives scripts stable,
    inspectable paths. Existing snapshots resume safely.
    """
    from huggingface_hub import snapshot_download

    root = Path(root)
    paths = []
    for resource in profile_resources(profile, include_evaluation):
        path = root / f"{resource.kind}s" / resource.local_name
        paths.append(path)
        if dry_run:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=resource.repo_id,
            repo_type=resource.kind,
            revision=resource.revision,
            local_dir=path,
            token=token,
        )
    return paths

