"""Experiment configuration (YAML) → typed sections."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from human_alignment.safety.data.tokenize import SAFE_RLHF_PROMPT


class _Loader(yaml.SafeLoader):
    """SafeLoader that also reads '1e-6' / '5E+4' as floats (YAML 1.1 requires a dot)."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?\.(?:inf|Inf|INF)
    |\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


def _yaml_load(text: str):
    return yaml.load(text, Loader=_Loader)


@dataclass
class LoRASpec:
    r: int = 16
    alpha: float = 32
    dropout: float = 0.0
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    modules_to_save: list[str] | None = None


@dataclass
class ModelSpec:
    policy: str = "PKU-Alignment/alpaca-7b-reproduced"
    ref: str | None = (
        None  # full FT: defaults to `policy`; LoRA: adapter-disabled policy
    )
    dtype: str = "bf16"
    attn_implementation: str = "sdpa"
    lora: LoRASpec | None = None
    trainable_substrings: list[str] | None = (
        None  # BFPO "selective": only params whose name contains these
    )


@dataclass
class DataSpec:
    source: str = (
        "pku"  # pku | alpaca | cpo_ultrasafety | cpsft_jsonl | cpsft_hf | pku_prompts
    )
    dataset: str = "PKU-Alignment/PKU-SafeRLHF-30K"
    revision: str | None = None
    train_split: str = "train"
    eval_split: str | None = None
    fraction: float | None = None
    shuffle_seed: int | None = None
    holdout: float | None = None
    holdout_seed: int | None = None
    limit: int | None = None  # debug / smoke tests
    eval_limit: int | None = None
    selection: str = "better"
    tokenization: str = "safe_rlhf"  # safe_rlhf | trl
    template: str = SAFE_RLHF_PROMPT
    max_length: int = 512
    max_prompt_length: int = 128
    paths: list[str] = field(default_factory=list)


@dataclass
class ExperimentConfig:
    method: str
    model: ModelSpec = field(default_factory=ModelSpec)
    data: DataSpec = field(default_factory=DataSpec)
    train: dict[str, Any] = field(default_factory=dict)
    method_args: dict[str, Any] = field(default_factory=dict)
    source: str | None = None


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _set_dotted(d: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def load_config(
    path: str | Path, overrides: list[str] | None = None
) -> ExperimentConfig:
    """YAML with optional ``base: other.yaml`` inheritance and ``key.sub=value`` CLI overrides."""
    path = Path(path)
    raw = _yaml_load(path.read_text()) or {}
    if "base" in raw:
        base_path = (path.parent / raw.pop("base")).resolve()
        base = load_config(base_path)
        raw = _merge(_config_to_dict(base), raw)
    for ov in overrides or []:
        key, _, value = ov.partition("=")
        _set_dotted(raw, key.strip(), _yaml_load(value))
    return config_from_dict(raw, source=str(path))


def _config_to_dict(cfg: ExperimentConfig) -> dict:
    from dataclasses import asdict

    d = asdict(cfg)
    d.pop("source", None)
    return d


def config_from_dict(raw: dict, source: str | None = None) -> ExperimentConfig:
    raw = copy.deepcopy(raw)
    if "method" not in raw:
        raise ValueError("config must define `method`")
    model = dict(raw.get("model") or {})
    lora = model.pop("lora", None)
    unknown = set(model) - set(ModelSpec.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown model keys: {sorted(unknown)}")
    data = dict(raw.get("data") or {})
    unknown = set(data) - set(DataSpec.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown data keys: {sorted(unknown)}")
    extra = set(raw) - {"method", "model", "data", "train", "method_args"}
    if extra:
        raise ValueError(f"unknown top-level keys: {sorted(extra)}")
    return ExperimentConfig(
        method=raw["method"],
        model=ModelSpec(**model, lora=LoRASpec(**lora) if lora else None),
        data=DataSpec(**data),
        train=dict(raw.get("train") or {}),
        method_args=dict(raw.get("method_args") or {}),
        source=source,
    )


def dump_config(cfg: ExperimentConfig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(_config_to_dict(cfg), sort_keys=False))

