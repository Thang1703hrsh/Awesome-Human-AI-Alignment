"""Public, lazy entry points for the integrated safety-alignment workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence


def safety_methods() -> tuple[str, ...]:
    """Return every executable safety-alignment stage."""

    from human_alignment.safety.registry import METHODS

    return tuple(METHODS)


def load_safety_config(
    config: str | Path | Mapping[str, Any] | Any,
    overrides: Sequence[str] | None = None,
):
    """Load a safety recipe from YAML, a mapping, or an existing typed config."""

    from human_alignment.safety.schema import ExperimentConfig, config_from_dict, load_config

    if isinstance(config, ExperimentConfig):
        if overrides:
            raise ValueError("overrides require a YAML path or mapping")
        return config
    if isinstance(config, Mapping):
        if overrides:
            raise ValueError("overrides require a YAML path")
        return config_from_dict(dict(config))
    return load_config(config, list(overrides or ()))


def run_safety_config(
    config: str | Path | Mapping[str, Any] | Any,
    overrides: Sequence[str] | None = None,
) -> Path:
    """Resolve and execute one integrated safety-alignment recipe."""

    from human_alignment.safety.registry import get_runner

    resolved = load_safety_config(config, overrides)
    return Path(get_runner(resolved.method)(resolved))


def run_safety_method(
    method: str,
    *,
    model: str | None = None,
    dataset: str | None = None,
    output_dir: str | None = None,
    model_options: Mapping[str, Any] | None = None,
    data_options: Mapping[str, Any] | None = None,
    train_options: Mapping[str, Any] | None = None,
    method_args: Mapping[str, Any] | None = None,
) -> Path:
    """Execute a safety stage without first writing a YAML recipe."""

    from human_alignment.safety.registry import METHODS

    if method not in METHODS:
        raise ValueError(f"unknown safety method {method!r}; known: {sorted(METHODS)}")
    raw: dict[str, Any] = {
        "method": method,
        "model": dict(model_options or {}),
        "data": dict(data_options or {}),
        "train": dict(train_options or {}),
        "method_args": dict(method_args or {}),
    }
    if model is not None:
        raw["model"]["policy"] = model
    if dataset is not None:
        raw["data"]["dataset"] = dataset
    if output_dir is not None:
        raw["train"]["output_dir"] = output_dir
    return run_safety_config(raw)
