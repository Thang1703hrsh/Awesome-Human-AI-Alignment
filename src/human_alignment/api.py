"""Small public facade for training and loading aligned models."""

from __future__ import annotations

from typing import Any, Mapping

from human_alignment.config import MethodRunConfig, load_config
from human_alignment.exceptions import ConfigurationError
from human_alignment.registry import default_registry
from human_alignment.results import AlignmentRun
from human_alignment.models.checkpoint import load_checkpoint
from human_alignment.specification.targets import AlignmentTarget


def _target(value: AlignmentTarget | Mapping[str, Any] | None) -> AlignmentTarget | None:
    if value is None or isinstance(value, AlignmentTarget):
        return value
    return AlignmentTarget(**dict(value))


def align(
    method: str,
    *,
    model: Any,
    dataset: Any,
    output_dir: str | None = None,
    config: Any = None,
    target: AlignmentTarget | Mapping[str, Any] | None = None,
    train: bool = True,
    **method_options: Any,
) -> AlignmentRun | Any:
    """Create a registered method and, by default, train it immediately."""

    method_id = method.lower()
    runnable = {
        "adpa",
        "ctpd",
        "dckd",
        "dpo",
        "grpo",
        "kto",
        "ppd",
        "ppo",
        "preference_distillation",
        "sft",
        "simpo",
        "tvkd",
        "vpd",
    }
    if method_id not in runnable:
        raise ConfigurationError(
            f"{method_id} is not a standalone training workflow; use AlignmentPipeline."
        )
    implementation = default_registry().create(
        method_id,
        model=model,
        dataset=dataset,
        output_dir=output_dir,
        config=config,
        target=_target(target),
        **method_options,
    )
    return implementation.train() if train else implementation


def run_config(path: str) -> AlignmentRun:
    run = load_config(path)
    return align(
        run.method,
        model=run.model,
        dataset=run.dataset,
        output_dir=run.output_dir,
        config=run.parameters,
        target=run.target,
        **dict(run.method_options),
    )


def load_model(model_id_or_path: str, **kwargs: Any) -> AlignmentRun:
    """Backward-compatible alias for :func:`load_checkpoint`."""

    return load_checkpoint(model_id_or_path, **kwargs)


def align_from_config(config: MethodRunConfig) -> AlignmentRun:
    return align(
        config.method,
        model=config.model,
        dataset=config.dataset,
        output_dir=config.output_dir,
        config=config.parameters,
        target=config.target,
        **dict(config.method_options),
    )
