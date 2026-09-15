"""Stable checkpoint-loading boundary independent of the execution backend."""

from __future__ import annotations

from typing import Any, Mapping

from human_alignment.config import CheckpointConfig
from human_alignment.exceptions import ConfigurationError
from human_alignment.results import AlignmentRun


def load_checkpoint(
    checkpoint: str,
    *,
    config: CheckpointConfig | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> AlignmentRun:
    """Load a base model, aligned checkpoint, or PEFT adapter."""

    if config is None:
        resolved = CheckpointConfig()
    elif isinstance(config, Mapping):
        resolved = CheckpointConfig(**dict(config))
    elif isinstance(config, CheckpointConfig):
        resolved = config
    else:
        raise ConfigurationError("config must be CheckpointConfig or a mapping")

    from human_alignment.integrations.transformers import load_transformers_model

    return load_transformers_model(checkpoint, config=resolved, **kwargs)
