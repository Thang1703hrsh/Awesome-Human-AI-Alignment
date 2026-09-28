"""Lifecycle facade for the integrated safety-alignment implementation."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from human_alignment.exceptions import ConfigurationError
from human_alignment.results import AlignmentRun
from human_alignment.types import TaxonomyTrace


SAFETY_METHOD_ALIASES = {
    "safety_sft": "sft",
    "safety_ppo": "ppo",
}

SAFETY_REINFORCEMENT_METHODS = {"saferlhf", "ppo", "morlhf", "can_dual"}
SAFETY_REWARD_METHODS = {"reward_model", "cost_model"}


class SafetyAlignment:
    """Run any migrated safety-alignment stage through a common package facade.

    ``config`` may be a YAML path, an ``ExperimentConfig``, or a mapping using
    the original ``model``/``data``/``train``/``method_args`` sections. Direct
    constructor arguments override the corresponding recipe values.
    """

    method_id = "safety_alignment"

    def __init__(
        self,
        model: Any = None,
        dataset: Any = None,
        output_dir: str | None = None,
        config: str | Path | Mapping[str, Any] | Any | None = None,
        target: Any = None,
        *,
        method: str | None = None,
        safety_method: str | None = None,
        overrides: Sequence[str] = (),
        model_options: Mapping[str, Any] | None = None,
        data_options: Mapping[str, Any] | None = None,
        train_options: Mapping[str, Any] | None = None,
        method_args: Mapping[str, Any] | None = None,
    ) -> None:
        requested = safety_method or method or self.method_id
        selected = SAFETY_METHOD_ALIASES.get(requested, requested)
        self.public_method = requested
        self.selected_method = selected
        self.model = model
        self.dataset = dataset
        self.output_dir = output_dir
        self.config = config
        self.target = target
        self.overrides = tuple(overrides)
        self.model_options = dict(model_options or {})
        self.data_options = dict(data_options or {})
        self.train_options = dict(train_options or {})
        self.method_args = dict(method_args or {})

    def _resolved_config(self):
        from human_alignment.safety.api import load_safety_config
        from human_alignment.safety.registry import METHODS
        from human_alignment.safety.schema import ExperimentConfig, config_from_dict

        if self.config is None:
            if self.selected_method == self.method_id:
                raise ConfigurationError("SafetyAlignment requires method or safety_method")
            raw = {
                "method": self.selected_method,
                "model": dict(self.model_options),
                "data": dict(self.data_options),
                "train": dict(self.train_options),
                "method_args": dict(self.method_args),
            }
            config = config_from_dict(raw)
        else:
            config = load_safety_config(self.config, self.overrides)
            raw = asdict(config)
            raw.pop("source", None)
            raw["method"] = (
                config.method if self.selected_method == self.method_id else self.selected_method
            )
            raw["model"].update(self.model_options)
            raw["data"].update(self.data_options)
            raw["train"].update(self.train_options)
            raw["method_args"].update(self.method_args)
            config = config_from_dict(raw, source=config.source)
        if config.method not in METHODS:
            raise ConfigurationError(
                f"Unknown safety method: {config.method}. Available: {', '.join(METHODS)}"
            )
        if self.model is not None:
            config.model.policy = str(self.model)
        if self.dataset is not None:
            config.data.dataset = str(self.dataset)
        if self.output_dir is not None:
            config.train["output_dir"] = self.output_dir
        return config

    def train(self) -> AlignmentRun:
        from human_alignment.safety.api import run_safety_config

        config = self._resolved_config()
        output = run_safety_config(config)
        mechanism = (
            "reinforcement_learning"
            if config.method in SAFETY_REINFORCEMENT_METHODS
            else "reward_verifier_modeling"
            if config.method in SAFETY_REWARD_METHODS
            else "supervised_alignment"
            if config.method in {"sft", "cpsft"}
            else "preference_optimization"
        )
        return AlignmentRun(
            model=str(output),
            method=self.public_method,
            output_dir=str(output),
            taxonomy_trace=TaxonomyTrace(
                specification=("safety_alignment",),
                supervision=("demonstrations_preferences",),
                mechanisms=(mechanism,),
            ),
            metadata={
                "backend": "integrated-safety",
                "safety_method": config.method,
                "recipe": config.source,
                "target": self.target,
            },
        )
