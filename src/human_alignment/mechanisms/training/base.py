"""Shared implementation for executable training-time alignment methods."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

from human_alignment.config import TrainingConfig
from human_alignment.exceptions import ConfigurationError
from human_alignment.results import AlignmentRun
from human_alignment.specification.targets import AlignmentSpecification
from human_alignment.supervision.datasets import prepare_dataset
from human_alignment.types import SupervisionSignal, TaxonomyTrace


class TrainingMethodBase:
    """Lifecycle-aware facade around a lazily loaded training backend."""

    method_id = "trl-method"
    backend_method = "unknown"
    dataset_kind = "instruction"
    config_type: type[TrainingConfig] = TrainingConfig
    mechanism_categories: tuple[str, ...] = ()
    supervision_categories: tuple[str, ...] = ()
    default_output_dir = "outputs/alignment"

    def __init__(
        self,
        model: Any = None,
        dataset: Any = None,
        output_dir: str | None = None,
        config: TrainingConfig | Mapping[str, Any] | None = None,
        target: Any = None,
        backend_options: Mapping[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.target = target
        self.backend_options = dict(backend_options or {})
        if config is None:
            self.config = self.config_type(output_dir=output_dir or self.default_output_dir)
        elif isinstance(config, Mapping):
            parameters = dict(config)
            parameters["output_dir"] = output_dir or parameters.get(
                "output_dir", self.default_output_dir
            )
            self.config = self.config_type(**parameters)
        elif isinstance(config, self.config_type):
            self.config = replace(config, output_dir=output_dir) if output_dir else config
        else:
            raise ConfigurationError(
                f"{self.method_id} requires {self.config_type.__name__} or a mapping"
            )

    def train(
        self,
        *,
        model: Any | None = None,
        dataset: Any | None = None,
        target: Any | None = None,
    ) -> AlignmentRun:
        selected_model = self.model if model is None else model
        selected_dataset = self.dataset if dataset is None else dataset
        selected_target = self.target if target is None else target
        if selected_model is None:
            raise ConfigurationError(f"{self.method_id} requires a model or model identifier")
        if selected_dataset is None:
            raise ConfigurationError(f"{self.method_id} requires a dataset")

        bundle = prepare_dataset(selected_dataset, self.dataset_kind)
        run = self._backend().train(
            method=self.backend_method,
            model=selected_model,
            dataset=bundle,
            config=self.config,
            options=self.backend_options,
        )
        run.method = self.method_id
        run.output_dir = self.config.output_dir
        run.taxonomy_trace = TaxonomyTrace(
            supervision=self.supervision_categories,
            mechanisms=self.mechanism_categories,
        )
        run.metadata = {
            **dict(run.metadata),
            "dataset_origin": bundle.origin,
            "specification_status": "explicit" if selected_target is not None else "implicit",
            "target": selected_target,
        }
        return run

    @staticmethod
    def _backend() -> Any:
        from human_alignment.integrations.trl import TRLBackend

        return TRLBackend()

    def fit(
        self,
        model: Any,
        supervision: Sequence[SupervisionSignal],
        specification: AlignmentSpecification,
    ) -> AlignmentRun:
        return self.train(model=model, dataset=supervision, target=specification.target)
