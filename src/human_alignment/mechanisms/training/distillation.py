"""Public facades for alignment and preference distillation."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from human_alignment.config import PreferenceDistillationConfig
from human_alignment.exceptions import ConfigurationError
from human_alignment.mechanisms.training.base import TrainingMethodBase
from human_alignment.results import AlignmentRun
from human_alignment.specification.targets import AlignmentSpecification
from human_alignment.types import SupervisionSignal, TaxonomyTrace


class AlignmentDistillation:
    """Framework-neutral callback adapter for custom distillation pipelines."""

    method_id = "alignment_distillation"

    def __init__(self, trainer: Callable[..., Any]) -> None:
        self.trainer = trainer

    def fit(
        self,
        model: Any,
        supervision: Sequence[SupervisionSignal],
        specification: AlignmentSpecification,
    ) -> AlignmentRun:
        outcome = self.trainer(model, supervision, specification)
        trained_model, metrics = outcome if isinstance(outcome, tuple) else (outcome, {})
        return AlignmentRun(
            model=trained_model,
            method=self.method_id,
            metrics=metrics,
            taxonomy_trace=TaxonomyTrace(mechanisms=("alignment_distillation",)),
        )


class PreferenceDistillation(TrainingMethodBase):
    """Train a student with one of six preference-distillation objectives.

    Teacher scores may already be present in the dataset. Otherwise, pass a
    ``teacher_model`` for VPD or PPD. CTPD additionally requires a
    ``reference_model`` unless it is supplied in ``backend_options``.
    """

    method_id = "preference_distillation"
    backend_method = "preference_distillation"
    dataset_kind = "preference_distillation"
    config_type = PreferenceDistillationConfig
    supervision_categories = ("demonstrations_preferences",)
    mechanism_categories = ("alignment_distillation",)
    default_output_dir = "outputs/preference-distillation"
    default_objective: str | None = None
    objective_defaults: Mapping[str, Mapping[str, Any]] = {
        "dckd": {
            "sft_weight": 1.0,
            "chosen_kd_weight": 0.1,
            "rejected_kd_weight": 0.1,
        },
        "tvkd": {
            "chosen_kd_weight": 0.0001,
            "rejected_kd_weight": 0.0001,
            "value_function": "entropy",
        },
        "adpa": {
            "objective_weight": 1.6,
            "sft_weight": 1.0,
        },
    }

    def __init__(
        self,
        model: Any = None,
        dataset: Any = None,
        output_dir: str | None = None,
        config: PreferenceDistillationConfig | Mapping[str, Any] | None = None,
        target: Any = None,
        teacher_model: Any = None,
        reference_model: Any = None,
        tokenizer: Any = None,
        backend_options: Mapping[str, Any] | None = None,
    ) -> None:
        selected_objective = self.default_objective
        if isinstance(config, Mapping):
            selected_objective = str(config.get("objective", selected_objective or "vpd"))
        elif isinstance(config, PreferenceDistillationConfig):
            selected_objective = config.objective
        elif selected_objective is None:
            selected_objective = "vpd"
        defaults = dict(self.objective_defaults.get(selected_objective, {}))

        parameters: PreferenceDistillationConfig | Mapping[str, Any] | None
        if config is None:
            parameters = defaults
        elif isinstance(config, Mapping):
            parameters = {**defaults, **dict(config)}
        else:
            parameters = config
        if self.default_objective is not None:
            if isinstance(parameters, Mapping):
                parameters = {**dict(parameters), "objective": self.default_objective}
            elif parameters.objective != self.default_objective:
                raise ConfigurationError(
                    f"{self.method_id} requires objective={self.default_objective!r}"
                )

        options = dict(backend_options or {})
        for key, value in {
            "teacher_model": teacher_model,
            "reference_model": reference_model,
            "tokenizer": tokenizer,
        }.items():
            if value is not None:
                if key in options:
                    raise ConfigurationError(f"Pass {key} directly or in backend_options, not both")
                options[key] = value

        super().__init__(
            model=model,
            dataset=dataset,
            output_dir=output_dir,
            config=parameters,
            target=target,
            backend_options=options,
        )

    @staticmethod
    def _backend() -> Any:
        from human_alignment.integrations.preference_distillation import (
            PreferenceDistillationBackend,
        )

        return PreferenceDistillationBackend()


class DCKD(PreferenceDistillation):
    method_id = "dckd"
    default_objective = "dckd"
    default_output_dir = "outputs/dckd"


class TVKD(PreferenceDistillation):
    method_id = "tvkd"
    default_objective = "tvkd"
    default_output_dir = "outputs/tvkd"


class ADPA(PreferenceDistillation):
    method_id = "adpa"
    default_objective = "adpa"
    default_output_dir = "outputs/adpa"


class CTPD(PreferenceDistillation):
    method_id = "ctpd"
    default_objective = "ctpd"
    default_output_dir = "outputs/ctpd"


class PPD(PreferenceDistillation):
    method_id = "ppd"
    default_objective = "ppd"
    default_output_dir = "outputs/ppd"


class VPD(PreferenceDistillation):
    method_id = "vpd"
    default_objective = "vpd"
    default_output_dir = "outputs/vpd"
