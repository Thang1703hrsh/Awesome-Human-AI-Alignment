"""Public workflow for preparing preference-distillation supervision."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

from human_alignment.config import DistillationObjective, PreferenceDistillationConfig
from human_alignment.exceptions import ConfigurationError


@dataclass(frozen=True)
class DistillationPreparationConfig:
    objective: DistillationObjective
    output_dir: str
    split: str | None = None
    top_k: int = 50
    num_responses: int = 4
    teacher_score_mode: Literal["sequence_logprob", "pairwise"] = "sequence_logprob"
    generation_backend: Literal["transformers", "vllm"] = "transformers"
    model_kwargs: Mapping[str, Any] = field(default_factory=dict)
    generation_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ConfigurationError("top_k must be positive")
        if self.num_responses < 2:
            raise ConfigurationError("num_responses must be at least two")
        if not self.output_dir:
            raise ConfigurationError("output_dir is required")
        if self.teacher_score_mode not in {"sequence_logprob", "pairwise"}:
            raise ConfigurationError("Unknown teacher_score_mode")
        if self.generation_backend not in {"transformers", "vllm"}:
            raise ConfigurationError("Unknown generation_backend")


@dataclass(frozen=True)
class PreparedDistillationDataset:
    objective: DistillationObjective
    path: Path
    dataset: Any


def prepare_preference_distillation_dataset(
    dataset: Any,
    *,
    objective: DistillationObjective,
    output_dir: str,
    teacher_model: Any = None,
    reference_model: Any = None,
    student_model: Any = None,
    tokenizer: Any = None,
    split: str | None = None,
    top_k: int = 50,
    num_responses: int = 4,
    teacher_score_mode: Literal["sequence_logprob", "pairwise"] = "sequence_logprob",
    generation_backend: Literal["transformers", "vllm"] = "transformers",
    model_kwargs: Mapping[str, Any] | None = None,
    generation_kwargs: Mapping[str, Any] | None = None,
    training_config: PreferenceDistillationConfig | None = None,
) -> PreparedDistillationDataset:
    """Prepare and persist the supervision consumed by one distillation method.

    DCKD and TVKD precompute compressed teacher token probabilities. ADPA
    computes teacher-reference token advantages. CTPD builds cross-tokenizer
    parent groups and weights. PPD and VPD optionally sample candidates from a
    student and score them with a teacher.
    """

    settings = DistillationPreparationConfig(
        objective=objective,
        output_dir=output_dir,
        split=split,
        top_k=top_k,
        num_responses=num_responses,
        teacher_score_mode=teacher_score_mode,
        generation_backend=generation_backend,
        model_kwargs=dict(model_kwargs or {}),
        generation_kwargs=dict(generation_kwargs or {}),
    )
    from human_alignment.integrations.distillation_data import DistillationDataBackend

    prepared = DistillationDataBackend().prepare(
        objective=settings.objective,
        dataset=dataset,
        output_dir=settings.output_dir,
        teacher_model=teacher_model,
        reference_model=reference_model,
        student_model=student_model,
        tokenizer=tokenizer,
        split=settings.split,
        top_k=settings.top_k,
        num_responses=settings.num_responses,
        teacher_score_mode=settings.teacher_score_mode,
        generation_backend=settings.generation_backend,
        model_kwargs=settings.model_kwargs,
        generation_kwargs=settings.generation_kwargs,
        config=training_config,
    )
    return PreparedDistillationDataset(
        objective=settings.objective,
        path=Path(settings.output_dir),
        dataset=prepared,
    )
