"""Advanced composition across the four alignment lifecycle dimensions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from human_alignment.assurance.base import AssuranceMethod
from human_alignment.mechanisms.base import InferenceMechanism, TrainingMethod
from human_alignment.results import PipelineResult
from human_alignment.specification.targets import AlignmentSpecification
from human_alignment.supervision.sources import FeedbackProvider
from human_alignment.types import EvaluationCase, GenerationRequest, InteractionContext


@dataclass
class AlignmentPipeline:
    specification: AlignmentSpecification
    supervision: Sequence[FeedbackProvider] = field(default_factory=tuple)
    training: Sequence[TrainingMethod] = field(default_factory=tuple)
    inference: InferenceMechanism | None = None
    assurance: Sequence[AssuranceMethod] = field(default_factory=tuple)

    def run(
        self,
        base_model: Any,
        *,
        contexts: Sequence[InteractionContext] = (),
        requests: Sequence[GenerationRequest] = (),
        evaluation_cases: Sequence[EvaluationCase] = (),
    ) -> PipelineResult:
        signals = []
        for provider in self.supervision:
            signals.extend(provider.collect(contexts, self.specification))

        current_model = base_model
        training_results = []
        for mechanism in self.training:
            result = mechanism.fit(current_model, signals, self.specification)
            training_results.append(result)
            current_model = result.model

        inference_results = []
        if self.inference is not None:
            inference_results = [
                self.inference.generate(current_model, request) for request in requests
            ]
        assurance_reports = [
            method.assess(current_model, evaluation_cases, baseline_model=base_model)
            for method in self.assurance
        ]
        return PipelineResult(
            current_model,
            tuple(signals),
            tuple(training_results),
            tuple(inference_results),
            tuple(assurance_reports),
        )
