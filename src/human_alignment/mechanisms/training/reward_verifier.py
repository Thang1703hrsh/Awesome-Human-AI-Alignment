"""Framework-neutral adapter for reward and verifier model training."""

from __future__ import annotations

from typing import Any, Callable, Sequence

from human_alignment.results import AlignmentRun
from human_alignment.specification.targets import AlignmentSpecification
from human_alignment.types import SupervisionSignal, TaxonomyTrace


class RewardVerifierModeling:
    method_id = "reward_verifier_modeling"

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
            taxonomy_trace=TaxonomyTrace(mechanisms=("reward_verifier_modeling",)),
        )
