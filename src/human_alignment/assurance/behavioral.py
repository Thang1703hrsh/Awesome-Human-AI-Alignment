"""Task- and behavior-level evaluation across explicit dimensions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from human_alignment.exceptions import AssuranceError
from human_alignment.results import AssuranceReport
from human_alignment.types import EvaluationCase, as_score, mean


@dataclass
class BehavioralEvaluation:
    evaluators: Mapping[str, Callable[[Any, EvaluationCase], bool | int | float]]
    thresholds: Mapping[str, float] | None = None
    name: str = "behavioral-evaluation"

    def assess(
        self,
        model: Any,
        cases: Sequence[EvaluationCase],
        *,
        baseline_model: Any | None = None,
    ) -> AssuranceReport:
        if not cases:
            raise AssuranceError("BehavioralEvaluation requires at least one evaluation case")
        if not self.evaluators:
            raise AssuranceError("BehavioralEvaluation requires at least one evaluator")

        metrics = {
            dimension: mean([as_score(evaluator(model, case)) for case in cases])
            for dimension, evaluator in self.evaluators.items()
        }
        thresholds = self.thresholds or {dimension: 0.5 for dimension in metrics}
        missing = set(metrics) - set(thresholds)
        if missing:
            raise AssuranceError(f"Missing thresholds for: {', '.join(sorted(missing))}")
        failures = [
            f"{dimension}={score:.4f} < {thresholds[dimension]:.4f}"
            for dimension, score in metrics.items()
            if score < thresholds[dimension]
        ]
        return AssuranceReport(
            method=self.name,
            metrics=metrics,
            passed=not failures,
            findings=tuple(failures),
            metadata={"case_count": len(cases)},
        )
