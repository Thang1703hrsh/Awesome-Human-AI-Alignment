"""Reliability checks for learned rewards, judges, and verifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from human_alignment.exceptions import AssuranceError
from human_alignment.results import AssuranceReport
from human_alignment.types import EvaluationCase, as_score, mean


@dataclass
class EvaluatorReliability:
    evaluator: Callable[[Any, EvaluationCase], bool | int | float]
    max_mae: float = 0.1
    binary_threshold: float = 0.5
    name: str = "evaluator-reliability"

    def assess(
        self,
        model: Any,
        cases: Sequence[EvaluationCase],
        *,
        baseline_model: Any | None = None,
    ) -> AssuranceReport:
        if not cases:
            raise AssuranceError("EvaluatorReliability requires labeled cases")
        predictions: list[float] = []
        references: list[float] = []
        for case in cases:
            if not isinstance(case.expected, (bool, int, float)):
                raise AssuranceError(
                    "Each reliability case needs a numeric or boolean expected value"
                )
            predictions.append(as_score(self.evaluator(model, case)))
            references.append(as_score(case.expected))

        errors = [
            abs(predicted - expected)
            for predicted, expected in zip(predictions, references)
        ]
        mae = mean(errors)
        agreement = mean(
            [
                float((predicted >= self.binary_threshold) == (expected >= self.binary_threshold))
                for predicted, expected in zip(predictions, references)
            ]
        )
        passed = mae <= self.max_mae
        findings = () if passed else (f"mean_absolute_error={mae:.4f} > {self.max_mae:.4f}",)
        return AssuranceReport(
            self.name,
            {"mean_absolute_error": mae, "binary_agreement": agreement},
            passed,
            findings,
            {"case_count": len(cases)},
        )
