"""Behavioral stress tests under controlled perturbations or shifts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from human_alignment.exceptions import AssuranceError
from human_alignment.results import AssuranceReport
from human_alignment.types import EvaluationCase, as_score, mean


@dataclass
class RobustnessEvaluation:
    scorer: Callable[[Any, EvaluationCase], bool | int | float]
    perturbations: Mapping[str, Callable[[EvaluationCase], EvaluationCase]]
    max_drop: float = 0.1
    name: str = "robustness-evaluation"

    def assess(
        self,
        model: Any,
        cases: Sequence[EvaluationCase],
        *,
        baseline_model: Any | None = None,
    ) -> AssuranceReport:
        if not cases or not self.perturbations:
            raise AssuranceError("RobustnessEvaluation requires cases and perturbations")
        nominal = mean([as_score(self.scorer(model, case)) for case in cases])
        metrics: dict[str, float] = {"nominal": nominal}
        findings: list[str] = []
        for shift_name, perturb in self.perturbations.items():
            shifted = mean([as_score(self.scorer(model, perturb(case))) for case in cases])
            drop = nominal - shifted
            metrics[f"shift/{shift_name}"] = shifted
            metrics[f"drop/{shift_name}"] = drop
            if drop > self.max_drop:
                findings.append(f"{shift_name} drop={drop:.4f} > {self.max_drop:.4f}")
        return AssuranceReport(
            self.name,
            metrics,
            not findings,
            tuple(findings),
            {"case_count": len(cases)},
        )
