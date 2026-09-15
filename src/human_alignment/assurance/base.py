"""Common assurance interface."""

from __future__ import annotations

from typing import Any, Protocol, Sequence

from human_alignment.results import AssuranceReport
from human_alignment.types import EvaluationCase


class AssuranceMethod(Protocol):
    name: str

    def assess(
        self,
        model: Any,
        cases: Sequence[EvaluationCase],
        *,
        baseline_model: Any | None = None,
    ) -> AssuranceReport: ...
