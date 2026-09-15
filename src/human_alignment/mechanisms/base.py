"""Common contracts for executable alignment mechanisms."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

from human_alignment.results import AlignmentRun, InferenceResult
from human_alignment.specification.targets import AlignmentSpecification
from human_alignment.types import Generation, GenerationRequest, SupervisionSignal


class TrainingMethod(Protocol):
    method_id: str

    def train(
        self,
        *,
        model: Any | None = None,
        dataset: Any | None = None,
        target: Any | None = None,
    ) -> AlignmentRun: ...

    def fit(
        self,
        model: Any,
        supervision: Sequence[SupervisionSignal],
        specification: AlignmentSpecification,
    ) -> AlignmentRun: ...


class InferenceMechanism(Protocol):
    name: str

    def generate(self, model: Any, request: GenerationRequest) -> InferenceResult: ...


def normalize_generation(value: Any, *, metadata: Mapping[str, Any] | None = None) -> Generation:
    if isinstance(value, Generation):
        return value
    return Generation(output=value, metadata=metadata or {})
