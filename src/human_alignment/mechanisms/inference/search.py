"""Verifier- or reward-guided candidate search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from human_alignment.mechanisms.base import normalize_generation
from human_alignment.results import InferenceResult
from human_alignment.types import Generation, GenerationRequest


@dataclass
class BestOfNSearch:
    generator: Callable[[Any, GenerationRequest], Any]
    scorer: Callable[[Any, Generation, GenerationRequest], float]
    n: int = 4
    name: str = "best-of-n-search"

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("BestOfNSearch.n must be at least 1")

    def generate(self, model: Any, request: GenerationRequest) -> InferenceResult:
        candidates = []
        for index in range(self.n):
            candidate = normalize_generation(
                self.generator(model, request), metadata={"candidate_index": index}
            )
            score = float(self.scorer(model, candidate, request))
            candidates.append(Generation(candidate.output, score, candidate.metadata))
        selected = max(
            candidates,
            key=lambda item: item.score if item.score is not None else float("-inf"),
        )
        return InferenceResult(
            (selected,),
            self.name,
            {"candidate_count": self.n, "candidate_scores": [c.score for c in candidates]},
        )
