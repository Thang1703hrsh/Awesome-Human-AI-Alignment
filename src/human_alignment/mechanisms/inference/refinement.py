"""Critique-and-revision loops executed without parameter updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from human_alignment.mechanisms.base import normalize_generation
from human_alignment.results import InferenceResult
from human_alignment.types import GenerationRequest


@dataclass
class IterativeRefinement:
    generator: Callable[[Any, GenerationRequest], Any]
    critic: Callable[[Any, Any, GenerationRequest], Any]
    reviser: Callable[[Any, Any, Any, GenerationRequest], Any]
    iterations: int = 1
    name: str = "iterative-refinement"

    def __post_init__(self) -> None:
        if self.iterations < 1:
            raise ValueError("IterativeRefinement.iterations must be at least 1")

    def generate(self, model: Any, request: GenerationRequest) -> InferenceResult:
        current = normalize_generation(self.generator(model, request))
        critiques: list[Any] = []
        for _ in range(self.iterations):
            critique = self.critic(model, current.output, request)
            critiques.append(critique)
            current = normalize_generation(
                self.reviser(model, current.output, critique, request),
                metadata={"refined": True},
            )
        return InferenceResult(
            (current,), self.name, metadata={"iterations": self.iterations, "critiques": critiques}
        )
