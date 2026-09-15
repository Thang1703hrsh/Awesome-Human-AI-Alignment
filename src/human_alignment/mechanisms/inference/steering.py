"""Representation, prompt, or decoding intervention at inference time."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from human_alignment.mechanisms.base import normalize_generation
from human_alignment.results import InferenceResult
from human_alignment.types import GenerationRequest


@dataclass
class InferenceSteering:
    generator: Callable[[Any, GenerationRequest], Any]
    steer: Callable[[Any, GenerationRequest], GenerationRequest]
    name: str = "inference-steering"

    def generate(self, model: Any, request: GenerationRequest) -> InferenceResult:
        steered_request = self.steer(model, request)
        generation = normalize_generation(
            self.generator(model, steered_request),
            metadata={"steered": True},
        )
        return InferenceResult((generation,), self.name)
