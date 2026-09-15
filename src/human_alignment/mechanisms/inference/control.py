"""Human or policy-based approval gates for model actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from human_alignment.mechanisms.base import normalize_generation
from human_alignment.results import InferenceResult
from human_alignment.types import GenerationRequest


@dataclass(frozen=True)
class ControlDecision:
    approved: bool
    replacement: Any = None
    reason: str = ""


@dataclass
class HumanControl:
    generator: Callable[[Any, GenerationRequest], Any]
    controller: Callable[[Any, GenerationRequest], ControlDecision | bool]
    blocked_output: Any = None
    name: str = "human-control"

    def generate(self, model: Any, request: GenerationRequest) -> InferenceResult:
        candidate = normalize_generation(self.generator(model, request))
        raw_decision = self.controller(candidate.output, request)
        decision = (
            raw_decision
            if isinstance(raw_decision, ControlDecision)
            else ControlDecision(approved=bool(raw_decision))
        )
        if decision.approved:
            output = candidate.output
        elif decision.replacement is not None:
            output = decision.replacement
        else:
            output = self.blocked_output
        generation = normalize_generation(
            output,
            metadata={"approved": decision.approved, "reason": decision.reason},
        )
        return InferenceResult((generation,), self.name)
