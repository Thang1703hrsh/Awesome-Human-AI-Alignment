"""Small, dependency-free domain types shared across the lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


class AlignmentStage(str, Enum):
    SPECIFICATION = "specification"
    SUPERVISION = "supervision"
    TRAINING = "training"
    INFERENCE = "inference"
    ASSURANCE = "assurance"


class FeedbackSource(str, Enum):
    HUMAN = "human"
    AI = "ai"
    PROGRAMMATIC = "programmatic"
    VERIFIABLE = "verifiable"
    HYBRID = "hybrid"


class FeedbackRepresentation(str, Enum):
    DEMONSTRATION = "demonstration"
    EVALUATION = "evaluation"
    PREFERENCE = "preference"
    RANKING = "ranking"
    CRITIQUE = "critique"
    PROCESS = "process"
    TRAJECTORY = "trajectory"
    REWARD = "reward"
    VERIFIER_OUTCOME = "verifier_outcome"
    RULE = "rule"


@dataclass(frozen=True)
class InteractionContext:
    input: Any
    id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AlignmentTarget:
    name: str
    description: str
    stakeholders: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    context_rules: Mapping[str, Any] = field(default_factory=dict)
    uncertainty: float = 0.0
    version: str = "1"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("AlignmentTarget.name must not be empty")
        if not 0.0 <= self.uncertainty <= 1.0:
            raise ValueError("AlignmentTarget.uncertainty must be in [0, 1]")


@dataclass(frozen=True)
class SupervisionSignal:
    context: InteractionContext
    source: FeedbackSource
    representation: FeedbackRepresentation
    payload: Any
    confidence: float = 1.0
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("SupervisionSignal.confidence must be in [0, 1]")


@dataclass(frozen=True)
class GenerationRequest:
    context: InteractionContext
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Generation:
    output: Any
    score: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationCase:
    context: InteractionContext
    expected: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaxonomyTrace:
    specification: tuple[str, ...] = ()
    supervision: tuple[str, ...] = ()
    mechanisms: tuple[str, ...] = ()
    assurance: tuple[str, ...] = ()


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot compute a mean over an empty collection")
    return sum(values) / len(values)


def as_score(value: bool | int | float) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"Expected a bool or numeric score, received {type(value).__name__}")
