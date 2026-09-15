"""Representations of feedback, kept distinct from their sources."""

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class Demonstration:
    output: Any
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Preference:
    options: tuple[Any, ...]
    preferred_index: int
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.options) < 2:
            raise ValueError("A preference requires at least two options")
        if not 0 <= self.preferred_index < len(self.options):
            raise ValueError("preferred_index is outside the available options")


@dataclass(frozen=True)
class Critique:
    response: Any
    text: str
    severity: float | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity is not None and not 0.0 <= self.severity <= 1.0:
            raise ValueError("Critique.severity must be in [0, 1]")


@dataclass(frozen=True)
class ProcessFeedback:
    steps: tuple[Any, ...]
    step_scores: tuple[float, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.step_scores and len(self.steps) != len(self.step_scores):
            raise ValueError("step_scores must be empty or match the number of steps")


@dataclass(frozen=True)
class TrajectoryFeedback:
    actions: tuple[Any, ...]
    outcome: Any
    reward: float | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)
