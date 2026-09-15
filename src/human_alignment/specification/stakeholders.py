"""Stakeholder descriptions used by an alignment specification."""

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class Stakeholder:
    id: str
    role: str
    weight: float = 1.0
    attributes: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Stakeholder.id must not be empty")
        if self.weight < 0:
            raise ValueError("Stakeholder.weight must be non-negative")
