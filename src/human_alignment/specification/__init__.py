"""Alignment target and stakeholder abstractions."""

from human_alignment.specification.stakeholders import Stakeholder
from human_alignment.specification.targets import (
    AlignmentSpecification,
    ResolvedTarget,
    preference_implied_target,
)

__all__ = [
    "AlignmentSpecification",
    "ResolvedTarget",
    "Stakeholder",
    "preference_implied_target",
]
