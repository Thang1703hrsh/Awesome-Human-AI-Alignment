"""Represent static and context-dependent alignment specifications."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping

from human_alignment.specification.stakeholders import Stakeholder
from human_alignment.types import AlignmentTarget, InteractionContext


@dataclass(frozen=True)
class ResolvedTarget:
    target: AlignmentTarget
    context: InteractionContext
    applied_rules: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AlignmentSpecification:
    """An explicit target plus optional context-sensitive resolution logic."""

    target: AlignmentTarget
    stakeholders: tuple[Stakeholder, ...] = ()
    resolver: Callable[[AlignmentTarget, InteractionContext], ResolvedTarget] | None = None

    def resolve(self, context: InteractionContext) -> ResolvedTarget:
        if self.resolver is not None:
            return self.resolver(self.target, context)

        applicable = {
            key: value
            for key, value in self.target.context_rules.items()
            if context.metadata.get(key) is not None
        }
        if not applicable:
            return ResolvedTarget(self.target, context)

        constraints = list(self.target.constraints)
        for value in applicable.values():
            if isinstance(value, str):
                constraints.append(value)
            elif isinstance(value, (list, tuple)):
                constraints.extend(str(item) for item in value)
        resolved = replace(self.target, constraints=tuple(dict.fromkeys(constraints)))
        return ResolvedTarget(resolved, context, applicable)


def preference_implied_target() -> AlignmentTarget:
    """Represent an unspecified target without conflating it with preference data."""

    return AlignmentTarget(
        name="preference-implied-target",
        description="The target is represented implicitly by the supplied preference data.",
        uncertainty=1.0,
    )
