"""Executable task requirements, separate from response quality/preferences."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping

from human_alignment.specification.targets import AlignmentSpecification, ResolvedTarget
from human_alignment.types import AlignmentTarget, InteractionContext


@dataclass(frozen=True)
class InstructionConstraint:
    id: str
    kind: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    scope: str = "final"

    def __post_init__(self):
        if not self.id or not self.kind or self.scope not in {"final", "reasoning", "both"}:
            raise ValueError("A constraint needs id, kind and final/reasoning/both scope")


class ConstraintChecker:
    """Small deterministic checker registry, NOT the complete IFEval/MOSAIC suite.

    Custom checkers have signature (text, parameters) -> bool. Unknown constraints
    and invalid configurations fail loudly instead of silently passing.
    """

    def __init__(self, custom: Mapping[str, Callable] | None = None):
        self.custom = dict(custom or {})

    def check(self, text: str, constraint: InstructionConstraint) -> bool:
        if not isinstance(text, str):
            raise TypeError("Constraint input must be text")
        kind, p = constraint.kind, constraint.parameters
        if kind in self.custom:
            result = self.custom[kind](text, p)
            if type(result) is not bool:
                raise TypeError("Custom checker must return bool")
            return result
        if kind in {"contains", "excludes", "starts_with", "ends_with"}:
            needle = p.get("text")
            if not isinstance(needle, str) or not needle:
                raise ValueError(f"{kind} requires nonempty text")
            if p.get("case_sensitive", True) is False:
                text, needle = text.casefold(), needle.casefold()
            return {"contains": lambda: needle in text, "excludes": lambda: needle not in text,
                    "starts_with": lambda: text.startswith(needle),
                    "ends_with": lambda: text.endswith(needle)}[kind]()
        if kind == "word_count":
            low, high = p.get("min", 0), p.get("max")
            if type(low) is not int or low < 0 or (high is not None and (
                type(high) is not int or high < low
            )):
                raise ValueError("word_count requires nonnegative integer bounds")
            count = len(re.findall(r"\S+", text))
            return count >= low and (high is None or count <= high)
        if kind == "json_object":
            try:
                value = json.loads(text, parse_constant=lambda x: (_ for _ in ()).throw(
                    ValueError(f"Invalid JSON constant {x}")))
            except ValueError:
                return False
            keys = p.get("required_keys", ())
            if isinstance(keys, str) or not all(isinstance(k, str) for k in keys):
                raise ValueError("required_keys must be a sequence of strings")
            return isinstance(value, dict) and set(keys) <= value.keys()
        raise ValueError(f"Unknown constraint kind: {kind}")


@dataclass(frozen=True)
class TaskSpecification:
    """Resolve task instructions into the existing AlignmentSpecification contract."""

    target: AlignmentTarget
    constraints: tuple[InstructionConstraint, ...] = ()

    def __post_init__(self):
        if len({c.id for c in self.constraints}) != len(self.constraints):
            raise ValueError("Constraint IDs must be unique within a task")

    def resolve(self, context: InteractionContext) -> ResolvedTarget:
        base = AlignmentSpecification(self.target).resolve(context)
        labels = tuple(f"{c.id}: {c.kind} {dict(c.parameters)} [{c.scope}]"
                       for c in self.constraints)
        return ResolvedTarget(replace(base.target, constraints=base.target.constraints + labels),
                              context, {**base.applied_rules, "instruction_constraints": self.constraints})

    def evaluate(self, final: str, *, reasoning: str | None = None,
                 checker: ConstraintChecker | None = None):
        checker = checker or ConstraintChecker()
        results = {}
        for constraint in self.constraints:
            scopes = ("reasoning", "final") if constraint.scope == "both" else (constraint.scope,)
            for scope in scopes:
                text = reasoning if scope == "reasoning" else final
                # Unavailable reasoning is not evidence of compliance.
                results[f"{constraint.id}:{scope}"] = (
                    None if text is None else checker.check(text, constraint))
        return results
