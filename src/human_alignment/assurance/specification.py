"""Metrics for task assistance, personalized ranking and temporal stability.

Adapters consume explicit observations, not inferred user success or hidden model
reasoning. See docs/ALIGNMENT_SPECIFICATION.md for paper-specific scope limits.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

from human_alignment.specification.tasks import ConstraintChecker


def _finite(value):
    if isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError("Expected a finite number")
    return float(value)


def _mean(values):
    values = list(values)
    if not values:
        raise ValueError("At least one observation is required")
    return sum(values) / len(values)


def _booleans(values):
    values = list(values)
    if not values or any(type(v) is not bool for v in values):
        raise ValueError("Expected a nonempty sequence of boolean outcomes")
    return values


class IFEvalReliability:
    """Paper reliable@k: ALL k distinct cousin prompts must pass in each group."""

    def evaluate(self, groups: Mapping[str, Sequence[bool]], k: int):
        if type(k) is not int or k <= 0 or not groups:
            raise ValueError("Positive k and nonempty cousin groups are required")
        values = [_booleans(group) for group in groups.values()]
        if any(len(group) != k for group in values):
            raise ValueError("Each explicitly selected cousin group must contain exactly k outcomes")
        return {f"reliable@{k}": _mean(all(group) for group in values),
                "prompt_accuracy": _mean(v for group in values for v in group),
                "groups": len(values)}


class MosaicEvaluation:
    """Granular compliance by constraint kind, position, and constraint count."""

    def __init__(self, checker: ConstraintChecker | None = None):
        self.checker = checker or ConstraintChecker()

    def evaluate(self, cases):
        groups = {"kind": defaultdict(list), "position": defaultdict(list),
                  "count": defaultdict(list)}
        prompts = []
        for task, response in cases:
            if not task.constraints or any(c.scope != "final" for c in task.constraints):
                raise ValueError("MOSAIC cases require nonempty final-response constraints")
            results = []
            for position, constraint in enumerate(task.constraints):
                passed = self.checker.check(response, constraint)
                results.append(passed)
                groups["kind"][constraint.kind].append(passed)
                groups["position"][position].append(passed)
                groups["count"][len(task.constraints)].append(passed)
            prompts.append(all(results))
        return {"prompt_accuracy": _mean(prompts),
                **{key: {group: _mean(values) for group, values in grouped.items()}
                   for key, grouped in groups.items()}}


class ReasonIFEvaluation:
    def __init__(self, checker=None):
        self.checker = checker or ConstraintChecker()

    def evaluate(self, cases):
        observed = {"reasoning": [], "final": []}
        missing = 0
        joint = []
        for task, final, reasoning in cases:
            if not task.constraints:
                raise ValueError("ReasonIF cases require constraints")
            results = task.evaluate(final, reasoning=reasoning, checker=self.checker)
            for key, value in results.items():
                if value is None:
                    missing += 1
                else:
                    observed[key.rsplit(":", 1)[1]].append(value)
            joint.append(all(v is True for v in results.values()))
        return {"joint_compliance": _mean(joint), "unobserved_constraints": missing,
                **{key: _mean(values) if values else None for key, values in observed.items()}}


class FeedbackResponsiveness:
    """FB-Bench-style correction/preservation analysis on caller-scored outcomes."""

    def evaluate(self, before: Sequence[bool], after: Sequence[bool]):
        before, after = _booleans(before), _booleans(after)
        if len(before) != len(after):
            raise ValueError("Before/after must be paired")
        failed, passed = before.count(False), before.count(True)
        corrections = sum(not b and a for b, a in zip(before, after))
        regressions = sum(b and not a for b, a in zip(before, after))
        return {"before_accuracy": _mean(before), "after_accuracy": _mean(after),
                "correction_rate": corrections / failed if failed else None,
                "preservation_rate": 1 - regressions / passed if passed else None,
                "corrections": corrections, "regressions": regressions}


class AssistanceOutcomeEvaluation:
    """Planorama-inspired distinction between liking a plan and task utility."""

    def evaluate(self, preference_scores, utility_scores):
        x, y = list(map(_finite, preference_scores)), list(map(_finite, utility_scores))
        if len(x) != len(y) or not x:
            raise ValueError("Preference/utility observations must be nonempty and paired")
        mx, my = _mean(x), _mean(y)
        cross = sum((a-mx)*(b-my) for a, b in zip(x, y))
        denominator = math.sqrt(sum((a-mx)**2 for a in x) * sum((b-my)**2 for b in y))
        return {"mean_preference": mx, "mean_observed_utility": my,
                "pearson_correlation": cross / denominator if denominator else None,
                "count": len(x)}


@dataclass(frozen=True)
class UserComparison:
    user_id: str
    model_a: str
    model_b: str
    score_a: float  # 1: A wins, 0: B wins, 0.5: tie.

    def __post_init__(self):
        if not self.user_id or not self.model_a or not self.model_b or self.model_a == self.model_b:
            raise ValueError("Comparison needs a user and two distinct models")
        if _finite(self.score_a) not in (0., 0.5, 1.):
            raise ValueError("score_a must be 0, 0.5 or 1")


class PersonalizedBenchmark:
    """User-specific Elo and L2-regularized Bradley-Terry rankings."""

    def evaluate(self, comparisons, *, elo_k=32., iterations=500, learning_rate=0.1, l2=0.01):
        if _finite(elo_k) <= 0 or _finite(learning_rate) <= 0 or _finite(l2) <= 0:
            raise ValueError("Elo K, learning rate and L2 must be positive")
        if type(iterations) is not int or iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        users = defaultdict(list)
        for row in comparisons:
            users[row.user_id].append(row)
        if not users:
            raise ValueError("No comparisons")
        output = {}
        for user, rows in users.items():
            names = sorted({m for r in rows for m in (r.model_a, r.model_b)})
            elo, bt = dict.fromkeys(names, 1000.), dict.fromkeys(names, 0.)
            for r in rows:
                exponent = max(-300., min(300., (elo[r.model_b]-elo[r.model_a])/400))
                expected = 1 / (1 + 10**exponent)
                delta = elo_k * (r.score_a-expected)
                elo[r.model_a] += delta
                elo[r.model_b] -= delta
            for _ in range(iterations):
                gradient = {m: l2*bt[m] for m in names}
                for r in rows:
                    margin = max(-700., min(700., bt[r.model_a]-bt[r.model_b]))
                    error = (1/(1+math.exp(-margin))-r.score_a) / len(rows)
                    gradient[r.model_a] += error
                    gradient[r.model_b] -= error
                bt = {m: bt[m]-learning_rate*gradient[m] for m in names}
            output[user] = {"elo": elo, "bradley_terry": bt, "comparisons": len(rows)}
        return output


@dataclass(frozen=True)
class TimedPreference:
    user_id: str
    item_id: str
    time: float
    choice: str
    context_id: str = "default"

    def __post_init__(self):
        if not self.user_id or not self.item_id or not self.choice or not self.context_id:
            raise ValueError("Nonempty user, item, choice and context are required")
        _finite(self.time)


class PreferenceStability:
    """Descriptive within-person retest agreement; no causal claim about drift."""

    def evaluate(self, observations):
        groups = defaultdict(list)
        for row in observations:
            groups[(row.user_id, row.item_id, row.context_id)].append(row)
        changes, lags = [], []
        for rows in groups.values():
            rows = sorted(rows, key=lambda x: x.time)
            if len({r.time for r in rows}) != len(rows):
                raise ValueError("Duplicate time for a user/item/context")
            for a, b in zip(rows, rows[1:]):
                changes.append(a.choice != b.choice)
                lags.append(b.time-a.time)
        return {"retest_agreement": 1-_mean(changes), "transitions": len(changes),
                "mean_time_gap": _mean(lags), "changed": sum(changes)}


class MoralChangeAnalysis(PreferenceStability):
    def evaluate(self, observations):
        report = super().evaluate(observations)
        return {**report, "interpretation": "Observed disagreement; change versus noise is not identified"}
