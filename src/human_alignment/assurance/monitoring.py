"""Stateful deployment checks and auditable event records."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

from human_alignment.exceptions import AssuranceError
from human_alignment.results import AssuranceReport
from human_alignment.types import EvaluationCase, as_score, mean


@dataclass(frozen=True)
class AuditEvent:
    timestamp: str
    event: Any
    scores: Mapping[str, float]
    passed: bool


@dataclass
class MonitoringAudit:
    checks: Mapping[str, Callable[[Any], bool | int | float]]
    thresholds: Mapping[str, float] | None = None
    name: str = "monitoring-audit"
    events: list[AuditEvent] = field(default_factory=list, init=False)

    def observe(self, event: Any) -> AuditEvent:
        if not self.checks:
            raise AssuranceError("MonitoringAudit requires at least one check")
        scores = {name: as_score(check(event)) for name, check in self.checks.items()}
        thresholds = self.thresholds or {name: 0.5 for name in scores}
        missing = set(scores) - set(thresholds)
        if missing:
            raise AssuranceError(f"Missing thresholds for: {', '.join(sorted(missing))}")
        record = AuditEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event=event,
            scores=scores,
            passed=all(score >= thresholds[name] for name, score in scores.items()),
        )
        self.events.append(record)
        return record

    def assess(
        self,
        model: Any,
        cases: Sequence[EvaluationCase],
        *,
        baseline_model: Any | None = None,
    ) -> AssuranceReport:
        if cases:
            for case in cases:
                self.observe(case)
        if not self.events:
            raise AssuranceError("MonitoringAudit has no events to summarize")
        dimensions = tuple(self.events[0].scores)
        metrics = {
            f"mean/{dimension}": mean([event.scores[dimension] for event in self.events])
            for dimension in dimensions
        }
        metrics["pass_rate"] = mean([float(event.passed) for event in self.events])
        failures = sum(not event.passed for event in self.events)
        findings = () if not failures else (f"{failures} of {len(self.events)} events failed",)
        return AssuranceReport(
            self.name,
            metrics,
            failures == 0,
            findings,
            {"event_count": len(self.events)},
        )
