"""Stable results returned by training, inference, assurance, and pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from human_alignment.exceptions import BackendUnavailableError
from human_alignment.types import EvaluationCase, Generation, SupervisionSignal, TaxonomyTrace


@dataclass
class AlignmentRun:
    model: Any
    tokenizer: Any = None
    method: str = "unknown"
    metrics: Mapping[str, float] = field(default_factory=dict)
    output_dir: str | None = None
    taxonomy_trace: TaxonomyTrace = field(default_factory=TaxonomyTrace)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    _generator: Callable[..., str] | None = field(default=None, repr=False)
    _saver: Callable[[str], None] | None = field(default=None, repr=False)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        if self._generator is None:
            raise BackendUnavailableError(
                "This run has no generation backend. Install the inference extra or "
                "construct the run through a bundled model backend."
            )
        return self._generator(prompt, **kwargs)

    def save(self, path: str | None = None) -> Path:
        destination = Path(path or self.output_dir or "outputs/aligned-model")
        if self._saver is not None:
            self._saver(str(destination))
        elif hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(destination)
            if self.tokenizer is not None and hasattr(self.tokenizer, "save_pretrained"):
                self.tokenizer.save_pretrained(destination)
        else:
            raise BackendUnavailableError("The current model does not support save_pretrained")
        self.output_dir = str(destination)
        return destination

    def evaluate(
        self,
        evaluator: Any,
        cases: Sequence[EvaluationCase],
        *,
        baseline_model: Any | None = None,
    ) -> AssuranceReport:
        """Evaluate this loaded or trained model through an assurance method."""

        return evaluator.assess(self.model, cases, baseline_model=baseline_model)


@dataclass(frozen=True)
class InferenceResult:
    generations: tuple[Generation, ...]
    mechanism: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AssuranceReport:
    method: str
    metrics: Mapping[str, float]
    passed: bool
    findings: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineResult:
    model: Any
    supervision: tuple[SupervisionSignal, ...] = ()
    training: tuple[AlignmentRun, ...] = ()
    inference: tuple[InferenceResult, ...] = ()
    assurance: tuple[AssuranceReport, ...] = ()
