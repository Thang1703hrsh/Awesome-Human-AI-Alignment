"""Feedback providers that acquire signals from different sources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

from human_alignment.specification.targets import AlignmentSpecification
from human_alignment.types import (
    FeedbackRepresentation,
    FeedbackSource,
    InteractionContext,
    SupervisionSignal,
)


class FeedbackProvider(Protocol):
    name: str

    def collect(
        self,
        contexts: Sequence[InteractionContext],
        specification: AlignmentSpecification,
    ) -> list[SupervisionSignal]: ...


@dataclass
class _CallableFeedback:
    callback: Callable[[InteractionContext, AlignmentSpecification], Any]
    representation: FeedbackRepresentation
    source: FeedbackSource
    name: str
    provenance: Mapping[str, Any] | None = None

    def collect(
        self,
        contexts: Sequence[InteractionContext],
        specification: AlignmentSpecification,
    ) -> list[SupervisionSignal]:
        signals = []
        for context in contexts:
            value = self.callback(context, specification)
            if isinstance(value, SupervisionSignal):
                signals.append(value)
            else:
                signals.append(
                    SupervisionSignal(
                        context=context,
                        source=self.source,
                        representation=self.representation,
                        payload=value,
                        provenance={"provider": self.name, **dict(self.provenance or {})},
                    )
                )
        return signals


class HumanFeedback(_CallableFeedback):
    def __init__(
        self,
        callback: Callable[[InteractionContext, AlignmentSpecification], Any] | None = None,
        representation: FeedbackRepresentation = FeedbackRepresentation.PREFERENCE,
        name: str = "human-feedback",
        **legacy: Any,
    ) -> None:
        callback = callback or legacy.pop("collector", None)
        if callback is None:
            raise TypeError("HumanFeedback requires a callback")
        super().__init__(callback, representation, FeedbackSource.HUMAN, name)


class AIFeedback(_CallableFeedback):
    def __init__(
        self,
        callback: Callable[[InteractionContext, AlignmentSpecification], Any] | None = None,
        representation: FeedbackRepresentation = FeedbackRepresentation.CRITIQUE,
        name: str = "ai-feedback",
        model_name: str | None = None,
        **legacy: Any,
    ) -> None:
        callback = callback or legacy.pop("generator", None)
        if callback is None:
            raise TypeError("AIFeedback requires a callback")
        provenance = {"model": model_name} if model_name else {}
        super().__init__(callback, representation, FeedbackSource.AI, name, provenance)
        self.model_name = model_name


class VerifiableFeedback(_CallableFeedback):
    def __init__(
        self,
        callback: Callable[[InteractionContext, AlignmentSpecification], Any] | None = None,
        representation: FeedbackRepresentation = FeedbackRepresentation.VERIFIER_OUTCOME,
        name: str = "verifiable-feedback",
        **legacy: Any,
    ) -> None:
        callback = callback or legacy.pop("verifier", None)
        if callback is None:
            raise TypeError("VerifiableFeedback requires a callback")
        source = (
            FeedbackSource.VERIFIABLE
            if representation == FeedbackRepresentation.VERIFIER_OUTCOME
            else FeedbackSource.PROGRAMMATIC
        )
        super().__init__(callback, representation, source, name)
