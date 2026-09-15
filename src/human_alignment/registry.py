"""Lazy method registry with multi-label taxonomy metadata."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable

from human_alignment.catalog import load_methods
from human_alignment.exceptions import RegistrationError
from human_alignment.types import AlignmentStage


class ImplementationStatus(str, Enum):
    CATALOG = "catalog"
    ADAPTER = "adapter"
    REFERENCE = "reference"
    NATIVE = "native"
    EXTERNAL = "external"


@dataclass(frozen=True)
class MethodMetadata:
    id: str
    name: str
    primary_category: str
    taxonomy_categories: tuple[str, ...]
    stage: AlignmentStage
    implementation_status: ImplementationStatus
    required_extra: str | None = None

    def __post_init__(self) -> None:
        if not self.id or not self.name or not self.primary_category:
            raise ValueError("Method id, name, and primary category are required")
        if self.primary_category not in self.taxonomy_categories:
            raise ValueError("primary_category must appear in taxonomy_categories")


@dataclass(frozen=True)
class RegisteredMethod:
    metadata: MethodMetadata
    factory: Callable[..., Any] | None = None


class MethodRegistry:
    def __init__(self) -> None:
        self._methods: dict[str, RegisteredMethod] = {}

    def register(
        self,
        metadata: MethodMetadata,
        factory: Callable[..., Any] | None = None,
        *,
        replace: bool = False,
    ) -> None:
        if metadata.id in self._methods and not replace:
            raise RegistrationError(f"Method is already registered: {metadata.id}")
        self._methods[metadata.id] = RegisteredMethod(metadata, factory)

    def get(self, method_id: str) -> RegisteredMethod:
        try:
            return self._methods[method_id]
        except KeyError as exc:
            available = ", ".join(sorted(self._methods))
            raise RegistrationError(
                f"Unknown method: {method_id}. Available methods: {available}"
            ) from exc

    def create(self, method_id: str, **parameters: Any) -> Any:
        registered = self.get(method_id)
        if registered.factory is None:
            raise RegistrationError(f"Method has no executable factory: {method_id}")
        return registered.factory(**parameters)

    def list(
        self,
        *,
        stage: AlignmentStage | None = None,
        taxonomy_category: str | None = None,
    ) -> tuple[RegisteredMethod, ...]:
        methods: Iterable[RegisteredMethod] = self._methods.values()
        if stage is not None:
            methods = (item for item in methods if item.metadata.stage == stage)
        if taxonomy_category is not None:
            methods = (
                item
                for item in methods
                if taxonomy_category in item.metadata.taxonomy_categories
            )
        return tuple(sorted(methods, key=lambda item: item.metadata.id))


def default_registry() -> MethodRegistry:
    from human_alignment.assurance import (
        AlignmentPreservation,
        BehavioralEvaluation,
        EvaluatorReliability,
        MonitoringAudit,
        RobustnessEvaluation,
    )
    from human_alignment.mechanisms.inference import (
        BestOfNSearch,
        HumanControl,
        InferenceSteering,
        IterativeRefinement,
    )
    from human_alignment.mechanisms.training import (
        ADPA,
        AlignmentDistillation,
        CTPD,
        DCKD,
        DPO,
        GRPO,
        KTO,
        PPO,
        PPD,
        PreferenceDistillation,
        RewardVerifierModeling,
        SFT,
        SimPO,
        TVKD,
        VPD,
    )

    factories = {
        "dpo": DPO,
        "sft": SFT,
        "kto": KTO,
        "simpo": SimPO,
        "ppo": PPO,
        "grpo": GRPO,
        "reward_verifier_modeling": RewardVerifierModeling,
        "alignment_distillation": AlignmentDistillation,
        "preference_distillation": PreferenceDistillation,
        "dckd": DCKD,
        "tvkd": TVKD,
        "adpa": ADPA,
        "ctpd": CTPD,
        "ppd": PPD,
        "vpd": VPD,
        "best_of_n": BestOfNSearch,
        "inference_steering": InferenceSteering,
        "iterative_refinement": IterativeRefinement,
        "human_control": HumanControl,
        "behavioral_evaluation": BehavioralEvaluation,
        "evaluator_reliability": EvaluatorReliability,
        "robustness_evaluation": RobustnessEvaluation,
        "alignment_preservation": AlignmentPreservation,
        "monitoring_audit": MonitoringAudit,
    }
    extras = {
        "dpo": "dpo",
        "sft": "sft",
        "kto": "trl",
        "simpo": "trl",
        "ppo": "trl",
        "grpo": "trl",
        "preference_distillation": "distillation",
        "dckd": "distillation",
        "tvkd": "distillation",
        "adpa": "distillation",
        "ctpd": "distillation",
        "ppd": "distillation",
        "vpd": "distillation",
    }
    registry = MethodRegistry()
    for row in load_methods():
        categories = tuple(row["taxonomy_categories"])
        metadata = MethodMetadata(
            id=row["id"],
            name=row["name"],
            primary_category=categories[-1],
            taxonomy_categories=categories,
            stage=AlignmentStage(row["stage"]),
            implementation_status=ImplementationStatus(row["implementation_status"]),
            required_extra=extras.get(row["id"]),
        )
        registry.register(metadata, factories.get(row["id"]))
    return registry
