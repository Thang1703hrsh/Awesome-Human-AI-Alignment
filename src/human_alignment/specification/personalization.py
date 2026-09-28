"""User-scoped targets and model-backed personalization workflows."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Sequence

from human_alignment.specification.targets import AlignmentSpecification, ResolvedTarget
from human_alignment.types import AlignmentTarget, InteractionContext


@dataclass(frozen=True)
class UserProfile:
    user_id: str
    preferences: tuple[str, ...] = ()
    version: int = 1
    evidence: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.user_id, str) or not self.user_id.strip():
            raise ValueError("A nonempty user_id is required")
        if type(self.version) is not int or self.version < 1:
            raise ValueError("Profile version must be a positive integer")
        if isinstance(self.preferences, str) or any(
            not isinstance(p, str) or not p.strip() for p in self.preferences
        ):
            raise ValueError("Preferences must be nonempty strings")


@dataclass
class PersonalizedSpecification:
    target: AlignmentTarget
    profiles: Mapping[str, UserProfile]

    def __post_init__(self):
        if any(key != profile.user_id for key, profile in self.profiles.items()):
            raise ValueError("Profile mapping keys must match user_id")

    def resolve(self, context: InteractionContext) -> ResolvedTarget:
        user = context.metadata.get("user_id")
        if user not in self.profiles:
            raise KeyError(f"No explicit profile for user {user!r}")
        profile = self.profiles[user]
        base = AlignmentSpecification(self.target).resolve(context)
        target = replace(base.target,
            constraints=base.target.constraints + profile.preferences,
            stakeholders=tuple(dict.fromkeys((*base.target.stakeholders, user))),
            version=f"{base.target.version}/user:{user}:{profile.version}")
        return ResolvedTarget(target, context,
                              {**base.applied_rules, "profile": profile})


@dataclass
class OPPU:
    """One training invocation per user's history, on an independent PEFT model.

    model_factory(user_id) must return a fresh base+adapter model. trainer(model,
    records, user_id) returns the trained adapter/model. No cross-user pooling.
    """

    model_factory: Callable[[str], Any]
    trainer: Callable[[Any, Sequence[Mapping[str, Any]], str], Any]

    def fit_users(self, histories: Mapping[str, Sequence[Mapping[str, Any]]]):
        if not histories or any(not user or not records for user, records in histories.items()):
            raise ValueError("Each user requires a nonempty training history")
        models, results = [], {}
        for user, records in histories.items():
            if any(row.get("user_id", user) != user for row in records):
                raise ValueError("Training records contain another user's data")
            model = self.model_factory(user)
            if model is None or any(model is prior for prior in models):
                raise ValueError("model_factory must return a fresh model for each user")
            models.append(model)  # Retain references to reliably detect shared objects.
            result = self.trainer(model, tuple(dict(row) for row in records), user)
            if result is None:
                raise ValueError("trainer must return the user's trained adapter/model")
            results[user] = result
        return results


@dataclass
class PROSE:
    """Infer, verify across writing samples, then iteratively refine preferences.

    Reference orchestration; caller supplies the LLM and task-specific prompts.
    Backend tasks: infer_preferences, verify_preferences, refine_preferences.
    """

    backend: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    rounds: int = 2

    def __post_init__(self):
        if type(self.rounds) is not int or self.rounds < 1:
            raise ValueError("rounds must be a positive integer")

    def infer(self, user_id: str, writing_samples: Sequence[str]):
        if isinstance(writing_samples, str) or len(writing_samples) < 2 or any(
            not isinstance(s, str) or not s.strip() for s in writing_samples
        ):
            raise ValueError("PROSE requires at least two nonempty writing samples")
        samples = tuple(writing_samples)
        result = self.backend({"task": "infer_preferences", "samples": samples})
        preferences = _preferences(result)
        evidence = []
        for iteration in range(self.rounds):
            checks = [self.backend({"task": "verify_preferences", "sample": sample,
                                   "preferences": preferences}) for sample in samples]
            if any(not isinstance(c, Mapping) or type(c.get("supported")) is not bool
                   or not isinstance(c.get("critique"), str) for c in checks):
                raise ValueError("Verification requires boolean supported and text critique")
            evidence.extend(c["critique"] for c in checks)
            if all(c["supported"] for c in checks):
                return UserProfile(user_id, preferences, evidence=tuple(evidence))
            if iteration + 1 < self.rounds:
                preferences = _preferences(self.backend({"task": "refine_preferences",
                    "samples": samples, "preferences": preferences, "checks": checks}))
        raise ValueError("Inferred preferences failed cross-sample verification")


def _preferences(result):
    if not isinstance(result, Mapping):
        raise ValueError("Preference backend must return a mapping")
    values = result.get("preferences")
    if not isinstance(values, (list, tuple)) or not values or any(
        not isinstance(p, str) or not p.strip() for p in values
    ):
        raise ValueError("Backend must return a nonempty preferences list")
    return tuple(dict.fromkeys(values))


@dataclass
class InteractionAlignment:
    """Reference inference loop for user-isolated conversational adaptation.

    This explicit profile loop is not the paper's trained implicit meta-skill.
    backend(request) returns response and preferences; history never crosses users.
    """

    backend: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    profiles: dict[str, UserProfile] = field(default_factory=dict)
    histories: dict[str, tuple[Mapping[str, str], ...]] = field(default_factory=dict)

    def respond(self, user_id: str, message: str):
        if not isinstance(message, str) or not message.strip():
            raise ValueError("Nonempty message required")
        profile = self.profiles.get(user_id, UserProfile(user_id))
        history = self.histories.get(user_id, ()) + ({"role": "user", "content": message},)
        result = self.backend({"task": "interact_to_align", "history": history,
                               "preferences": profile.preferences})
        preferences = _preferences(result)
        response = result.get("response")
        if not isinstance(response, str) or not response.strip():
            raise ValueError("Backend must return nonempty response")
        self.profiles[user_id] = UserProfile(user_id, preferences, profile.version + 1,
                                            profile.evidence + (message,))
        self.histories[user_id] = history + ({"role": "assistant", "content": response},)
        return response
