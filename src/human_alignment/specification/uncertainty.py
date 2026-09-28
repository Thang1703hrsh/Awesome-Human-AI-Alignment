"""Uncertain targets and paper-specific preference acquisition."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Callable

from human_alignment.specification.targets import AlignmentSpecification, ResolvedTarget
from human_alignment.types import AlignmentTarget, InteractionContext


def _finite(value):
    result = float(value)
    if isinstance(value, bool) or not math.isfinite(result):
        raise ValueError("Expected a finite number")
    return result


@dataclass
class UncertaintySpecification:
    target: AlignmentTarget
    estimator: Callable[[InteractionContext], float]
    evidence_kind: str = "user_supplied"

    def resolve(self, context):
        uncertainty = _finite(self.estimator(context))
        if not 0 <= uncertainty <= 1:
            raise ValueError("Target uncertainty must be in [0, 1]")
        base = AlignmentSpecification(self.target).resolve(context)
        return ResolvedTarget(replace(base.target, uncertainty=uncertainty), context,
                              {**base.applied_rules, "uncertainty_evidence": self.evidence_kind})


@dataclass
class ActivePreferenceLearning:
    """Muldrew et al.: select HIGH implicit reward gaps, optionally after entropy filtering."""

    beta: float = 0.1

    def __post_init__(self):
        if _finite(self.beta) <= 0:
            raise ValueError("beta must be positive")

    def select(self, records, count, *, entropy_pool_size=None):
        records = list(records)
        if type(count) is not int or not 0 < count <= len(records):
            raise ValueError("count must lie in [1, pool size]")
        candidates = list(enumerate(records))
        if entropy_pool_size is not None:
            if type(entropy_pool_size) is not int or not count <= entropy_pool_size <= len(records):
                raise ValueError("Entropy pool size must be between count and pool size")
            def entropy(row):
                values = [_finite(v) for v in row["sample_logps"]]
                if not values or any(v > 0 for v in values):
                    raise ValueError("Entropy requires nonempty sampled sequence log probabilities <= 0")
                return -sum(values)/len(values)
            candidates.sort(key=lambda pair: (-entropy(pair[1]), pair[0]))
            candidates = candidates[:entropy_pool_size]
        def certainty(row):
            policy, reference = row["policy_logps"], row["reference_logps"]
            if len(policy) != 2 or len(reference) != 2:
                raise ValueError("Two policy and reference sequence log probabilities required")
            values = [_finite(v) for v in (*policy, *reference)]
            if any(v > 0 for v in values):
                raise ValueError("Log probabilities cannot be positive")
            a, b, ar, br = values
            return self.beta * abs((a-ar)-(b-br))
        candidates.sort(key=lambda pair: (-certainty(pair[1]), pair[0]))
        return [index for index, _ in candidates[:count]]


@dataclass
class PILAF:
    """Token-wise PILAF sampling, Algorithm 1 / Section 5 (not T-PILAF).

    policy(prefix) and reference(prefix) return logits over the SAME vocabulary.
    Model/KV-cache batching is left to the backend. A pair is base/base with
    probability 1/2, otherwise plus/minus. No preference labels are fabricated.
    """

    beta: float = 0.1
    seed: int = 0

    def __post_init__(self):
        if _finite(self.beta) <= 0:
            raise ValueError("beta must be positive")
        self._rng = random.Random(self.seed)

    def logits(self, policy_logits, reference_logits, direction):
        if direction not in (-1, 0, 1) or len(policy_logits) != len(reference_logits):
            raise ValueError("Matching vocabularies and direction -1/0/1 are required")
        if not policy_logits:
            raise ValueError("Empty vocabulary")
        return [(1+direction*self.beta)*_finite(p)-direction*self.beta*_finite(r)
                for p, r in zip(policy_logits, reference_logits)]

    def sample_pair(self, prompt_tokens, policy, reference, *, max_new_tokens=64, eos_id=None):
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        directions = (0, 0) if self._rng.random() < 0.5 else (1, -1)
        outputs = []
        for direction in directions:
            prefix, response = list(prompt_tokens), []
            for _ in range(max_new_tokens):
                current = list(policy(tuple(prefix)))
                reference_values = list(reference(tuple(prefix))) if direction else current
                logits = self.logits(current, reference_values, direction)
                top = max(logits)
                weights = [math.exp(v-top) for v in logits]
                token = self._rng.choices(range(len(logits)), weights=weights, k=1)[0]
                prefix.append(token)
                response.append(token)
                if token == eos_id:
                    break
            outputs.append(tuple(response))
        return {"responses": tuple(outputs), "directions": directions}
