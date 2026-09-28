"""Executable AI feedback workflows with injectable model backends.

Backends receive a structured request mapping and return a mapping. Prompts are
original implementation prompts, not verbatim reproductions of paper prompts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from human_alignment.exceptions import DatasetFormatError
from human_alignment.supervision.paper_data import number
from human_alignment.supervision.signals import Demonstration, Preference
from human_alignment.types import (
    FeedbackRepresentation as Representation,
    FeedbackSource as Source,
    InteractionContext,
    SupervisionSignal,
)

Backend = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def _call(backend, task, **values):
    result = backend({"task": task, **values})
    if not isinstance(result, Mapping):
        raise DatasetFormatError(f"{task}: backend must return a mapping")
    return result


def _text(result, key):
    value = result.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DatasetFormatError(f"Backend must return nonempty {key}")
    return value


@dataclass
class LLMJudge:
    backend: Backend
    rubric: str = "Assess correctness, relevance, helpfulness, and safety."
    model_name: str | None = None
    swap_order: bool = True
    name: str = "llm_judge"

    def compare(self, context, responses, specification=None):
        if isinstance(responses, (str, bytes)) or len(responses) != 2:
            raise DatasetFormatError("Pairwise judging requires exactly two responses")
        if responses[0] == responses[1]:
            return None
        judgments = []
        votes = []
        for reverse in ((False, True) if self.swap_order else (False,)):
            options = tuple(reversed(responses)) if reverse else tuple(responses)
            result = _call(self.backend, "pairwise_judge", prompt=context.input,
                           responses=options, rubric=self.rubric,
                           target=specification.resolve(context).target.description
                           if specification else None,
                           instruction="Return winner: A, B, or tie; and rationale.")
            winner = result.get("winner")
            if winner not in ("A", "B", "tie"):
                raise DatasetFormatError("Judge winner must be A, B, or tie")
            votes.append(None if winner == "tie" else (int(winner == "B") ^ reverse))
            judgments.append(dict(result))
        if votes[0] is None or any(v != votes[0] for v in votes):
            return None  # Abstain on tie or position inconsistency.
        return SupervisionSignal(context, Source.AI, Representation.PREFERENCE,
            Preference(tuple(responses), votes[0]),
            provenance={"provider": self.name, "model": self.model_name,
                        "rubric": self.rubric, "judgments": judgments})

    def collect(self, contexts, specification):
        output = []
        for context in contexts:
            signal = self.compare(context, context.metadata["responses"], specification)
            if signal is not None:
                output.append(signal)
        return output


@dataclass
class RLAIF(LLMJudge):
    """AI preference probabilities, averaged over both presentation orders.

    Produces soft labels for reward learning; use pairwise_preferences() explicitly
    if a hard-label trainer is desired. Does not run policy optimization itself.
    """

    name: str = "rlaif"

    def compare(self, context, responses, specification=None):
        if isinstance(responses, (str, bytes)) or len(responses) != 2:
            raise DatasetFormatError("RLAIF requires exactly two responses")
        probabilities, judgments = [], []
        for reverse in ((False, True) if self.swap_order else (False,)):
            result = _call(self.backend, "preference_probability", prompt=context.input,
                           responses=tuple(reversed(responses)) if reverse else tuple(responses),
                           rubric=self.rubric,
                           target=specification.resolve(context).target.description
                           if specification else None,
                           instruction="Return probability_a, normalized over choices A and B.")
            p = number(result.get("probability_a"), "probability_a", 0, 1)
            probabilities.append(1 - p if reverse else p)
            judgments.append(dict(result))
        return SupervisionSignal(context, Source.AI, Representation.EVALUATION,
            {"responses": tuple(responses), "probability_a": sum(probabilities) / len(probabilities)},
            provenance={"provider": self.name, "model": self.model_name,
                        "judgments": judgments, "rubric": self.rubric})

    def pairwise_preferences(self, contexts, specification=None):
        output = []
        for signal in self.collect(contexts, specification):
            p = signal.payload["probability_a"]
            if p == 0.5 or signal.payload["responses"][0] == signal.payload["responses"][1]:
                continue
            output.append(SupervisionSignal(signal.context, Source.AI, Representation.PREFERENCE,
                Preference(signal.payload["responses"], int(p < 0.5)),
                provenance={**signal.provenance, "probability_a": p,
                            "conversion": "hard_label"}))
        return output


@dataclass
class ConstitutionalAI:
    backend: Backend
    principles: Sequence[str]
    model_name: str | None = None
    name: str = "constitutional_ai"

    def __post_init__(self):
        if isinstance(self.principles, str) or not self.principles or any(
            not isinstance(p, str) or not p.strip() for p in self.principles
        ):
            raise ValueError("A nonempty sequence of constitutional principles is required")

    def collect(self, contexts, specification):
        output = []
        for context in contexts:
            response = context.metadata.get("response")
            if response is None:
                response = _text(_call(self.backend, "generate", prompt=context.input), "response")
            history = []
            for principle in self.principles:
                critique = _text(_call(self.backend, "constitutional_critique",
                    prompt=context.input, response=response, principle=principle), "critique")
                revision = _text(_call(self.backend, "constitutional_revision",
                    prompt=context.input, response=response, critique=critique,
                    principle=principle), "response")
                history.append({"response": response, "critique": critique,
                                "revision": revision, "principle": principle})
                response = revision
            output.append(SupervisionSignal(context, Source.AI, Representation.DEMONSTRATION,
                Demonstration(response), provenance={"provider": self.name,
                "model": self.model_name, "revisions": history}))
        return output

    def preferences(self, contexts, specification=None):
        """RL-CAI comparison stage; does not assume a revision is always better."""
        return LLMJudge(self.backend, rubric="\n".join(self.principles),
                        model_name=self.model_name, name=self.name).collect(contexts, specification)


def _rouge_l(a: str, b: str) -> float:
    a, b = re.findall(r"\w+", a.lower()), re.findall(r"\w+", b.lower())
    if not a or not b:
        return 0.0
    previous = [0] * (len(b) + 1)
    for token in a:
        current = [0]
        for j, other in enumerate(b):
            current.append(previous[j] + 1 if token == other else
                           max(previous[j + 1], current[-1]))
        previous = current
    return 2 * previous[-1] / (len(a) + len(b))


@dataclass
class SelfInstruct:
    backend: Backend
    seeds: Sequence[str]
    similarity_threshold: float = 0.7
    model_name: str | None = None
    name: str = "self_instruct"

    def __post_init__(self):
        if isinstance(self.seeds, str) or not self.seeds or any(
            not isinstance(s, str) or not s.strip() for s in self.seeds
        ):
            raise ValueError("Self-Instruct requires seed instructions")
        number(self.similarity_threshold, "similarity_threshold", 0, 1)

    def generate(self, rounds: int = 1):
        if type(rounds) is not int or rounds < 1:
            raise ValueError("rounds must be a positive integer")
        pool, output = list(self.seeds), []
        for _ in range(rounds):
            proposal = _call(self.backend, "generate_instructions", seeds=tuple(pool))
            instructions = proposal.get("instructions")
            if not isinstance(instructions, (list, tuple)):
                raise DatasetFormatError("instructions must be a sequence of strings")
            for instruction in instructions:
                if not isinstance(instruction, str) or not instruction.strip():
                    continue
                if any(_rouge_l(instruction, seed) >= self.similarity_threshold for seed in pool):
                    continue
                classification = _call(self.backend, "classify_instruction", instruction=instruction)
                if type(classification.get("is_classification")) is not bool:
                    raise DatasetFormatError("is_classification must be boolean")
                instances = _call(self.backend, "generate_instances", instruction=instruction,
                                  is_classification=classification["is_classification"]).get("instances")
                if not isinstance(instances, (list, tuple)):
                    raise DatasetFormatError("instances must be a sequence")
                accepted = set()
                for instance in instances:
                    if not isinstance(instance, Mapping):
                        raise DatasetFormatError("Each instance must be a mapping")
                    completion = _text(instance, "output")
                    text_input = instance.get("input", "")
                    if not isinstance(text_input, str):
                        raise DatasetFormatError("Instance input must be a string")
                    key = (text_input, completion)
                    if key in accepted:
                        continue
                    accepted.add(key)
                    prompt = instruction + ("\n\n" + text_input if text_input else "")
                    output.append(SupervisionSignal(InteractionContext(prompt), Source.AI,
                        Representation.DEMONSTRATION, Demonstration(completion),
                        provenance={"provider": self.name, "model": self.model_name,
                                    "instruction": instruction, **dict(classification)}))
                if accepted:
                    pool.append(instruction)
        return output

    def collect(self, contexts, specification):
        if contexts:
            raise DatasetFormatError("Self-Instruct uses seeds; pass contexts=()")
        return self.generate()


@dataclass
class GEval:
    backend: Backend
    criteria: str
    min_score: int = 1
    max_score: int = 5
    model_name: str | None = None
    name: str = "g_eval"

    def __post_init__(self):
        if not self.criteria.strip() or type(self.min_score) is not int or \
                type(self.max_score) is not int or self.min_score >= self.max_score:
            raise ValueError("G-Eval needs criteria and an increasing integer score range")

    def collect(self, contexts, specification):
        steps = _text(_call(self.backend, "evaluation_steps", criteria=self.criteria), "steps")
        output = []
        for context in contexts:
            result = _call(self.backend, "score_distribution", prompt=context.input,
                           response=context.metadata["response"],
                           reference=context.metadata.get("reference"),
                           criteria=self.criteria, steps=steps,
                           min_score=self.min_score, max_score=self.max_score)
            distribution = result.get("score_probabilities")
            if not isinstance(distribution, Mapping) or not distribution:
                raise DatasetFormatError("G-Eval requires score_probabilities, not a scalar score")
            weighted, mass = 0.0, 0.0
            for score, probability in distribution.items():
                score = number(score, "score", self.min_score, self.max_score)
                if not score.is_integer():
                    raise DatasetFormatError("Score categories must be integers")
                probability = number(probability, "probability", 0, 1)
                weighted += score * probability
                mass += probability
            if mass <= 0 or mass > 1 + 1e-6:
                raise DatasetFormatError("Score probability mass must be in (0, 1]")
            output.append(SupervisionSignal(context, Source.AI, Representation.EVALUATION,
                {"score": weighted / mass, "score_probabilities": dict(distribution)},
                provenance={"provider": self.name, "model": self.model_name,
                            "criteria": self.criteria, "steps": steps,
                            "retained_probability_mass": mass}))
        return output


@dataclass
class SelfRewarding:
    """Iterative candidate generation, self-scoring, and preference training.

    backend_factory(model) must use THAT model for both generation and judging.
    train_round(model, signals, iteration) returns the updated model.
    """

    backend_factory: Callable[[Any], Backend]
    train_round: Callable[[Any, Sequence[SupervisionSignal], int], Any]
    num_candidates: int = 4
    name: str = "self_rewarding"

    def __post_init__(self):
        if type(self.num_candidates) is not int or self.num_candidates < 2:
            raise ValueError("num_candidates must be an integer >= 2")

    def run(self, model, contexts, *, rounds=1):
        if type(rounds) is not int or rounds < 1:
            raise ValueError("rounds must be a positive integer")
        contexts = tuple(contexts)
        history = []
        for iteration in range(rounds):
            backend = self.backend_factory(model)
            signals = []
            for context in contexts:
                responses = _call(backend, "generate_candidates", prompt=context.input,
                                  count=self.num_candidates).get("responses")
                if not isinstance(responses, (list, tuple)) or len(responses) != self.num_candidates:
                    raise DatasetFormatError("Candidate count does not match num_candidates")
                scores = []
                for response in responses:
                    if not isinstance(response, str) or not response.strip():
                        raise DatasetFormatError("Candidates must be nonempty strings")
                    result = _call(backend, "self_reward", prompt=context.input, response=response,
                                   instruction="Return a quality score from 0 to 5.")
                    scores.append(number(result.get("score"), "self reward", 0, 5))
                high = max(range(len(scores)), key=scores.__getitem__)
                low = min(range(len(scores)), key=scores.__getitem__)
                if scores[high] == scores[low] or responses[high] == responses[low]:
                    continue
                signals.append(SupervisionSignal(context, Source.AI, Representation.PREFERENCE,
                    Preference((responses[high], responses[low]), 0),
                    provenance={"provider": self.name, "iteration": iteration,
                                "candidate_scores": scores}))
            if not signals:
                raise DatasetFormatError("Self-rewarding round produced no strict preferences")
            model = self.train_round(model, signals, iteration)
            if model is None:
                raise ValueError("train_round must return the updated model")
            history.append(tuple(signals))
        return model, tuple(history)
