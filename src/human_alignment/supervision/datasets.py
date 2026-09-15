"""Dependency-free normalization of common alignment dataset formats."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from human_alignment.exceptions import DatasetFormatError
from human_alignment.supervision.signals import Demonstration, Preference
from human_alignment.types import SupervisionSignal


@dataclass(frozen=True)
class PreferenceExample:
    prompt: Any
    chosen: Any
    rejected: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InstructionExample:
    prompt: Any
    completion: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KTOExample:
    prompt: Any
    completion: Any
    label: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptExample:
    prompt: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreferenceDistillationExample:
    """A prompt, candidate responses, and optional teacher preference scores."""

    prompt: Any
    responses: tuple[Any, ...]
    teacher_scores: tuple[float, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.responses) < 2:
            raise DatasetFormatError("Preference distillation requires at least two responses")
        if self.teacher_scores and len(self.teacher_scores) != len(self.responses):
            raise DatasetFormatError("teacher_scores must match the number of responses")


def collect_preference_distillation_data(
    prompts: Iterable[Any],
    generator: Callable[[Any, int], Sequence[Any]],
    *,
    num_responses: int = 4,
    scorer: Callable[[Any, tuple[Any, ...]], Sequence[float]] | None = None,
) -> list[PreferenceDistillationExample]:
    """Collect multi-response supervision with an optional teacher scorer."""

    if num_responses < 2:
        raise DatasetFormatError("num_responses must be at least 2")
    examples = []
    for prompt in prompts:
        responses = tuple(generator(prompt, num_responses))
        if len(responses) != num_responses:
            raise DatasetFormatError(
                f"Generator returned {len(responses)} responses; expected {num_responses}"
            )
        scores = tuple(float(value) for value in scorer(prompt, responses)) if scorer else ()
        examples.append(PreferenceDistillationExample(prompt, responses, scores))
    return examples


@dataclass(frozen=True)
class DatasetBundle:
    data: Any
    kind: str
    origin: str


REQUIRED_FIELDS = {
    "preference": ("prompt", "chosen", "rejected"),
    "instruction": ("prompt", "completion"),
    "kto": ("prompt", "completion", "label"),
    "prompt": ("prompt",),
    "preference_distillation": (),
}


def _read_local(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise DatasetFormatError(
                        f"Invalid JSON on line {line_number} of {path}"
                    ) from exc
        return rows
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise DatasetFormatError("A JSON dataset must contain a top-level list")
        return value
    raise DatasetFormatError("Local datasets must use .json or .jsonl")


def _signal_record(signal: SupervisionSignal, kind: str) -> dict[str, Any]:
    payload = signal.payload
    if kind == "preference" and isinstance(payload, Preference):
        if len(payload.options) != 2:
            raise DatasetFormatError("Pairwise optimization requires exactly two options")
        rejected_index = 1 - payload.preferred_index
        return {
            "prompt": signal.context.input,
            "chosen": payload.options[payload.preferred_index],
            "rejected": payload.options[rejected_index],
        }
    if kind == "instruction" and isinstance(payload, Demonstration):
        return {"prompt": signal.context.input, "completion": payload.output}
    raise DatasetFormatError(
        f"Cannot convert {type(payload).__name__} supervision to a {kind} dataset"
    )


def _records(values: Iterable[Any], kind: str) -> list[dict[str, Any]]:
    records = []
    for value in values:
        if isinstance(value, SupervisionSignal):
            record = _signal_record(value, kind)
        elif is_dataclass(value):
            record = asdict(value)
        elif isinstance(value, Mapping):
            record = dict(value)
        else:
            raise DatasetFormatError(f"Unsupported dataset row: {type(value).__name__}")
        records.append(record)
    return records


def _validate(records: Sequence[Mapping[str, Any]], kind: str) -> None:
    required = REQUIRED_FIELDS.get(kind)
    if required is None:
        raise DatasetFormatError(f"Unknown dataset kind: {kind}")
    if not records:
        raise DatasetFormatError("The dataset is empty")
    for index, record in enumerate(records):
        if kind == "preference_distillation":
            has_ranked_responses = "prompt" in record and "responses" in record
            has_pair = "chosen" in record and "rejected" in record
            if not (has_ranked_responses or has_pair):
                raise DatasetFormatError(
                    f"Dataset row {index} must contain prompt/responses or chosen/rejected"
                )
            responses = record.get("responses")
            if responses is not None and (
                not isinstance(responses, Sequence)
                or isinstance(responses, (str, bytes))
                or len(responses) < 2
            ):
                raise DatasetFormatError(
                    f"Dataset row {index} requires at least two responses"
                )
            scores = record.get("teacher_scores")
            if scores is not None and len(scores) > 0:
                expected_score_count = len(responses) if responses is not None else 2
                if len(scores) != expected_score_count:
                    raise DatasetFormatError(
                        f"Dataset row {index} has mismatched responses and teacher_scores"
                    )
            continue
        missing = [field for field in required if field not in record]
        if missing:
            raise DatasetFormatError(
                f"Dataset row {index} is missing fields: {', '.join(missing)}"
            )


def prepare_dataset(source: Any, kind: str) -> DatasetBundle:
    """Normalize local data while deferring remote Hugging Face loading to TRL."""

    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.exists():
            if path.is_dir():
                return DatasetBundle(str(path), kind, str(path))
            records = _read_local(path)
            _validate(records, kind)
            return DatasetBundle(records, kind, str(path))
        if isinstance(source, str):
            if path.suffix.lower() in {".json", ".jsonl"}:
                raise DatasetFormatError(f"Dataset path does not exist: {path}")
            return DatasetBundle(source, kind, f"huggingface:{source}")
        raise DatasetFormatError(f"Dataset path does not exist: {path}")
    if hasattr(source, "column_names"):
        return DatasetBundle(source, kind, type(source).__name__)
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        records = _records(source, kind)
        _validate(records, kind)
        return DatasetBundle(records, kind, "memory")
    raise DatasetFormatError(f"Unsupported dataset source: {type(source).__name__}")
