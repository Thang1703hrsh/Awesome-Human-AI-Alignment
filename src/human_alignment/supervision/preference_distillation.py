"""Pure builders for preference-distillation supervision records."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from human_alignment.exceptions import DatasetFormatError


def merge_compressed_probabilities(
    record: Mapping[str, Any],
    chosen: Sequence[Mapping[str, Any]],
    rejected: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Attach DCKD/TVKD token distributions to a pairwise record."""

    if "chosen" not in record or "rejected" not in record:
        raise DatasetFormatError("Compressed pair supervision requires chosen and rejected")
    return {
        **dict(record),
        "teacher_chosen_probs": list(chosen),
        "teacher_rejected_probs": list(rejected),
    }


def attach_generated_response(
    record: Mapping[str, Any],
    response: Any,
    *,
    side: str = "rejected",
) -> dict[str, Any]:
    """Replace one side of a pair while preserving a chat conversation shape."""

    if side not in {"chosen", "rejected"}:
        raise ValueError("side must be 'chosen' or 'rejected'")
    current = record.get(side)
    if (
        isinstance(current, Sequence)
        and not isinstance(current, (str, bytes))
        and current
        and isinstance(current[-1], Mapping)
    ):
        replacement = [*current[:-1], {"role": "assistant", "content": str(response)}]
    else:
        replacement = response
    return {**dict(record), side: replacement}


def build_adpa_record(
    record: Mapping[str, Any],
    teacher: Sequence[Mapping[str, Any]],
    reference: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Attach sparse teacher-reference log-ratio advantages for ADPA."""

    from human_alignment.mechanisms.training.distillation_losses import (
        compressed_log_ratio_advantages,
    )

    return {
        **dict(record),
        "rejected_margin_logp_every": compressed_log_ratio_advantages(
            teacher, reference
        ),
    }


def common_parent_groups(
    student_offsets: Sequence[Sequence[int]],
    teacher_offsets: Sequence[Sequence[int]],
) -> tuple[list[list[int]], list[list[int]]]:
    """Group two tokenizations into spans delimited by shared character boundaries."""

    if not student_offsets or not teacher_offsets:
        raise DatasetFormatError("CTPD requires non-empty response tokenizations")
    end = max(int(student_offsets[-1][1]), int(teacher_offsets[-1][1]))
    student_boundaries = {0, end, *(int(offset[1]) for offset in student_offsets)}
    teacher_boundaries = {0, end, *(int(offset[1]) for offset in teacher_offsets)}
    boundaries = sorted(student_boundaries & teacher_boundaries)
    if boundaries[0] != 0 or boundaries[-1] != end:
        raise DatasetFormatError("Tokenizers do not cover the same response span")

    def groups(offsets: Sequence[Sequence[int]]) -> list[list[int]]:
        result = []
        for start, stop in zip(boundaries, boundaries[1:]):
            indices = [
                index
                for index, (token_start, token_stop) in enumerate(offsets)
                if int(token_stop) > start and int(token_stop) <= stop
            ]
            if not indices:
                raise DatasetFormatError("A shared CTPD span contains no tokens")
            result.append(indices)
        return result

    return groups(student_offsets), groups(teacher_offsets)
