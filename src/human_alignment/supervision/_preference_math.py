"""Dependency-free math helpers for preference-distillation supervision."""

from __future__ import annotations

from math import log
from typing import Any, Mapping, Sequence


def compressed_log_ratio_advantages(
    teacher: Sequence[Mapping[str, Any]],
    reference: Sequence[Mapping[str, Any]],
    *,
    epsilon: float = 1e-8,
) -> list[dict[str, list[float] | list[int]]]:
    """Build sparse teacher-reference log-probability margins for ADPA."""

    if len(teacher) != len(reference):
        raise ValueError("Teacher and reference token sequences must have equal lengths")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    output = []
    for teacher_item, reference_item in zip(teacher, reference):
        teacher_map = dict(zip(teacher_item.get("indices", ()), teacher_item.get("values", ())))
        reference_map = dict(
            zip(reference_item.get("indices", ()), reference_item.get("values", ()))
        )
        indices = sorted(set(teacher_map) | set(reference_map))
        margins = [
            log(max(float(teacher_map.get(index, epsilon)), epsilon))
            - log(max(float(reference_map.get(index, epsilon)), epsilon))
            for index in indices
        ]
        order = sorted(range(len(indices)), key=margins.__getitem__, reverse=True)
        output.append(
            {
                "indices": [int(indices[index]) for index in order],
                "values": [margins[index] for index in order],
            }
        )
    return output
