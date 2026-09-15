"""Load and validate packaged taxonomy metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from typing import Any, Mapping


EXPECTED_DIMENSIONS = {
    "alignment_specification",
    "alignment_supervision",
    "alignment_mechanisms",
    "alignment_assurance",
}


@dataclass(frozen=True)
class CatalogValidation:
    valid: bool
    errors: tuple[str, ...]
    dimension_count: int
    category_count: int


def _load(name: str) -> Any:
    resource = files("human_alignment.catalog").joinpath(f"data/{name}")
    return json.loads(resource.read_text(encoding="utf-8"))


def load_taxonomy() -> dict[str, Any]:
    return _load("taxonomy.json")


def load_methods() -> list[dict[str, Any]]:
    return _load("methods.json")


def categories(taxonomy: Mapping[str, Any] | None = None) -> tuple[dict[str, str], ...]:
    taxonomy = taxonomy or load_taxonomy()
    output = []
    for dimension in taxonomy.get("dimensions", []):
        for group in dimension.get("groups", []):
            for category in group.get("categories", []):
                output.append(
                    {
                        "id": category["id"],
                        "name": category["name"],
                        "group": group["name"],
                        "dimension": dimension["id"],
                    }
                )
    return tuple(output)


def validate_catalog() -> CatalogValidation:
    taxonomy = load_taxonomy()
    methods = load_methods()
    errors = []
    dimensions = taxonomy.get("dimensions", [])
    dimension_ids = [item.get("id") for item in dimensions]
    if set(dimension_ids) != EXPECTED_DIMENSIONS or len(dimension_ids) != 4:
        errors.append("Taxonomy must contain exactly the four lifecycle dimensions")
    if len(dimension_ids) != len(set(dimension_ids)):
        errors.append("Dimension IDs must be unique")

    category_rows = categories(taxonomy)
    category_ids = [item["id"] for item in category_rows]
    if len(category_ids) != 23:
        errors.append(f"Expected 23 terminal categories, found {len(category_ids)}")
    if len(category_ids) != len(set(category_ids)):
        errors.append("Terminal category IDs must be unique")

    required = {"id", "name", "taxonomy_categories", "stage", "implementation_status"}
    allowed_stages = {"supervision", "training", "inference", "assurance"}
    allowed_statuses = {"catalog", "adapter", "reference", "native", "external"}
    method_ids = [str(item.get("id", "")) for item in methods]
    if len(method_ids) != len(set(method_ids)):
        errors.append("Method IDs must be unique")
    known_categories = set(category_ids)
    for method in methods:
        missing = required - set(method)
        if missing:
            errors.append(f"Method {method.get('id', '<missing>')} lacks {sorted(missing)}")
        if method.get("stage") not in allowed_stages:
            errors.append(f"Method {method.get('id', '<missing>')} has an invalid stage")
        if method.get("implementation_status") not in allowed_statuses:
            errors.append(f"Method {method.get('id', '<missing>')} has an invalid status")
        unknown = set(method.get("taxonomy_categories", [])) - known_categories
        if unknown:
            errors.append(f"Method {method.get('id', '<missing>')} uses {sorted(unknown)}")
    return CatalogValidation(
        not errors,
        tuple(errors),
        len(dimensions),
        len(category_rows),
    )
