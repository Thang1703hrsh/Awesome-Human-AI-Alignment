"""Shared helpers: scorer adaptation, Wilson intervals, and dataset access."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

# A pointwise scorer maps (prompt, response) -> scalar. Objects exposing ``score_batch`` are batched.
PointwiseScorer = Callable[[str, str], float]


def score_pairs(scorer: Any, pairs: Sequence[tuple[str, str]]) -> list[float]:
    """Score ``(prompt, response)`` pairs with ``scorer.score_batch`` when available, else one by one."""
    if hasattr(scorer, "score_batch"):
        values = list(scorer.score_batch(list(pairs)))
    else:
        values = [scorer(prompt, response) for prompt, response in pairs]
    if len(values) != len(pairs):
        raise ValueError(f"scorer returned {len(values)} scores for {len(pairs)} inputs")
    return [float(v) for v in values]


def wilson_interval(successes: float, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson score interval for a proportion (the interval used in the survey's result tables)."""
    if n <= 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z / denom * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)


def load_hf_split(repo_id: str, split: str, config: str | None = None, revision: str | None = None) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised only without the optional extra
        raise ImportError("Loading benchmarks from the Hub requires `pip install datasets`.") from exc
    return [dict(row) for row in load_dataset(repo_id, config, split=split, revision=revision)]


def load_hf_json(repo_id: str, filename: str, revision: str | None = None) -> Any:
    """Read a JSON file from a dataset repository (for repositories that ship loading scripts)."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id, filename, repo_type="dataset", revision=revision)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def take(rows: Iterable[dict], limit: int | None) -> list[dict]:
    rows = list(rows)
    return rows if limit is None else rows[:limit]


def write_report(output_dir: str | Path, name: str, rows: list[dict], summary: dict) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    path = out / f"{name}_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path
