"""Dataset loading. Every loader returns plain row dicts in the PKU-SafeRLHF schema (or the method's own schema)."""

from __future__ import annotations

import json
from pathlib import Path

PKU_COLUMNS = (
    "prompt",
    "response_0",
    "response_1",
    "is_response_0_safe",
    "is_response_1_safe",
    "better_response_id",
    "safer_response_id",
)

# BFPO pins this revision of PKU-Alignment/PKU-SafeRLHF (bfpo/src/alignment/data.py:203-204).
BFPO_PKU_REVISION = "ff7ba91063016c78a225b0f74e1c0860bb18230f"


def load_pku(
    name: str = "PKU-Alignment/PKU-SafeRLHF-30K",
    split: str = "train",
    revision: str | None = None,
    fraction: float | None = None,
    shuffle_seed: int | None = None,
    limit: int | None = None,
    holdout: float | None = None,
    holdout_seed: int | None = None,
) -> list[dict]:
    """``holdout``: keep the ``train`` part of ``train_test_split(test_size=holdout, seed=holdout_seed)``
    (MODPO seed=0, CAN unseeded). ``fraction`` takes the first ``int(fraction * n)`` rows *before* shuffling
    (BFPO ``get_datasets``)."""
    ds = _load_hf_or_json(name, split, revision)
    if holdout is not None:
        ds = ds.train_test_split(test_size=holdout, seed=holdout_seed)["train"]
    if fraction is not None:
        ds = ds.select(range(int(fraction * len(ds))))
    if shuffle_seed is not None:
        ds = ds.shuffle(seed=shuffle_seed)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    missing = [c for c in PKU_COLUMNS if c not in ds.column_names]
    if missing:
        raise ValueError(f"{name} is missing PKU-SafeRLHF columns {missing}")
    return [{c: row[c] for c in PKU_COLUMNS} for row in ds]


def _load_hf_or_json(name: str, split: str, revision: str | None = None):
    from datasets import load_dataset

    if Path(name).suffix in (".json", ".jsonl"):
        return load_dataset("json", data_files=name, split="train")
    return load_dataset(name, split=split, revision=revision)


def unique_prompts(rows: list[dict]) -> list[str]:
    """safe-rlhf ``PromptOnlyDataset``: keep the first occurrence of each prompt, in order."""
    seen, out = set(), []
    for r in rows:
        if r["prompt"] not in seen:
            seen.add(r["prompt"])
            out.append(r["prompt"])
    return out


def load_alpaca(limit: int | None = None, path: str = "tatsu-lab/alpaca") -> list[dict]:
    """safe-rlhf ``alpaca`` raw dataset (``datasets/raw/alpaca.py``): input = instruction [+ ' ' + input]."""
    ds = _load_hf_or_json(path, "train")
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    rows = []
    for d in ds:
        prompt = (
            " ".join((d["instruction"], d["input"])) if d["input"] else d["instruction"]
        )
        rows.append({"prompt": prompt, "answer": d["output"]})
    return rows


def load_ultrafeedback_pairs(
    split: str = "train_prefs",
    shuffle_seed: int | None = 42,
    limit: int | None = None,
    path: str = "HuggingFaceH4/ultrafeedback_binarized",
) -> list[dict]:
    """``HuggingFaceH4/ultrafeedback_binarized`` as single-turn pairs with safe=1/1 (BFPO buffer)."""
    ds = _load_hf_or_json(path, split)
    if shuffle_seed is not None:
        ds = ds.shuffle(seed=shuffle_seed)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return [
        {
            "prompt": d["prompt"],
            "chosen": d["chosen"][-1]["content"],
            "rejected": d["rejected"][-1]["content"],
            "chosen_safe": True,
            "rejected_safe": True,
        }
        for d in ds
    ]


def load_json(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.suffix == ".jsonl":
        return [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
    return json.loads(path.read_text())

