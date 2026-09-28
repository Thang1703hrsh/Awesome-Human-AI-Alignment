"""Preflight checks for reproducible local and H100 runs."""

from __future__ import annotations

import importlib.metadata
import json
import shutil
import sys
from pathlib import Path


def collect_diagnostics(project_root: str | Path = ".") -> dict:
    root = Path(project_root).resolve()
    packages = {}
    for name in (
        "torch",
        "transformers",
        "peft",
        "datasets",
        "accelerate",
        "deepspeed",
        "numpy",
        "scipy",
        "PyYAML",
        "safetensors",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        import torch

        cuda = {
            "available": torch.cuda.is_available(),
            "version": torch.version.cuda,
            "devices": [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ],
        }
    except ImportError:
        cuda = {"available": False, "version": None, "devices": []}
    required = [
        root / "recipes/safety_alignment/methods",
        root / "src/human_alignment/safety",
        root / "tests",
    ]
    return {
        "python": sys.version.split()[0],
        "project_root": str(root),
        "packages": packages,
        "cuda": cuda,
        "commands": {
            name: shutil.which(name) for name in ("git", "nvcc", "nvidia-smi")
        },
        "paths": {str(path.relative_to(root)): path.exists() for path in required},
        "warnings": _warnings(root, packages, cuda),
    }


def _warnings(root: Path, packages: dict, cuda: dict) -> list[str]:
    warnings = []
    missing = [
        name
        for name, version in packages.items()
        if version is None and name != "deepspeed"
    ]
    if missing:
        warnings.append(f"missing required packages: {', '.join(missing)}")
    if not cuda["available"]:
        warnings.append("CUDA is unavailable: unit tests work, 7B training does not")
    if packages.get("deepspeed") is not None and shutil.which("nvcc") is None:
        warnings.append(
            "nvcc is unavailable: use zero2_offload_torch_adam.json or install a CUDA toolkit before "
            "using the default DeepSpeed CPU-Adam config"
        )
    return warnings


def diagnostics_text(report: dict) -> str:
    return json.dumps(report, indent=2)

