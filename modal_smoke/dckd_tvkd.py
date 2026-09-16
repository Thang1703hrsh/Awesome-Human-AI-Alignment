"""Isolated one-step DCKD and TVKD smoke tests on Modal.

Run from the repository root with one of:

    modal run modal_smoke/dckd_tvkd.py --method dckd
    modal run modal_smoke/dckd_tvkd.py --method tvkd
    modal run modal_smoke/dckd_tvkd.py --method all

The local repository is copied into the Modal image. All generated data and
checkpoints live in an ephemeral remote temporary directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal


APP_NAME = "human-alignment-distillation-smoke"
MODEL_ID = "sshleifer/tiny-gpt2"
REMOTE_SOURCE = "/opt/human-alignment/src"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "accelerate==1.2.1",
        "datasets==3.2.0",
        "safetensors==0.4.5",
        "torch==2.5.1",
        "transformers==4.47.1",
    )
    .add_local_dir(
        REPOSITORY_ROOT / "src",
        REMOTE_SOURCE,
        copy=True,
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
    .env(
        {
            "PYTHONPATH": REMOTE_SOURCE,
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)

app = modal.App(APP_NAME, image=image)


def _training_config(objective: str, output_dir: Path):
    from human_alignment import PreferenceDistillationConfig

    parameters = {
        "objective": objective,
        "output_dir": str(output_dir),
        "learning_rate": 1e-5,
        "epochs": 1.0,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "max_length": 64,
        "max_prompt_length": 32,
        "report_to": "none",
        "extra_args": {
            "max_steps": 1,
            "save_strategy": "no",
            "logging_steps": 1,
            "disable_tqdm": True,
            "dataloader_num_workers": 0,
        },
    }
    if objective == "dckd":
        parameters.update(
            chosen_kd_weight=0.1,
            rejected_kd_weight=0.1,
            sft_weight=0.0,
        )
    else:
        parameters.update(
            chosen_kd_weight=0.0001,
            rejected_kd_weight=0.0001,
            value_function="entropy",
            value_discount=1.0,
        )
    return PreferenceDistillationConfig(**parameters)


def _release_cuda() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@app.function(gpu="T4", cpu=0.5, memory=2_048, timeout=20 * 60)
def run_smoke(method: str = "all") -> dict[str, object]:
    """Prepare two rows and execute one DCKD and/or TVKD optimizer step."""

    import tempfile

    import torch

    from human_alignment import DCKD, TVKD, prepare_preference_distillation_dataset

    if method not in {"dckd", "tvkd", "all"}:
        raise ValueError("method must be one of: dckd, tvkd, all")
    if not torch.cuda.is_available():
        raise RuntimeError("Modal did not attach the requested GPU")

    torch.manual_seed(0)
    examples = [
        {
            "prompt": "Question: What is alignment? Answer:",
            "chosen": " Matching model behavior to intended goals.",
            "rejected": " Making every model response longer.",
        },
        {
            "prompt": "Question: How should uncertainty be handled? Answer:",
            "chosen": " State uncertainty clearly.",
            "rejected": " Always sound certain.",
        },
    ]

    results: dict[str, object] = {
        "cuda_device": torch.cuda.get_device_name(0),
        "model": MODEL_ID,
        "requested_method": method,
    }

    with tempfile.TemporaryDirectory(prefix="hai-modal-smoke-") as temporary:
        root = Path(temporary)
        prepared_path = root / "prepared-dckd-tvkd"
        prepared = prepare_preference_distillation_dataset(
            examples,
            objective="dckd",
            output_dir=str(prepared_path),
            teacher_model=MODEL_ID,
            tokenizer=MODEL_ID,
            top_k=4,
            training_config=_training_config("dckd", root / "unused-preparation-output"),
        )
        results["prepared_rows"] = len(prepared.dataset["train"])
        results["prepared_columns"] = sorted(prepared.dataset["train"].column_names)
        _release_cuda()

        student_model = MODEL_ID
        if method in {"dckd", "all"}:
            dckd_output = root / "dckd-checkpoint"
            dckd_run = DCKD(
                model=MODEL_ID,
                dataset=str(prepared_path),
                config=_training_config("dckd", dckd_output),
            ).train()
            results["dckd"] = {
                "metrics": dict(dckd_run.metrics),
                "checkpoint_created": (dckd_output / "config.json").exists(),
            }
            if method == "all":
                student_model = str(dckd_output)
            del dckd_run
            _release_cuda()

        if method in {"tvkd", "all"}:
            tvkd_output = root / "tvkd-checkpoint"
            tvkd_run = TVKD(
                model=student_model,
                dataset=str(prepared_path),
                config=_training_config("tvkd", tvkd_output),
            ).train()
            results["tvkd"] = {
                "metrics": dict(tvkd_run.metrics),
                "checkpoint_created": (tvkd_output / "config.json").exists(),
                "initialized_from_dckd": method == "all",
            }
            del tvkd_run
            _release_cuda()

    return results


@app.local_entrypoint()
def main(method: str = "all") -> None:
    result = run_smoke.remote(method)
    print(json.dumps(result, indent=2, sort_keys=True))
