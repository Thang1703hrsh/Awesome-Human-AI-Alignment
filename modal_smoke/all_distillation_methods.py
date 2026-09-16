"""One-step Modal smoke test for every preference-distillation workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal


APP_NAME = "human-alignment-all-distillation-smoke"
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


def _config(objective: str, output_dir: Path):
    from human_alignment import PreferenceDistillationConfig

    parameters: dict[str, Any] = {
        "objective": objective,
        "output_dir": str(output_dir),
        "learning_rate": 1e-5,
        "epochs": 1.0,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "max_length": 64,
        "max_prompt_length": 32,
        "report_to": "none",
        "exact_ranking_limit": 4,
        "extra_args": {
            "max_steps": 1,
            "save_strategy": "no",
            "logging_steps": 1,
            "disable_tqdm": True,
            "dataloader_num_workers": 0,
        },
    }
    if objective == "dckd":
        parameters.update(chosen_kd_weight=0.1, rejected_kd_weight=0.1)
    elif objective == "tvkd":
        parameters.update(
            chosen_kd_weight=0.0001,
            rejected_kd_weight=0.0001,
            value_function="entropy",
            value_discount=1.0,
        )
    elif objective == "ctpd":
        parameters.update(ctpd_beta=0.5)
    elif objective in {"ppd", "vpd"}:
        parameters.update(beta=1.0)
    return PreferenceDistillationConfig(**parameters)


def _release_cuda() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _record(run: Any, output_dir: Path, **metadata: Any) -> dict[str, Any]:
    return {
        "checkpoint_created": (output_dir / "config.json").exists(),
        "metrics": dict(run.metrics),
        **metadata,
    }


@app.function(gpu="T4", cpu=0.5, memory=2_048, timeout=20 * 60)
def run_all() -> dict[str, Any]:
    """Run one optimizer step for DCKD, TVKD, ADPA(+), CTPD, PPD, and VPD."""

    import tempfile

    import torch
    from transformers import AutoModelForCausalLM

    from human_alignment import (
        ADPA,
        CTPD,
        DCKD,
        PPD,
        TVKD,
        VPD,
        prepare_preference_distillation_dataset,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("Modal did not attach the requested GPU")
    torch.manual_seed(0)

    pairs = [
        {
            "prompt": "Question: What is alignment? Answer:",
            "chosen": " Matching model behavior to intended goals.",
            "rejected": " Making every response longer.",
        },
        {
            "prompt": "Question: How should uncertainty be handled? Answer:",
            "chosen": " State uncertainty clearly.",
            "rejected": " Always sound certain.",
        },
    ]
    prompts = [{"prompt": row["prompt"]} for row in pairs]
    device = torch.device("cuda")
    reference = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device).eval()
    teacher = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device).eval()
    with torch.no_grad():
        first_parameter = next(teacher.parameters())
        first_parameter.add_(0.02 * torch.randn_like(first_parameter))

    results: dict[str, Any] = {
        "cuda_device": torch.cuda.get_device_name(0),
        "model": MODEL_ID,
        "resources": {"gpu": "T4", "cpu": 0.5, "memory_mib": 2_048},
    }

    with tempfile.TemporaryDirectory(prefix="hai-all-distillation-") as temporary:
        root = Path(temporary)

        token_data = prepare_preference_distillation_dataset(
            pairs,
            objective="dckd",
            output_dir=str(root / "token-data"),
            teacher_model=teacher,
            tokenizer=MODEL_ID,
            top_k=4,
            training_config=_config("dckd", root / "unused-dckd-preparation"),
        )
        results["token_data_rows"] = len(token_data.dataset["train"])

        dckd_output = root / "dckd"
        dckd_run = DCKD(
            model=MODEL_ID,
            dataset=str(token_data.path),
            config=_config("dckd", dckd_output),
        ).train()
        results["dckd"] = _record(dckd_run, dckd_output)
        del dckd_run
        _release_cuda()

        tvkd_output = root / "tvkd"
        tvkd_run = TVKD(
            model=str(dckd_output),
            dataset=str(token_data.path),
            config=_config("tvkd", tvkd_output),
        ).train()
        results["tvkd"] = _record(
            tvkd_run,
            tvkd_output,
            initialized_from="dckd",
        )
        del tvkd_run
        _release_cuda()

        adpa_data = prepare_preference_distillation_dataset(
            pairs,
            objective="adpa",
            output_dir=str(root / "adpa-data"),
            teacher_model=teacher,
            reference_model=reference,
            tokenizer=MODEL_ID,
            top_k=4,
            training_config=_config("adpa", root / "unused-adpa-preparation"),
        )
        adpa_output = root / "adpa"
        adpa_run = ADPA(
            model=MODEL_ID,
            dataset=str(adpa_data.path),
            config=_config("adpa", adpa_output),
        ).train()
        results["adpa"] = _record(adpa_run, adpa_output)
        del adpa_run
        _release_cuda()

        adpa_plus_data = prepare_preference_distillation_dataset(
            pairs,
            objective="adpa",
            output_dir=str(root / "adpa-plus-data"),
            teacher_model=teacher,
            reference_model=str(dckd_output),
            student_model=str(dckd_output),
            tokenizer=str(dckd_output),
            top_k=4,
            generation_kwargs={
                "do_sample": False,
                "max_new_tokens": 6,
                "num_return_sequences": 1,
            },
            model_kwargs={"device_map": "cuda"},
            training_config=_config("adpa", root / "unused-adpa-plus-preparation"),
        )
        adpa_plus_output = root / "adpa-plus"
        adpa_plus_run = ADPA(
            model=str(dckd_output),
            dataset=str(adpa_plus_data.path),
            config=_config("adpa", adpa_plus_output),
        ).train()
        results["adpa_plus"] = _record(
            adpa_plus_run,
            adpa_plus_output,
            initialized_from="dckd",
            on_policy_data=True,
        )
        del adpa_plus_run
        _release_cuda()

        ctpd_data = prepare_preference_distillation_dataset(
            pairs,
            objective="ctpd",
            output_dir=str(root / "ctpd-data"),
            teacher_model=teacher,
            reference_model=reference,
            tokenizer=MODEL_ID,
            training_config=_config("ctpd", root / "unused-ctpd-preparation"),
        )
        ctpd_output = root / "ctpd"
        ctpd_run = CTPD(
            model=MODEL_ID,
            dataset=str(ctpd_data.path),
            reference_model=MODEL_ID,
            config=_config("ctpd", ctpd_output),
        ).train()
        results["ctpd"] = _record(ctpd_run, ctpd_output)
        del ctpd_run
        _release_cuda()

        ranking_data = prepare_preference_distillation_dataset(
            prompts,
            objective="vpd",
            output_dir=str(root / "ranking-data"),
            teacher_model=teacher,
            student_model=MODEL_ID,
            tokenizer=MODEL_ID,
            num_responses=3,
            teacher_score_mode="sequence_logprob",
            generation_kwargs={"max_new_tokens": 6},
            model_kwargs={"device_map": "cuda"},
            training_config=_config("vpd", root / "unused-ranking-preparation"),
        )
        results["ranking_data_rows"] = len(ranking_data.dataset["train"])

        for objective, method_class in (("vpd", VPD), ("ppd", PPD)):
            output = root / objective
            run = method_class(
                model=MODEL_ID,
                dataset=str(ranking_data.path),
                config=_config(objective, output),
            ).train()
            results[objective] = _record(run, output, response_count=3)
            del run
            _release_cuda()

        del teacher, reference
        _release_cuda()

    return results


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(run_all.remote(), indent=2, sort_keys=True))

