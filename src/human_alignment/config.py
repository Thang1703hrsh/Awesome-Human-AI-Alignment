"""Typed method configuration plus JSON/TOML loading."""

from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

from human_alignment.exceptions import ConfigurationError


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: str = "outputs/alignment"
    learning_rate: float = 1e-6
    epochs: float = 1.0
    batch_size: int = 2
    gradient_accumulation_steps: int = 1
    max_length: int | None = 1024
    seed: int = 42
    report_to: str = "none"
    resume_from_checkpoint: str | bool | None = None
    extra_args: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ConfigurationError("learning_rate must be positive")
        if self.epochs <= 0:
            raise ConfigurationError("epochs must be positive")
        if self.batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ConfigurationError("batch sizes and accumulation steps must be at least 1")
        if self.max_length is not None and self.max_length < 1:
            raise ConfigurationError("max_length must be positive or None")
        if isinstance(self.resume_from_checkpoint, str) and not self.resume_from_checkpoint:
            raise ConfigurationError("resume_from_checkpoint cannot be an empty string")

    def trainer_kwargs(self) -> dict[str, Any]:
        values = dict(self.extra_args)
        values.update({
            "output_dir": self.output_dir,
            "learning_rate": self.learning_rate,
            "num_train_epochs": self.epochs,
            "per_device_train_batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_length": self.max_length,
            "seed": self.seed,
            "report_to": self.report_to,
        })
        return values


@dataclass(frozen=True)
class DPOConfig(TrainingConfig):
    beta: float = 0.1
    loss_type: str = "sigmoid"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.beta <= 0:
            raise ConfigurationError("DPO beta must be positive")


@dataclass(frozen=True)
class SFTConfig(TrainingConfig):
    learning_rate: float = 2e-5
    packing: bool = False


@dataclass(frozen=True)
class KTOConfig(TrainingConfig):
    beta: float = 0.1

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.beta <= 0:
            raise ConfigurationError("KTO beta must be positive")


@dataclass(frozen=True)
class SimPOConfig(DPOConfig):
    loss_type: str = "sigmoid_norm"


@dataclass(frozen=True)
class PPOConfig(TrainingConfig):
    pass


@dataclass(frozen=True)
class GRPOConfig(TrainingConfig):
    beta: float = 0.0
    num_generations: int = 8
    max_completion_length: int = 512

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.beta < 0:
            raise ConfigurationError("GRPO beta must be non-negative")
        if self.num_generations < 2:
            raise ConfigurationError("GRPO num_generations must be at least 2")
        if self.max_completion_length < 1:
            raise ConfigurationError("GRPO max_completion_length must be positive")


DistillationObjective = Literal["dckd", "tvkd", "adpa", "ctpd", "ppd", "vpd"]
TVKDValueFunction = Literal["entropy", "soft"]


@dataclass(frozen=True)
class PreferenceDistillationConfig(TrainingConfig):
    """Configuration shared by the supported preference-distillation objectives."""

    objective: DistillationObjective = "vpd"
    beta: float = 10.0
    max_prompt_length: int = 512
    average_log_prob: bool = True
    format_plain_prompts: bool = True
    teacher_score_key: str | None = None
    teacher_logprob_score_key: str | None = None
    teacher_score_mix: float = 0.0
    objective_weight: float = 1.0
    sft_weight: float = 0.0
    chosen_kd_weight: float = 0.0
    rejected_kd_weight: float = 0.0
    ctpd_beta: float = 0.5
    value_function: TVKDValueFunction = "entropy"
    value_discount: float = 1.0
    value_temperature: float = 1.0
    student_value_weight: float = 0.0
    teacher_value_weight: float = 0.7
    exact_ranking_limit: int = 8

    def __post_init__(self) -> None:
        super().__post_init__()
        allowed = {"dckd", "tvkd", "adpa", "ctpd", "ppd", "vpd"}
        if self.objective not in allowed:
            raise ConfigurationError(
                f"Unknown distillation objective: {self.objective}. "
                f"Expected one of: {', '.join(sorted(allowed))}"
            )
        if self.beta <= 0 or self.ctpd_beta <= 0 or self.value_temperature <= 0:
            raise ConfigurationError("distillation temperatures and beta values must be positive")
        if self.max_prompt_length < 1:
            raise ConfigurationError("max_prompt_length must be positive")
        if self.max_length is not None and self.max_prompt_length > self.max_length:
            raise ConfigurationError("max_prompt_length cannot exceed max_length")
        if not 0.0 <= self.teacher_score_mix <= 1.0:
            raise ConfigurationError("teacher_score_mix must be in [0, 1]")
        if self.value_function not in {"entropy", "soft"}:
            raise ConfigurationError("value_function must be 'entropy' or 'soft'")
        if not 0.0 <= self.value_discount <= 1.0:
            raise ConfigurationError("value_discount must be in [0, 1]")
        if self.objective_weight <= 0:
            raise ConfigurationError("objective_weight must be positive")
        if min(
            self.sft_weight,
            self.chosen_kd_weight,
            self.rejected_kd_weight,
            self.student_value_weight,
            self.teacher_value_weight,
        ) < 0:
            raise ConfigurationError("distillation loss weights must be non-negative")
        if self.objective == "dckd" and self.chosen_kd_weight + self.rejected_kd_weight == 0:
            raise ConfigurationError("DCKD requires a positive chosen or rejected KD weight")
        if self.exact_ranking_limit < 2:
            raise ConfigurationError("exact_ranking_limit must be at least 2")


@dataclass(frozen=True)
class MethodRunConfig:
    method: str
    model: str
    dataset: str
    output_dir: str | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    target: Mapping[str, Any] | None = None
    method_options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CheckpointConfig:
    """Portable options for loading a local or Hugging Face checkpoint."""

    task: Literal["causal_lm", "sequence_classification"] = "causal_lm"
    adapter_path: str | None = None
    tokenizer_path: str | None = None
    device_map: Any = None
    torch_dtype: str | None = None
    trust_remote_code: bool = False
    revision: str | None = None
    merge_adapter: bool = False

    def __post_init__(self) -> None:
        if self.task not in {"causal_lm", "sequence_classification"}:
            raise ConfigurationError(f"Unsupported checkpoint task: {self.task}")
        if self.merge_adapter and not self.adapter_path:
            raise ConfigurationError("merge_adapter requires adapter_path")


def load_mapping(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ConfigurationError(f"Configuration file does not exist: {path}")
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() == ".toml":
        with path.open("rb") as stream:
            raw = tomllib.load(stream)
    else:
        raise ConfigurationError("Configuration must use .json or .toml")
    if not isinstance(raw, dict):
        raise ConfigurationError("Top-level configuration must be a mapping")
    return raw


def load_config(path: str | Path) -> MethodRunConfig:
    raw = load_mapping(path)
    required = ("method", "model", "dataset")
    missing = [key for key in required if not str(raw.get(key, "")).strip()]
    if missing:
        raise ConfigurationError(f"Configuration is missing: {', '.join(missing)}")
    parameters = raw.get("training", raw.get("parameters", {}))
    if not isinstance(parameters, Mapping):
        raise ConfigurationError("training parameters must be a mapping")
    target = raw.get("target")
    if target is not None and not isinstance(target, Mapping):
        raise ConfigurationError("target must be a mapping")
    method_options = raw.get("method_options", {})
    if not isinstance(method_options, Mapping):
        raise ConfigurationError("method_options must be a mapping")
    return MethodRunConfig(
        method=str(raw["method"]),
        model=str(raw["model"]),
        dataset=str(raw["dataset"]),
        output_dir=raw.get("output_dir"),
        parameters=dict(parameters),
        target=dict(target) if target else None,
        method_options=dict(method_options),
    )


def config_dict(config: TrainingConfig) -> dict[str, Any]:
    return asdict(config)
