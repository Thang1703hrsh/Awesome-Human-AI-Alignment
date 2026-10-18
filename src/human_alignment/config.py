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
class IPOConfig(DPOConfig):
    loss_type: str = "ipo"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.loss_type != "ipo":
            raise ConfigurationError("IPO requires loss_type='ipo'")


@dataclass(frozen=True)
class PreferenceOptimizationConfig(TrainingConfig):
    """Native BPO, TDPO, TIS-DPO, TI-DPO, and TokenRatio configuration.

    Defaults are starting points, not a reproduction of published benchmark scores.
    TIS-DPO consumes final token weights or estimates rank weights from two models.
    """

    beta: float = 0.1
    max_prompt_length: int = 128
    alpha: float = 0.5
    tdpo2: bool = True
    tis_token_level: bool = False
    generator: str = "sba"
    bregman_lambda: float = 0.2
    bregman_scale: float = 4.0
    log_ratio_clip: float = 30.0
    label_smoothing: float = 0.0
    baseline_clip: float = 10.0
    baseline_learning_rate: float = 1e-4
    importance_mix: float = 0.8
    prior_sigma_div: float = 4.0
    triplet_weight: float = 0.1
    triplet_margin: float = 0.1
    anchor_max_new_tokens: int = 64
    dp3o_alpha: float = 0.2
    dp3o_reward_type: str = "proxy"
    beta_dpo_a: float = 0.6
    beta_dpo_filter_ratio: float = 0.2
    beta_dpo_momentum: float = 0.9
    epsilon: float = 0.01
    sam_rho: float = 0.05
    sam_adaptive: bool = False
    ra_dpo_variant: int = 1
    ra_dpo_alpha: float = 0.5
    ra_dpo_confidence: float = 0.95
    ra_dpo_split_vocabulary: bool = True
    adpo_granularity: str = "adaptive"
    adpo_segments: int = 256
    adpo_static_k: int = 32
    adpo_segment_reduce: str = "sum"

    def __post_init__(self) -> None:
        import math

        super().__post_init__()
        if not 0.0 <= self.beta_dpo_filter_ratio < 1.0 or not 0.0 <= self.beta_dpo_momentum < 1.0:
            raise ConfigurationError("beta_dpo_filter_ratio and beta_dpo_momentum must be in [0, 1)")
        if not 0.0 < self.epsilon < 1.0 or self.sam_rho <= 0:
            raise ConfigurationError("epsilon must be in (0, 1) and sam_rho positive")
        if self.ra_dpo_variant not in {1, 2} or not 0.0 < self.ra_dpo_confidence < 1.0:
            raise ConfigurationError("ra_dpo_variant must be 1 or 2 and ra_dpo_confidence in (0, 1)")
        if (self.adpo_granularity not in {"adaptive", "static", "pass"}
                or self.adpo_segment_reduce not in {"sum", "mean"}):
            raise ConfigurationError(
                "adpo_granularity must be adaptive/static/pass and adpo_segment_reduce sum/mean")
        if self.adpo_segments < 1 or self.adpo_static_k < 1:
            raise ConfigurationError("adpo_segments and adpo_static_k must be positive")
        if not 0.0 <= self.dp3o_alpha <= 1.0:
            raise ConfigurationError("dp3o_alpha must be in [0, 1]")
        if self.dp3o_reward_type not in {"proxy", "golden"}:
            raise ConfigurationError("dp3o_reward_type must be 'proxy' or 'golden'")
        for name in ("beta", "bregman_lambda", "bregman_scale", "log_ratio_clip",
                     "baseline_clip", "baseline_learning_rate", "prior_sigma_div"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ConfigurationError(f"{name} must be finite and positive")
        for name in ("alpha", "triplet_weight", "triplet_margin"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ConfigurationError(f"{name} must be finite and non-negative")
        if not 0 <= self.label_smoothing <= 0.5 or not 0 <= self.importance_mix <= 1:
            raise ConfigurationError("Invalid label_smoothing or importance_mix")
        if self.generator not in {"sba", "ba", "lsif", "kliep", "logistic"}:
            raise ConfigurationError(f"Unknown Bregman generator: {self.generator}")
        if self.max_length is None or not 1 <= self.max_prompt_length < self.max_length:
            raise ConfigurationError("Require 1 <= max_prompt_length < max_length")
        if self.anchor_max_new_tokens < 1:
            raise ConfigurationError("anchor_max_new_tokens must be positive")


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


@dataclass(frozen=True)
class PBSDConfig(TrainingConfig):
    """Preference-Based Self-Distillation (Yu et al., 2026).

    Defaults follow the authors' ``scripts/run_pbsd.sh`` (Qwen3-4B-Instruct-2507,
    LoRA r=64/alpha=128, ``beta=0.1``, sampling T=1.1/top-p 0.95/top-k 20).
    ``max_length`` bounds tokenized prompts.
    """

    learning_rate: float = 1e-5
    max_length: int | None = 20000
    beta: float = 0.1
    pairwise_loss_type: str = "logistic_gap"
    normalize_logprob_by_length: bool = False
    correctness_gate: bool = True
    teacher_use_reference: bool = False
    task: str = "math"
    student_think_mode: str = "auto"
    teacher_think_mode: str = "auto"
    student_max_new_tokens: int = 4096
    teacher_max_new_tokens: int = 4096
    temperature: float = 1.1
    top_p: float = 0.95
    top_k: int = 20
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.pairwise_loss_type not in {"logistic_gap", "mse_to_zero", "mse"}:
            raise ConfigurationError(f"Unknown pairwise_loss_type: {self.pairwise_loss_type}")
        if self.task not in {"math", "tooluse"}:
            raise ConfigurationError("PBSD task must be 'math' or 'tooluse'")
        for name in ("student_think_mode", "teacher_think_mode"):
            if getattr(self, name) not in {"auto", "enabled", "disabled"}:
                raise ConfigurationError(f"{name} must be 'auto', 'enabled', or 'disabled'")
        if self.beta <= 0 or self.temperature <= 0 or not 0 < self.top_p <= 1 or self.top_k < 0:
            raise ConfigurationError("Invalid PBSD beta or sampling parameters")
        if min(self.student_max_new_tokens, self.teacher_max_new_tokens) < 1:
            raise ConfigurationError("PBSD generation lengths must be positive")
        if self.use_lora and (self.lora_r < 1 or self.lora_alpha < 1 or not self.lora_target_modules):
            raise ConfigurationError("PBSD LoRA requires positive rank/alpha and target modules")
        object.__setattr__(self, "lora_target_modules", tuple(self.lora_target_modules))


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
    mcq_confidence_filter: bool = False
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
        if self.mcq_confidence_filter and self.objective not in {"vpd", "ppd"}:
            raise ConfigurationError("mcq_confidence_filter applies only to VPD and PPD")
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
