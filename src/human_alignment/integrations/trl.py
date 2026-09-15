"""Centralized, lazy integration with Hugging Face TRL trainers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from human_alignment.config import (
    DPOConfig,
    GRPOConfig,
    KTOConfig,
    PPOConfig,
    SFTConfig,
    TrainingConfig,
)
from human_alignment.exceptions import BackendUnavailableError, ConfigurationError
from human_alignment.integrations.transformers import make_generator
from human_alignment.results import AlignmentRun
from human_alignment.supervision.datasets import DatasetBundle


def _require_dependencies() -> tuple[Any, Any]:
    try:
        import datasets
        import trl
    except ImportError as exc:
        raise BackendUnavailableError(
            "TRL training requires `pip install human-ai-alignment[trl]`."
        ) from exc
    return datasets, trl


def _materialize(bundle: DatasetBundle, datasets_module: Any, options: dict[str, Any]) -> Any:
    if isinstance(bundle.data, str):
        split = options.pop("dataset_split", "train")
        if Path(bundle.data).is_dir():
            loaded = datasets_module.load_from_disk(bundle.data)
            return loaded[split] if hasattr(loaded, "keys") and split in loaded else loaded
        return datasets_module.load_dataset(bundle.data, split=split)
    if isinstance(bundle.data, list):
        return datasets_module.Dataset.from_list(bundle.data)
    return bundle.data


def _metrics(output: Any) -> dict[str, float]:
    raw = getattr(output, "metrics", {}) or {}
    return {
        str(key): float(value)
        for key, value in raw.items()
        if isinstance(value, (int, float))
    }


def _run_from_trainer(
    trainer: Any,
    output_dir: str,
    *,
    resume_from_checkpoint: str | bool | None = None,
) -> AlignmentRun:
    output = (
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        if resume_from_checkpoint is not None
        else trainer.train()
    )
    trainer.save_model(output_dir)
    model = trainer.model
    tokenizer = getattr(trainer, "processing_class", None) or getattr(
        trainer, "tokenizer", None
    )

    def save(path: str) -> None:
        trainer.save_model(path)
        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(path)

    generator = make_generator(model, tokenizer) if tokenizer is not None else None
    return AlignmentRun(
        model=model,
        tokenizer=tokenizer,
        metrics=_metrics(output),
        output_dir=output_dir,
        metadata={"backend": "trl"},
        _generator=generator,
        _saver=save,
    )


class TRLBackend:
    def train(
        self,
        *,
        method: str,
        model: Any,
        dataset: DatasetBundle,
        config: TrainingConfig,
        options: Mapping[str, Any] | None = None,
    ) -> AlignmentRun:
        datasets_module, trl = _require_dependencies()
        options_dict = dict(options or {})
        train_dataset = _materialize(dataset, datasets_module, options_dict)
        if method == "dpo":
            return self._dpo(trl, model, train_dataset, config, options_dict)
        if method == "sft":
            return self._sft(trl, model, train_dataset, config, options_dict)
        if method == "kto":
            return self._kto(trl, model, train_dataset, config, options_dict)
        if method == "grpo":
            return self._grpo(trl, model, train_dataset, config, options_dict)
        if method == "ppo":
            return self._ppo(model, train_dataset, config, options_dict)
        raise ConfigurationError(f"Unsupported TRL method: {method}")

    @staticmethod
    def _dpo(
        trl: Any,
        model: Any,
        dataset: Any,
        config: TrainingConfig,
        options: dict[str, Any],
    ) -> AlignmentRun:
        if not isinstance(config, DPOConfig):
            raise ConfigurationError("DPO requires DPOConfig")
        kwargs = config.trainer_kwargs()
        kwargs.pop("beta", None)
        kwargs.pop("loss_type", None)
        args = trl.DPOConfig(**kwargs, beta=config.beta, loss_type=config.loss_type)
        trainer = trl.DPOTrainer(model=model, args=args, train_dataset=dataset, **options)
        return _run_from_trainer(
            trainer,
            config.output_dir,
            resume_from_checkpoint=config.resume_from_checkpoint,
        )

    @staticmethod
    def _sft(
        trl: Any,
        model: Any,
        dataset: Any,
        config: TrainingConfig,
        options: dict[str, Any],
    ) -> AlignmentRun:
        if not isinstance(config, SFTConfig):
            raise ConfigurationError("SFT requires SFTConfig")
        kwargs = config.trainer_kwargs()
        kwargs.pop("packing", None)
        args = trl.SFTConfig(**kwargs, packing=config.packing)
        trainer = trl.SFTTrainer(model=model, args=args, train_dataset=dataset, **options)
        return _run_from_trainer(
            trainer,
            config.output_dir,
            resume_from_checkpoint=config.resume_from_checkpoint,
        )

    @staticmethod
    def _kto(
        trl: Any,
        model: Any,
        dataset: Any,
        config: TrainingConfig,
        options: dict[str, Any],
    ) -> AlignmentRun:
        if not isinstance(config, KTOConfig):
            raise ConfigurationError("KTO requires KTOConfig")
        try:
            config_class, trainer_class = trl.KTOConfig, trl.KTOTrainer
        except AttributeError:
            try:
                from trl.experimental.kto import KTOConfig as config_class
                from trl.experimental.kto import KTOTrainer as trainer_class
            except ImportError as exc:
                raise BackendUnavailableError(
                    "The installed TRL version does not expose KTOTrainer."
                ) from exc
        kwargs = config.trainer_kwargs()
        kwargs.pop("beta", None)
        args = config_class(**kwargs, beta=config.beta)
        trainer = trainer_class(model=model, args=args, train_dataset=dataset, **options)
        return _run_from_trainer(
            trainer,
            config.output_dir,
            resume_from_checkpoint=config.resume_from_checkpoint,
        )

    @staticmethod
    def _grpo(
        trl: Any,
        model: Any,
        dataset: Any,
        config: TrainingConfig,
        options: dict[str, Any],
    ) -> AlignmentRun:
        if not isinstance(config, GRPOConfig):
            raise ConfigurationError("GRPO requires GRPOConfig")
        reward_funcs = options.pop("reward_funcs", None)
        if reward_funcs is None:
            raise ConfigurationError("GRPO requires backend_options={'reward_funcs': ...}")
        kwargs = config.trainer_kwargs()
        kwargs.pop("max_length", None)
        kwargs.pop("beta", None)
        kwargs.pop("num_generations", None)
        kwargs.pop("max_completion_length", None)
        args = trl.GRPOConfig(
            **kwargs,
            beta=config.beta,
            num_generations=config.num_generations,
            max_completion_length=config.max_completion_length,
        )
        trainer = trl.GRPOTrainer(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=dataset,
            **options,
        )
        return _run_from_trainer(
            trainer,
            config.output_dir,
            resume_from_checkpoint=config.resume_from_checkpoint,
        )

    @staticmethod
    def _ppo(
        model: Any,
        dataset: Any,
        config: TrainingConfig,
        options: dict[str, Any],
    ) -> AlignmentRun:
        if not isinstance(config, PPOConfig):
            raise ConfigurationError("PPO requires PPOConfig")
        try:
            from trl.experimental.ppo import PPOConfig as TRLPPOConfig
            from trl.experimental.ppo import PPOTrainer
        except ImportError as exc:
            raise BackendUnavailableError(
                "The installed TRL version does not expose experimental PPO."
            ) from exc
        required = ("processing_class", "reward_model", "value_model")
        missing = [key for key in required if key not in options]
        if missing:
            raise ConfigurationError(
                "PPO backend_options is missing: " + ", ".join(missing)
            )
        kwargs = config.trainer_kwargs()
        kwargs.pop("max_length", None)
        args = TRLPPOConfig(**kwargs)
        trainer = PPOTrainer(
            args=args,
            model=model,
            train_dataset=dataset,
            **options,
        )
        return _run_from_trainer(
            trainer,
            config.output_dir,
            resume_from_checkpoint=config.resume_from_checkpoint,
        )
