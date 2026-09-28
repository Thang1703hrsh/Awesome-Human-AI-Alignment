"""Lazy Transformers backend for token-level and Bregman preference methods."""

import copy
import json
from dataclasses import asdict
from pathlib import Path

from human_alignment.exceptions import BackendUnavailableError, ConfigurationError
from human_alignment.results import AlignmentRun


class PreferenceBackend:
    def train(self, *, method, model, dataset, config, options=None):
        try:
            import accelerate  # noqa: F401
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed
            from safetensors.torch import save_file
        except ImportError as exc:
            raise BackendUnavailableError("Install native training dependencies with pip install -e '.[preference]'") from exc
        from human_alignment.integrations.transformers import make_generator
        from human_alignment.mechanisms.training.preference.data import PreferenceCollator
        from human_alignment.mechanisms.training.preference.trainer import PreferenceModel, NativePreferenceTrainer

        opts = dict(options or {})
        tokenizer = opts.pop("tokenizer", None)
        reference = opts.pop("reference_model", None)
        positive = opts.pop("positive_model", None)
        negative = opts.pop("negative_model", None)
        model_kwargs = dict(opts.pop("model_kwargs", {}))
        peft_config = opts.pop("peft_config", None)
        split = opts.pop("dataset_split", "train")
        if opts:
            raise ConfigurationError(f"Unknown preference backend options: {', '.join(sorted(opts))}")
        if (positive is None) != (negative is None):
            raise ConfigurationError("Supply both positive_model and negative_model for TIS-DPO")
        if method != "tis_dpo" and positive is not None:
            raise ConfigurationError("Contrastive weight models are only used by TIS-DPO")
        kwargs = config.trainer_kwargs()
        kwargs.pop("max_length", None)
        kwargs.update(remove_unused_columns=False, label_names=[], save_safetensors=False)
        # Checkpoint files contain the wrapper, optimizer and TBPO baseline. Export
        # below separately produces a standard HF policy directory for inference.
        args = TrainingArguments(**kwargs)
        if method == "ti_dpo" and args.gradient_checkpointing:
            # autograd.grad on embeddings is incompatible with reentrant checkpointing.
            args.gradient_checkpointing_kwargs = {
                **(args.gradient_checkpointing_kwargs or {}), "use_reentrant": False,
            }
        if args.deepspeed or args.fsdp:
            raise ConfigurationError("Native preference backend supports single device/DDP; DeepSpeed/FSDP are not supported")
        if method == "ti_dpo" and args.world_size > 1:
            raise ConfigurationError("TI-DPO attribution/anchor training currently requires one process")
        set_seed(config.seed)
        source = model if isinstance(model, (str, Path)) else None
        if tokenizer is None:
            if source is None:
                raise ConfigurationError("Supply backend_options={'tokenizer': tokenizer} with a model object")
            tokenizer = AutoTokenizer.from_pretrained(str(source))
        elif isinstance(tokenizer, (str, Path)):
            tokenizer = AutoTokenizer.from_pretrained(str(tokenizer))
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        def load(value):
            return AutoModelForCausalLM.from_pretrained(str(value), **model_kwargs) if isinstance(value, (str, Path)) else value

        policy = load(model)
        if reference is policy:
            raise ConfigurationError("Reference and trainable policy must be separate models")
        reference = load(reference) if reference is not None else (load(source) if source is not None else copy.deepcopy(policy))
        positive, negative = load(positive), load(negative)
        if any(value is policy for value in (reference, positive, negative)):
            raise ConfigurationError("Frozen auxiliary models must not be the trainable policy")
        if peft_config is not None:
            try:
                from peft import get_peft_model
            except ImportError as exc:
                raise BackendUnavailableError("Install the peft extra for LoRA training") from exc
            policy = get_peft_model(policy, peft_config)
        for module in policy.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        rows = dataset.data
        if isinstance(rows, str):
            try:
                from datasets import load_dataset, load_from_disk
            except ImportError as exc:
                raise BackendUnavailableError("Install datasets to read Hub datasets or saved Arrow directories") from exc
            if Path(rows).is_dir():
                rows = load_from_disk(rows)
                if hasattr(rows, "keys"):
                    rows = rows[split]
            else:
                rows = load_dataset(rows, split=split)
        if not len(rows):
            raise ConfigurationError("Preference dataset must not be empty")
        collator = PreferenceCollator(tokenizer, config.max_length, config.max_prompt_length)
        # Preflight one row before allocating optimizer state.
        collator([rows[0]])
        wrapped = PreferenceModel(policy, use_baseline=method == "tbpo_q")
        trainer = NativePreferenceTrainer(model=wrapped, args=args, train_dataset=rows,
            processing_class=tokenizer, data_collator=collator, method=method,
            preference_config=config, reference_model=reference,
            positive_model=positive, negative_model=negative)
        output = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
        wrapped = trainer.accelerator.unwrap_model(trainer.model)

        def save(path):
            if trainer.is_world_process_zero():
                dest = Path(path)
                dest.mkdir(parents=True, exist_ok=True)
                wrapped.policy.save_pretrained(dest)
                tokenizer.save_pretrained(dest)
                if wrapped.baseline is not None:
                    save_file({k: v.detach().cpu().contiguous() for k, v in wrapped.baseline.state_dict().items()},
                              str(dest / "baseline_head.safetensors"))
                (dest / "preference_config.json").write_text(
                    json.dumps({"method": method, "config": asdict(config)}, indent=2, default=str), encoding="utf-8")

        save(config.output_dir)
        return AlignmentRun(model=wrapped.policy, tokenizer=tokenizer, output_dir=config.output_dir,
            metrics={k: float(v) for k, v in output.metrics.items() if isinstance(v, (int, float))},
            metadata={"backend": "native-preference", "method": method},
            _generator=make_generator(wrapped.policy, tokenizer), _saver=save)
