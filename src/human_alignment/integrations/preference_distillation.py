"""Lazy Transformers backend for preference-distillation objectives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from human_alignment.config import PreferenceDistillationConfig
from human_alignment.exceptions import (
    BackendUnavailableError,
    ConfigurationError,
    DatasetFormatError,
)
from human_alignment.integrations.transformers import make_generator
from human_alignment.results import AlignmentRun
from human_alignment.supervision.datasets import DatasetBundle


_PASSTHROUGH_FIELDS = (
    "teacher_chosen_probs",
    "chosen_compressed_probs",
    "teacher_rejected_probs",
    "rejected_compressed_probs",
    "rejected_margin_logp_every",
    "chosen_ctpd_parent_list",
    "rejected_ctpd_parent_list",
    "chosen_ctpd_weight",
    "rejected_ctpd_weight",
)


def _require_dependencies() -> tuple[Any, Any]:
    try:
        import accelerate
        import torch
        import transformers
        from transformers import Trainer

        _ = accelerate, Trainer
    except Exception as exc:
        raise BackendUnavailableError(
            "Preference distillation requires `pip install human-ai-alignment[distillation]`."
        ) from exc
    return torch, transformers


def _materialize(source: Any, *, split: str) -> Any:
    if not isinstance(source, str):
        return source
    try:
        import datasets
    except ImportError as exc:
        raise BackendUnavailableError(
            "Loading a Hugging Face dataset ID requires the `datasets` package."
        ) from exc
    if Path(source).is_dir():
        loaded = datasets.load_from_disk(source)
        return loaded[split] if hasattr(loaded, "keys") and split in loaded else loaded
    return datasets.load_dataset(source, split=split)


def _is_messages(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and all(
            isinstance(item, Mapping) and "role" in item and "content" in item
            for item in value
        )
    )


def _apply_messages(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    generate: bool,
) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise DatasetFormatError(
            "Chat-formatted data requires a tokenizer with apply_chat_template"
        )
    return tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=generate,
    )


def _pair_from_conversations(tokenizer: Any, chosen: Any, rejected: Any) -> tuple[str, list[str]]:
    if not (_is_messages(chosen) and _is_messages(rejected)):
        raise DatasetFormatError(
            "Without an explicit prompt, chosen and rejected must be chat-message sequences"
        )
    if not chosen or not rejected:
        raise DatasetFormatError("Chosen and rejected conversations must not be empty")
    chosen_prompt = list(chosen[:-1])
    rejected_prompt = list(rejected[:-1])
    if chosen_prompt != rejected_prompt:
        raise DatasetFormatError("Chosen and rejected conversations must share the same prompt")
    prompt = _apply_messages(tokenizer, chosen_prompt, generate=True)
    responses = []
    for conversation in (chosen, rejected):
        full = _apply_messages(tokenizer, conversation, generate=False)
        if full.startswith(prompt):
            responses.append(full[len(prompt) :])
        else:
            responses.append(str(conversation[-1]["content"]))
    return prompt, responses


def _response_text(tokenizer: Any, prompt: Any, response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, Mapping) and "content" in response:
        return str(response["content"])
    if _is_messages(response):
        if _is_messages(prompt):
            prompt_text = _apply_messages(tokenizer, prompt, generate=True)
            prompt_messages = list(prompt)
            response_messages = list(response)
            full_messages = (
                response_messages
                if response_messages[: len(prompt_messages)] == prompt_messages
                else [*prompt_messages, *response_messages]
            )
            full = _apply_messages(tokenizer, full_messages, generate=False)
            if full.startswith(prompt_text):
                return full[len(prompt_text) :]
        return str(response[-1]["content"])
    return str(response)


def _teacher_scores(
    row: Mapping[str, Any], config: PreferenceDistillationConfig
) -> list[float] | None:
    if config.teacher_score_key:
        values = row.get(config.teacher_score_key)
        if values is None:
            raise DatasetFormatError(
                f"Configured teacher score column is absent: {config.teacher_score_key}"
            )
        base = [float(value) for value in values]
    elif "teacher_scores" in row:
        base = [float(value) for value in row["teacher_scores"]]
    elif "scores_mcq" in row:
        base = [float(value) for value in row["scores_mcq"]]
    elif "scores" in row:
        base = [float(value) for value in row["scores"]]
    else:
        return None
    if not base:
        return None

    logprob_key = config.teacher_logprob_score_key
    if logprob_key is None and "scores_avglogp" in row:
        logprob_key = "scores_avglogp"
    if logprob_key and config.teacher_score_mix > 0:
        auxiliary = row.get(logprob_key)
        if auxiliary is None:
            raise DatasetFormatError(f"Teacher log-probability column is absent: {logprob_key}")
        if len(auxiliary) != len(base):
            raise DatasetFormatError("Teacher score columns must have equal lengths")
        weight = config.teacher_score_mix
        base = [
            weight * float(other) + (1.0 - weight) * score
            for score, other in zip(base, auxiliary)
        ]
    return base


def _normalize_row(
    row: Mapping[str, Any],
    tokenizer: Any,
    config: PreferenceDistillationConfig,
) -> dict[str, Any]:
    row = dict(row)
    if "responses" in row:
        prompt_value = row.get("prompt")
        if prompt_value is None:
            raise DatasetFormatError("A multi-response row requires prompt")
        if _is_messages(prompt_value):
            prompt = _apply_messages(tokenizer, prompt_value, generate=True)
        elif config.format_plain_prompts and getattr(tokenizer, "chat_template", None):
            prompt = _apply_messages(
                tokenizer, [{"role": "user", "content": str(prompt_value)}], generate=True
            )
        else:
            prompt = str(prompt_value)
        responses = [_response_text(tokenizer, prompt_value, value) for value in row["responses"]]
    elif "chosen" in row and "rejected" in row:
        if "prompt" not in row:
            prompt, responses = _pair_from_conversations(tokenizer, row["chosen"], row["rejected"])
        else:
            prompt_value = row["prompt"]
            if _is_messages(prompt_value):
                prompt = _apply_messages(tokenizer, prompt_value, generate=True)
            elif config.format_plain_prompts and getattr(tokenizer, "chat_template", None):
                prompt = _apply_messages(
                    tokenizer,
                    [{"role": "user", "content": str(prompt_value)}],
                    generate=True,
                )
            else:
                prompt = str(prompt_value)
            responses = [
                _response_text(tokenizer, prompt_value, row["chosen"]),
                _response_text(tokenizer, prompt_value, row["rejected"]),
            ]
    else:
        raise DatasetFormatError("Expected prompt/responses or chosen/rejected columns")

    if len(responses) < 2:
        raise DatasetFormatError("Preference distillation requires at least two responses")
    scores = _teacher_scores(row, config)
    if scores is not None and len(scores) != len(responses):
        raise DatasetFormatError("Teacher scores must match the number of responses")

    normalized = {"prompt": prompt, "responses": responses, "teacher_scores": scores}
    for key in _PASSTHROUGH_FIELDS:
        if key in row:
            normalized[key] = row[key]
    return normalized


def _tokenize_response(
    tokenizer: Any,
    prompt: str,
    response: str,
    config: PreferenceDistillationConfig,
) -> dict[str, list[int]]:
    full = tokenizer(prompt + response, add_special_tokens=False)
    prompt_only = tokenizer(prompt, add_special_tokens=False)
    full_ids = list(full["input_ids"])
    prompt_ids = list(prompt_only["input_ids"])
    boundary = len(prompt_ids)
    if full_ids[:boundary] != prompt_ids:
        boundary = max(0, boundary - 1)
    if full_ids[:boundary] != prompt_ids[:boundary]:
        raise DatasetFormatError(
            "Prompt and prompt-plus-response tokenizations differ by more than one token"
        )
    response_ids = full_ids[boundary:]
    prompt_ids = full_ids[:boundary]

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None and (not response_ids or response_ids[-1] != eos_token_id):
        response_ids.append(int(eos_token_id))
    if config.max_length is not None:
        prompt_ids = prompt_ids[-config.max_prompt_length :]
        response_budget = config.max_length - len(prompt_ids)
        response_ids = response_ids[:response_budget]
    if not response_ids:
        raise DatasetFormatError("A response became empty after tokenization or truncation")

    input_ids = prompt_ids + response_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + response_ids,
    }


class _TokenizedDataset:
    def __init__(self, data: Any, tokenizer: Any, config: PreferenceDistillationConfig) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, Any]:
        normalized = _normalize_row(self.data[index], self.tokenizer, self.config)
        tokenized = [
            _tokenize_response(self.tokenizer, normalized["prompt"], response, self.config)
            for response in normalized["responses"]
        ]
        return {**normalized, "tokenized_responses": tokenized}


@dataclass
class _DistillationCollator:
    torch: Any
    pad_token_id: int

    def __call__(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        batch_size = len(rows)
        response_count = max(len(row["tokenized_responses"]) for row in rows)
        sequence_length = max(
            len(response["input_ids"])
            for row in rows
            for response in row["tokenized_responses"]
        )
        input_ids = self.torch.full(
            (batch_size, response_count, sequence_length),
            self.pad_token_id,
            dtype=self.torch.long,
        )
        attention_mask = self.torch.zeros_like(input_ids)
        labels = self.torch.full_like(input_ids, -100)
        response_mask = self.torch.zeros((batch_size, response_count), dtype=self.torch.bool)
        teacher_scores = self.torch.full(
            (batch_size, response_count), float("nan"), dtype=self.torch.float
        )

        for row_index, row in enumerate(rows):
            for response_index, response in enumerate(row["tokenized_responses"]):
                length = len(response["input_ids"])
                input_ids[row_index, response_index, :length] = self.torch.tensor(
                    response["input_ids"], dtype=self.torch.long
                )
                attention_mask[row_index, response_index, :length] = 1
                labels[row_index, response_index, :length] = self.torch.tensor(
                    response["labels"], dtype=self.torch.long
                )
                response_mask[row_index, response_index] = True
            if row["teacher_scores"] is not None:
                count = len(row["teacher_scores"])
                teacher_scores[row_index, :count] = self.torch.tensor(
                    row["teacher_scores"], dtype=self.torch.float
                )

        batch: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "response_mask": response_mask,
            "teacher_scores": teacher_scores,
        }
        for key in _PASSTHROUGH_FIELDS:
            values = [row.get(key) for row in rows]
            if any(value is not None for value in values):
                if not all(value is not None for value in values):
                    raise DatasetFormatError(
                        f"Column {key} must be present in every row of a batch"
                    )
                batch[key] = values
        return batch


def _teacher_field(inputs: Mapping[str, Any], preferred: str, legacy: str) -> Any:
    value = inputs.get(preferred)
    return value if value is not None else inputs.get(legacy)


def _make_trainer_class(torch: Any, transformers: Any) -> type[Any]:
    from human_alignment.mechanisms.training.distillation_losses import (
        advantage_expectation_loss,
        causal_token_log_probs,
        compressed_forward_kl,
        ctpd_loss,
        ppd_loss,
        tvkd_loss,
        vpd_loss,
    )

    class PreferenceDistillationTrainer(transformers.Trainer):
        def __init__(
            self,
            *args: Any,
            distillation_config: PreferenceDistillationConfig,
            teacher_model: Any = None,
            reference_model: Any = None,
            **kwargs: Any,
        ) -> None:
            self.distillation_config = distillation_config
            self.teacher_model = teacher_model
            self.reference_model = reference_model
            for auxiliary in (teacher_model, reference_model):
                if auxiliary is not None:
                    auxiliary.eval()
                    for parameter in auxiliary.parameters():
                        parameter.requires_grad_(False)
            super().__init__(*args, **kwargs)

        @staticmethod
        def _auxiliary_logits(model: Any, input_ids: Any, attention_mask: Any, device: Any) -> Any:
            try:
                auxiliary_device = next(model.parameters()).device
            except StopIteration:
                auxiliary_device = device
            with torch.no_grad():
                output = model(
                    input_ids=input_ids.to(auxiliary_device),
                    attention_mask=attention_mask.to(auxiliary_device),
                    use_cache=False,
                )
            return output.logits.to(device)

        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            del num_items_in_batch
            config = self.distillation_config
            batch_size, response_count, sequence_length = inputs["input_ids"].shape
            flat_input_ids = inputs["input_ids"].reshape(-1, sequence_length)
            flat_attention = inputs["attention_mask"].reshape(-1, sequence_length)
            flat_labels = inputs["labels"].reshape(-1, sequence_length)
            outputs = model(
                input_ids=flat_input_ids,
                attention_mask=flat_attention,
                use_cache=False,
            )
            sequence_logps, token_logps, _ = causal_token_log_probs(
                outputs.logits,
                flat_labels,
                average=config.average_log_prob,
            )
            student_scores = sequence_logps.reshape(batch_size, response_count)
            response_mask = inputs["response_mask"]

            if config.objective in {"vpd", "ppd"}:
                teacher_scores = inputs["teacher_scores"].to(student_scores.device)
                missing = response_mask & teacher_scores.isnan()
                if missing.any():
                    if self.teacher_model is None:
                        raise ConfigurationError(
                            "VPD/PPD requires teacher scores in the dataset or teacher_model"
                        )
                    teacher_logits = self._auxiliary_logits(
                        self.teacher_model,
                        flat_input_ids,
                        flat_attention,
                        outputs.logits.device,
                    )
                    inferred_scores, _, _ = causal_token_log_probs(
                        teacher_logits,
                        flat_labels,
                        average=config.average_log_prob,
                    )
                    inferred_scores = inferred_scores.reshape(batch_size, response_count)
                    teacher_scores = torch.where(missing, inferred_scores, teacher_scores)
                ranking_losses = []
                for row in range(batch_size):
                    count = int(response_mask[row].sum())
                    student_row = student_scores[row, :count].unsqueeze(0)
                    teacher_row = teacher_scores[row, :count].unsqueeze(0)
                    if config.objective == "vpd":
                        ranking_losses.append(vpd_loss(student_row, teacher_row, beta=config.beta))
                    else:
                        ranking_losses.append(
                            ppd_loss(
                                student_row,
                                teacher_row,
                                beta=config.beta,
                                exact_ranking_limit=config.exact_ranking_limit,
                            )
                        )
                loss = torch.cat(ranking_losses).mean()
            else:
                if response_count != 2 or not response_mask.all():
                    raise DatasetFormatError(
                        f"{config.objective.upper()} requires exactly chosen and rejected responses"
                    )
                vocab_size = outputs.logits.shape[-1]
                shifted_probs = outputs.logits[:, :-1, :].softmax(-1)
                shifted_labels = flat_labels[:, 1:]
                shifted_probs = shifted_probs.reshape(
                    batch_size, response_count, sequence_length - 1, vocab_size
                )
                shifted_labels = shifted_labels.reshape(
                    batch_size, response_count, sequence_length - 1
                )
                chosen_probs, rejected_probs = shifted_probs[:, 0], shifted_probs[:, 1]
                chosen_labels, rejected_labels = shifted_labels[:, 0], shifted_labels[:, 1]

                chosen_teacher = _teacher_field(
                    inputs, "teacher_chosen_probs", "chosen_compressed_probs"
                )
                rejected_teacher = _teacher_field(
                    inputs, "teacher_rejected_probs", "rejected_compressed_probs"
                )
                if config.objective == "dckd":
                    if chosen_teacher is None or rejected_teacher is None:
                        raise DatasetFormatError(
                            "DCKD requires teacher_chosen_probs and teacher_rejected_probs"
                        )
                    loss = config.chosen_kd_weight * compressed_forward_kl(
                        chosen_probs, chosen_labels, chosen_teacher
                    )
                    loss = loss + config.rejected_kd_weight * compressed_forward_kl(
                        rejected_probs, rejected_labels, rejected_teacher
                    )
                elif config.objective == "tvkd":
                    if chosen_teacher is None or rejected_teacher is None:
                        raise DatasetFormatError(
                            "TVKD requires teacher_chosen_probs and teacher_rejected_probs"
                        )
                    loss = tvkd_loss(
                        chosen_probs,
                        rejected_probs,
                        chosen_labels,
                        rejected_labels,
                        chosen_teacher,
                        rejected_teacher,
                        value_function=config.value_function,
                        value_discount=config.value_discount,
                        value_temperature=config.value_temperature,
                        student_value_weight=config.student_value_weight,
                        teacher_value_weight=config.teacher_value_weight,
                    ).mean()
                    if config.chosen_kd_weight > 0:
                        loss = loss + config.chosen_kd_weight * compressed_forward_kl(
                            chosen_probs, chosen_labels, chosen_teacher
                        )
                    if config.rejected_kd_weight > 0:
                        loss = loss + config.rejected_kd_weight * compressed_forward_kl(
                            rejected_probs, rejected_labels, rejected_teacher
                        )
                elif config.objective == "adpa":
                    advantages = inputs.get("rejected_margin_logp_every")
                    if advantages is None:
                        raise DatasetFormatError("ADPA requires rejected_margin_logp_every")
                    loss = advantage_expectation_loss(
                        rejected_probs, rejected_labels, advantages
                    )
                elif config.objective == "ctpd":
                    if self.reference_model is None:
                        raise ConfigurationError("CTPD requires reference_model")
                    required = (
                        "chosen_ctpd_parent_list",
                        "rejected_ctpd_parent_list",
                        "chosen_ctpd_weight",
                        "rejected_ctpd_weight",
                    )
                    missing = [key for key in required if key not in inputs]
                    if missing:
                        raise DatasetFormatError(
                            "CTPD dataset is missing: " + ", ".join(missing)
                        )
                    reference_logits = self._auxiliary_logits(
                        self.reference_model,
                        flat_input_ids,
                        flat_attention,
                        outputs.logits.device,
                    )
                    _, reference_token_logps, _ = causal_token_log_probs(
                        reference_logits,
                        flat_labels,
                        average=config.average_log_prob,
                    )
                    policy_tokens = token_logps.reshape(
                        batch_size, response_count, sequence_length - 1
                    )
                    reference_tokens = reference_token_logps.reshape(
                        batch_size, response_count, sequence_length - 1
                    )
                    chosen_weights = torch.nn.utils.rnn.pad_sequence(
                        [
                            torch.as_tensor(value, device=outputs.logits.device)
                            for value in inputs["chosen_ctpd_weight"]
                        ],
                        batch_first=True,
                    )
                    rejected_weights = torch.nn.utils.rnn.pad_sequence(
                        [
                            torch.as_tensor(value, device=outputs.logits.device)
                            for value in inputs["rejected_ctpd_weight"]
                        ],
                        batch_first=True,
                    )
                    loss = ctpd_loss(
                        policy_tokens[:, 0],
                        policy_tokens[:, 1],
                        reference_tokens[:, 0],
                        reference_tokens[:, 1],
                        inputs["chosen_ctpd_parent_list"],
                        inputs["rejected_ctpd_parent_list"],
                        chosen_weights,
                        rejected_weights,
                        beta=config.ctpd_beta,
                    ).mean()
                else:
                    raise ConfigurationError(f"Unsupported objective: {config.objective}")

            loss = config.objective_weight * loss

            if config.sft_weight > 0:
                chosen_flat_index = (
                    torch.arange(batch_size, device=outputs.logits.device) * response_count
                )
                chosen_logits = outputs.logits.index_select(0, chosen_flat_index)
                chosen_labels_full = flat_labels.index_select(0, chosen_flat_index)
                sft_loss = torch.nn.functional.cross_entropy(
                    chosen_logits[:, :-1, :].reshape(-1, chosen_logits.shape[-1]),
                    chosen_labels_full[:, 1:].reshape(-1),
                    ignore_index=-100,
                )
                loss = loss + config.sft_weight * sft_loss

            return (loss, outputs) if return_outputs else loss

    return PreferenceDistillationTrainer


def _load_model(value: Any, loader: Any, kwargs: Mapping[str, Any]) -> Any:
    return loader.from_pretrained(value, **dict(kwargs)) if isinstance(value, str) else value


class PreferenceDistillationBackend:
    """Execute VPD, PPD, DCKD, TVKD, ADPA, or CTPD with Transformers."""

    def train(
        self,
        *,
        method: str,
        model: Any,
        dataset: DatasetBundle,
        config: PreferenceDistillationConfig,
        options: Mapping[str, Any] | None = None,
    ) -> AlignmentRun:
        if method != "preference_distillation":
            raise ConfigurationError(f"Unsupported distillation backend method: {method}")
        if not isinstance(config, PreferenceDistillationConfig):
            raise ConfigurationError(
                "Preference distillation requires PreferenceDistillationConfig"
            )

        torch, transformers = _require_dependencies()
        options = dict(options or {})
        split = str(options.pop("dataset_split", "train"))
        eval_source = options.pop("eval_dataset", None)
        eval_split = str(options.pop("eval_dataset_split", "test"))
        tokenizer = options.pop("tokenizer", None)
        teacher_value = options.pop("teacher_model", None)
        reference_value = options.pop("reference_model", None)
        model_kwargs = dict(options.pop("model_kwargs", {}))
        teacher_kwargs = dict(options.pop("teacher_model_kwargs", {}))
        reference_kwargs = dict(options.pop("reference_model_kwargs", {}))
        trainer_options = dict(options.pop("trainer_options", {}))
        if options:
            raise ConfigurationError(
                "Unknown preference-distillation backend options: " + ", ".join(sorted(options))
            )

        model_loader = transformers.AutoModelForCausalLM
        student = _load_model(model, model_loader, model_kwargs)
        tokenizer_source = model if isinstance(model, str) else getattr(
            getattr(student, "config", None), "_name_or_path", None
        )
        if tokenizer is None:
            if not tokenizer_source:
                raise ConfigurationError(
                    "Pass tokenizer when the model object has no config._name_or_path"
                )
            tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_source)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ConfigurationError("Tokenizer requires a pad_token_id or eos_token_id")
            tokenizer.pad_token_id = tokenizer.eos_token_id

        teacher = (
            _load_model(teacher_value, model_loader, teacher_kwargs)
            if teacher_value is not None
            else None
        )
        reference = (
            _load_model(reference_value, model_loader, reference_kwargs)
            if reference_value is not None
            else None
        )
        train_data = _materialize(dataset.data, split=split)
        eval_data = _materialize(eval_source, split=eval_split) if eval_source is not None else None
        train_dataset = _TokenizedDataset(train_data, tokenizer, config)
        eval_dataset = (
            _TokenizedDataset(eval_data, tokenizer, config)
            if eval_data is not None
            else None
        )

        training_kwargs = config.trainer_kwargs()
        training_kwargs.pop("max_length", None)
        training_kwargs["remove_unused_columns"] = False
        training_args = transformers.TrainingArguments(**training_kwargs)
        trainer_class = _make_trainer_class(torch, transformers)
        import inspect

        processor_argument = (
            "processing_class"
            if "processing_class" in inspect.signature(transformers.Trainer.__init__).parameters
            else "tokenizer"
        )
        trainer = trainer_class(
            model=student,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=_DistillationCollator(torch, int(tokenizer.pad_token_id)),
            distillation_config=config,
            teacher_model=teacher,
            reference_model=reference,
            **{processor_argument: tokenizer},
            **trainer_options,
        )
        train_output = (
            trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
            if config.resume_from_checkpoint is not None
            else trainer.train()
        )
        trainer.save_model(config.output_dir)
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(config.output_dir)
        raw_metrics = getattr(train_output, "metrics", {}) or {}
        metrics = {
            str(key): float(value)
            for key, value in raw_metrics.items()
            if isinstance(value, (int, float))
        }

        def save(path: str) -> None:
            trainer.save_model(path)
            if hasattr(tokenizer, "save_pretrained"):
                tokenizer.save_pretrained(path)

        return AlignmentRun(
            model=trainer.model,
            tokenizer=tokenizer,
            metrics=metrics,
            output_dir=config.output_dir,
            metadata={
                "backend": "transformers-preference-distillation",
                "objective": config.objective,
            },
            _generator=make_generator(trainer.model, tokenizer),
            _saver=save,
        )
