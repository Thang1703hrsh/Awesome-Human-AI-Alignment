"""Hugging Face backend for preparing preference-distillation datasets."""

from __future__ import annotations

from itertools import combinations
from math import log
from pathlib import Path
from typing import Any, Mapping, Sequence

from human_alignment.config import PreferenceDistillationConfig
from human_alignment.exceptions import BackendUnavailableError, ConfigurationError
from human_alignment.integrations.preference_distillation import (
    _apply_messages,
    _is_messages,
    _normalize_row,
    _tokenize_response,
)
from human_alignment.mechanisms.training.distillation_losses import (
    causal_token_log_probs,
    compress_probabilities,
)
from human_alignment.supervision.preference_distillation import (
    attach_generated_response,
    build_adpa_record,
    common_parent_groups,
    merge_compressed_probabilities,
)


def _dependencies() -> tuple[Any, Any, Any]:
    try:
        import datasets
        import torch
        import transformers
    except ImportError as exc:
        raise BackendUnavailableError(
            "Data preparation requires `pip install human-ai-alignment[distillation]`."
        ) from exc
    return datasets, torch, transformers


def _load_source(datasets: Any, source: Any, split: str | None) -> Any:
    if not isinstance(source, (str, Path)):
        return source
    path = Path(source)
    if path.exists():
        loaded = datasets.load_from_disk(str(path))
        return loaded[split] if split and hasattr(loaded, "keys") and split in loaded else loaded
    if split:
        return datasets.load_dataset(str(source), split=split)
    return datasets.load_dataset(str(source))


def _as_splits(data: Any, datasets: Any) -> dict[str, Any]:
    if isinstance(data, datasets.DatasetDict):
        result = {}
        for name, split in data.items():
            aliases = {"train_prefs": "train", "test_prefs": "test"}
            output_name = aliases.get(name, name)
            result[output_name] = split
        return result
    return {"train": data}


def _device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return None


def _model_input(torch: Any, tokenized: Mapping[str, Sequence[int]], device: Any) -> dict[str, Any]:
    values = {
        "input_ids": torch.tensor([tokenized["input_ids"]], dtype=torch.long),
        "attention_mask": torch.tensor([tokenized["attention_mask"]], dtype=torch.long),
    }
    return {key: value.to(device) for key, value in values.items()} if device else values


def _compressed_response(
    model: Any,
    tokenizer: Any,
    prompt: str,
    response: str,
    config: PreferenceDistillationConfig,
    *,
    top_k: int,
    torch: Any,
) -> list[dict[str, Any]]:
    tokenized = _tokenize_response(tokenizer, prompt, response, config)
    inputs = _model_input(torch, tokenized, _device(model))
    labels = torch.tensor(
        [tokenized["labels"]], dtype=torch.long, device=inputs["input_ids"].device
    )
    with torch.no_grad():
        logits = model(**inputs, use_cache=False).logits
    valid = labels[:, 1:].ne(-100)[0]
    return compress_probabilities(logits[0, :-1][valid].float().cpu(), top_k=top_k)


def _sequence_score(
    model: Any,
    tokenizer: Any,
    prompt: str,
    response: str,
    config: PreferenceDistillationConfig,
    torch: Any,
) -> float:
    tokenized = _tokenize_response(tokenizer, prompt, response, config)
    inputs = _model_input(torch, tokenized, _device(model))
    labels = torch.tensor(
        [tokenized["labels"]], dtype=torch.long, device=inputs["input_ids"].device
    )
    with torch.no_grad():
        logits = model(**inputs, use_cache=False).logits
    scores, _, _ = causal_token_log_probs(logits, labels, average=True)
    return float(scores[0].detach().cpu())


def _pairwise_scores(
    model: Any,
    tokenizer: Any,
    prompt: str,
    responses: Sequence[str],
    torch: Any,
) -> list[float]:
    """Score candidates through order-balanced pairwise A/B judgments."""

    if len(responses) < 2:
        raise ConfigurationError("Pairwise scoring requires at least two responses")
    totals = [0.0] * len(responses)
    counts = [0] * len(responses)
    option_ids = []
    for option in (" A", " B"):
        ids = tokenizer(option, add_special_tokens=False)["input_ids"]
        if not ids:
            raise ConfigurationError("Teacher tokenizer cannot encode pairwise options")
        option_ids.append(int(ids[-1]))
    if option_ids[0] == option_ids[1]:
        raise ConfigurationError(
            "Teacher tokenizer maps pairwise options A and B to the same final token"
        )

    for left, right in combinations(range(len(responses)), 2):
        for first, second in ((left, right), (right, left)):
            judge_prompt = (
                "Choose the response that better follows the instruction. Reply with A or B.\n\n"
                f"Instruction:\n{prompt}\n\nA:\n{responses[first]}\n\n"
                f"B:\n{responses[second]}\n\nAnswer:"
            )
            encoded = tokenizer(judge_prompt, return_tensors="pt")
            device = _device(model)
            if device:
                encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                logits = model(**encoded, use_cache=False).logits[0, -1].float()
            probabilities = logits[option_ids].softmax(-1).detach().cpu().tolist()
            totals[first] += float(probabilities[0])
            totals[second] += float(probabilities[1])
            counts[first] += 1
            counts[second] += 1
    return [log(max(total / max(count, 1), 1e-8)) for total, count in zip(totals, counts)]


def _generate_responses(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    count: int,
    generation_kwargs: Mapping[str, Any],
    torch: Any,
) -> list[str]:
    encoded = tokenizer(prompt, return_tensors="pt")
    device = _device(model)
    if device:
        encoded = {key: value.to(device) for key, value in encoded.items()}
    parameters = {
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 0.9,
        "max_new_tokens": 512,
        "num_return_sequences": count,
        **dict(generation_kwargs),
    }
    with torch.no_grad():
        output = model.generate(**encoded, **parameters)
    prompt_length = encoded["input_ids"].shape[1]
    return [tokenizer.decode(ids[prompt_length:], skip_special_tokens=True) for ids in output]


def _generate_with_engine(
    engine: Any,
    prompt: str,
    *,
    count: int,
    generation_kwargs: Mapping[str, Any],
) -> list[str]:
    values = engine.generate_many(prompt, n=count, **dict(generation_kwargs))
    return [str(value) for value in values]


def _offsets(tokenizer: Any, text: str) -> list[tuple[int, int]]:
    try:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    except Exception as exc:
        raise ConfigurationError("CTPD requires fast tokenizers with offset mappings") from exc
    if "offset_mapping" not in encoded:
        raise ConfigurationError("CTPD requires fast tokenizers with offset mappings")
    return [tuple(map(int, offset)) for offset in encoded["offset_mapping"]]


def _token_scores(model: Any, tokenizer: Any, text: str, torch: Any) -> list[float]:
    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    device = _device(model)
    if device:
        encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded, use_cache=False).logits[0]
    ids = encoded["input_ids"][0]
    logps = logits[:-1].log_softmax(-1).gather(1, ids[1:, None]).squeeze(1)
    return [0.0, *[float(value) for value in logps.detach().cpu()]]


def _ctpd_side(
    prompt: str,
    response: str,
    *,
    student_tokenizer: Any,
    teacher_tokenizer: Any,
    teacher: Any,
    reference: Any,
    positive: bool,
    torch: Any,
) -> tuple[list[list[int]], list[float]]:
    student_full = prompt + response
    teacher_full = prompt + response
    prompt_chars = len(prompt)
    student_full_offsets = _offsets(student_tokenizer, student_full)
    teacher_full_offsets = _offsets(teacher_tokenizer, teacher_full)
    student_positions = [
        index for index, (_, stop) in enumerate(student_full_offsets) if stop > prompt_chars
    ]
    teacher_positions = [
        index for index, (_, stop) in enumerate(teacher_full_offsets) if stop > prompt_chars
    ]
    student_response_offsets = [
        (max(0, student_full_offsets[index][0] - prompt_chars),
         student_full_offsets[index][1] - prompt_chars)
        for index in student_positions
    ]
    teacher_response_offsets = [
        (max(0, teacher_full_offsets[index][0] - prompt_chars),
         teacher_full_offsets[index][1] - prompt_chars)
        for index in teacher_positions
    ]
    student_groups, teacher_groups = common_parent_groups(
        student_response_offsets, teacher_response_offsets
    )
    teacher_logps = _token_scores(teacher, teacher_tokenizer, teacher_full, torch)
    reference_logps = _token_scores(reference, teacher_tokenizer, teacher_full, torch)
    deltas = []
    for group in teacher_groups:
        positions = [teacher_positions[index] for index in group]
        deltas.append(
            sum(teacher_logps[index] - reference_logps[index] for index in positions)
        )
    sign = 1.0 if positive else -1.0
    weights = torch.exp(torch.tensor(deltas).mul(sign).clamp(-0.5, 1.5))
    rounded = (torch.round(weights * 100) / 100).tolist()
    if rounded:
        rounded[-1] = 0.0
    parent_positions = [[student_positions[index] for index in group] for group in student_groups]
    return parent_positions, [float(value) for value in rounded]


class DistillationDataBackend:
    """Prepare reusable datasets for all migrated preference-distillation objectives."""

    def prepare(
        self,
        *,
        objective: str,
        dataset: Any,
        output_dir: str,
        teacher_model: Any = None,
        reference_model: Any = None,
        student_model: Any = None,
        tokenizer: Any = None,
        split: str | None = None,
        top_k: int = 50,
        num_responses: int = 4,
        teacher_score_mode: str = "sequence_logprob",
        generation_backend: str = "transformers",
        model_kwargs: Mapping[str, Any] | None = None,
        generation_kwargs: Mapping[str, Any] | None = None,
        config: PreferenceDistillationConfig | None = None,
    ) -> Any:
        datasets, torch, transformers = _dependencies()
        if objective not in {"dckd", "tvkd", "adpa", "ctpd", "ppd", "vpd"}:
            raise ConfigurationError(f"Unknown distillation objective: {objective}")
        if config is None:
            defaults: dict[str, Any] = {"objective": objective}
            if objective == "dckd":
                defaults.update(chosen_kd_weight=0.1, rejected_kd_weight=0.1)
            config = PreferenceDistillationConfig(**defaults)
        loading = dict(model_kwargs or {})
        loader = transformers.AutoModelForCausalLM

        def load(value: Any) -> Any:
            if isinstance(value, str):
                return loader.from_pretrained(value, **loading).eval()
            return value

        if generation_backend not in {"transformers", "vllm"}:
            raise ConfigurationError(f"Unknown generation backend: {generation_backend}")
        generation_engine = None
        if generation_backend == "vllm" and student_model is not None:
            if not isinstance(student_model, str):
                raise ConfigurationError("vLLM generation requires a student model identifier")
            from human_alignment.integrations.vllm import VLLMGenerator

            generation_engine = VLLMGenerator(student_model)

        teacher = load(teacher_model) if teacher_model is not None else None
        reference = load(reference_model) if reference_model is not None else None
        needs_student_generation = objective in {"adpa", "ppd", "vpd"}
        student = (
            load(student_model)
            if student_model is not None
            and generation_backend == "transformers"
            and needs_student_generation
            else None
        )
        for model in (teacher, reference, student):
            if model is not None:
                model.eval()
        tokenizer_source = tokenizer or student_model or teacher_model
        if tokenizer_source is None:
            raise ConfigurationError("Pass tokenizer, student_model, or teacher_model")
        student_tokenizer = (
            transformers.AutoTokenizer.from_pretrained(tokenizer_source)
            if isinstance(tokenizer_source, str)
            else tokenizer_source
        )
        teacher_tokenizer = (
            transformers.AutoTokenizer.from_pretrained(teacher_model)
            if isinstance(teacher_model, str)
            else student_tokenizer
        )
        source = _load_source(datasets, dataset, split)
        output_splits = {}

        for split_name, source_split in _as_splits(source, datasets).items():
            prepared = []
            for raw in source_split:
                if (
                    objective in {"vpd", "ppd"}
                    and (student is not None or generation_engine is not None)
                    and "responses" not in raw
                    and not ("chosen" in raw and "rejected" in raw)
                ):
                    prompt_value = raw.get("prompt")
                    if prompt_value is None:
                        raise ConfigurationError("Candidate generation requires a prompt column")
                    prompt = (
                        _apply_messages(student_tokenizer, prompt_value, generate=True)
                        if _is_messages(prompt_value)
                        else str(prompt_value)
                    )
                    responses: Sequence[str] = ()
                else:
                    normalized = _normalize_row(raw, student_tokenizer, config)
                    prompt = normalized["prompt"]
                    responses = normalized["responses"]
                record = dict(raw)
                if objective in {"vpd", "ppd"}:
                    if generation_engine is not None:
                        responses = _generate_with_engine(
                            generation_engine,
                            prompt,
                            count=num_responses,
                            generation_kwargs=generation_kwargs or {},
                        )
                    elif student is not None:
                        responses = _generate_responses(
                            student,
                            student_tokenizer,
                            prompt,
                            count=num_responses,
                            generation_kwargs=generation_kwargs or {},
                            torch=torch,
                        )
                    if teacher is None:
                        raise ConfigurationError("VPD/PPD preparation requires teacher_model")
                    if teacher_score_mode == "pairwise":
                        scores = _pairwise_scores(
                            teacher, teacher_tokenizer, prompt, responses, torch
                        )
                    elif teacher_score_mode == "sequence_logprob":
                        scores = [
                            _sequence_score(
                                teacher, teacher_tokenizer, prompt, response, config, torch
                            )
                            for response in responses
                        ]
                    else:
                        raise ConfigurationError(
                            f"Unknown teacher score mode: {teacher_score_mode}"
                        )
                    record.update(prompt=prompt, responses=responses, teacher_scores=scores)
                elif objective in {"dckd", "tvkd"}:
                    if teacher is None:
                        raise ConfigurationError("DCKD/TVKD preparation requires teacher_model")
                    if len(responses) != 2:
                        raise ConfigurationError("DCKD/TVKD requires pairwise data")
                    compressed = [
                        _compressed_response(
                            teacher,
                            student_tokenizer,
                            prompt,
                            response,
                            config,
                            top_k=top_k,
                            torch=torch,
                        )
                        for response in responses
                    ]
                    record = merge_compressed_probabilities(record, *compressed)
                elif objective == "adpa":
                    if teacher is None or reference is None:
                        raise ConfigurationError("ADPA preparation requires teacher and reference")
                    if generation_engine is not None:
                        generated = _generate_with_engine(
                            generation_engine,
                            prompt,
                            count=1,
                            generation_kwargs=generation_kwargs or {},
                        )[0]
                    elif student is not None:
                        generated = _generate_responses(
                            student,
                            student_tokenizer,
                            prompt,
                            count=1,
                            generation_kwargs=generation_kwargs or {},
                            torch=torch,
                        )[0]
                    else:
                        generated = None
                    if generated is not None:
                        record = attach_generated_response(record, generated, side="rejected")
                        normalized = _normalize_row(record, student_tokenizer, config)
                        prompt = normalized["prompt"]
                        responses = normalized["responses"]
                    teacher_probs = _compressed_response(
                        teacher,
                        student_tokenizer,
                        prompt,
                        responses[1],
                        config,
                        top_k=top_k,
                        torch=torch,
                    )
                    reference_probs = _compressed_response(
                        reference,
                        student_tokenizer,
                        prompt,
                        responses[1],
                        config,
                        top_k=top_k,
                        torch=torch,
                    )
                    record = build_adpa_record(record, teacher_probs, reference_probs)
                else:
                    if teacher is None or reference is None:
                        raise ConfigurationError("CTPD preparation requires teacher and reference")
                    for index, side in enumerate(("chosen", "rejected")):
                        parents, weights = _ctpd_side(
                            prompt,
                            responses[index],
                            student_tokenizer=student_tokenizer,
                            teacher_tokenizer=teacher_tokenizer,
                            teacher=teacher,
                            reference=reference,
                            positive=side == "chosen",
                            torch=torch,
                        )
                        record[f"{side}_ctpd_parent_list"] = parents
                        record[f"{side}_ctpd_weight"] = weights
                prepared.append(record)
            output_splits[split_name] = datasets.Dataset.from_list(prepared)

        output = datasets.DatasetDict(output_splits)
        Path(output_dir).parent.mkdir(parents=True, exist_ok=True)
        output.save_to_disk(output_dir)
        return output
