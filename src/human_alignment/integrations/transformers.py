"""Model loading and generation through Hugging Face Transformers."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from human_alignment.config import CheckpointConfig
from human_alignment.exceptions import BackendUnavailableError
from human_alignment.results import AlignmentRun


def _require_transformers() -> Any:
    try:
        import transformers
    except ImportError as exc:
        raise BackendUnavailableError(
            "Transformers support requires `pip install human-ai-alignment[inference]`."
        ) from exc
    return transformers


def make_generator(model: Any, tokenizer: Any) -> Callable[..., str]:
    def generate(prompt: str, **parameters: Any) -> str:
        parameters = {"max_new_tokens": 128, **parameters}
        inputs = tokenizer(prompt, return_tensors="pt")
        try:
            device = next(model.parameters()).device
            inputs = {key: value.to(device) for key, value in inputs.items()}
        except (AttributeError, StopIteration):
            pass
        output_ids = model.generate(**inputs, **parameters)
        prompt_length = inputs["input_ids"].shape[1]
        return tokenizer.decode(output_ids[0][prompt_length:], skip_special_tokens=True)

    return generate


def load_transformers_model(
    model_id_or_path: str,
    *,
    config: CheckpointConfig | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
    tokenizer_kwargs: Mapping[str, Any] | None = None,
) -> AlignmentRun:
    transformers = _require_transformers()
    config = config or CheckpointConfig()
    loading = dict(model_kwargs or {})
    for key, value in {
        "device_map": config.device_map,
        "trust_remote_code": config.trust_remote_code,
        "revision": config.revision,
    }.items():
        if value is not None:
            loading.setdefault(key, value)
    if config.torch_dtype:
        if config.torch_dtype == "auto":
            loading.setdefault("torch_dtype", "auto")
        else:
            try:
                import torch
            except ImportError as exc:
                raise BackendUnavailableError("torch_dtype requires PyTorch") from exc
            try:
                loading.setdefault("torch_dtype", getattr(torch, config.torch_dtype))
            except AttributeError as exc:
                raise ValueError(f"Unknown torch dtype: {config.torch_dtype}") from exc

    loader = (
        transformers.AutoModelForCausalLM
        if config.task == "causal_lm"
        else transformers.AutoModelForSequenceClassification
    )
    model = loader.from_pretrained(model_id_or_path, **loading)
    if config.adapter_path:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise BackendUnavailableError(
                "Loading an adapter requires `pip install human-ai-alignment[peft]`."
            ) from exc
        model = PeftModel.from_pretrained(model, config.adapter_path)
        if config.merge_adapter:
            model = model.merge_and_unload()
    model.eval()

    tokenizer_source = config.tokenizer_path or model_id_or_path
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_source, **dict(tokenizer_kwargs or {})
    )

    def save(path: str) -> None:
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)

    return AlignmentRun(
        model=model,
        tokenizer=tokenizer,
        method="loaded-model",
        metadata={
            "source": model_id_or_path,
            "adapter": config.adapter_path,
            "task": config.task,
        },
        _generator=make_generator(model, tokenizer) if config.task == "causal_lm" else None,
        _saver=save,
    )
