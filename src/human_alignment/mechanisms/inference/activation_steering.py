"""Contrastive Activation Addition (CAA; Rimsky et al., 2024) for Hugging Face decoder-only models.

Port of ``nrimsky/CAA``: the steering vector for layer ℓ is the mean difference between residual-stream outputs of
decoder block ℓ on positive and negative completions of the same prompts (``generate_vectors.py``), read at one token
position (the source uses −2, the answer letter of ``(A``/``(B``; −1 here by default for free-text completions).
During generation ``multiplier · v`` is added to the block output from the end of the instruction onward
(``add_vector_from_position``): the last prompt token and every generated token. The base weights never change.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

from human_alignment.results import InferenceResult
from human_alignment.types import Generation, GenerationRequest


def decoder_layers(model: Any) -> Any:
    """The list of decoder blocks for common HF architectures (Llama/Qwen/Mistral/Gemma, GPT-2, GPT-NeoX, OPT)."""
    root = model
    for attr in ("base_model", "model"):  # unwrap PEFT wrappers
        if hasattr(root, "peft_config") and hasattr(root, attr):
            root = getattr(root, attr)
    for path in ("model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers", "layers"):
        node = root
        try:
            for part in path.split("."):
                node = getattr(node, part)
        except AttributeError:
            continue
        return node
    raise ValueError(f"cannot locate decoder layers on {type(model).__name__}")


def _hidden(output: Any) -> Any:
    return output[0] if isinstance(output, tuple) else output


def _replace_hidden(output: Any, hidden: Any) -> Any:
    return (hidden,) + tuple(output[1:]) if isinstance(output, tuple) else hidden


def encode(tokenizer: Any, text: str, add_special_tokens: bool, device: Any) -> dict:
    """Tokenize ``text`` for a decoder-only model (drops ``token_type_ids``, which decoders reject)."""
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=add_special_tokens)
    return {k: enc[k].to(device) for k in ("input_ids", "attention_mask") if k in enc}


def render_prompt(tokenizer: Any, prompt: str) -> tuple[str, bool]:
    """Prompt text ending where the assistant turn starts, and whether a chat template was applied."""
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        return text, True
    return prompt, False


def compute_caa_vector(
    model: Any,
    tokenizer: Any,
    examples: Sequence[tuple[str, str, str]],
    layer: int,
    position: int = -1,
):
    """Mean of ``h_ℓ(prompt + positive)[position] − h_ℓ(prompt + negative)[position]`` over ``examples``."""
    import torch

    if not examples:
        raise ValueError("CAA needs at least one (prompt, positive, negative) example")
    block = decoder_layers(model)[layer]
    captured: list[Any] = []
    handle = block.register_forward_hook(lambda _m, _i, out: captured.append(_hidden(out).detach()))
    device = next(model.parameters()).device
    diffs = []
    try:
        with torch.no_grad():
            for prompt, positive, negative in examples:
                prefix, templated = render_prompt(tokenizer, prompt)
                acts = []
                for completion in (positive, negative):
                    captured.clear()
                    model(**encode(tokenizer, prefix + completion, not templated, device))
                    acts.append(captured[-1][0, position].float())
                diffs.append(acts[0] - acts[1])
    finally:
        handle.remove()
    return torch.stack(diffs).mean(dim=0)


@dataclass
class ContrastiveActivationAddition:
    """Steer generation by adding ``multiplier · vector`` to the output of decoder block ``layer``."""

    model: Any
    tokenizer: Any
    vector: Any
    layer: int
    multiplier: float = 1.0
    apply_to: str = "instruction_end"  # "instruction_end" (source behaviour) or "all" prompt positions
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    name: str = "caa"

    def __post_init__(self) -> None:
        if self.apply_to not in ("instruction_end", "all"):
            raise ValueError("apply_to must be 'instruction_end' or 'all'")

    @classmethod
    def from_examples(
        cls, model: Any, tokenizer: Any, examples: Sequence[tuple[str, str, str]], layer: int,
        position: int = -1, **kwargs: Any,
    ) -> "ContrastiveActivationAddition":
        return cls(model, tokenizer, compute_caa_vector(model, tokenizer, examples, layer, position), layer, **kwargs)

    @contextlib.contextmanager
    def steering(self, multiplier: float | None = None) -> Iterator[None]:
        """Register the steering hook for the duration of the block (inputs must be left-padded)."""
        scale = self.multiplier if multiplier is None else multiplier
        block = decoder_layers(self.model)[self.layer]

        def hook(_module, _inputs, output):
            hidden = _hidden(output)
            delta = (scale * self.vector).to(device=hidden.device, dtype=hidden.dtype)
            if self.apply_to == "instruction_end" and hidden.shape[1] > 1:
                hidden = hidden.clone()
                hidden[:, -1, :] += delta  # prefill: only the last prompt token, i.e. the end of the instruction
            else:
                hidden = hidden + delta  # every prompt position, or a generated token under the KV cache
            return _replace_hidden(output, hidden)

        handle = block.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    def generate_text(self, prompt: str, multiplier: float | None = None, **generate_kwargs: Any) -> str:
        import torch

        text, templated = render_prompt(self.tokenizer, prompt)
        enc = encode(self.tokenizer, text, not templated, next(self.model.parameters()).device)
        kwargs = dict(max_new_tokens=self.max_new_tokens, pad_token_id=self.tokenizer.pad_token_id
                      or self.tokenizer.eos_token_id)
        if self.temperature > 0:
            kwargs.update(do_sample=True, temperature=self.temperature, top_p=self.top_p)
        else:
            kwargs.update(do_sample=False)
        kwargs.update(generate_kwargs)
        with self.steering(multiplier), torch.no_grad():
            out = self.model.generate(**enc, **kwargs)
        return self.tokenizer.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True)

    def generate(self, model: Any, request: GenerationRequest) -> InferenceResult:
        """``InferenceMechanism`` entry point; ``model`` is ignored in favour of the wrapped model."""
        output = self.generate_text(str(request.context.input), **dict(request.parameters))
        return InferenceResult(
            (Generation(output, metadata={"layer": self.layer, "multiplier": self.multiplier}),),
            self.name,
            {"vector_norm": float(self.vector.norm())},
        )
