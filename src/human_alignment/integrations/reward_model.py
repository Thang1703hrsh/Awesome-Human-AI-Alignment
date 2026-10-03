"""Scalar reward models through Hugging Face Transformers.

One scorer serves evaluator benchmarks (RewardBench 2, JudgeBench) and reward-guided inference (best-of-N, ARGS).
A conversation ``[user: prompt, assistant: response]`` is rendered with the reward model's chat template, as in
RewardBench's ``prepare_dialogue_from_tokenizer``; tokenizers without a template fall back to ``prompt + response``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from human_alignment.exceptions import BackendUnavailableError


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise BackendUnavailableError(
            "Reward models require `pip install human-ai-alignment[inference]`."
        ) from exc
    return torch


@dataclass
class TransformersRewardModel:
    """Wrap ``AutoModelForSequenceClassification`` (single-logit head) or an already loaded model/tokenizer pair."""

    model: Any
    tokenizer: Any
    max_length: int = 4096
    batch_size: int = 8
    device: Any = None
    _template: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self._template = bool(getattr(self.tokenizer, "chat_template", None))
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        if self.device is None:
            try:
                self.device = next(self.model.parameters()).device
            except (AttributeError, StopIteration):
                self.device = "cpu"

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        *,
        dtype: str = "bf16",
        device: str | None = None,
        max_length: int = 4096,
        batch_size: int = 8,
        **model_kwargs: Any,
    ) -> "TransformersRewardModel":
        torch = _torch()
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForSequenceClassification.from_pretrained(
            path, dtype=torch_dtype, **model_kwargs
        ).to(device)
        model.eval()
        return cls(model, AutoTokenizer.from_pretrained(path), max_length, batch_size, device)

    def render(self, prompt: str, response: str) -> str:
        if self._template:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
                tokenize=False,
            )
        return prompt + response

    def score_batch(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """Scalar reward for each ``(prompt, response)``; texts are batched with right padding."""
        torch = _torch()
        texts = [self.render(p, r) for p, r in pairs]
        scores: list[float] = []
        padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"
        try:
            for start in range(0, len(texts), self.batch_size):
                enc = self.tokenizer(
                    texts[start : start + self.batch_size],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    add_special_tokens=not self._template,
                )
                with torch.no_grad():
                    logits = self.model(
                        **{k: enc[k].to(self.device) for k in ("input_ids", "attention_mask")}
                    ).logits
                if logits.shape[-1] != 1:
                    raise ValueError(
                        f"expected a single-logit reward head, got {logits.shape[-1]} outputs"
                    )
                scores.extend(logits[:, 0].float().cpu().tolist())
        finally:
            self.tokenizer.padding_side = padding_side
        return scores

    def score(self, prompt: str, response: str) -> float:
        return self.score_batch([(prompt, response)])[0]

    __call__ = score
