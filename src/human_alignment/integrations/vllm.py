"""Optional high-throughput generation through vLLM."""

from __future__ import annotations

from typing import Any, Mapping

from human_alignment.exceptions import BackendUnavailableError


class VLLMGenerator:
    def __init__(
        self,
        model: str,
        *,
        model_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            from vllm import LLM
        except ImportError as exc:
            raise BackendUnavailableError(
                "vLLM support requires `pip install human-ai-alignment[vllm]`."
            ) from exc
        self._llm = LLM(model=model, **dict(model_kwargs or {}))

    def generate(self, prompt: str, **sampling_kwargs: Any) -> str:
        return self.generate_many(prompt, n=1, **sampling_kwargs)[0]

    def generate_many(self, prompt: str, *, n: int = 4, **sampling_kwargs: Any) -> tuple[str, ...]:
        """Generate multiple candidates for preference-distillation data collection."""

        if n < 1:
            raise ValueError("n must be at least 1")
        from vllm import SamplingParams

        outputs = self._llm.generate([prompt], SamplingParams(n=n, **sampling_kwargs))
        return tuple(candidate.text for candidate in outputs[0].outputs)
