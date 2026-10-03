"""Reward-guided inference with a scalar reward model: best-of-N selection and ARGS decoding.

``RewardModelBestOfN`` samples N responses and returns the one the reward model scores highest.

``ARGSDecoding`` ports ``deeplearning-wisc/args`` (Khanov et al., 2024, ``argsearch.py``): at each step the policy's
top-k next tokens (by raw logit) are scored as ``logit + w · r(prompt, response so far + token)``; ``greedy`` takes
the argmax and ``sampling`` draws from ``softmax(score / temperature)`` (the source's ``topk`` method). The source
scores token ids with a reward model that shares the policy tokenizer; here candidates are decoded to text and
scored with the reward model's own template, so any reward model works at the cost of k reward evaluations per
token. The policy is decoded with its KV cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from human_alignment.benchmarks.common import score_pairs
from human_alignment.mechanisms.inference.activation_steering import encode, render_prompt
from human_alignment.results import InferenceResult
from human_alignment.types import Generation, GenerationRequest


def _device(model: Any) -> Any:
    return next(model.parameters()).device


@dataclass
class RewardModelBestOfN:
    model: Any
    tokenizer: Any
    reward_model: Any  # scorer(prompt, response) -> float, or an object with score_batch
    n: int = 4
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 1.0
    name: str = "reward-best-of-n"

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("n must be at least 1")
        if self.temperature <= 0:
            raise ValueError("best-of-N needs sampling (temperature > 0)")

    def candidates(self, prompt: str) -> list[str]:
        import torch

        text, templated = render_prompt(self.tokenizer, prompt)
        enc = encode(self.tokenizer, text, not templated, _device(self.model))
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=self.n,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        return self.tokenizer.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)

    def generate(self, model: Any, request: GenerationRequest) -> InferenceResult:
        prompt = str(request.context.input)
        texts = self.candidates(prompt)
        scores = score_pairs(self.reward_model, [(prompt, t) for t in texts])
        best = max(range(len(texts)), key=scores.__getitem__)
        return InferenceResult(
            (Generation(texts[best], scores[best], {"candidate_index": best}),),
            self.name,
            {"candidate_count": self.n, "candidate_scores": scores, "candidates": texts},
        )


@dataclass
class ARGSDecoding:
    model: Any
    tokenizer: Any
    reward_model: Any
    weight: float = 1.0
    topk: int = 10
    max_new_tokens: int = 128
    method: str = "greedy"  # "greedy" or "sampling"
    temperature: float = 0.7
    seed: int | None = None
    name: str = "args"

    def __post_init__(self) -> None:
        if self.method not in ("greedy", "sampling"):
            raise ValueError("method must be 'greedy' or 'sampling'")
        if self.topk < 1:
            raise ValueError("topk must be at least 1")

    def decode(self, prompt: str) -> tuple[str, list[float]]:
        """Return the response and the reward of each chosen prefix."""
        import torch

        text, templated = render_prompt(self.tokenizer, prompt)
        input_ids = encode(self.tokenizer, text, not templated, _device(self.model))["input_ids"]
        generator = None
        if self.seed is not None:
            generator = torch.Generator(device=input_ids.device).manual_seed(self.seed)
        generated: list[int] = []
        rewards: list[float] = []
        past, step_input = None, input_ids
        eos = self.tokenizer.eos_token_id
        with torch.no_grad():
            for _ in range(self.max_new_tokens):
                out = self.model(input_ids=step_input, past_key_values=past, use_cache=True)
                past = out.past_key_values
                logits = out.logits[0, -1].float()
                top_logits, top_tokens = torch.topk(logits, k=min(self.topk, logits.shape[-1]))
                texts = [
                    self.tokenizer.decode(generated + [int(t)], skip_special_tokens=True) for t in top_tokens
                ]
                reward = torch.tensor(
                    score_pairs(self.reward_model, [(prompt, t) for t in texts]), device=logits.device
                )
                scores = top_logits + self.weight * reward
                if self.method == "greedy":
                    index = int(torch.argmax(scores))
                else:
                    probs = torch.softmax(scores / self.temperature, dim=-1)
                    index = int(torch.multinomial(probs, 1, generator=generator))
                token = int(top_tokens[index])
                generated.append(token)
                rewards.append(float(reward[index]))
                if token == eos:
                    break
                step_input = top_tokens[index].view(1, 1)
        return self.tokenizer.decode(generated, skip_special_tokens=True), rewards

    def generate(self, model: Any, request: GenerationRequest) -> InferenceResult:
        output, rewards = self.decode(str(request.context.input))
        return InferenceResult(
            (Generation(output, rewards[-1] if rewards else None, {"weight": self.weight, "topk": self.topk}),),
            self.name,
            {"tokens": len(rewards), "prefix_rewards": rewards},
        )
