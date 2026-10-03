"""Batched chat-template generation for response-level benchmarks (AlpacaEval model outputs)."""

from __future__ import annotations

from typing import Sequence


def chat_generate(
    model_path: str,
    prompts: Sequence[str],
    *,
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    top_p: float = 1.0,
    batch_size: int = 8,
    dtype: str = "bf16",
    seed: int = 42,
) -> list[str]:
    """Generate one response per prompt; instruct tokenizers get their chat template, others the raw prompt."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch_dtype).to(device).eval()
    templated = bool(getattr(tokenizer, "chat_template", None))
    outputs: list[str] = []
    for start in range(0, len(prompts), batch_size):
        chunk = list(prompts[start : start + batch_size])
        texts = (
            [
                tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
                for p in chunk
            ]
            if templated
            else chunk
        )
        enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=not templated)
        enc = {k: enc[k].to(device) for k in ("input_ids", "attention_mask")}
        kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)
        if temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            kwargs.update(do_sample=False)
        with torch.no_grad():
            generated = model.generate(**enc, **kwargs)
        outputs.extend(
            tokenizer.batch_decode(generated[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        )
    return [o.strip() for o in outputs]
