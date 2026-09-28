"""Deterministic batched generation on local JSON or Hugging Face datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch

from human_alignment.safety.data.tokenize import SAFE_RLHF_PROMPT, format_prompt

PROMPT_COLUMNS = ("prompt", "instruction", "question", "input")


def prompt_from_row(row: dict, column: str | None = None) -> str:
    if column:
        value = row.get(column)
        if not isinstance(value, str):
            raise ValueError(f"prompt column {column!r} is absent or not text")
        return value
    for key in PROMPT_COLUMNS:
        if isinstance(row.get(key), str) and row[key].strip():
            return row[key]
    raise ValueError(f"could not find a prompt column among {PROMPT_COLUMNS}")


def load_prompt_rows(
    dataset: str,
    split: str = "test",
    config: str | None = None,
    prompt_column: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    from datasets import load_dataset

    path = Path(dataset)
    if path.suffix in (".json", ".jsonl"):
        ds = load_dataset("json", data_files=str(path), split="train")
    else:
        ds = load_dataset(dataset, config, split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return [
        {
            "id": row.get("id", i),
            "prompt": prompt_from_row(row, prompt_column),
            "source": dataset,
        }
        for i, row in enumerate(ds)
    ]


def _load_model_and_tokenizer(model_path: str, dtype: str = "bf16"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[dtype]
    adapter_config = Path(model_path) / "adapter_config.json"
    if adapter_config.exists():
        from peft import AutoPeftModelForCausalLM

        model = AutoPeftModelForCausalLM.from_pretrained(model_path, dtype=torch_dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, tokenizer, device


@torch.no_grad()
def generate_rows(
    model_path: str,
    rows: Iterable[dict],
    output: str | Path,
    batch_size: int = 8,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int = 42,
    template: str = SAFE_RLHF_PROMPT,
    dtype: str = "bf16",
) -> Path:
    model, tokenizer, device = _load_model_and_tokenizer(model_path, dtype)
    torch.manual_seed(seed)
    rows = list(rows)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as fh:
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            prompts = [format_prompt(r["prompt"], template) for r in chunk]
            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=tokenizer.model_max_length,
            ).to(device)
            kwargs = dict(
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            if temperature > 0:
                kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
            else:
                kwargs.update(do_sample=False)
            generated = model.generate(**enc, **kwargs)
            lengths = enc["attention_mask"].sum(dim=1)
            # Left-padded generation appends after the padded input width, not after each non-pad length.
            responses = tokenizer.batch_decode(
                generated[:, enc["input_ids"].shape[1] :], skip_special_tokens=True
            )
            for row, response, prompt_len in zip(chunk, responses, lengths.tolist()):
                record = {
                    **row,
                    "response": response.strip(),
                    "model": model_path,
                    "prompt_tokens": prompt_len,
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output


def read_jsonl(path: str | Path) -> list[dict]:
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]

