"""Tokenization of prompts/pairs in the two conventions used by the reference code.

``safe_rlhf``: ``tokenize(prompt + answer + eos)`` with special tokens, right-truncated at ``max_length``
(``safe_rlhf/datasets/preference.py:58-59``, ``base.py:338-357``). Prompt length for label masking is
``len(tokenize(prompt))`` (``datasets/supervised.py``).

``trl``: port of TRL 0.8.x ``DPOTrainer.build_tokenized_answer`` + ``tokenize_row`` (decoder-only branch),
used by SACPO, CAN, CPO, BFPO, MODPO: BOS + prompt, answer + EOS, prompt truncated (keep_end) then answer.
"""

from __future__ import annotations

from typing import Literal

import torch

from human_alignment.safety.losses.logps import IGNORE_INDEX

PROMPT_BEGIN = "BEGINNING OF CONVERSATION: "
PROMPT_USER = "USER: {input} "
PROMPT_ASSISTANT = "ASSISTANT:"
SAFE_RLHF_PROMPT = PROMPT_BEGIN + PROMPT_USER + PROMPT_ASSISTANT

TokenizationStyle = Literal["safe_rlhf", "trl"]


def format_prompt(user_input: str, template: str = SAFE_RLHF_PROMPT) -> str:
    return template.format(input=user_input)


def _tokenize(tokenizer, text: str, max_length: int | None) -> list[int]:
    return tokenizer(
        text,
        add_special_tokens=True,
        truncation=max_length is not None,
        max_length=max_length,
    )["input_ids"]


def safe_rlhf_encode(
    tokenizer, prompt: str, answer: str, max_length: int
) -> dict[str, list[int]]:
    input_ids = _tokenize(tokenizer, prompt + answer + tokenizer.eos_token, max_length)
    prompt_len = len(_tokenize(tokenizer, prompt, max_length))
    labels = list(input_ids)
    labels[:prompt_len] = [IGNORE_INDEX] * min(prompt_len, len(labels))
    return {"input_ids": input_ids, "labels": labels}


def _build_tokenized_answer(
    tokenizer, prompt: str, answer: str
) -> dict[str, list[int]]:
    full = tokenizer(prompt + answer, add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    start = len(prompt_ids)
    if prompt_ids != full[:start]:
        start -= 1
    return {"prompt_input_ids": full[:start], "input_ids": full[start:]}


def trl_encode_pair(
    tokenizer,
    prompt: str,
    chosen: str,
    rejected: str,
    max_length: int,
    max_prompt_length: int,
) -> dict[str, list[int]]:
    c = _build_tokenized_answer(tokenizer, prompt, chosen)
    r = _build_tokenized_answer(tokenizer, prompt, rejected)
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    p_len = min(len(c["prompt_input_ids"]), len(r["prompt_input_ids"]))
    p_ids = p_ids[:p_len]
    n_diff = sum(a != b for a, b in zip(c["prompt_input_ids"], r["prompt_input_ids"]))
    if n_diff > 1 or abs(len(c["prompt_input_ids"]) - len(r["prompt_input_ids"])) > 1:
        raise ValueError(
            "Chosen and rejected prompt_input_ids might only differ on the last token."
        )
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    prompt_toks = {"prompt_input_ids": bos + p_ids}
    for t in (c, r):
        t["prompt_input_ids"] = bos + t["prompt_input_ids"]
        t["input_ids"] = t["input_ids"] + [tokenizer.eos_token_id]
    longer = max(len(c["input_ids"]), len(r["input_ids"]))
    for t in (c, r, prompt_toks):
        if len(t["prompt_input_ids"]) + longer > max_length:
            t["prompt_input_ids"] = t["prompt_input_ids"][-max_prompt_length:]
    for t in (c, r):
        if len(t["prompt_input_ids"]) + longer > max_length:
            t["input_ids"] = t["input_ids"][: max_length - max_prompt_length]
    out = {"prompt_input_ids": prompt_toks["prompt_input_ids"]}
    for name, t in (("chosen", c), ("rejected", r)):
        ids = t["prompt_input_ids"] + t["input_ids"]
        labels = list(ids)
        labels[: len(t["prompt_input_ids"])] = [IGNORE_INDEX] * len(
            t["prompt_input_ids"]
        )
        out[f"{name}_input_ids"] = ids
        out[f"{name}_labels"] = labels
    return out


def encode_pair(
    tokenizer,
    pair: dict,
    style: TokenizationStyle,
    max_length: int,
    max_prompt_length: int = 128,
    template: str = SAFE_RLHF_PROMPT,
) -> dict[str, list[int]]:
    prompt = format_prompt(pair["prompt"], template)
    if style == "safe_rlhf":
        c = safe_rlhf_encode(tokenizer, prompt, pair["chosen"], max_length)
        r = safe_rlhf_encode(tokenizer, prompt, pair["rejected"], max_length)
        if c["input_ids"] == r["input_ids"]:
            raise ValueError(
                "Two responses get the same `input_ids` after tokenization."
            )
        return {
            "chosen_input_ids": c["input_ids"],
            "chosen_labels": c["labels"],
            "rejected_input_ids": r["input_ids"],
            "rejected_labels": r["labels"],
        }
    if style == "trl":
        return trl_encode_pair(
            tokenizer,
            prompt,
            pair["chosen"],
            pair["rejected"],
            max_length,
            max_prompt_length,
        )
    raise ValueError(f"Unknown tokenization style {style!r}")


def _pad(seqs: list[list[int]], value: int) -> torch.Tensor:
    return torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(s, dtype=torch.long) for s in seqs],
        batch_first=True,
        padding_value=value,
    )


class PairCollator:
    """Right-pads chosen and rejected *together* to one length, as safe-rlhf's ``PreferenceCollator`` does
    (this also fixes whether the safe-rlhf "one past EOS" position exists). Extra per-example scalar fields
    listed in ``extra_keys`` are stacked as tensors."""

    def __init__(self, pad_token_id: int, extra_keys: tuple[str, ...] = ()):
        self.pad_token_id = pad_token_id
        self.extra_keys = extra_keys

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        n = len(features)
        ids = _pad(
            [f["chosen_input_ids"] for f in features]
            + [f["rejected_input_ids"] for f in features],
            self.pad_token_id,
        )
        labels = _pad(
            [f["chosen_labels"] for f in features]
            + [f["rejected_labels"] for f in features],
            IGNORE_INDEX,
        )
        mask = _pad(
            [[1] * len(f["chosen_input_ids"]) for f in features]
            + [[1] * len(f["rejected_input_ids"]) for f in features],
            0,
        ).bool()
        batch = {
            "chosen_input_ids": ids[:n],
            "rejected_input_ids": ids[n:],
            "chosen_attention_mask": mask[:n],
            "rejected_attention_mask": mask[n:],
            "chosen_labels": labels[:n],
            "rejected_labels": labels[n:],
        }
        for key in self.extra_keys:
            batch[key] = torch.tensor([f[key] for f in features])
        return batch


class SFTCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        return {
            "input_ids": _pad([f["input_ids"] for f in features], self.pad_token_id),
            "labels": _pad([f["labels"] for f in features], IGNORE_INDEX),
            "attention_mask": _pad(
                [[1] * len(f["input_ids"]) for f in features], 0
            ).bool(),
        }

