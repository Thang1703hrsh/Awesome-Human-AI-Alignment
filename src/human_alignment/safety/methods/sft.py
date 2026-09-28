"""Supervised fine-tuning.

``safe_rlhf`` style (SFT baseline, ``safe_rlhf/datasets/supervised.py``): prompt (template) masked, CE on the
answer + EOS. ``cpsft`` style (CPO, ``CPO/src/CPSFT/cpsft/train_sft.py``): ``train_on_inputs=True`` so CE covers the
whole text; EOS appended when the text is shorter than the cutoff.
Loss = mean over label tokens of each micro-batch (``model(...).loss``), averaged over gradient accumulation,
which is what DeepSpeed/HF did in the references (we disable HF's token-count loss normalisation).
"""

from __future__ import annotations

from transformers import Trainer

from human_alignment.safety.data.tokenize import SAFE_RLHF_PROMPT, format_prompt, safe_rlhf_encode
from human_alignment.safety.losses.logps import IGNORE_INDEX


def sft_features_safe_rlhf(
    tokenizer,
    prompt: str,
    answer: str,
    max_length: int,
    template: str = SAFE_RLHF_PROMPT,
) -> dict:
    return safe_rlhf_encode(
        tokenizer, format_prompt(prompt, template), answer, max_length
    )


def sft_features_cpsft(tokenizer, text: str, cutoff_len: int) -> dict:
    ids = tokenizer(text, truncation=True, max_length=cutoff_len)["input_ids"]
    if ids[-1] != tokenizer.eos_token_id and len(ids) < cutoff_len:
        ids = ids + [tokenizer.eos_token_id]
    return {"input_ids": ids, "labels": list(ids)}


class SFTTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        out = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            use_cache=False,
        )
        return (out.loss, out) if return_outputs else out.loss


__all__ = ["SFTTrainer", "sft_features_safe_rlhf", "sft_features_cpsft", "IGNORE_INDEX"]

