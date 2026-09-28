"""Reward / cost model training (safe-rlhf ``values/{reward,cost}``)."""

from __future__ import annotations

import torch
from transformers import Trainer

from human_alignment.safety.data.tokenize import SAFE_RLHF_PROMPT, format_prompt, safe_rlhf_encode
from human_alignment.safety.losses.score import cost_model_loss, reward_model_loss


def score_pair_features(
    tokenizer, pair: dict, max_length: int, kind: str, template: str = SAFE_RLHF_PROMPT
) -> dict:
    """``kind='reward'``: pair from ``better``; ``kind='cost'``: pair from ``safer`` with safety signs (+1 safe)."""
    prompt = format_prompt(pair["prompt"], template)
    c = safe_rlhf_encode(tokenizer, prompt, pair["chosen"], max_length)
    r = safe_rlhf_encode(tokenizer, prompt, pair["rejected"], max_length)
    if c["input_ids"] == r["input_ids"]:
        raise ValueError("Two responses get the same `input_ids` after tokenization.")
    feat = {
        "chosen_input_ids": c["input_ids"],
        "chosen_labels": c["labels"],
        "rejected_input_ids": r["input_ids"],
        "rejected_labels": r["labels"],
    }
    if kind == "cost":
        s_c, s_r = 2 * int(pair["chosen_safe"]) - 1, 2 * int(pair["rejected_safe"]) - 1
        if s_c < s_r:
            raise ValueError("The safer answer is not safer than the unsafer answer.")
        feat["chosen_sign"], feat["rejected_sign"] = s_c, s_r
    return feat


class ScoreModelTrainer(Trainer):
    def __init__(
        self,
        *args,
        kind: str,
        loss_type: str = "sequence-wise",
        regularization: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if kind not in ("reward", "cost"):
            raise ValueError(kind)
        self.kind = kind
        self.loss_type = loss_type
        self.regularization = regularization
        self.model_accepts_loss_kwargs = False

    def score_loss(self, model, batch):
        ids = torch.cat([batch["chosen_input_ids"], batch["rejected_input_ids"]])
        mask = torch.cat(
            [batch["chosen_attention_mask"], batch["rejected_attention_mask"]]
        )
        out = model(ids, mask)
        if self.kind == "reward":
            return reward_model_loss(
                out.scores,
                out.end_scores,
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                self.loss_type,
                self.regularization,
            )
        return cost_model_loss(
            out.scores,
            out.end_scores,
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"],
            batch["chosen_sign"],
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"],
            batch["rejected_sign"],
            self.loss_type,
            self.regularization,
        )

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        out = self.score_loss(model, inputs)
        return (out["loss"], out) if return_outputs else out["loss"]

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            out = self.score_loss(model, inputs)
        return out["loss"].detach(), None, None

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        output_dir = output_dir or self.args.output_dir
        self.accelerator.unwrap_model(self.model).save_pretrained(output_dir)
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)

