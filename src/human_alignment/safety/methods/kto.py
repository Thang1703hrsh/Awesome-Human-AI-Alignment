"""KTO as used by SACPO (TRL ``KTOTrainer`` at commit 4219cbf, pinned in ``sacpo/requirements.txt``).

Data: unpaired rows ``{prompt, completion, label}`` (``sacpo/src/util.py:get_pku_by_{helpfulness,safety}``).
KL term: completions shuffled within chunks of 1000 (``get_KL_dataset``, python ``random.sample``) give mismatched
(prompt, completion) pairs; desirable/undesirable subsets are interleaved (``all_exhausted``) then shuffled.
SACPO sets desirable_weight = 1, undesirable_weight = n_desirable / n_undesirable.
"""

from __future__ import annotations

import random

import torch
from datasets import Dataset, interleave_datasets
from transformers import Trainer

from human_alignment.safety.data.tokenize import _build_tokenized_answer, _pad
from human_alignment.safety.losses import logps as LP
from human_alignment.safety.losses import preference as P
from human_alignment.safety.losses.logps import IGNORE_INDEX
from human_alignment.safety.methods.pairwise import adapter_disabled


def kto_rows(rows, by: str) -> list[dict]:
    """``by='better'``: (better, True), (worse, False) per pair; ``by='safe'``: each response with its safety label."""
    out = []
    for r in rows:
        if by == "better":
            b = int(r["better_response_id"])
            out.append(
                {"prompt": r["prompt"], "completion": r[f"response_{b}"], "label": True}
            )
            out.append(
                {
                    "prompt": r["prompt"],
                    "completion": r[f"response_{1 - b}"],
                    "label": False,
                }
            )
        elif by == "safe":
            out.append(
                {
                    "prompt": r["prompt"],
                    "completion": r["response_0"],
                    "label": bool(r["is_response_0_safe"]),
                }
            )
            out.append(
                {
                    "prompt": r["prompt"],
                    "completion": r["response_1"],
                    "label": bool(r["is_response_1_safe"]),
                }
            )
        else:
            raise ValueError(by)
    return out


def class_weights(rows: list[dict]) -> tuple[float, float]:
    n_des = sum(1 for r in rows if r["label"])
    return 1.0, float(n_des) / float(len(rows) - n_des)


def trl_encode_single(
    tokenizer, prompt: str, completion: str, max_length: int, max_prompt_length: int
):
    """``KTOTrainer.tokenize_row`` (decoder-only branch)."""
    c = _build_tokenized_answer(tokenizer, prompt, completion)
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    p_ids = bos + c["prompt_input_ids"]
    ans = c["input_ids"] + [tokenizer.eos_token_id]
    if len(p_ids) + len(ans) > max_length:
        p_ids = p_ids[-max_prompt_length:]
    if len(p_ids) + len(ans) > max_length:
        ans = ans[: max_length - max_prompt_length]
    ids = p_ids + ans
    labels = [IGNORE_INDEX] * len(p_ids) + ans
    return ids, labels


def build_kto_dataset(
    rows: list[dict],
    tokenizer,
    template: str,
    max_length: int,
    max_prompt_length: int,
    seed: int,
    data_seed: int | None = None,
) -> Dataset:
    """Tokenise rows + KL rows, interleave desirable/undesirable, shuffle — same order of operations as TRL."""
    random.seed(seed)
    kl_completions = []
    comps = [r["completion"] for r in rows]
    for start in range(0, len(comps), 1000):
        chunk = comps[start : start + 1000]
        kl_completions += random.sample(chunk, len(chunk))
    feats = {
        "input_ids": [],
        "labels": [],
        "kl_input_ids": [],
        "kl_labels": [],
        "label": [],
    }
    for r, kl_c in zip(rows, kl_completions):
        prompt = template.format(input=r["prompt"])
        ids, labels = trl_encode_single(
            tokenizer, prompt, r["completion"], max_length, max_prompt_length
        )
        kl_ids, kl_labels = trl_encode_single(
            tokenizer, prompt, kl_c, max_length, max_prompt_length
        )
        feats["input_ids"].append(ids)
        feats["labels"].append(labels)
        feats["kl_input_ids"].append(kl_ids)
        feats["kl_labels"].append(kl_labels)
        feats["label"].append(bool(r["label"]))
    ds = Dataset.from_dict(feats)
    desirable = ds.filter(lambda x: x["label"])
    undesirable = ds.filter(lambda x: not x["label"])
    inter = interleave_datasets(
        [desirable, undesirable], stopping_strategy="all_exhausted"
    )
    return inter.shuffle(seed=data_seed if data_seed is not None else seed)


class KTOCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        out = {}
        for prefix in ("", "kl_"):
            ids = [f[f"{prefix}input_ids"] for f in features]
            out[f"{prefix}input_ids"] = _pad(ids, self.pad_token_id)
            out[f"{prefix}labels"] = _pad(
                [f[f"{prefix}labels"] for f in features], IGNORE_INDEX
            )
            out[f"{prefix}attention_mask"] = _pad([[1] * len(i) for i in ids], 0).bool()
        out["label"] = torch.tensor([bool(f["label"]) for f in features])
        return out


def _logps(model, ids, mask, labels):
    return LP.masked_sequence_logps(
        model(input_ids=ids, attention_mask=mask, use_cache=False).logits, labels
    )


class KTOTrainer(Trainer):
    def __init__(
        self,
        *args,
        beta: float,
        desirable_weight: float,
        undesirable_weight: float,
        ref_model=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.desirable_weight = desirable_weight
        self.undesirable_weight = undesirable_weight
        self.ref_model = ref_model
        self.model_accepts_loss_kwargs = False
        if ref_model is not None:
            ref_model.eval().requires_grad_(False).to(self.args.device)

    def kto_batch_loss(self, model, batch):
        policy = _logps(
            model, batch["input_ids"], batch["attention_mask"], batch["labels"]
        )
        with torch.no_grad():
            policy_kl = _logps(
                model,
                batch["kl_input_ids"],
                batch["kl_attention_mask"],
                batch["kl_labels"],
            )
            ctx = adapter_disabled(model) if self.ref_model is None else torch.no_grad()
            ref_model = model if self.ref_model is None else self.ref_model
            with ctx:
                ref = _logps(
                    ref_model,
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["labels"],
                )
                ref_kl = _logps(
                    ref_model,
                    batch["kl_input_ids"],
                    batch["kl_attention_mask"],
                    batch["kl_labels"],
                )
        kl = P.kto_kl_estimate(policy_kl, ref_kl)
        losses, rewards = P.kto_loss(
            policy,
            ref,
            batch["label"],
            kl,
            self.beta,
            self.desirable_weight,
            self.undesirable_weight,
        )
        n_des = int(batch["label"].sum())
        # TRL appends a NaN placeholder when one side is empty and takes .mean() over it: the gradient is
        # sum/(B + 1); we reproduce that scaling with a finite loss.
        denom = losses.numel() + int(n_des == 0) + int(n_des == losses.numel())
        loss = losses.sum() / denom
        return loss, {"kl": kl, "rewards/mean": rewards.mean()}

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        loss, metrics = self.kto_batch_loss(model, inputs)
        return (loss, metrics) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss, _ = self.kto_batch_loss(model, inputs)
        return loss.detach(), None, None

