"""One trainer for every offline pairwise method (HF ``Trainer`` subclass with a custom loss).

DPO-{Helpful,Harmless,SafeBetter}, SACPO (DPO stages), MoCAN/PeCAN (DPO on relabeled pairs), CDPO, SafeDPO, BSO,
BFPO (+ buffer replay), MODPO (margin = implicit reward of a second adapter/model), MidPO experts (margin from a
score model). Chosen and rejected are run through the policy as one concatenated batch, as in all references.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn as nn
from transformers import Trainer

from human_alignment.safety.losses import logps as LP
from human_alignment.safety.losses import preference as P

LossName = Literal["dpo", "safedpo", "bso", "bfpo", "modpo", "midpo_expert"]


@dataclass
class PairwiseLossConfig:
    loss: LossName = "dpo"
    beta: float = 0.1
    logp_mode: Literal["masked", "safe_rlhf"] = "masked"
    # SafeDPO
    delta: float = 10.0
    # BSO
    safety_penalty: float = 30.0
    bso_generator: str = "sba"
    bso_lambda: float = 0.2
    bso_s: float = 4.0
    bso_penalty_inside_beta: bool = False
    # BFPO
    b1: float = 3.0
    alpha: float = 0.5
    # MODPO
    modpo_w: list[float] = field(default_factory=lambda: [0.5, 0.5])
    modpo_loss_type: str = "sigmoid"
    margin_beta: float = 0.1
    # MidPO experts
    midpo_expert: str = "safety"


def sequence_logps(
    model: nn.Module, batch: dict[str, torch.Tensor], mode: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (chosen_logps, rejected_logps) for one concatenated forward pass."""
    n = batch["chosen_input_ids"].size(0)
    input_ids = torch.cat([batch["chosen_input_ids"], batch["rejected_input_ids"]])
    attention_mask = torch.cat(
        [batch["chosen_attention_mask"], batch["rejected_attention_mask"]]
    )
    logits = model(
        input_ids=input_ids, attention_mask=attention_mask, use_cache=False
    ).logits
    if mode == "masked":
        labels = torch.cat([batch["chosen_labels"], batch["rejected_labels"]])
        lps = LP.masked_sequence_logps(logits, labels)
        return lps[:n], lps[n:]
    if mode == "safe_rlhf":
        token_lp = LP.token_log_probs(logits, input_ids)
        spans_c, spans_r = LP.safe_rlhf_pair_spans(
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"],
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"],
        )
        return LP.sum_over_spans(token_lp[:n], spans_c), LP.sum_over_spans(
            token_lp[n:], spans_r
        )
    raise ValueError(f"Unknown logp_mode {mode!r}")


def adapter_disabled(model: nn.Module):
    unwrapped = getattr(model, "module", model)
    if hasattr(unwrapped, "disable_adapter"):
        return unwrapped.disable_adapter()
    raise ValueError("ref_model is None but the policy has no adapters to disable")


@contextlib.contextmanager
def active_adapter(model: nn.Module, name: str):
    unwrapped = getattr(model, "module", model)
    previous = unwrapped.active_adapter
    unwrapped.set_adapter(name)
    try:
        yield
    finally:
        unwrapped.set_adapter(previous)


class PairwiseTrainer(Trainer):
    def __init__(
        self,
        *args,
        loss_config: PairwiseLossConfig,
        ref_model: nn.Module | None = None,
        margin_model: nn.Module | None = None,
        margin_adapter: str | None = None,
        score_model: nn.Module | None = None,
        buffer_dataset=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_config = loss_config
        self.ref_model = ref_model
        self.margin_model = margin_model
        self.margin_adapter = margin_adapter
        self.score_model = score_model
        self.buffer_dataset = buffer_dataset
        self._buffer_iter = None
        self._buffer_epoch = None
        self.model_accepts_loss_kwargs = False
        for extra in (ref_model, margin_model, score_model):
            if extra is not None:
                extra.eval()
                extra.requires_grad_(False)
                extra.to(self.args.device)

    # ------------------------------------------------------------------ reference / margin quantities
    @torch.no_grad()
    def _ref_logps(self, model, batch):
        mode = self.loss_config.logp_mode
        if self.ref_model is not None:
            return sequence_logps(self.ref_model, batch, mode)
        with adapter_disabled(model):
            return sequence_logps(model, batch, mode)

    @torch.no_grad()
    def _modpo_margin(self, model, batch, ref_c, ref_r):
        """ImplicitRewardWrapper: β_m·(logπ_margin − logπ_ref), masked (SFT-style) log-probs."""
        if self.margin_model is not None:
            m_c, m_r = sequence_logps(self.margin_model, batch, "masked")
        elif self.margin_adapter is not None:
            with active_adapter(model, self.margin_adapter):
                m_c, m_r = sequence_logps(model, batch, "masked")
        else:
            raise ValueError("MODPO needs margin_model or margin_adapter")
        if self.loss_config.logp_mode != "masked":
            ref_c, ref_r = self._ref_logps_masked(model, batch)
        b = self.loss_config.margin_beta
        return (b * (m_c - ref_c)).unsqueeze(-1), (b * (m_r - ref_r)).unsqueeze(-1)

    @torch.no_grad()
    def _ref_logps_masked(self, model, batch):
        if self.ref_model is not None:
            return sequence_logps(self.ref_model, batch, "masked")
        with adapter_disabled(model):
            return sequence_logps(model, batch, "masked")

    @torch.no_grad()
    def _midpo_scores(self, batch):
        ids = torch.cat([batch["chosen_input_ids"], batch["rejected_input_ids"]])
        mask = torch.cat(
            [batch["chosen_attention_mask"], batch["rejected_attention_mask"]]
        )
        end = self.score_model(ids, mask).end_scores
        n = batch["chosen_input_ids"].size(0)
        return end[:n], end[n:]

    # ------------------------------------------------------------------ loss
    def pair_loss(self, model, batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cfg = self.loss_config
        pol_c, pol_r = sequence_logps(model, batch, cfg.logp_mode)
        ref_c, ref_r = self._ref_logps(model, batch)
        if cfg.loss == "dpo":
            out = P.dpo_loss(pol_c, pol_r, ref_c, ref_r, cfg.beta)
        elif cfg.loss == "safedpo":
            out = P.safedpo_loss(
                pol_c,
                pol_r,
                ref_c,
                ref_r,
                1 - batch["chosen_safe"].long(),
                1 - batch["rejected_safe"].long(),
                cfg.beta,
                cfg.delta,
            )
        elif cfg.loss == "bso":
            out = P.bso_loss(
                pol_c,
                pol_r,
                ref_c,
                ref_r,
                1 - batch["chosen_safe"].long(),
                1 - batch["rejected_safe"].long(),
                cfg.beta,
                cfg.safety_penalty,
                cfg.bso_generator,
                cfg.bso_lambda,
                cfg.bso_s,
                cfg.bso_penalty_inside_beta,
            )
        elif cfg.loss == "bfpo":
            out = P.bfpo_loss(
                pol_c,
                pol_r,
                ref_c,
                ref_r,
                batch["chosen_safe"].long(),
                batch["rejected_safe"].long(),
                cfg.beta,
                cfg.b1,
                cfg.alpha,
            )
        elif cfg.loss == "modpo":
            m_c, m_r = self._modpo_margin(model, batch, ref_c, ref_r)
            out = P.modpo_loss(
                pol_c,
                pol_r,
                ref_c,
                ref_r,
                m_c,
                m_r,
                cfg.modpo_w,
                cfg.beta,
                cfg.modpo_loss_type,
            )
        elif cfg.loss == "midpo_expert":
            s_c, s_r = self._midpo_scores(batch)
            out = P.midpo_expert_loss(
                pol_c, pol_r, ref_c, ref_r, s_c, s_r, cfg.beta, cfg.midpo_expert
            )
        else:
            raise ValueError(f"Unknown loss {cfg.loss!r}")
        metrics = {
            "rewards/chosen": out.chosen_rewards.mean(),
            "rewards/rejected": out.rejected_rewards.mean(),
            "rewards/accuracy": (out.chosen_rewards > out.rejected_rewards)
            .float()
            .mean(),
            "rewards/margin": (out.chosen_rewards - out.rejected_rewards).mean(),
        }
        return out.losses.mean(), metrics

    def _next_buffer_batch(self):
        epoch = int(self.state.epoch or 0)
        if self._buffer_iter is None or self._buffer_epoch != epoch:
            self._buffer_iter = iter(self.get_train_dataloader_for(self.buffer_dataset))
            self._buffer_epoch = epoch
        try:
            return next(self._buffer_iter)
        except StopIteration:
            self._buffer_iter = iter(self.get_train_dataloader_for(self.buffer_dataset))
            return next(self._buffer_iter)

    def get_train_dataloader_for(self, dataset):
        original = self.train_dataset
        self.train_dataset = dataset
        try:
            return self.get_train_dataloader()
        finally:
            self.train_dataset = original

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        loss, metrics = self.pair_loss(model, inputs)
        if self.buffer_dataset is not None and model.training:
            buffer_batch = self._prepare_inputs(self._next_buffer_batch())
            buffer_loss, _ = self.pair_loss(model, buffer_batch)
            metrics["buffer_loss"] = buffer_loss.detach()
            loss = loss + buffer_loss
        if (
            model.training
            and self.state.global_step % max(self.args.logging_steps, 1) == 0
        ):
            self._stash_metrics(metrics)
        return (loss, metrics) if return_outputs else loss

    def _stash_metrics(self, metrics):
        self._pending_metrics = {
            k: float(v.detach().float().mean()) for k, v in metrics.items()
        }

    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        pending = getattr(self, "_pending_metrics", None)
        if pending and "loss" in logs:
            logs = {**logs, **pending}
        super().log(logs, *args, **kwargs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss, _ = self.pair_loss(model, inputs)
        return loss.detach(), None, None


def pairwise_features(
    tokenizer,
    pair: dict,
    style: str,
    max_length: int,
    max_prompt_length: int,
    template: str,
) -> dict[str, Any]:
    from human_alignment.safety.data.tokenize import encode_pair

    feat = encode_pair(tokenizer, pair, style, max_length, max_prompt_length, template)
    feat["chosen_safe"] = int(pair.get("chosen_safe", True))
    feat["rejected_safe"] = int(pair.get("rejected_safe", True))
    return feat

