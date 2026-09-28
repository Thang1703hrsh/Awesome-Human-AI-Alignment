"""HF Trainer implementation for native preference objectives."""

import torch
from torch import nn
from transformers import Trainer

from human_alignment.mechanisms.training.preference.losses import (
    importance_weights, preference_loss, rank_weights, token_statistics, triplet_loss,
)


class PreferenceModel(nn.Module):
    """Keep the TBPO-Q baseline in the optimizer and resumable Trainer state."""

    def __init__(self, policy, use_baseline=False):
        super().__init__()
        self.policy = policy
        self.config = policy.config
        self.baseline = nn.Linear(policy.config.hidden_size, 1) if use_baseline else None

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        out = self.policy(input_ids=input_ids, inputs_embeds=inputs_embeds,
                          attention_mask=attention_mask, use_cache=False,
                          output_hidden_states=self.baseline is not None)
        baselines = None
        if self.baseline is not None:
            # State features are detached, as in the released Q-TBPO implementation.
            hidden = out.hidden_states[-1].detach().to(self.baseline.weight.dtype)
            baselines = self.baseline(hidden).squeeze(-1)
        return {"logits": out.logits, "baselines": baselines}

    def gradient_checkpointing_enable(self, **kwargs):
        return self.policy.gradient_checkpointing_enable(**kwargs)


def attribution(model, ids, attention, labels, config):
    """Gradient of max next-token logit at the last valid position, wrt embeddings."""
    was_training = model.training
    model.eval()
    try:
        with torch.enable_grad():
            embeds = model.get_input_embeddings()(ids).detach().requires_grad_(True)
            logits = model(inputs_embeds=embeds, attention_mask=attention, use_cache=False).logits
            positions = torch.arange(ids.shape[1], device=ids.device).expand_as(ids)
            last = positions.masked_fill(~attention.bool(), -1).max(-1).values
            target = logits[torch.arange(len(ids), device=ids.device), last].max(-1).values.sum()
            grad, = torch.autograd.grad(target, embeds)
        return importance_weights(grad.detach().float().abs().sum(-1), labels.ne(-100),
                                  config.importance_mix, config.prior_sigma_div)
    finally:
        model.train(was_training)


class NativePreferenceTrainer(Trainer):
    def __init__(self, *args, method, preference_config, reference_model,
                 positive_model=None, negative_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = method
        self.preference_config = preference_config
        self.model_accepts_loss_kwargs = False
        self.reference_model = self._prepare_frozen(reference_model)
        self.positive_model = self._prepare_frozen(positive_model) if positive_model is not None else None
        self.negative_model = self._prepare_frozen(negative_model) if negative_model is not None else None

    def _prepare_frozen(self, model):
        model.requires_grad_(False)
        model.eval()
        return self.accelerator.prepare_model(model, evaluation_mode=True)

    def create_optimizer(self):
        if self.optimizer is None:
            wrapped = self.accelerator.unwrap_model(self.model)
            decay = set(self.get_decay_parameter_names(wrapped.policy))
            groups = [
                {"params": [p for name, p in wrapped.policy.named_parameters()
                            if p.requires_grad and (name in decay) == use_decay],
                 "weight_decay": self.args.weight_decay if use_decay else 0.0}
                for use_decay in (True, False)
            ]
            if wrapped.baseline is not None:
                groups.append({"params": wrapped.baseline.parameters(),
                               "lr": self.preference_config.baseline_learning_rate,
                               "weight_decay": 0.0})
            optimizer_cls, kwargs = self.get_optimizer_cls_and_kwargs(self.args, self.model)
            self.optimizer = optimizer_cls(groups, **kwargs)
        return self.optimizer

    def _anchor_ratios(self, policy, ids, labels):
        """Sample from each prompt, then score the sampled response with gradients."""
        prompts = [row[:int((lab != -100).nonzero()[0])] for row, lab in zip(ids, labels)]
        width = max(len(row) for row in prompts)
        tokenizer = self.processing_class
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        inputs = ids.new_full((len(prompts), width), pad)
        attention = ids.new_zeros(inputs.shape)
        for i, row in enumerate(prompts):
            inputs[i, -len(row):] = row
            attention[i, -len(row):] = 1
        cfg = self.preference_config
        was_training = policy.training
        policy.eval()
        try:
            with torch.no_grad():
                generated = policy.generate(input_ids=inputs, attention_mask=attention,
                    max_new_tokens=min(cfg.anchor_max_new_tokens, cfg.max_length - width),
                    do_sample=True, temperature=0.8, top_p=0.95, top_k=50,
                    pad_token_id=pad, eos_token_id=tokenizer.eos_token_id)
        finally:
            policy.train(was_training)
        anchor_labels = generated.clone()
        anchor_labels[:, :width] = -100
        mask = torch.ones_like(generated)
        mask[:, :width] = attention
        # Include the first EOS; mask later generation padding even when PAD == EOS.
        for i in range(len(generated)):
            eos = (generated[i, width:] == tokenizer.eos_token_id).nonzero()
            if len(eos):
                end = width + int(eos[0]) + 1
                anchor_labels[i, end:] = -100
                mask[i, end:] = 0
        logits = policy(input_ids=generated, attention_mask=mask, use_cache=False).logits
        with torch.no_grad():
            reference = self.reference_model(input_ids=generated, attention_mask=mask, use_cache=False).logits
        ratio, _, _, valid = token_statistics(logits, reference, anchor_labels)
        return ratio, valid

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        ids, attention, labels = (torch.cat([inputs[f"chosen_{key}"], inputs[f"rejected_{key}"]])
                                  for key in ("input_ids", "attention_mask", "labels"))
        cfg = self.preference_config
        policy = self.accelerator.unwrap_model(model).policy
        weights = None
        if self.method == "ti_dpo":
            weights = attribution(policy, ids, attention, labels, cfg)
        elif self.method == "tis_dpo":
            if "chosen_weights" in inputs:
                weights = torch.cat([inputs["chosen_weights"], inputs["rejected_weights"]])
            else:
                if self.positive_model is None or self.negative_model is None:
                    raise ValueError("TIS-DPO requires chosen_weights/rejected_weights or positive_model and negative_model")
                with torch.no_grad():
                    pos = self.positive_model(input_ids=ids, attention_mask=attention, use_cache=False).logits
                    neg = self.negative_model(input_ids=ids, attention_mask=attention, use_cache=False).logits
                    # Released rank transformation: chosen log p_pos/p_neg; rejected opposite.
                    scores, _, _, valid = token_statistics(pos, neg, labels)
                    scores[len(ids) // 2:] *= -1
                    # Upstream assigns EOS zero importance.
                    valid &= labels[:, 1:].ne(self.processing_class.eos_token_id)
                    weights = torch.zeros_like(labels, dtype=torch.float32)
                    weights[:, 1:] = rank_weights(scores, valid)
        output = model(input_ids=ids, attention_mask=attention)
        with torch.no_grad():
            reference = self.reference_model(input_ids=ids, attention_mask=attention, use_cache=False).logits
        losses = preference_loss(self.method, output["logits"], reference, labels, cfg,
                                 weights=weights, baselines=output["baselines"])
        if self.method == "ti_dpo" and cfg.triplet_weight:
            n = len(ids) // 2
            ratios, _, _, valid = token_statistics(output["logits"], reference, labels)
            anchor, anchor_mask = self._anchor_ratios(policy, ids[:n], labels[:n])
            losses = losses + cfg.triplet_weight * triplet_loss(
                anchor, ratios[:n], ratios[n:], anchor_mask, valid[:n], valid[n:], cfg.triplet_margin)
        loss = losses.mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite preference loss; check logits, weights, and Bregman scaling")
        return (loss, {"loss": loss}) if return_outputs else loss
