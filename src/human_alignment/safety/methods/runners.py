"""Runners: one function per method family, driven by an ``ExperimentConfig``.

Precision rule (single GPU): trainable parameters are always fp32 master weights; frozen weights are bf16 and all
forward passes run under bf16 autocast (HF ``bf16=True``). Full fine-tuning of a 7B model therefore needs DeepSpeed
ZeRO with CPU optimizer offload (``train.deepspeed: recipes/safety_alignment/deepspeed/zero2_offload.json``), which keeps bf16 weights
on the GPU and fp32 master weights + Adam states on the CPU — the same setup safe-rlhf uses with ``--offload``.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset

from human_alignment.safety.data import cpo as CPO
from human_alignment.safety.data import pairs as D
from human_alignment.safety.data import sources as S
from human_alignment.safety.data.tokenize import PairCollator, SFTCollator, format_prompt
from human_alignment.safety.schema import ExperimentConfig, dump_config

log = logging.getLogger("human_alignment.safety")


# ============================================================ shared helpers
def _dtype(name):
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def output_dir(cfg: ExperimentConfig) -> Path:
    return Path(cfg.train.get("output_dir", f"output/{cfg.method}"))


def training_args(cfg: ExperimentConfig, **forced):
    from transformers import TrainingArguments

    defaults = dict(
        output_dir=str(output_dir(cfg)),
        remove_unused_columns=False,
        report_to=[],
        logging_steps=10,
        save_strategy="no",
        bf16=torch.cuda.is_available(),
        tf32=torch.cuda.is_available(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        lr_scheduler_type="cosine",
        dataloader_num_workers=2,
        seed=42,
        data_seed=None,
    )
    args = {
        **defaults,
        **{k: v for k, v in cfg.train.items() if k != "param_groups"},
        **forced,
    }
    if not torch.cuda.is_available():
        args.update(bf16=False, tf32=False, use_cpu=True)
    return TrainingArguments(**args)


def safe_rlhf_optimizer_mixin(trainer_cls):
    """safe-rlhf optimiser grouping: no weight decay for names containing 'bias' or 'layernorm.weight';
    warmup steps = int(total * ratio) (``trainers/base.py`` / ``rl_trainer.py:_init_train_engine``)."""

    class _Mixed(trainer_cls):
        def get_decay_parameter_names(self, model):
            return [
                n
                for n, p in model.named_parameters()
                if not any(nd in n.lower() for nd in ("bias", "layernorm.weight"))
            ]

        def create_scheduler(self, num_training_steps, optimizer=None):
            # transformers >= 5 stores a ratio as 0 < warmup_steps < 1 (and rounds up); older versions use
            # warmup_ratio. safe-rlhf rounds down.
            ratio = self.args.warmup_steps if 0 < self.args.warmup_steps < 1 else None
            if ratio is None and self.args.warmup_steps == 0:
                ratio = getattr(self.args, "warmup_ratio", None) or None
            if ratio:
                self.args.warmup_steps = int(num_training_steps * ratio)
                if getattr(self.args, "warmup_ratio", None):
                    self.args.warmup_ratio = 0.0
            return super().create_scheduler(num_training_steps, optimizer)

    _Mixed.__name__ = f"SafeRLHFOpt{trainer_cls.__name__}"
    return _Mixed


def maybe_mixin(cfg, trainer_cls):
    return (
        safe_rlhf_optimizer_mixin(trainer_cls)
        if cfg.train.get("param_groups") == "safe_rlhf"
        else trainer_cls
    )


def load_tokenizer(cfg: ExperimentConfig, path: str | None = None):
    from transformers import AutoTokenizer

    from human_alignment.safety.methods.models import add_missing_special_tokens

    tok = AutoTokenizer.from_pretrained(
        path or cfg.model.policy, model_max_length=cfg.data.max_length
    )
    add_missing_special_tokens(tok)
    return tok


def _uses_deepspeed(cfg) -> bool:
    return bool(cfg.train.get("deepspeed"))


def load_policy(cfg: ExperimentConfig, tokenizer, path: str | None = None):
    """Returns the trainable policy (full, LoRA, or selective) with fp32 trainable params."""
    from transformers import AutoModelForCausalLM

    from human_alignment.safety.methods.models import add_missing_special_tokens

    path = path or cfg.model.policy
    full_ft = cfg.model.lora is None and not cfg.model.trainable_substrings
    if full_ft and not _uses_deepspeed(cfg):
        dtype = torch.float32
        if torch.cuda.is_available():
            log.warning(
                "Full fine-tuning without DeepSpeed: loading the policy in fp32 (needs ~16 bytes/param). "
                "For 7B models on one H100 set train.deepspeed to "
                "recipes/safety_alignment/deepspeed/zero2_offload.json."
            )
    else:
        dtype = _dtype(cfg.model.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=dtype, attn_implementation=cfg.model.attn_implementation
    )
    add_missing_special_tokens(tokenizer, model)
    model.config.use_cache = False
    if cfg.model.lora is not None:
        from peft import LoraConfig, get_peft_model

        spec = cfg.model.lora
        model = get_peft_model(
            model,
            LoraConfig(
                r=spec.r,
                lora_alpha=spec.alpha,
                lora_dropout=spec.dropout,
                target_modules=list(spec.target_modules),
                modules_to_save=spec.modules_to_save,
                task_type="CAUSAL_LM",
            ),
        )
        if cfg.train.get("gradient_checkpointing", True):
            model.enable_input_require_grads()
    elif cfg.model.trainable_substrings:
        for n, p in model.named_parameters():
            p.requires_grad_(any(s in n for s in cfg.model.trainable_substrings))
    for p in model.parameters():
        if (
            p.requires_grad
            and p.dtype != torch.float32
            and not (full_ft and _uses_deepspeed(cfg))
        ):
            p.data = p.data.float()
    return model


def load_ref(cfg: ExperimentConfig, tokenizer):
    from transformers import AutoModelForCausalLM

    from human_alignment.safety.methods.models import add_missing_special_tokens

    if cfg.model.lora is not None and cfg.model.ref is None:
        return None
    ref = AutoModelForCausalLM.from_pretrained(
        cfg.model.ref or cfg.model.policy,
        dtype=_dtype(cfg.model.dtype),
        attn_implementation=cfg.model.attn_implementation,
    )
    add_missing_special_tokens(tokenizer, ref)
    ref.config.use_cache = False
    return ref


def _pku_rows(cfg, split=None, limit=None):
    return S.load_pku(
        cfg.data.dataset,
        split or cfg.data.train_split,
        cfg.data.revision,
        cfg.data.fraction,
        cfg.data.shuffle_seed,
        limit if limit is not None else cfg.data.limit,
        cfg.data.holdout,
        cfg.data.holdout_seed,
    )


def _eval_rows(cfg):
    if not cfg.data.eval_split:
        return None
    return S.load_pku(
        cfg.data.dataset,
        cfg.data.eval_split,
        cfg.data.revision,
        limit=cfg.data.eval_limit,
    )


def _pair_dataset(cfg, tokenizer, pairs: list[dict]) -> Dataset:
    from human_alignment.safety.methods.pairwise import pairwise_features

    feats, skipped = [], 0
    for p in pairs:
        try:
            feats.append(
                pairwise_features(
                    tokenizer,
                    p,
                    cfg.data.tokenization,
                    cfg.data.max_length,
                    cfg.data.max_prompt_length,
                    cfg.data.template,
                )
            )
        except ValueError:
            skipped += 1
    if skipped:
        log.warning("skipped %d pairs whose responses tokenize identically", skipped)
    return Dataset.from_list(feats)


def _finish(cfg, trainer, tokenizer, extra: dict | None = None):
    out = output_dir(cfg)
    trainer.save_model(str(out))
    if tokenizer is not None:
        tokenizer.save_pretrained(str(out))
    dump_config(cfg, out / "experiment.yaml")
    if extra:
        (out / "run_info.json").write_text(json.dumps(extra, indent=2, default=str))
    return out


# ============================================================ SFT
def run_sft(cfg: ExperimentConfig):
    from human_alignment.safety.methods.sft import (
        SFTTrainer,
        sft_features_cpsft,
        sft_features_safe_rlhf,
    )

    tok = load_tokenizer(cfg)
    if cfg.data.source == "alpaca":
        rows = S.load_alpaca(
            cfg.data.limit,
            cfg.data.dataset
            if cfg.data.dataset.endswith((".json", ".jsonl"))
            else "tatsu-lab/alpaca",
        )
        feats = [
            sft_features_safe_rlhf(
                tok, r["prompt"], r["answer"], cfg.data.max_length, cfg.data.template
            )
            for r in rows
        ]
    elif cfg.data.source == "pku":
        pairs = D.build_pairs(_pku_rows(cfg), cfg.data.selection)
        feats = [
            sft_features_safe_rlhf(
                tok, p["prompt"], p["chosen"], cfg.data.max_length, cfg.data.template
            )
            for p in pairs
        ]
    elif cfg.data.source == "cpsft_jsonl":
        lines = [
            ln
            for p in cfg.data.paths
            for ln in Path(p).read_text().splitlines()
            if ln.strip()
        ]
        feats = [
            sft_features_cpsft(tok, CPO.cpsft_text(ex), cfg.data.max_length)
            for ex in CPO.cpsft_examples(lines)
        ]
    elif cfg.data.source == "cpsft_hf":
        if cfg.data.paths:
            lines = [
                ln
                for p in cfg.data.paths
                for ln in Path(p).read_text().splitlines()
                if ln.strip()
            ]
            examples = CPO.cpsft_examples(lines)
        else:
            from datasets import load_dataset

            ds = load_dataset(cfg.data.dataset, split=cfg.data.train_split)
            if cfg.data.limit is not None:
                ds = ds.select(range(min(cfg.data.limit, len(ds))))
            examples = CPO.cpsft_examples_from_rows(list(ds))
        feats = [
            sft_features_cpsft(tok, CPO.cpsft_text(ex), cfg.data.max_length)
            for ex in examples
        ]
    else:
        raise ValueError(f"SFT source {cfg.data.source!r}")
    model = load_policy(cfg, tok)
    trainer_cls = maybe_mixin(cfg, SFTTrainer)
    trainer = trainer_cls(
        model=model,
        args=training_args(cfg),
        train_dataset=Dataset.from_list(feats),
        data_collator=SFTCollator(tok.pad_token_id),
        processing_class=tok,
    )
    trainer.train()
    return _finish(cfg, trainer, tok)


# ============================================================ reward / cost models
def run_score_model(cfg: ExperimentConfig):
    from human_alignment.safety.methods.models import load_score_model
    from human_alignment.safety.methods.score_trainer import ScoreModelTrainer, score_pair_features

    kind = cfg.method_args.get(
        "kind", "reward" if cfg.method == "reward_model" else "cost"
    )
    tok = load_tokenizer(cfg)
    selection = "better" if kind == "reward" else "safer"
    feats, dropped = [], 0
    for p in D.build_pairs(_pku_rows(cfg), selection):
        try:
            feats.append(
                score_pair_features(
                    tok, p, cfg.data.max_length, kind, cfg.data.template
                )
            )
        except ValueError:
            dropped += 1
    if dropped:
        log.warning(
            "dropped %d pairs (identical tokens or inconsistent safety labels)", dropped
        )
    model = load_score_model(
        cfg.model.policy,
        kind,
        dtype="fp32" if not _uses_deepspeed(cfg) else cfg.model.dtype,
        attn_implementation=cfg.model.attn_implementation,
        from_causal_lm=True,
    )
    extra = ("chosen_sign", "rejected_sign") if kind == "cost" else ()
    trainer_cls = maybe_mixin(cfg, ScoreModelTrainer)
    trainer = trainer_cls(
        model=model,
        args=training_args(cfg),
        train_dataset=Dataset.from_list(feats),
        data_collator=PairCollator(tok.pad_token_id, extra),
        processing_class=tok,
        kind=kind,
        loss_type=cfg.method_args.get("loss_type", "sequence-wise"),
        regularization=cfg.method_args.get("regularization", 0.001),
    )
    trainer.train()
    return _finish(cfg, trainer, tok)


# ============================================================ pairwise family
def _mocan_pairs(cfg, tok, rows):
    """CAN collect.py: score PROMPT + ' ' + response (no EOS) with RM and CM; relabel with λ."""
    from human_alignment.safety.methods.models import load_score_model

    a = cfg.method_args
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    scores = {}
    for key, path in (("reward", a["reward_model"]), ("cost", a["cost_model"])):
        sm = load_score_model(path, key, dtype=cfg.model.dtype).to(dev).eval()
        vals = {0: [], 1: []}
        for i in range(0, len(rows), a.get("score_batch_size", 16)):
            chunk = rows[i : i + a.get("score_batch_size", 16)]
            for j in (0, 1):
                texts = [
                    format_prompt(r["prompt"]) + " " + r[f"response_{j}"] for r in chunk
                ]
                enc = tok(
                    texts,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=cfg.data.max_length,
                    truncation=True,
                )
                ids = enc["input_ids"].to(dev)
                mask = ids.ne(tok.pad_token_id) & ids.ne(tok.unk_token_id)
                with (
                    torch.no_grad(),
                    torch.autocast(torch.device(dev).type, dtype=torch.bfloat16),
                ):
                    vals[j] += sm(ids, mask).end_scores.float().tolist()
        scores[key] = vals
        del sm
        torch.cuda.empty_cache()
    torch.manual_seed(cfg.train.get("seed", 42))
    logits = D.mocan_logits(
        scores["reward"][0],
        scores["reward"][1],
        scores["cost"][0],
        scores["cost"][1],
        a["lam"],
    )
    return D.bernoulli_relabel(rows, logits)


def _pecan_pairs(cfg, tok, rows):
    """CAN log_prob_alg2.py + dpo_alg2.py: sequence log-probs of the helpful-DPO, safe-DPO and reference models."""
    from transformers import AutoModelForCausalLM

    from human_alignment.safety.data.tokenize import trl_encode_pair
    from human_alignment.safety.losses.logps import masked_sequence_logps

    a = cfg.method_args
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lp = {}
    for key, path in (
        ("help", a["helpful_model"]),
        ("safety", a["safe_model"]),
        ("ref", cfg.model.ref or cfg.model.policy),
    ):
        m = (
            AutoModelForCausalLM.from_pretrained(path, dtype=_dtype(cfg.model.dtype))
            .to(dev)
            .eval()
        )
        vals = {0: [], 1: []}
        for r in rows:
            enc = trl_encode_pair(
                tok,
                format_prompt(r["prompt"]),
                r["response_0"],
                r["response_1"],
                cfg.data.max_length,
                cfg.data.max_prompt_length,
            )
            for j, name in ((0, "chosen"), (1, "rejected")):
                ids = torch.tensor([enc[f"{name}_input_ids"]], device=dev)
                labels = torch.tensor([enc[f"{name}_labels"]], device=dev)
                with (
                    torch.no_grad(),
                    torch.autocast(torch.device(dev).type, dtype=torch.bfloat16),
                ):
                    vals[j].append(
                        masked_sequence_logps(m(input_ids=ids).logits, labels).item()
                    )
        lp[key] = vals
        del m
        torch.cuda.empty_cache()
    torch.manual_seed(cfg.train.get("seed", 42))
    logits = D.pecan_logits(
        lp["help"][0],
        lp["help"][1],
        lp["safety"][0],
        lp["safety"][1],
        lp["ref"][0],
        lp["ref"][1],
        a["lam"],
    )
    return D.bernoulli_relabel(rows, logits)


def _pairs_for(cfg, tok) -> list[dict]:
    src = cfg.data.source
    if src == "pku":
        rows = _pku_rows(cfg)
        if cfg.method == "mocan":
            return _mocan_pairs(cfg, tok, rows)
        if cfg.method == "pecan":
            return _pecan_pairs(cfg, tok, rows)
        return list(D.build_pairs(rows, cfg.data.selection))
    if src == "cpo_ultrasafety":
        if cfg.data.paths:
            data = S.load_json(cfg.data.paths[0])
        else:
            from datasets import load_dataset

            data = list(load_dataset(cfg.data.dataset, split=cfg.data.train_split))
        if data and "completions" in data[0]:
            data = CPO.flatten_annotated_rows(data)
        a = cfg.method_args
        pairs = CPO.build_cdpo_pairs(
            data,
            a.get("has_harmless", CPO.ULTRASAFETY_CFG["has_harmless"]),
            a.get("random_cfg", CPO.ULTRASAFETY_CFG["random_cfg"]),
            a.get("variant", "ultrasafety"),
            rng=random.Random(cfg.train.get("seed", 42)),
        )
        pairs = CPO.further_process(pairs)
        if cfg.data.limit:
            pairs = pairs[: cfg.data.limit]
        return [
            {
                "prompt": p["prompt"],
                "chosen": p["chosen"],
                "rejected": p["rejected"],
                "chosen_safe": True,
                "rejected_safe": True,
            }
            for p in pairs
        ]
    raise ValueError(f"pairwise source {src!r}")


LOSS_FOR_METHOD = {
    "dpo": "dpo",
    "dpo_helpful": "dpo",
    "dpo_harmless": "dpo",
    "dpo_safebetter": "dpo",
    "sacpo_dpo": "dpo",
    "mocan": "dpo",
    "pecan": "dpo",
    "cdpo": "dpo",
    "modpo_margin": "dpo",
    "safedpo": "safedpo",
    "bso": "bso",
    "bfpo": "bfpo",
    "modpo": "modpo",
    "midpo_safety_expert": "midpo_expert",
    "midpo_helpfulness_expert": "midpo_expert",
}


def run_pairwise(cfg: ExperimentConfig):
    from human_alignment.safety.methods.pairwise import PairwiseLossConfig, PairwiseTrainer

    tok = load_tokenizer(cfg)
    pairs = _pairs_for(cfg, tok)
    train_ds = _pair_dataset(cfg, tok, pairs)
    eval_ds = None
    if cfg.data.source == "pku" and cfg.data.eval_split:
        eval_ds = _pair_dataset(
            cfg, tok, list(D.build_pairs(_eval_rows(cfg), cfg.data.selection))
        )
    margs = {
        k: v
        for k, v in cfg.method_args.items()
        if k in PairwiseLossConfig.__dataclass_fields__
    }
    loss_cfg = PairwiseLossConfig(
        loss=LOSS_FOR_METHOD[cfg.method],
        logp_mode="masked" if cfg.data.tokenization == "trl" else "safe_rlhf",
        **margs,
    )
    if cfg.method.startswith("midpo_"):
        loss_cfg.midpo_expert = "safety" if "safety" in cfg.method else "helpfulness"
    model = load_policy(cfg, tok)
    kwargs: dict[str, Any] = {"ref_model": load_ref(cfg, tok)}
    if cfg.method == "modpo":
        model.load_adapter(
            cfg.method_args["margin_adapter"], adapter_name="margin_reward"
        )
        model.set_adapter("default")
        kwargs["margin_adapter"] = "margin_reward"
    if cfg.method.startswith("midpo_"):
        from human_alignment.safety.methods.models import load_score_model

        kwargs["score_model"] = load_score_model(
            cfg.method_args["score_model"], dtype=cfg.model.dtype
        )
    if cfg.method == "bfpo" and cfg.method_args.get("use_buffer", True):
        buf = S.load_ultrafeedback_pairs(
            cfg.method_args.get("buffer_split", "train_prefs"),
            limit=cfg.method_args.get("buffer_limit"),
            path=cfg.method_args.get(
                "buffer_dataset", "HuggingFaceH4/ultrafeedback_binarized"
            ),
        )
        kwargs["buffer_dataset"] = _pair_dataset(cfg, tok, buf)
    trainer_cls = maybe_mixin(cfg, PairwiseTrainer)
    trainer = trainer_cls(
        model=model,
        args=training_args(cfg),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=PairCollator(tok.pad_token_id, ("chosen_safe", "rejected_safe")),
        processing_class=tok,
        loss_config=loss_cfg,
        **kwargs,
    )
    trainer.train()
    return _finish(cfg, trainer, tok, {"num_train_pairs": len(train_ds)})


# ============================================================ KTO (SACPO)
def run_kto(cfg: ExperimentConfig):
    from human_alignment.safety.methods.kto import (
        KTOCollator,
        KTOTrainer,
        build_kto_dataset,
        class_weights,
        kto_rows,
    )

    tok = load_tokenizer(cfg)
    rows = kto_rows(_pku_rows(cfg), cfg.method_args.get("label_by", "safe"))
    dw, uw = class_weights(rows)
    seed = cfg.train.get("seed", 42)
    ds = build_kto_dataset(
        rows,
        tok,
        cfg.data.template,
        cfg.data.max_length,
        cfg.data.max_prompt_length,
        seed,
        cfg.train.get("data_seed"),
    )
    model = load_policy(cfg, tok)
    trainer = maybe_mixin(cfg, KTOTrainer)(
        model=model,
        args=training_args(cfg),
        train_dataset=ds,
        data_collator=KTOCollator(tok.pad_token_id),
        processing_class=tok,
        beta=cfg.method_args.get("beta", 0.1),
        desirable_weight=dw,
        undesirable_weight=uw,
        ref_model=load_ref(cfg, tok),
    )
    trainer.train()
    return _finish(
        cfg, trainer, tok, {"desirable_weight": dw, "undesirable_weight": uw}
    )


# ============================================================ MidPO router
def run_midpo_router(cfg: ExperimentConfig):
    from transformers import AutoModelForCausalLM

    from human_alignment.safety.methods.midpo import MidPORouter, MidPORouterTrainer
    from human_alignment.safety.methods.pairwise import PairwiseLossConfig

    a = cfg.method_args
    tok = load_tokenizer(cfg)
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model.policy,
        dtype=_dtype(cfg.model.dtype),
        attn_implementation=cfg.model.attn_implementation,
    )
    base.config.use_cache = False
    router = MidPORouter(
        base,
        r=a.get("expert_r", 16),
        expert_scale=a.get("expert_scale", 1.0),
        router_hidden=a.get("router_hidden", 512),
        router_nonlinearity=a.get("router_nonlinearity", False),
        reg_source=a.get("reg_source", "layer_input"),
    )
    router.load_experts(a["safety_expert"], a["helpfulness_expert"])
    router.freeze_all_but_router()
    train_ds = _pair_dataset(cfg, tok, _pairs_for(cfg, tok))
    trainer = maybe_mixin(cfg, MidPORouterTrainer)(
        model=router,
        args=training_args(cfg),
        train_dataset=train_ds,
        data_collator=PairCollator(tok.pad_token_id, ("chosen_safe", "rejected_safe")),
        processing_class=tok,
        loss_config=PairwiseLossConfig(
            loss="dpo", beta=a.get("beta", 0.1), logp_mode="safe_rlhf"
        ),
    )
    trainer.train()
    return _finish(cfg, trainer, tok)


# ============================================================ PPO family
def _prompt_dataset(cfg, tok):
    prompts = S.unique_prompts(_pku_rows(cfg))
    feats = []
    for p in prompts:
        ids = tok(
            format_prompt(p, cfg.data.template),
            truncation=True,
            max_length=cfg.data.max_length,
        )["input_ids"]
        feats.append({"input_ids": ids})
    return feats


def run_saferlhf(cfg: ExperimentConfig):
    from dataclasses import fields

    from transformers import AutoModelForCausalLM

    from human_alignment.safety.methods.models import load_score_model
    from human_alignment.safety.methods.ppo import PPOConfig, SafeRLHFTrainer
    from human_alignment.safety.methods.sft import sft_features_safe_rlhf

    a = dict(cfg.method_args)
    tok = load_tokenizer(cfg)
    names = {f.name for f in fields(PPOConfig)}
    ppo_cfg = PPOConfig(
        **{k: v for k, v in a.items() if k in names},
        output_dir=str(output_dir(cfg)),
        max_length=cfg.data.max_length,
        seed=cfg.train.get("seed", 42),
    )
    actor = AutoModelForCausalLM.from_pretrained(
        cfg.model.policy,
        dtype=_dtype(cfg.model.dtype),
        attn_implementation=cfg.model.attn_implementation,
    )
    reward = load_score_model(a["reward_model"], "reward", dtype=cfg.model.dtype)
    cost = (
        load_score_model(a["cost_model"], "cost", dtype=cfg.model.dtype)
        if cfg.method == "saferlhf"
        else None
    )
    ptx = None
    if a.get("ptx", True) and ppo_cfg.ptx_coeff > 0:
        rows = S.load_alpaca(
            a.get("ptx_limit"), a.get("ptx_dataset", "tatsu-lab/alpaca")
        )
        ptx = [
            sft_features_safe_rlhf(tok, r["prompt"], r["answer"], cfg.data.max_length)
            for r in rows
        ]
    trainer = SafeRLHFTrainer(
        ppo_cfg,
        actor,
        tok,
        reward,
        cost,
        _prompt_dataset(cfg, tok),
        ptx_dataset=ptx,
        ptx_collate=SFTCollator(tok.pad_token_id),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    trainer.train(max_steps=a.get("max_steps"))
    trainer.save()
    dump_config(cfg, output_dir(cfg) / "experiment.yaml")
    return output_dir(cfg)


def run_morlhf(cfg: ExperimentConfig):
    from dataclasses import fields

    from transformers import AutoModelForCausalLM

    from human_alignment.safety.methods.models import load_score_model
    from human_alignment.safety.methods.morlhf import (
        MORLHFConfig,
        MORLHFTrainer,
        score_model_reward_fn,
    )

    a = dict(cfg.method_args)
    tok = load_tokenizer(cfg)
    names = {f.name for f in fields(MORLHFConfig)}
    m_cfg = MORLHFConfig(
        **{k: v for k, v in a.items() if k in names}, output_dir=str(output_dir(cfg))
    )
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    policy = AutoModelForCausalLM.from_pretrained(
        cfg.model.policy,
        dtype=_dtype(cfg.model.dtype),
        attn_implementation=cfg.model.attn_implementation,
    )
    reward = (
        load_score_model(a["reward_model"], "reward", dtype=cfg.model.dtype)
        .to(dev)
        .eval()
    )
    cost = (
        load_score_model(a["cost_model"], "cost", dtype=cfg.model.dtype).to(dev).eval()
    )
    fns = [
        score_model_reward_fn(reward, tok.eos_token_id, device=dev),
        score_model_reward_fn(cost, tok.eos_token_id, negate=True, device=dev),
    ]
    prompts = [p for p in _prompt_dataset(cfg, tok) if 8 <= len(p["input_ids"]) <= 256]
    trainer = MORLHFTrainer(m_cfg, policy, tok, fns, prompts, device=dev)
    trainer.train(max_steps=a.get("max_steps"))
    trainer.save()
    dump_config(cfg, output_dir(cfg) / "experiment.yaml")
    return output_dir(cfg)


# ============================================================ merging / CAN dual
def run_merge(cfg: ExperimentConfig):
    from human_alignment.safety.methods.merge import merge_linear_checkpoints

    a = cfg.method_args
    merge_linear_checkpoints(
        a["models"], a["weights"], str(output_dir(cfg)), _dtype(cfg.model.dtype)
    )
    dump_config(cfg, output_dir(cfg) / "experiment.yaml")
    return output_dir(cfg)


def run_can_dual(cfg: ExperimentConfig):
    """Sample K responses per prompt from π_ref, score with RM/CM (safety = −cost), solve the dual for λ*."""
    import numpy as np
    from transformers import AutoModelForCausalLM

    from human_alignment.safety.methods.can_dual import solve_dual
    from human_alignment.safety.methods.models import load_score_model

    a = cfg.method_args
    out = output_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    scores_file = out / "ref_sample_scores.npz"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.get("scores_file"):
        data = np.load(a["scores_file"])
        help_s, safe_s = data["helpfulness"], data["safety"]
    else:
        tok = load_tokenizer(cfg)
        tok.padding_side = "left"
        prompts = S.unique_prompts(_pku_rows(cfg))[: a.get("num_prompts", 1000)]
        k = a.get("num_samples", 64)
        model = (
            AutoModelForCausalLM.from_pretrained(
                cfg.model.ref or cfg.model.policy, dtype=_dtype(cfg.model.dtype)
            )
            .to(dev)
            .eval()
        )
        seqs = []
        for i in range(0, len(prompts), a.get("gen_batch_size", 8)):
            batch = [
                format_prompt(p, cfg.data.template)
                for p in prompts[i : i + a.get("gen_batch_size", 8)]
            ]
            enc = tok(batch, return_tensors="pt", padding=True).to(dev)
            with torch.no_grad():
                g = model.generate(
                    **enc,
                    do_sample=True,
                    temperature=a.get("temperature", 1.0),
                    top_p=a.get("top_p", 1.0),
                    max_new_tokens=a.get("max_new_tokens", 256),
                    num_return_sequences=k,
                    pad_token_id=tok.pad_token_id,
                )
            seqs.append(g.cpu())
        del model
        torch.cuda.empty_cache()
        scores = {}
        for key, path in (("reward", a["reward_model"]), ("cost", a["cost_model"])):
            sm = load_score_model(path, key, dtype=cfg.model.dtype).to(dev).eval()
            vals = []
            for g in seqs:
                g = g.to(dev)
                mask = g.ne(tok.pad_token_id) & g.ne(tok.unk_token_id)
                with (
                    torch.no_grad(),
                    torch.autocast(torch.device(dev).type, dtype=torch.bfloat16),
                ):
                    vals.append(sm(g, mask).end_scores.float().cpu())
            scores[key] = torch.cat(vals).view(len(prompts), k).numpy()
            del sm
        help_s, safe_s = scores["reward"], -scores["cost"]
        np.savez(scores_file, helpfulness=help_s, safety=safe_s)
    res = solve_dual(
        help_s,
        safe_s,
        a.get("threshold", 0.0),
        a.get("kl_coeff", 0.1),
        lam_init=a.get("lam_init", 1.0),
        lr=a.get("lr"),
        num_iters=a.get("num_iters", 200),
        err=a.get("err", 1e-5),
    )
    info = {
        "lam_star": res.lam_star,
        "converged": res.converged,
        "expected_helpfulness": res.helpfulness_trajectory[-1]
        if res.helpfulness_trajectory
        else None,
        "expected_safety": res.safety_trajectory[-1] if res.safety_trajectory else None,
    }
    (out / "dual.json").write_text(json.dumps(info, indent=2, default=float))
    log.info("CAN dual: %s", info)
    return out

