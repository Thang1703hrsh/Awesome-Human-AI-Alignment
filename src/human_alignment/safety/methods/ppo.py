"""SafeRLHF (PPO-Lagrangian) and plain PPO on one GPU, following ``safe_rlhf/trainers/rl_trainer.py`` and
``safe_rlhf/algorithms/{ppo,ppo_lag}/trainer.py`` step by step.

Memory layout (user decision: LoRA actor + critics): the actor is a LoRA adapter on the SFT model and the reference
policy is the same model with the adapter disabled. Each critic is a LoRA adapter + its own score head on the backbone
of the reward (cost) model it is initialised from (safe-rlhf initialises critics from the RM/CM); the frozen RM/CM
score is the same backbone with the adapter disabled. Three 7B backbones in bf16 in total.

DeepSpeed engine semantics are reproduced by ``_Engine``: ``backward`` scales by 1/GA, ``step`` only updates at the
accumulation boundary (clip → optimizer → scheduler → zero_grad). With PTX the actor's GA is doubled, exactly like
``RLTrainer.init_engines``.
"""

from __future__ import annotations

import copy
import itertools
import json
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from transformers import GenerationConfig, get_scheduler

from human_alignment.safety.losses import ppo as PL
from human_alignment.safety.losses.logps import gather_log_probabilities
from human_alignment.safety.methods.models import ScoreModel


@dataclass
class PPOConfig:
    max_length: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    num_return_sequences: int = 1
    epochs: int = 1
    update_iters: int = 1
    per_device_prompt_batch_size: int = 16
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    actor_lr: float = 1e-5
    actor_weight_decay: float = 0.01
    actor_lr_scheduler_type: str = "cosine"
    actor_lr_warmup_ratio: float = 0.03
    critic_lr: float = 5e-6
    critic_weight_decay: float = 0.0
    critic_lr_scheduler_type: str = "constant"
    critic_lr_warmup_ratio: float = 0.03
    kl_coeff: float = 0.01
    clip_range_ratio: float = 0.2
    clip_range_score: float = 50.0
    clip_range_value: float = 5.0
    ptx_coeff: float = 16.0
    gamma: float = 1.0
    gae_lambda: float = 0.95
    max_grad_norm: float = 1.0
    adam_betas: tuple[float, float] = (0.9, 0.95)
    # PPO-Lagrangian
    threshold: float = 0.0
    lambda_init: float = 1.0
    lambda_lr: float = 0.1
    lambda_max: float | None = 5.0
    lambda_update_delay_steps: int = 0
    episode_cost_window_size: int = 128
    # Emulating N data-parallel GPUs with gradient accumulation: the reference updates λ once per data-parallel
    # step (one rl_step on every rank at once), so update λ on every N-th rl_step and use a window of 128·N.
    lambda_update_interval: int = 1
    # LoRA (single-H100 adaptation)
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.0
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    gradient_checkpointing: bool = True
    bf16: bool = True
    seed: int = 42
    save_interval: int = 1_000_000
    output_dir: str = "output/ppo"


def safe_rlhf_param_groups(module: nn.Module, weight_decay: float):
    """``safe_rlhf.utils.get_optimizer_grouped_parameters``."""
    no_decay = {"bias", "layernorm.weight"}
    named = [(n.lower(), p) for n, p in module.named_parameters() if p.requires_grad]
    return [
        {
            "params": [p for n, p in named if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in named if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]


class _Engine:
    """Minimal stand-in for a DeepSpeed engine's backward/step accumulation contract."""

    def __init__(
        self,
        params_module: nn.Module,
        lr: float,
        weight_decay: float,
        betas,
        scheduler_type: str,
        warmup_ratio: float,
        total_micro_steps: int,
        grad_accum: int,
        max_grad_norm: float,
    ):
        groups = [
            g
            for g in safe_rlhf_param_groups(params_module, weight_decay)
            if g["params"]
        ]
        self.params = [p for g in groups for p in g["params"]]
        self.optimizer = torch.optim.AdamW(groups, lr=lr, betas=betas, eps=1e-8)
        update_steps = total_micro_steps // grad_accum
        self.scheduler = get_scheduler(
            scheduler_type,
            self.optimizer,
            num_warmup_steps=int(update_steps * warmup_ratio),
            num_training_steps=update_steps,
        )
        self.grad_accum = grad_accum
        self.max_grad_norm = max_grad_norm
        self.micro_steps = 0

    def backward(self, loss: torch.Tensor) -> None:
        (loss / self.grad_accum if self.grad_accum > 1 else loss).backward()

    def step(self) -> bool:
        self.micro_steps += 1
        if self.micro_steps % self.grad_accum != 0:
            return False
        if self.max_grad_norm and self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        return True


class LoRAScoreCritic(nn.Module):
    """Frozen score model (adapter off) + critic (LoRA adapter on, separate head initialised from the score head)."""

    def __init__(self, score_model: ScoreModel, lora_config):
        super().__init__()
        from peft import get_peft_model

        self.score_model = score_model
        score_model.requires_grad_(False)
        self.backbone = get_peft_model(score_model.model, lora_config)
        self.critic_head = (
            copy.deepcopy(score_model.score_head).float().requires_grad_(True)
        )

    @torch.no_grad()
    def score(self, input_ids, attention_mask) -> torch.Tensor:
        with self.backbone.disable_adapter():
            hidden = self.backbone(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            ).last_hidden_state
        return self.score_model.score_from_hidden(hidden, attention_mask).end_scores

    def values(self, input_ids, attention_mask) -> torch.Tensor:
        hidden = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).last_hidden_state
        return self.score_model.score_from_hidden(
            hidden, attention_mask, head=self.critic_head
        ).scores

    def trainable_module(self) -> nn.Module:
        return self


def _lora_config(cfg: PPOConfig, task_type: str | None):
    from peft import LoraConfig

    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        task_type=task_type,
    )


def prompt_collate(pad_token_id: int):
    """Left padding for generation (``PromptOnlyCollator``)."""

    def collate(features):
        seqs = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        ids = torch.nn.utils.rnn.pad_sequence(
            [s.flip(0) for s in seqs], batch_first=True, padding_value=pad_token_id
        ).flip(1)
        mask = torch.nn.utils.rnn.pad_sequence(
            [torch.ones_like(s, dtype=torch.bool).flip(0) for s in seqs],
            batch_first=True,
            padding_value=False,
        ).flip(1)
        return {"input_ids": ids, "attention_mask": mask}

    return collate


class SafeRLHFTrainer:
    """PPO (``cost_critic is None``) or PPO-Lagrangian (SafeRLHF)."""

    def __init__(
        self,
        cfg: PPOConfig,
        actor: nn.Module,
        tokenizer,
        reward: ScoreModel,
        cost: ScoreModel | None,
        prompt_dataset,
        ptx_dataset=None,
        ptx_collate=None,
        device: str | torch.device = "cuda",
    ):
        from peft import get_peft_model

        torch.manual_seed(cfg.seed)
        self.cfg = cfg
        self.device = torch.device(device)
        self.tokenizer = tokenizer
        self.lagrangian = cost is not None
        actor.requires_grad_(False)
        self.actor = get_peft_model(actor, _lora_config(cfg, "CAUSAL_LM")).to(
            self.device
        )
        self.reward_critic = LoRAScoreCritic(reward, _lora_config(cfg, None)).to(
            self.device
        )
        self.cost_critic = (
            LoRAScoreCritic(cost, _lora_config(cfg, None)).to(self.device)
            if cost is not None
            else None
        )
        if cfg.gradient_checkpointing:
            for m in (
                self.actor,
                self.reward_critic.backbone,
                *([self.cost_critic.backbone] if cost else []),
            ):
                m.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )

        self.prompt_loader = torch.utils.data.DataLoader(
            prompt_dataset,
            batch_size=cfg.per_device_prompt_batch_size,
            shuffle=True,
            collate_fn=prompt_collate(tokenizer.pad_token_id),
            generator=torch.Generator().manual_seed(cfg.seed),
        )
        self.use_ptx = ptx_dataset is not None
        self.ptx_loader = (
            torch.utils.data.DataLoader(
                ptx_dataset,
                batch_size=cfg.per_device_prompt_batch_size,
                shuffle=True,
                collate_fn=ptx_collate,
                generator=torch.Generator().manual_seed(cfg.seed),
            )
            if self.use_ptx
            else None
        )

        self.total_training_steps = int(
            len(self.prompt_loader)
            * cfg.epochs
            * cfg.update_iters
            * cfg.per_device_prompt_batch_size
            * cfg.num_return_sequences
            // cfg.per_device_train_batch_size
        )
        actor_ga, actor_steps = (
            cfg.gradient_accumulation_steps,
            self.total_training_steps,
        )
        if self.use_ptx:
            actor_ga, actor_steps = actor_ga * 2, actor_steps * 2
        self.actor_engine = _Engine(
            self.actor,
            cfg.actor_lr,
            cfg.actor_weight_decay,
            cfg.adam_betas,
            cfg.actor_lr_scheduler_type,
            cfg.actor_lr_warmup_ratio,
            actor_steps,
            actor_ga,
            cfg.max_grad_norm,
        )
        self.reward_critic_engine = _Engine(
            self.reward_critic,
            cfg.critic_lr,
            cfg.critic_weight_decay,
            cfg.adam_betas,
            cfg.critic_lr_scheduler_type,
            cfg.critic_lr_warmup_ratio,
            self.total_training_steps,
            cfg.gradient_accumulation_steps,
            cfg.max_grad_norm,
        )
        self.cost_critic_engine = (
            _Engine(
                self.cost_critic,
                cfg.critic_lr,
                cfg.critic_weight_decay,
                cfg.adam_betas,
                cfg.critic_lr_scheduler_type,
                cfg.critic_lr_warmup_ratio,
                self.total_training_steps,
                cfg.gradient_accumulation_steps,
                cfg.max_grad_norm,
            )
            if self.lagrangian
            else None
        )
        if self.lagrangian:
            self.lagrange = PL.LagrangeMultiplier(
                cfg.lambda_init,
                cfg.lambda_lr,
                cfg.lambda_max,
                cfg.threshold,
                cfg.lambda_update_delay_steps,
                device=self.device,
            )
            self.episode_costs: deque = deque(maxlen=cfg.episode_cost_window_size)
        self.generation_config = GenerationConfig(
            max_length=cfg.max_length,
            num_return_sequences=cfg.num_return_sequences,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            repetition_penalty=cfg.repetition_penalty,
            do_sample=True,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        self.global_step = 0
        self._rl_steps = 0
        self.history: list[dict] = []

    # ------------------------------------------------------------------ helpers
    def _autocast(self):
        return torch.autocast(
            self.device.type, dtype=torch.bfloat16, enabled=self.cfg.bf16
        )

    def _log_probs(self, input_ids, attention_mask):
        logits = self.actor(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).logits
        return gather_log_probabilities(logits[:, :-1], input_ids[:, 1:])

    def set_train(self, mode: bool):
        for m in (self.actor, self.reward_critic, self.cost_critic):
            if m is not None:
                m.train(mode)

    # ------------------------------------------------------------------ rollout
    @torch.no_grad()
    def rollout(self, batch) -> list[dict]:
        input_ids = batch["input_ids"].to(self.device)
        with self._autocast():
            sequences = self.actor.generate(
                input_ids=input_ids,
                attention_mask=batch["attention_mask"].to(self.device),
                generation_config=self.generation_config,
            )
        sequences = (
            sequences.contiguous()
            .view(input_ids.size(0), self.cfg.num_return_sequences, -1)
            .transpose(0, 1)
        )
        out = []
        for seq in sequences:
            mask = torch.logical_and(
                seq.ne(self.tokenizer.pad_token_id), seq.ne(self.tokenizer.unk_token_id)
            )
            out.append(self.post_rollout(input_ids, seq, mask))
        return out

    @torch.no_grad()
    def post_rollout(self, prompt, sequence, attention_mask) -> dict:
        with self._autocast():
            log_probs = self._log_probs(sequence, attention_mask)
            with self.actor.disable_adapter():
                ref_log_probs = self._log_probs(sequence, attention_mask)
            reward = self.reward_critic.score(sequence, attention_mask)
            reward_values = self.reward_critic.values(sequence, attention_mask)[:, :-1]
            item = {
                "prompt": prompt,
                "log_probs": log_probs.float(),
                "ref_log_probs": ref_log_probs.float(),
                "reward": reward.float(),
                "reward_values": reward_values.float(),
                "input_ids": sequence,
                "attention_mask": attention_mask,
            }
            if self.lagrangian:
                cost = self.cost_critic.score(sequence, attention_mask)
                item["cost"] = cost.float()
                item["cost_values"] = self.cost_critic.values(sequence, attention_mask)[
                    :, :-1
                ].float()
                self.episode_costs.extend(cost.tolist())
        return item

    # ------------------------------------------------------------------ update
    def rl_step(self, rl_batch: dict) -> dict:
        cfg = self.cfg
        info = {}
        if self.lagrangian:
            episode_cost = torch.tensor(
                list(self.episode_costs), device=self.device
            ).mean()
            if self._rl_steps % cfg.lambda_update_interval == 0:
                self.lagrange.update(
                    episode_cost, self.global_step // cfg.lambda_update_interval
                )
            info["train/episode_cost"] = episode_cost.item()
            info["train/lambda"] = self.lagrange.value
        self._rl_steps += 1
        prompt, input_ids, attention_mask = (
            rl_batch["prompt"],
            rl_batch["input_ids"],
            rl_batch["attention_mask"],
        )
        old_log_probs, ref_log_probs = rl_batch["log_probs"], rl_batch["ref_log_probs"]
        start = prompt.size(-1) - 1
        sequence_mask = attention_mask[:, 1:]
        with torch.no_grad():
            if self.lagrangian:
                old_rewards, old_costs = PL.kl_shaped_reward_and_cost(
                    rl_batch["reward"],
                    rl_batch["cost"],
                    old_log_probs,
                    ref_log_probs,
                    sequence_mask,
                    cfg.kl_coeff,
                    cfg.clip_range_score,
                )
                c_adv, c_ret = PL.gae_safe_rlhf(
                    rl_batch["cost_values"],
                    old_costs,
                    sequence_mask,
                    start,
                    cfg.gamma,
                    cfg.gae_lambda,
                )
            else:
                old_rewards = PL.kl_shaped_reward(
                    rl_batch["reward"],
                    old_log_probs,
                    ref_log_probs,
                    sequence_mask,
                    cfg.kl_coeff,
                    cfg.clip_range_score,
                )
            r_adv, r_ret = PL.gae_safe_rlhf(
                rl_batch["reward_values"],
                old_rewards,
                sequence_mask,
                start,
                cfg.gamma,
                cfg.gae_lambda,
            )
        mask = sequence_mask[:, start:]
        with self._autocast():
            log_probs = self._log_probs(input_ids, attention_mask).float()
        if self.lagrangian:
            actor_loss = PL.ppo_lag_actor_loss(
                log_probs[:, start:],
                old_log_probs[:, start:],
                r_adv,
                c_adv,
                mask,
                self.lagrange.value,
                cfg.clip_range_ratio,
            )
        else:
            actor_loss = PL.ppo_actor_loss(
                log_probs[:, start:],
                old_log_probs[:, start:],
                r_adv,
                mask,
                cfg.clip_range_ratio,
            )
        self.actor_engine.backward(actor_loss)
        self.actor_engine.step()

        with self._autocast():
            reward_values = self.reward_critic.values(input_ids, attention_mask)[
                :, :-1
            ].float()
        reward_critic_loss = PL.critic_loss(
            reward_values[:, start:],
            rl_batch["reward_values"][:, start:],
            r_ret,
            mask,
            cfg.clip_range_value,
        )
        self.reward_critic_engine.backward(reward_critic_loss)
        self.reward_critic_engine.step()
        info.update(
            {
                "train/actor_loss": actor_loss.item(),
                "train/reward_critic_loss": reward_critic_loss.item(),
                "train/reward": rl_batch["reward"].mean().item(),
                "train/kl_divergence": (
                    (old_log_probs - ref_log_probs) * sequence_mask
                )[:, start:]
                .sum(-1)
                .mean()
                .item(),
                "train/actor_lr": self.actor_engine.optimizer.param_groups[0]["lr"],
            }
        )
        if self.lagrangian:
            with self._autocast():
                cost_values = self.cost_critic.values(input_ids, attention_mask)[
                    :, :-1
                ].float()
            cost_critic_loss = PL.critic_loss(
                cost_values[:, start:],
                rl_batch["cost_values"][:, start:],
                c_ret,
                mask,
                cfg.clip_range_value,
            )
            self.cost_critic_engine.backward(cost_critic_loss)
            self.cost_critic_engine.step()
            info.update(
                {
                    "train/cost_critic_loss": cost_critic_loss.item(),
                    "train/cost": rl_batch["cost"].mean().item(),
                }
            )
        return info

    def ptx_step(self, ptx_batch: dict) -> dict:
        ptx_batch = {k: v.to(self.device) for k, v in ptx_batch.items()}
        with self._autocast():
            ptx_loss = self.actor(
                input_ids=ptx_batch["input_ids"],
                attention_mask=ptx_batch["attention_mask"],
                labels=ptx_batch["labels"],
                use_cache=False,
            ).loss
        self.actor_engine.backward(self.cfg.ptx_coeff * ptx_loss)
        self.actor_engine.step()
        return {"train/ptx_loss": ptx_loss.item()}

    def _split(self, batch: dict) -> Iterable[dict]:
        n, mb = batch["input_ids"].size(0), self.cfg.per_device_train_batch_size
        for i in range(0, n, mb):
            yield {k: v[i : i + mb] for k, v in batch.items()}

    def train(self, max_steps: int | None = None, log_fn=print) -> list[dict]:
        n_prompt, n_ptx = (
            len(self.prompt_loader),
            len(self.ptx_loader) if self.use_ptx else 1,
        )
        replicas = (n_prompt + n_ptx - 1) // n_ptx
        for _ in range(self.cfg.epochs):
            ptx_iter = (
                itertools.chain.from_iterable([self.ptx_loader] * replicas)
                if self.use_ptx
                else itertools.repeat(None)
            )
            for prompt_batch, ptx_batch in zip(self.prompt_loader, ptx_iter):
                self.set_train(False)
                rl_batches = [
                    rb for mb in self._split(prompt_batch) for rb in self.rollout(mb)
                ]
                ptx_batches = (
                    list(self._split(ptx_batch))
                    if self.use_ptx
                    else [None] * len(rl_batches)
                )
                self.set_train(True)
                for _ in range(self.cfg.update_iters):
                    for rl_batch, ptx_mb in zip(rl_batches, ptx_batches):
                        info = self.rl_step(rl_batch)
                        if self.use_ptx:
                            info.update(self.ptx_step(ptx_mb))
                        self.global_step += 1
                        info["step"] = self.global_step
                        self.history.append(info)
                        log_fn(info)
                        if self.global_step % self.cfg.save_interval == 0:
                            self.save(
                                Path(self.cfg.output_dir) / f"step_{self.global_step}"
                            )
                        if max_steps is not None and self.global_step >= max_steps:
                            return self.history
        return self.history

    def save(self, output_dir: str | Path | None = None) -> None:
        out = Path(output_dir or self.cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.actor.save_pretrained(out / "actor_lora")
        self.tokenizer.save_pretrained(out / "actor_lora")
        state = {"global_step": self.global_step}
        if self.lagrangian:
            state["lambda"] = self.lagrange.value
        (out / "trainer_state.json").write_text(
            json.dumps({**state, "config": asdict(self.cfg)}, indent=2, default=str)
        )


def merge_actor(actor_lora_dir: str, output_dir: str, dtype=torch.bfloat16) -> None:
    """Merge the trained actor adapter into the SFT weights (for evaluation / serving)."""
    from peft import AutoPeftModelForCausalLM

    model = AutoPeftModelForCausalLM.from_pretrained(
        actor_lora_dir, dtype=dtype
    ).merge_and_unload()
    model.save_pretrained(output_dir)

