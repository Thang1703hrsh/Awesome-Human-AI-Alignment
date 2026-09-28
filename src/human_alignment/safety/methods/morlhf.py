"""MORLHF: linear-scalarised multi-objective PPO, as run in RiC ``ppo/morlhf.py`` on TRL 0.8.0 ``PPOTrainer``.

Policy = LoRA (r=64, α=128, dropout 0.05) on the SFT model + a trainable value head (dropout 0.1 → Linear) on the
last hidden state, sharing the backbone (``AutoModelForCausalLMWithValueHead``); reference = adapter disabled.
Per batch: generate → clean text → truncate → score → ``PPOTrainer.step`` (rewards with adaptive-KL penalty,
whitened GAE advantages, ``ppo_epochs`` passes over shuffled minibatches, early stop on policy KL, Adam, grad-clip).

Safety adaptation (documented): objectives are the beaver reward (helpfulness) and the negated beaver cost
(harmlessness) on PKU-SafeRLHF prompts instead of RiC's HH-RLHF GPT-2 reward models.
Reward = round(Σ_k w_k·r_k, 2) as in ``morlhf.py``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from human_alignment.safety.losses import ppo as PL
from human_alignment.safety.losses.logps import gather_log_probabilities


@dataclass
class MORLHFConfig:
    preference: list[float] = field(default_factory=lambda: [0.5, 0.5])
    epochs: int = 1
    learning_rate: float = 1e-5
    batch_size: int = 64
    mini_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    ppo_epochs: int = 4
    init_kl_coef: float = 0.2
    target: float = 3.0
    horizon: float = 10000
    early_stopping: bool = True
    target_kl: float = 1.0
    max_grad_norm: float = 0.5
    gamma: float = 1.0
    lam: float = 0.95
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    vf_coef: float = 0.1
    ratio_threshold: float = 10.0
    whiten_rewards: bool = False
    value_head_dropout: float = 0.1
    max_new_tokens: int = 128
    temperature: float = 0.7
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = (
        None  # peft default for llama: q_proj, v_proj
    )
    stop_strings: list[str] = field(
        default_factory=lambda: [
            "\n\nHuman:",
            "\nHuman:",
            "\n\nAssistant:",
            "\nAssistant:",
            "\n\n\n",
            "###",
            "USER:",
            "BEGINNING OF CONVERSATION",
        ]
    )
    strip_mode: str = "charset"
    bf16: bool = True
    seed: int = 8888
    output_dir: str = "output/morlhf"


class ValueHead(nn.Module):
    """TRL 0.8.0 ``ValueHead``: dropout → Linear(hidden, 1), fp32."""

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.summary = nn.Linear(hidden_size, 1)

    def forward(self, hidden):
        out = self.dropout(hidden)
        if out.dtype != self.summary.weight.dtype:
            out = out.to(self.summary.weight.dtype)
        return self.summary(out)


def clean_response(
    text: str, stop_strings: list[str], strip_mode: str = "charset"
) -> str:
    """``morlhf.py`` response cleaning, then cut at the first turn marker.

    ``charset`` reproduces the original ``str.strip('[PAD] ')``/``strip('<unk>')``/``strip('<s>')`` calls, which
    remove *characters* from those sets at both ends (e.g. 'Answer' → 'wer'); ``tokens`` removes only the literal
    special-token strings.
    """
    if strip_mode == "charset":
        response = text.strip("[PAD] ").strip("<unk>")
        response = response.strip("<s>").strip("</s>")
    elif strip_mode == "tokens":
        response = text
        for tok in ("[PAD]", "<unk>", "<s>", "</s>", "<pad>"):
            response = response.replace(tok, "")
        response = response.strip()
    else:
        raise ValueError(strip_mode)
    for s in stop_strings:
        response = response.split(s)[0].strip()
    return response


class MORLHFTrainer:
    def __init__(
        self,
        cfg: MORLHFConfig,
        policy: nn.Module,
        tokenizer,
        reward_fns: list,
        prompt_dataset,
        device="cuda",
    ):
        """``reward_fns[k](prompt_ids_list, response_ids_list) -> list[float]``; ``prompt_dataset`` rows have
        ``input_ids`` (templated prompt tokens)."""
        from peft import LoraConfig, get_peft_model

        if len(reward_fns) != len(cfg.preference):
            raise ValueError("one preference weight per reward function")
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.cfg = cfg
        self.device = torch.device(device)
        self.tokenizer = tokenizer
        policy.requires_grad_(False)
        lora = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=cfg.lora_target_modules,
        )
        self.model = get_peft_model(policy, lora).to(self.device)
        self.v_head = ValueHead(policy.config.hidden_size, cfg.value_head_dropout).to(
            self.device
        )
        self.params = [p for p in self.model.parameters() if p.requires_grad] + list(
            self.v_head.parameters()
        )
        self.optimizer = torch.optim.Adam(self.params, lr=cfg.learning_rate)
        self.kl_ctl = PL.AdaptiveKLController(cfg.init_kl_coef, cfg.target, cfg.horizon)
        self.reward_fns = reward_fns
        self.loader = torch.utils.data.DataLoader(
            prompt_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=lambda b: [r["input_ids"] for r in b],
            generator=torch.Generator().manual_seed(cfg.seed),
        )
        self.history: list[dict] = []

    def _autocast(self):
        return torch.autocast(
            self.device.type, dtype=torch.bfloat16, enabled=self.cfg.bf16
        )

    # ------------------------------------------------------------------ generation (TRL _generate_batched)
    @torch.no_grad()
    def generate(
        self, queries: list[torch.Tensor], batch_size: int = 4
    ) -> list[torch.Tensor]:
        tok = self.tokenizer
        side = tok.padding_side
        tok.padding_side = "left"
        outputs = []
        try:
            for i in range(0, len(queries), batch_size):
                batch = queries[i : i + batch_size]
                enc = tok.pad({"input_ids": batch}, return_tensors="pt").to(self.device)
                with self._autocast():
                    gen = self.model.generate(
                        **enc,
                        max_new_tokens=self.cfg.max_new_tokens,
                        min_length=-1,
                        top_k=0,
                        top_p=1.0,
                        do_sample=True,
                        temperature=self.cfg.temperature,
                        pad_token_id=tok.eos_token_id,
                        begin_suppress_tokens=[tok.eos_token_id],
                    )
                for g, m in zip(gen, enc["attention_mask"]):
                    out = g[(1 - m).sum() :][m.sum() :]
                    if tok.eos_token_id in out:
                        out = out[
                            : torch.nonzero(out == tok.eos_token_id)[0, 0].item() + 1
                        ]
                    outputs.append(out)
        finally:
            tok.padding_side = side
        return outputs

    def postprocess_responses(
        self, responses: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[str]]:
        texts = self.tokenizer.batch_decode(responses)
        clean = [
            clean_response(t, self.cfg.stop_strings, self.cfg.strip_mode) for t in texts
        ]
        lengths = [len(self.tokenizer.encode(t)) for t in clean]
        return [r[: max(n, 2)] for r, n in zip(responses, lengths)], clean

    # ------------------------------------------------------------------ PPO step (TRL PPOTrainer.step)
    def _forward(self, input_ids, attention_mask):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        values = self.v_head(out.hidden_states[-1]).squeeze(-1)
        return out.logits.float(), values

    def batched_forward_pass(
        self, queries, responses, input_ids, attention_mask, ref: bool = False
    ):
        fbs = self.cfg.mini_batch_size
        all_lp, all_v, all_m = [], [], []
        for i in range(0, len(queries), fbs):
            ids, am = input_ids[i : i + fbs], attention_mask[i : i + fbs]
            with self._autocast():
                if ref:
                    with self.model.disable_adapter():
                        logits, values = self._forward(ids, am)
                else:
                    logits, values = self._forward(ids, am)
            lp = gather_log_probabilities(logits[:, :-1], ids[:, 1:])
            masks = torch.zeros_like(am, dtype=torch.long)
            masks[:, :-1] = am[:, 1:]
            for j in range(ids.size(0)):
                start = len(queries[i + j]) - 1
                if am[j, 0] == 0:
                    start += am[j].nonzero()[0].item()
                end = start + len(responses[i + j])
                masks[j, :start] = 0
                masks[j, end:] = 0
            all_lp.append(lp)
            all_v.append(values[:, :-1].float())
            all_m.append(masks[:, :-1])
        return torch.cat(all_lp), torch.cat(all_v), torch.cat(all_m)

    def _model_inputs(self, queries, responses):
        seqs = [torch.cat([q, r]) for q, r in zip(queries, responses)]
        enc = self.tokenizer.pad({"input_ids": seqs}, return_tensors="pt")
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)

    def step(self, queries, responses, scores) -> dict:
        cfg = self.cfg
        input_ids, attention_mask = self._model_inputs(queries, responses)
        self.model.eval()
        self.v_head.eval()
        with torch.no_grad():
            logprobs, values, masks = self.batched_forward_pass(
                queries, responses, input_ids, attention_mask
            )
            ref_logprobs, _, _ = self.batched_forward_pass(
                queries, responses, input_ids, attention_mask, ref=True
            )
            rewards, _, kls = PL.trl_compute_rewards(
                scores, logprobs, ref_logprobs, masks, self.kl_ctl.value
            )
            values, advantages, returns = PL.trl_compute_advantages(
                values, rewards, masks, cfg.gamma, cfg.lam, cfg.whiten_rewards
            )
        bs = len(queries)
        backward_bs = cfg.mini_batch_size * cfg.gradient_accumulation_steps
        early_stop = False
        last = None
        for _ in range(cfg.ppo_epochs):
            if early_stop:
                break
            b_inds = np.random.permutation(bs)
            for bstart in range(0, bs, backward_bs):
                binds = b_inds[bstart : bstart + backward_bs]
                for mstart in range(0, backward_bs, cfg.mini_batch_size):
                    minds = binds[mstart : mstart + cfg.mini_batch_size]
                    self.model.train()
                    self.v_head.train()
                    new_lp, vpreds, _ = self.batched_forward_pass(
                        [queries[i] for i in minds],
                        [responses[i] for i in minds],
                        input_ids[minds],
                        attention_mask[minds],
                    )
                    last = PL.trl_ppo_loss(
                        logprobs[minds],
                        values[minds],
                        vpreds,
                        new_lp,
                        masks[minds],
                        advantages[minds],
                        returns[minds],
                        cfg.cliprange,
                        cfg.cliprange_value,
                        cfg.vf_coef,
                        cfg.ratio_threshold,
                    )
                    loss = last.pg_loss + last.vf_loss
                    (loss / cfg.gradient_accumulation_steps).backward()
                    if (
                        mstart // cfg.mini_batch_size + 1
                    ) % cfg.gradient_accumulation_steps == 0:
                        if cfg.max_grad_norm is not None:
                            torch.nn.utils.clip_grad_norm_(
                                self.params, cfg.max_grad_norm
                            )
                        self.optimizer.step()
                        self.optimizer.zero_grad()
            if (
                cfg.early_stopping
                and last is not None
                and last.policykl > 1.5 * cfg.target_kl
            ):
                self.optimizer.zero_grad()
                early_stop = True
        mean_kl = ((logprobs - ref_logprobs) * masks).sum(-1).mean().item()
        self.kl_ctl.update(mean_kl, cfg.batch_size)
        return {
            "objective/kl": mean_kl,
            "objective/kl_coef": self.kl_ctl.value,
            "ppo/policykl": float(last.policykl) if last is not None else 0.0,
            "ppo/early_stop": early_stop,
        }

    def score(self, queries, responses) -> list[torch.Tensor]:
        per_objective = [fn(queries, responses) for fn in self.reward_fns]
        w = self.cfg.preference
        return [
            torch.tensor(
                round(sum(w[k] * per_objective[k][j] for k in range(len(w))), 2),
                device=self.device,
            )
            for j in range(len(queries))
        ]

    def train(self, max_steps: int | None = None, log_fn=print) -> list[dict]:
        step = 0
        for _ in range(self.cfg.epochs):
            for batch in self.loader:
                queries = [torch.as_tensor(q, device=self.device) for q in batch]
                responses = self.generate(queries)
                responses, _ = self.postprocess_responses(responses)
                scores = self.score(queries, responses)
                stats = self.step(queries, responses, scores)
                stats["env/reward_mean"] = float(torch.stack(scores).mean())
                step += 1
                stats["step"] = step
                self.history.append(stats)
                log_fn(stats)
                if max_steps is not None and step >= max_steps:
                    return self.history
        return self.history

    def save(self, output_dir: str | None = None) -> None:
        out = Path(output_dir or self.cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out / "policy_lora")
        torch.save(self.v_head.state_dict(), out / "value_head.pt")
        (out / "config.json").write_text(
            json.dumps(asdict(self.cfg), indent=2, default=str)
        )


def score_model_reward_fn(
    score_model, eos_token_id: int, negate: bool = False, device="cuda"
):
    """Wrap a beaver-style score model: score prompt + response (+EOS if missing) at the end token."""

    @torch.no_grad()
    def fn(queries, responses):
        seqs = []
        for q, r in zip(queries, responses):
            s = torch.cat([q, r])
            if s[-1].item() != eos_token_id:
                s = torch.cat([s, s.new_tensor([eos_token_id])])
            seqs.append(s)
        ids = torch.nn.utils.rnn.pad_sequence(
            seqs, batch_first=True, padding_value=0
        ).to(device)
        mask = torch.nn.utils.rnn.pad_sequence(
            [torch.ones_like(s, dtype=torch.bool) for s in seqs],
            batch_first=True,
            padding_value=False,
        ).to(device)
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            end = score_model(ids, mask).end_scores.float()
        return (-end if negate else end).tolist()

    return fn

