"""Model / tokenizer loading shared by all methods.

Score models follow safe-rlhf's ``AutoModelForScore`` layout (``safe_rlhf/models/score_model``): a decoder
backbone under ``model.*`` and ``score_head = Linear(hidden, 1, bias=True)``; per-token scores and the score at the
last attended token (``end_scores``). Checkpoints are saved in the same layout (``architectures: [LlamaForScore]``)
so they interoperate with ``PKU-Alignment/beaver-7b-*-{reward,cost}``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def _dtype(name: str | torch.dtype) -> torch.dtype:
    return (
        name
        if isinstance(name, torch.dtype)
        else {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }[name]
    )


def load_tokenizer(
    path: str, padding_side: str = "right", model_max_length: int = 512
) -> PreTrainedTokenizerBase:
    tok = AutoTokenizer.from_pretrained(
        path, padding_side=padding_side, model_max_length=model_max_length
    )
    return tok


def add_missing_special_tokens(tokenizer, model: nn.Module | None = None) -> None:
    """safe-rlhf ``resize_tokenizer_embedding``: add missing pad/eos/bos/unk and init new rows to the mean."""
    special = {}
    if tokenizer.pad_token is None:
        special["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special["unk_token"] = DEFAULT_UNK_TOKEN
    num_new = tokenizer.add_special_tokens(special)
    if model is None or num_new == 0:
        return
    model.resize_token_embeddings(len(tokenizer))
    for emb in (model.get_input_embeddings(), model.get_output_embeddings()):
        if emb is None:
            continue
        with torch.no_grad():
            emb.weight[-num_new:] = emb.weight[:-num_new].mean(dim=0, keepdim=True)


def load_causal_lm(
    path: str, dtype="bf16", attn_implementation: str = "sdpa", device=None
):
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=_dtype(dtype), attn_implementation=attn_implementation
    )
    if device is not None:
        model.to(device)
    return model


@dataclass
class ScoreOutput:
    scores: torch.Tensor  # (B, L) float32
    end_scores: torch.Tensor  # (B,)   float32
    end_index: torch.Tensor  # (B,)


def end_index_of(attention_mask: torch.Tensor) -> torch.Tensor:
    return torch.cat([m.nonzero()[-1] for m in attention_mask])


class ScoreModel(nn.Module):
    """Backbone + linear score head. ``score_type`` is metadata (reward / cost / critic)."""

    def __init__(
        self, backbone: nn.Module, score_head: nn.Linear, score_type: str = "reward"
    ):
        super().__init__()
        self.model = backbone
        self.score_head = score_head
        self.score_type = score_type

    @property
    def config(self):
        return self.model.config

    def hidden_states(self, input_ids, attention_mask):
        return self.model(
            input_ids, attention_mask=attention_mask, use_cache=False
        ).last_hidden_state

    def score_from_hidden(
        self, hidden, attention_mask, head: nn.Linear | None = None
    ) -> ScoreOutput:
        head = head or self.score_head
        scores = head(hidden.to(head.weight.dtype)).float().squeeze(-1)
        end_index = end_index_of(attention_mask)
        end_scores = scores.gather(1, end_index.to(scores.device).unsqueeze(1)).squeeze(
            1
        )
        return ScoreOutput(scores, end_scores, end_index)

    def forward(self, input_ids, attention_mask) -> ScoreOutput:
        return self.score_from_hidden(
            self.hidden_states(input_ids, attention_mask), attention_mask
        )

    def gradient_checkpointing_enable(self, **kw):
        self.model.gradient_checkpointing_enable(**kw)

    # -------------------------------------------------------------- persistence (safe-rlhf layout)
    def save_pretrained(self, output_dir: str) -> None:
        from safetensors.torch import save_file

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        state = {
            f"model.{k}": v.detach().contiguous().cpu()
            for k, v in self.model.state_dict().items()
        }
        state.update(
            {
                f"score_head.{k}": v.detach().contiguous().cpu()
                for k, v in self.score_head.state_dict().items()
            }
        )
        save_file(state, str(out / "model.safetensors"), metadata={"format": "pt"})
        cfg = self.model.config.to_dict()
        arch = type(self.model).__name__.replace("Model", "ForScore")
        cfg.update(
            {
                "architectures": [arch],
                "score_dim": 1,
                "score_bias": self.score_head.bias is not None,
                "score_type": self.score_type,
                "do_normalize": False,
                "normalizer_type": None,
            }
        )
        (out / "config.json").write_text(json.dumps(cfg, indent=2, default=str))


def _read_checkpoint_tensors(path: str, prefix: str) -> dict[str, torch.Tensor]:
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    root = Path(path)
    if not root.exists():
        root = Path(
            snapshot_download(
                path, allow_patterns=["*.json", "*.safetensors", "*.bin", "*.model"]
            )
        )
    out: dict[str, torch.Tensor] = {}
    for f in sorted(root.glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as sf:
            for k in sf.keys():
                if k.startswith(prefix):
                    out[k[len(prefix) :]] = sf.get_tensor(k)
    if not out:
        for f in sorted(root.glob("*.bin")):
            sd = torch.load(str(f), map_location="cpu", weights_only=True)
            out.update(
                {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}
            )
    return out


def load_score_model(
    path: str,
    score_type: str = "reward",
    dtype="bf16",
    attn_implementation: str = "sdpa",
    from_causal_lm: bool | None = None,
) -> ScoreModel:
    """Load a safe-rlhf score model, or initialise one from a causal LM (e.g. the SFT model for RM training).

    New heads are initialised as in safe-rlhf: ``post_init`` → Linear weight ~ N(0, initializer_range), bias 0.
    """
    config = AutoConfig.from_pretrained(path)
    backbone = AutoModel.from_pretrained(
        path, dtype=_dtype(dtype), attn_implementation=attn_implementation
    )
    head = nn.Linear(config.hidden_size, 1, bias=True)
    head_state = {} if from_causal_lm else _read_checkpoint_tensors(path, "score_head.")
    if head_state:
        head.load_state_dict(head_state)
    else:
        if from_causal_lm is False:
            raise ValueError(f"{path} has no score_head weights")
        nn.init.normal_(
            head.weight, mean=0.0, std=getattr(config, "initializer_range", 0.02)
        )
        nn.init.zeros_(head.bias)
    head.to(_dtype(dtype))
    return ScoreModel(backbone, head, score_type=score_type)

