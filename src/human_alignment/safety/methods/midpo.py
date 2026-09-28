"""MidPO router (EMNLP-F 2025; ``MidPO/safe_rlhf/algorithms/mdpo/mdpo_router/llama_with_router.py``).

Every decoder layer's MLP becomes
    z   = act(gate(x)) * up(x)
    out = down(z) + a(z)·lora_safe(z) + b(z)·lora_help(z),      a, b = sigmoid(L2(L1(L0(z))))   (no activations)
Regulariser (``LlamaModel.forward:967-1023``): computed from a *second* ``mlp(h)`` call on the decoder layer's input
residual stream ``h`` (not the post-attention input used for routing): mean over layers/tokens (padding included) of
|a| and |b − 1|. Experts are the trained PEFT LoRA A/B matrices copied without the ``alpha/r`` scale
(``mdpo_router/trainer.py:53-121``). Both quirks are the defaults and can be switched off.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
import torch.nn as nn


class ExpertLoRA(nn.Module):
    """``Lora_Layer``: linear2(linear1(z)), no bias, optional scale."""

    def __init__(self, in_features: int, r: int, out_features: int, scale: float = 1.0):
        super().__init__()
        self.linear1 = nn.Linear(in_features, r, bias=False)
        self.linear2 = nn.Linear(r, out_features, bias=False)
        self.scale = scale

    def forward(self, z):
        out = self.linear2(self.linear1(z))
        return out if self.scale == 1.0 else out * self.scale


class RouterNN(nn.Module):
    def __init__(self, in_features: int, hidden: int = 512, nonlinearity: bool = False):
        super().__init__()
        self.linear_0 = nn.Linear(in_features, hidden, bias=True)
        self.linear_1 = nn.Linear(hidden, in_features)
        self.linear_2 = nn.Linear(in_features, 1)
        self.act = nn.GELU() if nonlinearity else nn.Identity()

    def forward(self, z):
        return torch.sigmoid(
            self.linear_2(self.act(self.linear_1(self.act(self.linear_0(z)))))
        )


class RoutedMLP(nn.Module):
    def __init__(
        self,
        mlp: nn.Module,
        layer_idx: int,
        container: "MidPORouter",
        r: int,
        expert_scale: float,
        router_hidden: int,
        router_nonlinearity: bool,
    ):
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj, self.act_fn = (
            mlp.gate_proj,
            mlp.up_proj,
            mlp.down_proj,
            mlp.act_fn,
        )
        inter, hidden = mlp.down_proj.in_features, mlp.down_proj.out_features
        self.lora_0 = ExpertLoRA(inter, r, hidden, expert_scale)  # safety expert
        self.lora_1 = ExpertLoRA(inter, r, hidden, expert_scale)  # helpfulness expert
        self.alpha_ = RouterNN(inter, router_hidden, router_nonlinearity)
        self.beta_ = RouterNN(inter, router_hidden, router_nonlinearity)
        self.layer_idx = layer_idx
        self._container = [
            container
        ]  # list: avoid registering the container as a submodule

    def intermediate(self, x):
        return self.act_fn(self.gate_proj(x)) * self.up_proj(x)

    def _gates(self, z):
        zr = z.to(self.alpha_.linear_0.weight.dtype)
        return self.alpha_(zr), self.beta_(zr)

    def forward(self, x):
        container = self._container[0]
        z = self.intermediate(x)
        base = self.down_proj(z)
        if not container.routing_enabled:
            return base
        a, b = self._gates(z)
        ze = z.to(self.lora_0.linear1.weight.dtype)
        out = base + (a * self.lora_0(ze) + b * self.lora_1(ze)).to(base.dtype)
        if container.reg_source == "mlp_input":
            container.regs[self.layer_idx] = (a.abs(), (b - 1).abs())
        elif container._layer_inputs.get(self.layer_idx) is not None:
            a_in, b_in = self._gates(
                self.intermediate(container._layer_inputs[self.layer_idx])
            )
            container.regs[self.layer_idx] = (a_in.abs(), (b_in - 1).abs())
        return out


class MidPORouter(nn.Module):
    """Installs routed MLPs into a HF decoder-only model in place; ``self.model`` stays a normal CausalLM."""

    def __init__(
        self,
        model: nn.Module,
        r: int = 16,
        expert_scale: float = 1.0,
        router_hidden: int = 512,
        router_nonlinearity: bool = False,
        reg_source: str = "layer_input",
        dtype=torch.float32,
    ):
        super().__init__()
        if reg_source not in ("layer_input", "mlp_input"):
            raise ValueError(reg_source)
        self.model = model
        self.reg_source = reg_source
        self.routing_enabled = True
        self.regs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._layer_inputs: dict[int, torch.Tensor] = {}
        layers = model.model.layers
        self.routed = nn.ModuleList()
        for i, layer in enumerate(layers):
            routed = RoutedMLP(
                layer.mlp, i, self, r, expert_scale, router_hidden, router_nonlinearity
            )
            for sub in (routed.lora_0, routed.lora_1, routed.alpha_, routed.beta_):
                sub.to(dtype)
            layer.mlp = routed
            self.routed.append(routed)
            layer.register_forward_pre_hook(self._make_input_hook(i), with_kwargs=True)

    def _make_input_hook(self, idx):
        def hook(module, args, kwargs):
            hidden = args[0] if args else kwargs["hidden_states"]
            self._layer_inputs[idx] = hidden

        return hook

    def forward(self, input_ids, attention_mask=None, **kwargs):
        self.regs = {}
        self._layer_inputs = {}
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        self._layer_inputs = {}
        return out

    def reg_loss(self) -> torch.Tensor:
        """reg_alpha + reg_beta with the reference reduction: stack per-layer (B,T,1) tensors, global mean."""
        n = len(self.routed)
        if len(self.regs) != n:
            raise RuntimeError(
                f"expected router regularisers from {n} layers, got {len(self.regs)}"
            )
        reg_a = torch.stack([self.regs[i][0] for i in range(n)]).mean()
        reg_b = torch.stack([self.regs[i][1] for i in range(n)]).mean()
        return reg_a + reg_b

    def router_parameters(self):
        for m in self.routed:
            yield from m.alpha_.parameters()
            yield from m.beta_.parameters()

    def freeze_all_but_router(self):
        self.requires_grad_(False)
        for p in self.router_parameters():
            p.requires_grad_(True)

    def load_experts(self, safety_adapter: str, helpfulness_adapter: str):
        for expert_attr, path in (
            ("lora_0", safety_adapter),
            ("lora_1", helpfulness_adapter),
        ):
            a, b = read_down_proj_lora(path)
            for i, m in enumerate(self.routed):
                exp = getattr(m, expert_attr)
                with torch.no_grad():
                    exp.linear1.weight.copy_(a[i])
                    exp.linear2.weight.copy_(b[i])

    def router_state_dict(self):
        return {
            f"{i}.{name}": t
            for i, m in enumerate(self.routed)
            for name, t in {
                **{f"alpha_.{k}": v for k, v in m.alpha_.state_dict().items()},
                **{f"beta_.{k}": v for k, v in m.beta_.state_dict().items()},
            }.items()
        }

    def load_router_state_dict(self, state):
        for i, m in enumerate(self.routed):
            m.alpha_.load_state_dict(
                {
                    k[len(f"{i}.alpha_.") :]: v
                    for k, v in state.items()
                    if k.startswith(f"{i}.alpha_.")
                }
            )
            m.beta_.load_state_dict(
                {
                    k[len(f"{i}.beta_.") :]: v
                    for k, v in state.items()
                    if k.startswith(f"{i}.beta_.")
                }
            )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        kw = dict(gradient_checkpointing_kwargs or {})
        kw["use_reentrant"] = False  # side-channel regularisers need a live graph
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kw)
        self.model.enable_input_require_grads()

    @property
    def config(self):
        return self.model.config


_LORA_KEY = re.compile(
    r"layers\.(\d+)\.mlp\.down_proj\.lora_([AB])(?:\.[^.]+)?\.weight$"
)


def read_down_proj_lora(
    adapter_dir: str,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    from safetensors import safe_open

    path = Path(adapter_dir)
    f = path / "adapter_model.safetensors" if path.is_dir() else path
    a, b = {}, {}
    with safe_open(str(f), framework="pt") as sf:
        for k in sf.keys():
            m = _LORA_KEY.search(k)
            if m:
                (a if m.group(2) == "A" else b)[int(m.group(1))] = sf.get_tensor(k)
    if not a or a.keys() != b.keys():
        raise ValueError(f"no down_proj LoRA A/B pairs found in {f}")
    return a, b


class routing_disabled:
    def __init__(self, router: MidPORouter):
        self.router = router

    def __enter__(self):
        self.prev = self.router.routing_enabled
        self.router.routing_enabled = False

    def __exit__(self, *exc):
        self.router.routing_enabled = self.prev


def _make_router_trainer():
    from human_alignment.safety.losses import preference as P
    from human_alignment.safety.methods.pairwise import PairwiseTrainer, sequence_logps

    class MidPORouterTrainer(PairwiseTrainer):
        """Router loss (``mdpo_router/trainer.py:232-237``): DPO(β) on the routed model vs. the SFT base
        (= the same model with routing disabled) + mean|a| + mean|b − 1|."""

        def pair_loss(self, model, batch):
            router = getattr(model, "module", model)
            pol_c, pol_r = sequence_logps(model, batch, "safe_rlhf")
            reg = router.reg_loss()
            with torch.no_grad(), routing_disabled(router):
                ref_c, ref_r = sequence_logps(model, batch, "safe_rlhf")
            out = P.dpo_loss(pol_c, pol_r, ref_c, ref_r, self.loss_config.beta)
            metrics = {
                "router/reg": reg.detach(),
                "rewards/accuracy": (out.chosen_rewards > out.rejected_rewards)
                .float()
                .mean(),
                "rewards/margin": (out.chosen_rewards - out.rejected_rewards).mean(),
            }
            return out.losses.mean() + reg, metrics

        def save_model(self, output_dir=None, _internal_call=False):
            output_dir = Path(output_dir or self.args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            router = self.accelerator.unwrap_model(self.model)
            torch.save(
                {k: v.detach().cpu() for k, v in router.router_state_dict().items()},
                output_dir / "router.pth",
            )

    return MidPORouterTrainer


def __getattr__(name):
    if name == "MidPORouterTrainer":
        return _make_router_trainer()
    raise AttributeError(name)

