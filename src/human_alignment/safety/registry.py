"""Method registry: name to runner, reference implementation, and description."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class MethodSpec:
    runner: str
    source: str
    description: str


METHODS: dict[str, MethodSpec] = {
    "sft": MethodSpec(
        "run_sft",
        "safe-rlhf/safe_rlhf/finetune",
        "Supervised fine-tuning (prompt masked)",
    ),
    "cpsft": MethodSpec(
        "run_sft", "CPO/src/CPSFT", "CPO stage 1: control-token SFT on whole text"
    ),
    "reward_model": MethodSpec(
        "run_score_model",
        "safe-rlhf/safe_rlhf/values/reward",
        "Bradley-Terry reward model",
    ),
    "cost_model": MethodSpec(
        "run_score_model",
        "safe-rlhf/safe_rlhf/values/cost",
        "Cost model with safety-sign loss",
    ),
    "dpo_helpful": MethodSpec(
        "run_pairwise", "safe-rlhf/safe_rlhf/algorithms/dpo", "DPO on `better` pairs"
    ),
    "dpo_harmless": MethodSpec(
        "run_pairwise", "safe-rlhf/safe_rlhf/algorithms/dpo", "DPO on `safer` pairs"
    ),
    "dpo_safebetter": MethodSpec(
        "run_pairwise",
        "safe-rlhf/safe_rlhf/algorithms/dpo",
        "DPO on `better` pairs whose preferred response is safe",
    ),
    "saferlhf": MethodSpec(
        "run_saferlhf",
        "safe-rlhf/safe_rlhf/algorithms/ppo_lag",
        "PPO-Lagrangian (LoRA)",
    ),
    "ppo": MethodSpec(
        "run_saferlhf",
        "safe-rlhf/safe_rlhf/algorithms/ppo",
        "PPO on the reward only (LoRA)",
    ),
    "morlhf": MethodSpec(
        "run_morlhf",
        "RiC/ppo/morlhf.py",
        "Linear-scalarised multi-objective PPO (TRL 0.8.0)",
    ),
    "sacpo_dpo": MethodSpec(
        "run_pairwise",
        "sacpo/src/train/*_dpo.py",
        "SACPO stage: DPO from the previous stage",
    ),
    "sacpo_kto": MethodSpec(
        "run_kto",
        "sacpo/src/train/*_kto.py",
        "SACPO stage: KTO from the previous stage",
    ),
    "p_sacpo": MethodSpec(
        "run_merge",
        "sacpo/config/merge",
        "P-SACPO: linear merge of stage-1 and SACPO models",
    ),
    "can_dual": MethodSpec(
        "run_can_dual",
        "CAN/safe_rlhf/trainers/model_based.py",
        "CAN: solve the dual for lambda*",
    ),
    "mocan": MethodSpec(
        "run_pairwise",
        "CAN/safe_rlhf/algorithms/cdpo/dpo.py",
        "MoCAN: RM/CM relabel + DPO",
    ),
    "pecan": MethodSpec(
        "run_pairwise",
        "CAN/safe_rlhf/algorithms/cdpo/dpo_alg2.py",
        "PeCAN: implicit-reward relabel + DPO",
    ),
    "modpo_margin": MethodSpec(
        "run_pairwise",
        "modpo/scripts/examples/dpo/dpo.py",
        "MODPO stage 1: DPO margin model",
    ),
    "modpo": MethodSpec(
        "run_pairwise",
        "modpo/src/trainer/modpo_trainer.py",
        "MODPO with a DPO margin adapter",
    ),
    "cdpo": MethodSpec(
        "run_pairwise", "CPO/src/CDPO", "CPO stage 2: DPO on controllable pairs"
    ),
    "bfpo": MethodSpec(
        "run_pairwise",
        "bfpo/src/alignment/trainer/bfpo.py",
        "BFPO + helpfulness buffer replay",
    ),
    "midpo_safety_expert": MethodSpec(
        "run_pairwise", "MidPO/.../mdpo_safety_expert", "MidPO safety expert (LoRA)"
    ),
    "midpo_helpfulness_expert": MethodSpec(
        "run_pairwise",
        "MidPO/.../mdpo_helpfulness_expert",
        "MidPO helpfulness expert (LoRA)",
    ),
    "midpo_router": MethodSpec(
        "run_midpo_router",
        "MidPO/.../mdpo_router",
        "MidPO router over two frozen experts",
    ),
    "safedpo": MethodSpec(
        "run_pairwise", "paper 2505.20065 section 3", "SafeDPO: T(D) + safety margin delta"
    ),
    "bso": MethodSpec(
        "run_pairwise", "paper 2605.12339 section 3", "BSO: Bregman ratio matching (SBA_lambda)"
    ),
}


def get_runner(method: str) -> Callable:
    from human_alignment.safety.methods import runners

    if method not in METHODS:
        raise KeyError(f"unknown method {method!r}; known: {sorted(METHODS)}")
    return getattr(runners, METHODS[method].runner)

