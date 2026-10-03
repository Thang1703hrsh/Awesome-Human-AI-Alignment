# Reproduction guide

This guide mirrors the operational layout of the sibling `Preference-Distillation` project: every family has an
explicit resource profile, dependency-ordered run script, checked recipe, and a shared evaluation path. The default
unified benchmark uses `PKU-Alignment/alpaca-7b-reproduced` on one H100; deviations from an author's original
backbone are called out below and in `PORTING_PLAN.md`.

## 1. Environment and inputs

```bash
scripts/safety_alignment/tools/setup.sh
source .venv/bin/activate
scripts/safety_alignment/tools/download.sh all --include-evaluation
hai-align safety doctor --root .
```

`hai-align safety prepare PROFILE --dry-run` prints the exact repositories and destinations without downloading them.
WildGuardMix and some base models are gated; accept their terms and set `HF_TOKEN` before downloading.

## 2. Dependency-ordered training

| Family | Prepare | Train | Produced dependency |
|---|---|---|---|
| SFT | `scripts/safety_alignment/tools/download.sh sft` | `scripts/safety_alignment/run/run_sft.sh` | shared SFT policy |
| reward/cost | `scripts/safety_alignment/tools/download.sh reward_cost` | `scripts/safety_alignment/run/run_reward_cost.sh` | optional local RM/CM |
| DPO baselines | `scripts/safety_alignment/tools/download.sh dpo` | `scripts/safety_alignment/run/run_dpo.sh` | helpful/safe policies used by PeCAN |
| SafeDPO | `scripts/safety_alignment/tools/download.sh safedpo` | `scripts/safety_alignment/run/run_safedpo.sh` | standalone |
| BSO | `scripts/safety_alignment/tools/download.sh bso` | `scripts/safety_alignment/run/run_bso.sh` | standalone |
| SafeRLHF | `scripts/safety_alignment/tools/download.sh saferlhf` | `scripts/safety_alignment/run/run_saferlhf.sh` | actor LoRA |
| PPO | `scripts/safety_alignment/tools/download.sh ppo` | `scripts/safety_alignment/run/run_ppo.sh` | actor LoRA |
| MORLHF | `scripts/safety_alignment/tools/download.sh morlhf` | `scripts/safety_alignment/run/run_morlhf.sh` | policies for five preference weights |
| SACPO | `scripts/safety_alignment/tools/download.sh sacpo` | `scripts/safety_alignment/run/run_sacpo.sh` | helpful stage, then DPO/KTO safety stages |
| P-SACPO | SACPO outputs | `scripts/safety_alignment/run/run_p_sacpo.sh` | three linear merges |
| MODPO | `scripts/safety_alignment/tools/download.sh modpo` | `scripts/safety_alignment/run/run_modpo.sh` | margin adapter, then three weights |
| CAN | DPO outputs + `scripts/safety_alignment/tools/download.sh can` | `scripts/safety_alignment/run/run_can.sh` | dual λ, MoCAN and PeCAN |
| CPO | `scripts/safety_alignment/tools/download.sh cpo` | `scripts/safety_alignment/run/run_cpo.sh` | CPSFT then CDPO |
| BFPO | `scripts/safety_alignment/tools/download.sh bfpo` | `scripts/safety_alignment/run/run_bfpo.sh` | standalone |
| MidPO | `scripts/safety_alignment/tools/download.sh midpo` | `scripts/safety_alignment/run/run_midpo.sh` | two experts, then router |

All run scripts forward additional dotted overrides:

```bash
scripts/safety_alignment/run/run_safedpo.sh data.limit=1024 train.max_steps=20 method_args.delta=5
BETAS="0.1 0.01" scripts/safety_alignment/run/run_sacpo.sh
WEIGHTS="0.3 0.7" scripts/safety_alignment/run/run_morlhf.sh
```

## 3. Paper-to-recipe matrix

| Method | Author backbone/data | Unified recipe | Main paper evaluation |
|---|---|---|---|
| SafeRLHF | Alpaca-7B; PKU-SafeRLHF; separate reward/cost models | `saferlhf.yaml` | unified RM/CM, GPT-4/human pairwise Elo |
| DPO baselines | shared SafeDPO SFT; PKU-SafeRLHF-30K | `dpo_*.yaml` | reward, negative cost, cost≤0 ratio |
| SACPO/P-SACPO | Alpaca-7B; PKU-SafeRLHF-30K | `sacpo_*.yaml`, `p_sacpo.yaml` | helpfulness/harmlessness scores and GPT comparisons |
| MoCAN/PeCAN | Alpaca-7B; PKU-SafeRLHF-30K | `can_dual.yaml`, `mocan.yaml`, `pecan.yaml` | reward/cost frontier and judge comparisons |
| MODPO | Alpaca-7B; PKU-SafeRLHF-10K | `modpo_margin.yaml`, `modpo.yaml` | win rates against SFT for helpfulness/harmlessness |
| CPO | Mistral/Zephyr; UltraFeedback + UltraSafety | `cpsft.yaml`, `cdpo.yaml` | MT-Bench, HaluEval 2.0, HackaPrompt, GPT-4 3H scores |
| BFPO | Mistral-7B/Zephyr; UltraChat, UltraFeedback, PKU | `bfpo.yaml` | lm-eval helpfulness suite + generative/discriminative safety |
| MidPO | Alpaca-7B; PKU-SafeRLHF | `midpo_*` | PKU, Do-Not-Answer, WildGuardMix; RM and LLM scores |
| SafeDPO | Alpaca-7B; PKU-SafeRLHF-30K | `safedpo.yaml` | unified RM/CM, GPT judge, XSTest over-refusal |
| BSO | Qwen2.5-0.5B and Llama-3.2-3B; PKU-30K | `bso_qwen2.5_0.5b.yaml`, `bso_llama3.2_3b.yaml` (authors' runs); `bso.yaml` (shared backbone) | unified RM/CM; XSTest over-refusal and pointwise LLM judges |
| MORLHF | Llama-2-7B; HH-RLHF + GPT-2 reward models | `morlhf.yaml` | per-objective rewards over preference weights |

The unified recipes intentionally keep one common Alpaca-7B backbone for controlled comparison. BSO's own runs are
reproduced from the authors' code rather than by overriding the backbone, because they also change tokenization
(chat template, pairs over 2048 tokens dropped), optimiser (RMSprop, lr 1e-6 / 7e-7, global batch 16, grad-clip 10)
and precision (fp32 policy, fp16 reference):

```bash
hai-align safety prepare bso_reference                       # Qwen2.5-0.5B, Llama-3.2-3B (gated), PKU-30K, scorers, XSTest
bash scripts/safety_alignment/examples/bso_reference.sh qwen  # train + PKU reward/cost + XSTest (judge if a key is set)
bash scripts/safety_alignment/examples/bso_reference.sh llama 30 0.3
```

Known differences from the source: HF `Trainer` shuffles pairs per epoch, whereas the source groups pairs by prompt
before shuffling; HF rounds the 5% warmup up rather than down; `generate` strips leading/trailing whitespace from
responses before scoring. The source trains with the chat template but evaluates with the safe-rlhf
`BEGINNING OF CONVERSATION:` template; the script reproduces that choice. RiC/MORLHF's original HH-RLHF setting is documented but the default recipe uses
PKU prompts and Beaver reward/negative-cost to remain a safety-alignment comparison.

## 4. Test levels

```bash
scripts/safety_alignment/tools/test.sh                         # compile, lint (when available), and regression suite
bash scripts/safety_alignment/examples/smoke_h100.sh           # 7B downloads, plumbing and peak GPU memory
python -m unittest -v tests.test_safety_alignment              # focused safety integration suite
```

The first level must pass before a commit. The H100 smoke test should pass before launching any full recipe. Real-data
tests require network access and are intentionally excluded from the default test marker.

