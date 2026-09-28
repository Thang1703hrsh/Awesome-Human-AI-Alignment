# Integrated Safety Alignment

This codebase provides one tested interface for safety-alignment training and evaluation. It contains executable
stages for:

- Supervised fine-tuning, reward modeling, and cost modeling
- Helpful, harmless, and safe-better DPO baselines
- SafeRLHF (PPO-Lagrangian), reward-only PPO, and multi-objective RLHF
- SACPO, KTO-SACPO, and parameter-space P-SACPO
- CAN dual optimization, MoCAN, and PeCAN
- MODPO, CPO/CPSFT/CDPO, BFPO, and MidPO
- Paper-faithful SafeDPO and Bregman Safety Optimization (BSO)

The implementation exposes 25 executable stages through a shared CLI, YAML recipes, dependency-ordered bash entry
points, resource download profiles, and a common reward/cost evaluation pipeline.

---

## Environment Setup

Create an isolated environment and install the project dependencies:

```bash
python -m pip install -e ".[safety]"
```

To create a dedicated environment with all training, evaluation, and test extras, run
`scripts/safety_alignment/tools/setup.sh` on Linux or WSL.

The setup script installs the package, training/test/evaluation extras, runs `pip check`, and prints a CUDA and
dependency diagnostic report.

Main dependencies:

- `torch`
- `transformers`
- `peft`
- `datasets`
- `accelerate`
- `deepspeed`

The exact environment used by the standalone test suite is recorded in
`recipes/safety_alignment/requirements-tested.txt`. Install the PyTorch wheel
appropriate for the CUDA version on the training machine.

Verify the installation:

```bash
hai-align safety doctor --root .
hai-align safety list
python -m unittest discover -s tests -v
```

For full fine-tuning of a 7B model on one H100, the default recipes use DeepSpeed ZeRO-2 with CPU optimizer offload.
The default `recipes/safety_alignment/deepspeed/zero2_offload.json` requires `nvcc`. When a CUDA toolkit is unavailable, use:

```bash
train.deepspeed=recipes/safety_alignment/deepspeed/zero2_offload_torch_adam.json
```

---

## Models and Datasets

Default unified models:

| Purpose | Model |
| --- | --- |
| Shared policy / SFT reference | `PKU-Alignment/alpaca-7b-reproduced` |
| Helpful reward model | `PKU-Alignment/beaver-7b-v1.0-reward` |
| Safety cost model | `PKU-Alignment/beaver-7b-v1.0-cost` |
| Unified helpfulness scorer | `PKU-Alignment/beaver-7b-unified-reward` |
| Unified safety scorer | `PKU-Alignment/beaver-7b-unified-cost` |

Default training datasets:

| Stage / method | Dataset |
| --- | --- |
| SFT | `tatsu-lab/alpaca` |
| Reward / cost models | `PKU-Alignment/PKU-SafeRLHF` |
| DPO, SafeDPO, BSO | `PKU-Alignment/PKU-SafeRLHF-30K` |
| SafeRLHF / PPO | `PKU-Alignment/PKU-SafeRLHF` + `tatsu-lab/alpaca` PTX |
| MORLHF | `PKU-Alignment/PKU-SafeRLHF` |
| SACPO / CAN | `PKU-Alignment/PKU-SafeRLHF-30K` |
| MODPO | `PKU-Alignment/PKU-SafeRLHF-10K` |
| CPSFT | `openbmb/UltraFeedback` |
| CDPO | `openbmb/UltraSafety` |
| BFPO | `PKU-Alignment/PKU-SafeRLHF` + `HuggingFaceH4/ultrafeedback_binarized` |
| MidPO | `PKU-Alignment/PKU-SafeRLHF` |

Download all external inputs, including evaluation datasets:

```bash
hai-align safety prepare all --include-evaluation
```

Download only the inputs for one family:

```bash
hai-align safety prepare safedpo
hai-align safety prepare cpo
hai-align safety prepare midpo --include-evaluation
```

Preview exact repositories and local destinations without downloading:

```bash
hai-align safety prepare all --include-evaluation --dry-run
```

Some base models, WildGuardMix, and XSTest are gated. Accept their Hugging Face terms and export `HF_TOKEN` before
downloading.

---

## Default Output Directories

| Stage / method | Output |
| --- | --- |
| SFT | `output/sft` |
| Reward model | `output/reward_model` |
| Cost model | `output/cost_model` |
| Helpful / harmless / safe-better DPO | `output/dpo_*` |
| SafeDPO | `output/safedpo` |
| BSO | `output/bso` |
| SafeRLHF | `output/saferlhf` |
| PPO | `output/ppo` |
| MORLHF | `output/morlhf_w*` |
| SACPO | `output/sacpo/*` |
| CAN / MoCAN / PeCAN | `output/can/*` |
| MODPO | `output/modpo/*` |
| CPSFT / CDPO | `output/cpo/*` |
| BFPO | `output/bfpo` |
| MidPO | `output/midpo/*` |

Every run writes its resolved recipe to `experiment.yaml`. Multi-stage methods also write their intermediate models
and method-specific metadata into the output directory.

---

## Training Procedure

All entry points forward dotted configuration overrides to `hai-align safety run`.

### 1. Train the Shared SFT Policy

```bash
scripts/safety_alignment/run/run_sft.sh
```

Recipe: `recipes/safety_alignment/methods/sft.yaml`
Dataset: `tatsu-lab/alpaca`
Output: `output/sft`

The released `PKU-Alignment/alpaca-7b-reproduced` checkpoint is the default shared policy, so retraining SFT is
optional for most experiments.

### 2. Train Reward and Cost Models

```bash
scripts/safety_alignment/run/run_reward_cost.sh
```

Recipes: `recipes/safety_alignment/methods/reward_model.yaml`, `recipes/safety_alignment/methods/cost_model.yaml`
Dataset: `PKU-Alignment/PKU-SafeRLHF`
Outputs: `output/reward_model`, `output/cost_model`

Released Beaver reward and cost models are used by default in downstream recipes. Point downstream
`method_args.reward_model` and `method_args.cost_model` to these local outputs when reproducing the complete pipeline.

### 3. Train DPO Baselines

```bash
scripts/safety_alignment/run/run_dpo.sh
```

This trains:

- `dpo_helpful` on `better` pairs
- `dpo_harmless` on `safer` pairs
- `dpo_safebetter` on helpful pairs whose preferred answer is safe

The helpful and harmless checkpoints are also inputs to PeCAN.

### 4. Run SafeDPO

```bash
scripts/safety_alignment/run/run_safedpo.sh
```

Recipe: `recipes/safety_alignment/methods/safedpo.yaml`
Dataset: `PKU-Alignment/PKU-SafeRLHF-30K`
Default safety margin: `delta=10`

Example margin override:

```bash
scripts/safety_alignment/run/run_safedpo.sh method_args.delta=5 train.output_dir=output/safedpo_d5
```

### 5. Run Bregman Safety Optimization

```bash
scripts/safety_alignment/run/run_bso.sh
```

Recipe: `recipes/safety_alignment/methods/bso.yaml`
Default generator: shifted-Bregman (`sba`)
Default paper parameters: `C=30`, `lambda=0.2`, `s=4`

### 6. Run SafeRLHF / PPO-Lagrangian

```bash
scripts/safety_alignment/run/run_saferlhf.sh
```

Recipe: `recipes/safety_alignment/methods/saferlhf.yaml`
Inputs: policy, reward model, cost model, PKU prompts, and Alpaca PTX data
Output: `output/saferlhf`

### 7. Run Reward-Only PPO

```bash
scripts/safety_alignment/run/run_ppo.sh
```

Recipe: `recipes/safety_alignment/methods/ppo.yaml`
Output: `output/ppo`

This shares the SafeRLHF trainer and disables the cost/Lagrange branch.

### 8. Run Multi-Objective RLHF

```bash
scripts/safety_alignment/run/run_morlhf.sh
```

By default, the script trains five preference weights:

```text
0.1, 0.3, 0.5, 0.7, 0.9
```

Override the sweep with:

```bash
WEIGHTS="0.3 0.7" scripts/safety_alignment/run/run_morlhf.sh
```

### 9. Run SACPO

```bash
scripts/safety_alignment/run/run_sacpo.sh
```

Execution order:

1. Helpful DPO stage
2. Safety DPO stages for every beta
3. Safety KTO stages for every beta

Default beta sweep:

```text
0.1, 0.05, 0.025, 0.01
```

Use a smaller sweep with:

```bash
BETAS="0.1 0.01" scripts/safety_alignment/run/run_sacpo.sh
```

### 10. Merge P-SACPO Models

Run SACPO first, then:

```bash
scripts/safety_alignment/run/run_p_sacpo.sh
```

The script produces parameter-space merges for safety weights `0.25`, `0.5`, and `0.75`.

### 11. Run MODPO

```bash
scripts/safety_alignment/run/run_modpo.sh
```

Execution order:

1. Train the safer-DPO margin adapter
2. Train MODPO for preference weights `0.1`, `0.5`, and `0.9`

Override the sweep with:

```bash
WEIGHTS="0.3 0.7" scripts/safety_alignment/run/run_modpo.sh
```

### 12. Run CAN, MoCAN, and PeCAN

Train the helpful and harmless DPO baselines first:

```bash
scripts/safety_alignment/run/run_dpo.sh
scripts/safety_alignment/run/run_can.sh
```

Execution order:

1. Sample and score reference-policy responses
2. Solve the CAN dual problem for `lambda*`
3. Train MoCAN using reward/cost-model relabeling
4. Train PeCAN using implicit rewards from helpful and harmless DPO models

### 13. Run CPO

```bash
scripts/safety_alignment/run/run_cpo.sh
```

Execution order:

1. CPSFT converts released UltraFeedback annotations into honesty-control examples
2. CDPO converts UltraSafety annotations into controllable preference pairs
3. CDPO starts from the CPSFT checkpoint

No manual JSONL shard preparation is required.

### 14. Run BFPO

```bash
scripts/safety_alignment/run/run_bfpo.sh
```

Recipe: `recipes/safety_alignment/methods/bfpo.yaml`
Preference data: pinned revision of `PKU-Alignment/PKU-SafeRLHF`
Replay buffer: `HuggingFaceH4/ultrafeedback_binarized`

### 15. Run MidPO

```bash
scripts/safety_alignment/run/run_midpo.sh
```

Execution order:

1. Train the safety LoRA expert
2. Train the helpfulness LoRA expert
3. Freeze both experts and train the layer-wise router

Outputs: `output/midpo/safety_expert`, `output/midpo/helpfulness_expert`, and `output/midpo/router`.

---

## Method Cheat Sheet

| Family | Prepare command | Train command | Main recipe(s) |
| --- | --- | --- | --- |
| SFT | `hai-align safety prepare sft` | `scripts/safety_alignment/run/run_sft.sh` | `recipes/safety_alignment/methods/sft.yaml` |
| Reward / cost | `hai-align safety prepare reward_cost` | `scripts/safety_alignment/run/run_reward_cost.sh` | `recipes/safety_alignment/methods/reward_model.yaml`, `recipes/safety_alignment/methods/cost_model.yaml` |
| DPO baselines | `hai-align safety prepare dpo` | `scripts/safety_alignment/run/run_dpo.sh` | `recipes/safety_alignment/methods/dpo_*.yaml` |
| SafeDPO | `hai-align safety prepare safedpo` | `scripts/safety_alignment/run/run_safedpo.sh` | `recipes/safety_alignment/methods/safedpo.yaml` |
| BSO | `hai-align safety prepare bso` | `scripts/safety_alignment/run/run_bso.sh` | `recipes/safety_alignment/methods/bso.yaml` |
| SafeRLHF | `hai-align safety prepare saferlhf` | `scripts/safety_alignment/run/run_saferlhf.sh` | `recipes/safety_alignment/methods/saferlhf.yaml` |
| PPO | `hai-align safety prepare ppo` | `scripts/safety_alignment/run/run_ppo.sh` | `recipes/safety_alignment/methods/ppo.yaml` |
| MORLHF | `hai-align safety prepare morlhf` | `scripts/safety_alignment/run/run_morlhf.sh` | `recipes/safety_alignment/methods/morlhf.yaml` |
| SACPO | `hai-align safety prepare sacpo` | `scripts/safety_alignment/run/run_sacpo.sh` | `recipes/safety_alignment/methods/sacpo_*.yaml` |
| P-SACPO | SACPO outputs | `scripts/safety_alignment/run/run_p_sacpo.sh` | `recipes/safety_alignment/methods/p_sacpo.yaml` |
| MODPO | `hai-align safety prepare modpo` | `scripts/safety_alignment/run/run_modpo.sh` | `recipes/safety_alignment/methods/modpo_margin.yaml`, `recipes/safety_alignment/methods/modpo.yaml` |
| CAN | DPO outputs + `hai-align safety prepare can` | `scripts/safety_alignment/run/run_can.sh` | `recipes/safety_alignment/methods/can_dual.yaml`, `recipes/safety_alignment/methods/mocan.yaml`, `recipes/safety_alignment/methods/pecan.yaml` |
| CPO | `hai-align safety prepare cpo` | `scripts/safety_alignment/run/run_cpo.sh` | `recipes/safety_alignment/methods/cpsft.yaml`, `recipes/safety_alignment/methods/cdpo.yaml` |
| BFPO | `hai-align safety prepare bfpo` | `scripts/safety_alignment/run/run_bfpo.sh` | `recipes/safety_alignment/methods/bfpo.yaml` |
| MidPO | `hai-align safety prepare midpo` | `scripts/safety_alignment/run/run_midpo.sh` | `recipes/safety_alignment/methods/midpo_*.yaml` |

---

## Evaluation

Generate candidate and baseline responses, pair them, and compute the standard safety-alignment report:

```bash
BASELINE_MODEL=PKU-Alignment/alpaca-7b-reproduced \
  scripts/safety_alignment/tools/evaluate.sh output/safedpo pku
```

Available profiles:

| Profile | Dataset | Purpose |
| --- | --- | --- |
| `pku` | `PKU-Alignment/PKU-SafeRLHF-30K` | Common helpfulness/safety frontier |
| `beaver` | `PKU-Alignment/BeaverTails-Evaluation` | Held-out SafeRLHF safety prompts |
| `do_not_answer` | `LibrAI/do-not-answer` | Harmful-instruction evaluation |
| `wildguard` | `allenai/wildguardmix`, `wildguardtest` | Out-of-distribution safety prompts |
| `xstest` | `walledai/XSTest` | Over-refusal and unsafe-prompt evaluation |

The standard report contains:

- Mean helpfulness reward
- Mean safety cost
- Safe rate: `mean(cost <= 0)`
- Candidate-vs-reference helpfulness win rate
- Candidate-vs-reference harmlessness win rate
- Joint win rate

Use the unified SafeDPO/BSO score models with:

```bash
REWARD_MODEL=PKU-Alignment/beaver-7b-unified-reward \
COST_MODEL=PKU-Alignment/beaver-7b-unified-cost \
BASELINE_MODEL=PKU-Alignment/alpaca-7b-reproduced \
  scripts/safety_alignment/tools/evaluate.sh output/bso pku
```

Run the optional blinded LLM judge after candidate and reference generations are paired:

```bash
export OPENAI_API_KEY=...
hai-align safety judge evaluation/safedpo/pku/paired.jsonl \
  --model YOUR_JUDGE_MODEL \
  --output-dir evaluation/safedpo/pku/judge
```

The judge randomizes response order, evaluates helpfulness and safety independently, retains raw judgments, and
reports win rates with ties excluded from the denominator.

Method-specific benchmark requirements for CPO, BFPO, MidPO, SafeDPO, and MORLHF are documented in
`docs/safety_alignment/EVALUATION.md`.

---

## Testing and Reproducibility

Run the complete local suite:

```bash
python -m unittest discover -s tests -v
```

This checks catalog and registry integration, recipe inheritance and overrides, the nested CLI, differentiability
of the DPO/SafeDPO/BSO objectives, safety data transforms, resource resolution, and evaluation summaries.

The helper also compiles the source tree and runs Ruff when it is installed:

```bash
bash scripts/safety_alignment/tools/test.sh
```

For only the focused safety regression suite, run
`bash scripts/safety_alignment/examples/test_local.sh`.

Before a full training launch on H100, run the real-model smoke suite:

```bash
bash scripts/safety_alignment/examples/smoke_h100.sh
```

Run a full family in dependency order:

```bash
bash scripts/safety_alignment/examples/full_runs_h100.sh sacpo
```

The standalone source project contained a larger author-source equivalence suite. The integrated repository keeps
focused regression coverage in `tests/test_safety_alignment.py`; retain the upstream checkout separately only if
you need its historical cross-repository comparison fixtures.

---

## Project Layout

```text
recipes/safety_alignment/methods/        Paper and unified training recipes
recipes/safety_alignment/deepspeed/      Single-H100 ZeRO-2 configurations
scripts/safety_alignment/run/       One entry point per method family
scripts/safety_alignment/tools/     Setup, download, test, and evaluation commands
src/human_alignment/safety/         Shared data, losses, trainers, CLI, and evaluators
tests/test_safety_alignment.py      Integrated safety regression tests
scripts/safety_alignment/examples/ Local and H100 orchestration
docs/safety_alignment/SOURCE_MAP.md Author-source and formula mapping
docs/safety_alignment/PORTING_PLAN.md Reproduced quirks, deviations, and open ambiguities
docs/safety_alignment/REPRODUCTION.md Paper-to-recipe reproduction guide
docs/safety_alignment/EVALUATION.md Common and method-specific evaluation protocol
```

---

## Detailed Documentation

- `docs/safety_alignment/SOURCE_MAP.md`: mapping from each implementation to paper equations and author files.
- `docs/safety_alignment/PORTING_PLAN.md`: compatibility decisions, faithful quirks, single-H100 changes, and unresolved paper details.
- `docs/safety_alignment/REPRODUCTION.md`: model/dataset matrix and dependency-ordered execution guide.
- `docs/safety_alignment/EVALUATION.md`: standard scorer report, LLM judge protocol, and method-specific benchmark requirements.

