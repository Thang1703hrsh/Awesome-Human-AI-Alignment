# Alignment Specification: executable reference components

This extension covers the **22 uncommented citations** supplied for Task &
Assistance, Personalized Alignment, and Uncertainty & Drift. WDPO and KLDPO have
separate IDs (23 registry entries). Commented citations and ARC-BPO are excluded.

These are **reference components and data adapters, not 22 reproduced papers**.
No pretrained weights, original benchmark datasets, or published scores are
bundled. Some papers describe datasets or empirical studies rather than training
algorithms. Their components therefore belong to supervision or assurance while
retaining their Alignment Specification taxonomy category.

## Coverage and sources

| Citation / primary source | Registry ID | Implemented scope; remaining work |
|---|---|---|
| [Scaling Instruction-Finetuned LMs](https://www.jmlr.org/papers/v25/23-0870.html) | `flan_mixture` | Seeded weighted collection sampling into SFT examples. Supply actual FLAN collections, published mixture weights, templates and training configuration. |
| [Parrot](https://aclanthology.org/2024.acl-long.525/) | `parrot` | Multi-turn SFT export preserving earlier conversation. Normalized messages input; does not implement synthetic conversation-tree generation. |
| [A Good Plan is Hard to Find](https://aclanthology.org/2025.emnlp-main.585/) | `good_plan` | Separate measured preference and observed utility, with paired correlation. Planorama-inspired descriptive analysis, not its user-study interface or protocol. |
| [IFEval++](https://aclanthology.org/2026.acl-long.354/) | `ifeval_pp` | Group reliability: all k explicitly supplied cousin prompts must pass. Does not generate cousin prompts or bundle the original checker suite. [Official code](https://github.com/jianshuod/IFEval-pp). |
| [MOSAIC](https://aclanthology.org/2026.eacl-long.62/) | `mosaic` | Compliance aggregation by constraint kind, order, and count. Small extensible checker registry, not the full published constraint inventory or benchmark. |
| [ReasonIF](https://aclanthology.org/2026.findings-acl.1456/) | `reasonif` | Separate observed reasoning/final compliance. Missing reasoning is unknown, never a pass. Supply authorized, actually observed traces and benchmark-specific checks. |
| [FB-Bench](https://aclanthology.org/2025.emnlp-main.471/) | `fb_bench` | Before/after correction and preservation metrics on caller-scored outputs. Not the full evaluation protocol. [Official benchmark](https://github.com/PKU-Baichuan-MLSystemLab/FB-Bench). |
| [VPL](https://arxiv.org/abs/2408.10075) | `vpl` | Differentiable sampled Bradley-Terry negative ELBO with Gaussian posterior KL. Supply latent encoder, reparameterized samples, reward network and user-grouped batches. [Official LLM code](https://github.com/WEIRDLabUW/vpl_llm). |
| [OPPU](https://aclanthology.org/2024.emnlp-main.372/) | `oppu` | Independent per-user model factory/trainer calls. Supply PEFT construction, actual optimizer, task formatting and adapter persistence. No automatic cross-user pooling. |
| [PRISM](https://huggingface.co/datasets/HannahRoseKirk/prism-alignment) | `prism` | Official `conversation_history` schema to within-user/turn preference pairs. Subsequent context follows `if_chosen`, not the highest score. Demographic analysis and dataset loading remain external. |
| [PAD](https://arxiv.org/abs/2410.04070) | `pad` | Top-k base-model candidate filtering then personalized reward-adjusted decoding (Eq. 13). Supply trained preference-conditioned scalar PersRM rewards and generation loop. [Official project](https://github.com/zjuruizhechen/PAD). |
| [PROSE](https://proceedings.mlr.press/v267/aroca-ouellette25a.html) | `prose` | Structured infer/verify/refine orchestration across writing samples. Caller provides backend and task prompts; no claim that model assertions prove a preference is correct. |
| [FPPS](https://aclanthology.org/2026.findings-acl.395/) | `fpps` | Hard, soft and mixed hidden-state transformations (Eqs. 5–9). Supply trained factuality probe, selected layers, personalization/factual vectors and model hooks. |
| [Personalized Benchmarking](https://aclanthology.org/2026.findings-acl.31/) | `personalized_benchmark` | Per-user Elo and regularized Bradley-Terry fitting. Elo is input-order dependent; disconnected comparison graphs cannot establish a global empirical ordering. No learned user-feature ranking predictor. |
| [Interaction-based Alignment](https://aclanthology.org/2025.coling-main.511/) | `interaction_alignment` | **Related explicit-profile adaptation**, not reproduction of the paper's implicit conversational meta-skill, SFT/RL training or ALOE benchmark. Backend required. |
| [Distributional Preference Learning](https://arxiv.org/abs/2312.08358) | `distributional_preference` | Gaussian/probit and categorical reward-distribution preference likelihoods. Models hidden-context/aleatoric uncertainty, not automatically epistemic uncertainty. Supply distributional reward heads. [Official code](https://github.com/cassidylaidlaw/hidden-context). |
| [Active Preference Learning](https://arxiv.org/abs/2402.08114) | `active_preference` | Select highest absolute implicit reward gaps, optionally after high predictive-entropy filtering. Requires policy/reference sequence log probabilities and separately acquired labels. |
| [Stability of Moral Preferences](https://ojs.aaai.org/index.php/AIES/article/view/31626) | `preference_stability` | Within-user/item/context consecutive retest agreement and lag. Descriptive utility, not reproduction of the original statistical study. |
| [COPR](https://aclanthology.org/2025.findings-acl.281/) | `copr` | Set-normalized log-probability MSE, normalized replay Lagrangian and dual update. Caller must construct paper-specific optimal targets, replay memory, thresholds and current-task SFT term. Not an end-to-end continual trainer. |
| [Moral Change or Noise?](https://ojs.aaai.org/index.php/AAAI/article/view/41083) | `moral_change` | Retest report explicitly does not identify genuine change versus noise. No causal/noise decomposition is inferred from flip counts. |
| [PILAF](https://proceedings.mlr.press/v267/feng25g.html) | `pilaf` | Practical token-logit plus/minus policy mixture sampler with 50% base/base pairs. Requires same vocabulary and aligned token IDs; not theoretical T-PILAF or a labeler. |
| [WDPO / KLDPO](https://arxiv.org/abs/2502.01930) | `wdpo`, `kldpo` | First-order Wasserstein input-gradient penalty and fixed-temperature KL-DRO envelope gradient. Supply unreduced DPO losses. No radius-to-temperature solver or distributed global-batch aggregation. [Official code](https://github.com/TheBlackCat22/distributionally_robust_dpo). |

## Modules and lifecycle

- `specification/tasks.py`: `TaskSpecification`, scoped `InstructionConstraint`, custom checkers.
- `specification/personalization.py`: explicit `UserProfile`, `PersonalizedSpecification`, user-isolated workflows.
- `specification/uncertainty.py`: externally estimated target uncertainty, active acquisition and PILAF.
- `supervision/specification_data.py`: FLAN, Parrot, PRISM preparation.
- `mechanisms/training/specification_losses.py`: differentiable low-level losses.
- `mechanisms/inference/personalized.py`: PAD/FPPS kernels.
- `assurance/specification.py`: task, personalized ranking, retest metrics.

The three specification objects implement the existing `resolve(context)`
contract. They preserve base target constraints. `PersonalizedSpecification`
requires an explicit `context.metadata['user_id']`; absent users fail rather than
receiving another user's profile. Stored profile preferences are data, not a
security policy; downstream prompting must not let them override safety rules.

Registry entry points expose **different component APIs**, not a universal
`.fit()` trainer. For example:

```python
from human_alignment.registry import default_registry

registry = default_registry()
metric = registry.create("ifeval_pp")
print(metric.evaluate({"question_a": [True, True], "question_b": [True, False]}, k=2))
# reliable@2 = 0.5; prompt_accuracy = 0.75
```

Run `python recipes/specification/demo.py` after `pip install -e .` for a
dependency-free task/profile/uncertainty example. Torch kernels additionally need
`pip install -e ".[specification]"`; imports and the base registry stay usable
without Torch. No network/model calls occur during registry construction.

## Objective contracts

VPL expects `reward_margins[samples,batch]` computed from latent-conditioned
chosen-minus-rejected rewards, and Gaussian posterior parameters `[batch,latent]`.
The caller must sample latents with a differentiable reparameterization; this
module cannot validate that provenance from numerical margins alone.

Distributional Gaussian loss expects **positive variances**, not standard
deviations. Categorical logits use a shared ordered support; equal-bin comparisons
contribute half probability. Gaussian preference likelihood uses a stable log-CDF.

COPR `fit_loss(policy_logps, target_logps)` normalizes over the supplied response
set. Targets are detached. `loss(current_loss, replay_errors, thresholds,
log_multipliers)` assumes each replay error has already been reduced within its
task. `current_loss` includes the caller's current fitting and SFT terms. At the
first task with no replay, use current loss directly. Store historical targets
and update dual variables separately using detached constraint violations.

WDPO expects independent examples: input embeddings must not include cross-batch
coupling. Pass both chosen and rejected embedding tensors. Second derivatives
are necessary; some fused attention/backends do not support them. KL-DRO uses
detached softmax weights: its returned value is a weighted surrogate with the
envelope gradient, not a report of the exact robust objective value.

```python
import torch
from human_alignment.registry import default_registry

# Sequence log probabilities must exclude prompt/padding and use the same
# response masking for policy and frozen reference.
policy_chosen = torch.tensor([-2., -3.], requires_grad=True)
policy_rejected = torch.tensor([-4., -2.], requires_grad=True)
reference_margin = torch.tensor([0., 0.])
losses = torch.nn.functional.softplus(
    -0.1 * (policy_chosen - policy_rejected - reference_margin))
loss = default_registry().create("kldpo", temperature=1.).loss(losses)
loss.backward()
```

## Backend contracts and privacy

`PROSE.backend(request)` receives tasks `infer_preferences`, `verify_preferences`
or `refine_preferences`. Inference/refinement must return a nonempty string list
under `preferences`; verification returns boolean `supported` and text `critique`.
Failure after the configured rounds raises an error rather than publishing an
unverified profile. This is a reference orchestration, not the paper's complete
prompt collection.

`InteractionAlignment.backend(request)` receives `history` and `preferences` and
returns nonempty `response` plus `preferences`. Histories remain user-scoped in
memory. The backend itself must be stateless or isolate sessions; these classes
cannot enforce isolation inside an external model service. Obtain consent before
sending user writing/history to any remote backend, and implement retention and
deletion policies before deployment. No remote backend is configured by default.

## Verification

`python -m unittest discover -s tests -q` exercises deterministic metrics,
dataset/history provenance, isolation, missing reasoning, acquisition ordering,
PILAF sampling, Torch formulas and a small optimizer step. These tests establish
component behavior, **not benchmark reproduction or trained-model performance**.
