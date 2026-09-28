# SOURCE_MAP — method → paper → reference code → our code → test

Bundled version-pinned TRL references live in `tests/reference_sources/`. Other author paths are resolved from an
optional sibling checkout during development; tests that need an unavailable author source skip cleanly in a
standalone clone.

| Method | Paper | Reference code | Our code | Verified by |
|---|---|---|---|---|
| SFT | 2310.12773 | `safe-rlhf/safe_rlhf/finetune`, `datasets/supervised.py` | `methods/sft.py` | `test_trainers_tiny::test_sft_trainer_masks_prompt`, `test_data::test_safe_rlhf_encode` |
| Reward / cost model | 2310.12773 | `safe-rlhf/safe_rlhf/values/{reward,cost}/trainer.py` | `losses/score.py`, `methods/score_trainer.py` | `test_score_losses` (vs. source) |
| DPO-Helpful/Harmless/SafeBetter | 2305.18290; SafeDPO §5 | `safe-rlhf/safe_rlhf/algorithms/dpo/trainer.py` | `losses/preference.py::dpo_loss`, `data/pairs.py` | `test_losses_vs_reference::test_safe_rlhf_dpo_full_loss_matches_reference` |
| SafeRLHF (PPO-Lag) | 2310.12773 | `safe-rlhf/safe_rlhf/algorithms/ppo_lag`, `trainers/rl_trainer.py` | `losses/ppo.py` §A, `methods/ppo.py` | `test_ppo_losses` (vs. source), `test_ppo_trainer_tiny` |
| PPO | 2310.12773 | `safe-rlhf/safe_rlhf/algorithms/ppo` | same | same |
| MORLHF | RiC (ICML'24) baseline | `RiC/ppo/morlhf.py` + TRL 0.8.0 `PPOTrainer` | `losses/ppo.py` §B, `methods/morlhf.py` | `test_ppo_losses::test_trl_*` (vs. TRL 0.8.0), `test_morlhf_tiny` |
| SACPO | 2404.11049 | `sacpo/src/train/*_{dpo,kto}.py` (stock TRL) | pairwise / `methods/kto.py` | `test_losses_vs_reference::test_kto_matches_pinned_trl`, `test_data::test_trl_encode_pair_matches_trl_086` |
| P-SACPO | 2404.11049 | `sacpo/config/merge/linear_*.yaml` (mergekit linear) | `methods/merge.py` | `test_can_dual_and_merge::test_merge_*` |
| MoCAN | 2405.19544 | `CAN/safe_rlhf/algorithms/cdpo/dpo.py`, `trainers/model_based.py` | `data/pairs.py::mocan_logits`, `methods/can_dual.py` | `test_data::test_mocan_relabel_matches_reference`, `test_can_dual_and_merge::test_dual_matches_reference` |
| PeCAN | 2405.19544 | `CAN/safe_rlhf/algorithms/cdpo/dpo_alg2.py` | `data/pairs.py::pecan_logits` | `test_data::test_pecan_relabel_matches_reference` |
| MODPO | 2310.03708 | `modpo/src/trainer/modpo_trainer.py`, `src/utils/reward.py` | `losses/preference.py::modpo_loss`, pairwise trainer | `test_losses_vs_reference::test_modpo_matches_reference`, `test_trainers_tiny::test_modpo_with_margin_adapter` |
| CPO / CDPO | 2402.19085 | `CPO/src/CDPO/data_preparation`, `CPO/src/CPSFT` | `data/cpo.py` | `test_cpo_data` (identical pairs on the real UltraSafety file) |
| BFPO | 2408.15313 | `bfpo/src/alignment/trainer/bfpo.py`, buffer in `trainer.py` | `losses/preference.py::bfpo_loss`, pairwise trainer | `test_losses_vs_reference::test_bfpo_matches_reference`, `test_trainers_tiny::test_bfpo_buffer_adds_loss` |
| MidPO | EMNLP-F 2025 | `MidPO/safe_rlhf/algorithms/mdpo/*` | `losses/preference.py::midpo_expert_loss`, `methods/midpo.py` | `test_losses_vs_reference::test_midpo_expert_full_loss_matches_reference`, `test_midpo_router` |
| SafeDPO | 2505.20065 (no code) | paper §3, Eq. 11–12, App. B.1 | `safedpo_loss`, `pairs.safedpo_transform` | `test_losses_vs_reference::test_safedpo_*`, `test_real_data` |
| BSO | 2605.12339 (no code) | paper §3, Eq. 12–24, App. D, F | `bso_loss`, `bso_per_sample` | `test_losses_vs_reference::test_bso_*` |

Not ported (outside the method list): RiC itself (`RiC/ric`), PPO reward shaping, the RiC text-to-image variant.

## Core formulas

Notation: `Δπ(y) = log πθ(y|x) − log πref(y|x)` summed over response tokens; `h = Δπ(y_w) − Δπ(y_l)`.

* **DPO** `−logσ(β h)`, β = 0.1.
* **SafeDPO** on T(D) (keep if y_w safe; swap if y_w unsafe and y_l safe; drop if both unsafe):
  `−logσ(β h − (h̃_l − h̃_w) Δ)`, Δ = 10.
* **BSO** `log R = −(β h + C (s_w − s_l))`, `ℓ = h'(R)R − h(R) − h'(1/R)`; SBA_λ: `ℓ = [λR^{1+λ} − (1+λ)R^{−λ} + 1]/(sλ(λ+1))`,
  C = 30, λ = 0.2, s = 4, on T(D).
* **BFPO** `(h − (1/β)(b1·b3·s_w − b3·s_l − α))²`, b1 = 3, b3 = 1/(b1−1), α = 0.5, s = 1 for safe; plus the same
  loss on an UltraFeedback buffer batch every step.
* **MODPO** `R(y) = (1/w0)[β Δπ(y) − w[1:]·m(y)]`, `−logσ(R_w − R_l)`, m = β_m(log π_margin − log π_ref).
* **MidPO experts** safety `−logσ(β h + min(0, S_w − S_l))`, helpfulness `−logσ(β h − max(0, R_w − R_l))`.
  **Router**: `out = down(z) + a(z) lora_s(z) + b(z) lora_h(z)`, loss `DPO + mean|a| + mean|b − 1|`.
* **CAN** dual `D(λ) = β E_x log E_{π_ref} exp((r + λ(g − b))/β)`; MoCAN label ~ Bernoulli(σ(w1 − w0)),
  `w = r − λ c`; PeCAN `w = (log π_help − log π_ref) + λ (log π_safe − log π_ref)`; then DPO.
* **SACPO** DPO/KTO on helpfulness, then DPO/KTO on safety starting from and referenced to the stage-1 model.
  **P-SACPO** `θ = (1−q) θ_stage1 + q θ_SACPO`, q ∈ {0.25, 0.5, 0.75}.
* **CPO** pairs ranked by `R = −|H_resp − H_cond| (+ help + honesty terms)`, control tags prepended, vanilla DPO.
* **SafeRLHF** reward stream `−β KL`, cost stream `+β KL`, GAE (γ=1, λ=0.95), `A = (A_r − λ A_c)/(1+λ)`, PPO clip
  0.2, clipped critics; `log λ ← log λ + lr (J_c − d) λ`, λ ≤ 5.
* **MORLHF** reward `round(Σ w_k r_k, 2)`, TRL 0.8.0 PPO (adaptive KL 0.2 → target 3, whitened advantages, 4 epochs).

