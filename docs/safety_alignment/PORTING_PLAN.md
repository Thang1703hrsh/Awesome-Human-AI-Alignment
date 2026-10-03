# PORTING_PLAN — decisions, reproduced quirks, deviations, open questions

## Decisions (confirmed with the user, 2026-09-25)
* Base model for every config: `PKU-Alignment/alpaca-7b-reproduced`; target hardware: 1× H100 80 GB.
* SafeRLHF / PPO / MORLHF: LoRA actor and LoRA critics (full-FT PPO with six 7B models does not fit one GPU).
* No TRL dependency: each repository pins a different, mutually incompatible TRL; all losses are native PyTorch.
* Default behaviour reproduces the author code **including its quirks**; each quirk has a flag where it matters.

## Author quirks reproduced by default
| Where | Quirk | Switch |
|---|---|---|
| safe-rlhf DPO, SafeDPO/BSO/DPO-* configs, MidPO | sequence log-prob = Σ lp[diverge : end+1]: skips the first diverging token, adds one position past EOS | `data.tokenization: trl` → masked convention |
| MidPO router | regulariser from a second `mlp(h)` on the decoder-layer *input*, averaged over all tokens incl. padding | `method_args.reg_source: mlp_input` |
| MidPO router | expert LoRA copied without peft's `alpha/r` (=2) scale | `method_args.expert_scale: 2.0` |
| MidPO router | router MLP has no activations (3 stacked Linear + sigmoid), width 512 | `method_args.router_nonlinearity: true` |
| CAN dual | lr = 2·lr, restart decay `lr /= 2**n` compounds, `num_iters` kwarg | — |
| MoCAN / PeCAN | relabeled prompts fed to DPO without the conversation template; PeCAN uses raw log-ratios (no β) | `data.template` |
| MoCAN | RM/CM scored on `PROMPT + ' ' + response` without EOS | — |
| PeCAN | log-probs = TRL `precompute_ref_log_probs` (masked sums) on templated prompts, response_0 as "chosen" (`evaluate/collect_log_probs_dpo.py`) | — |
| CPO | tag spelled `helplessness`; UltraFeedback R2 = `-abs(x) - y` (precedence); default-safe tag stripped | — |
| TRL KTO (SACPO) | with a one-sided batch the loss is divided by B+1 (NaN placeholder in the mean) | — |
| RiC MORLHF | `str.strip('[PAD] ')` etc. strip *character sets* (e.g. 'Answer' → 'wer'); kept length = len(encode(clean)) incl. BOS | `method_args.strip_mode: tokens` |
| safe-rlhf PPO | with PTX the actor GA doubles, so each loss is scaled by 1/(2·GA) | — |
| BFPO | `x[int_tensor]` used as a mask in logged metrics | not reproduced (metrics only) |

## Deviations required by one H100 or by the chosen base model
| Method | Original | Here | Why |
|---|---|---|---|
| All | 2–8 GPUs | 1 GPU, same *global* batch via gradient accumulation | hardware |
| Full-FT methods | DeepSpeed ZeRO-3 / FSDP | DeepSpeed ZeRO-2 + CPU optimizer offload | single GPU; same fp32-master numerics |
| SafeRLHF, PPO | full FT, 6 models | LoRA r64 actor/critics; critic = LoRA + own head on the RM/CM backbone | memory (user decision) |
| SafeRLHF, PPO | 1 λ update per data-parallel step, cost window 128 per rank | `lambda_update_interval = GA`, window 128·8 | keeps λ dynamics per sample identical |
| MoCAN, PeCAN, CDPO | 4-bit QLoRA (bitsandbytes / unsloth), paged / 8-bit Adam | bf16 LoRA, AdamW | H100 has the memory; avoids unsloth |
| CDPO, BFPO | Mistral / zephyr base with chat templates | alpaca-7b with the PKU template | base model chosen by the user |
| MORLHF | HH-RLHF prompts, GPT-2 reward models | PKU prompts, beaver reward and −cost | safety setting |
| CPSFT | cutoff 8192 | `max_length` 2048 (configurable) | memory |
| Log-softmax | model dtype (bf16 under DeepSpeed) | fp32 upcast of bf16 logits | numerical precision only |

## Open questions (no reference code, or the reference is ambiguous)
1. **BSO scale of C — resolved.** The authors' SafeBPO code computes `log R_safe = β(h_l − h_w) − C(s_w − s_l)`, i.e.
   C outside β as in Eq. 12/18, so `bso_penalty_inside_beta: false` is the reference behaviour (the flag remains for
   ablations). The code also clamps log R to [−30, 30] before ℓ_h; with C = 30 a safe-winner pair starts exactly on
   the bound and its gradient is zero whenever β·h < 0, which is why the authors' scripts raise the upper bound to
   min(C + 10, ⌊85/(1+λ)⌋) for C > 30. Their own runs (Qwen2.5-0.5B, Llama-3.2-3B) are `bso_qwen2.5_0.5b.yaml` and
   `bso_llama3.2_3b.yaml`; `bso.yaml` keeps the shared Alpaca-7B backbone with SafeDPO Table 4 optimiser settings.
2. **MidPO safety score model** is `/root/.../safety-reward-eval` in the repo. The sign convention of the two expert
   losses implies a cost-like score, so the config uses `beaver-7b-unified-cost`.
3. **MidPO / SafeDPO GPU count** is not stated; configs assume 8 GPUs for the global batch.
4. **PeCAN λ**: the preference-based dual notebook is missing from the CAN repo; `can_dual` implements the
   model-based dual (MoCAN). Pass `method_args.lam` explicitly for PeCAN.
5. **Datasets moved on the Hub**: `PKU-Alignment/PKU-SafeRLHF` was re-released in 2024; safe-rlhf used the earlier
   version. BFPO pins revision `ff7ba91…` (set in `bfpo.yaml`); pin `data.revision` elsewhere if you need the old data.
6. **CAN hold-out split** uses `train_test_split(test_size=0.1)` without a seed in the original; configs keep it
   unseeded (`holdout_seed: null`), so the exact subset differs run to run exactly as in the original.

## Memory plan for 1× H100 80 GB (7B, bf16, gradient checkpointing, seq 512)
| Family | GPU | CPU RAM |
|---|---|---|
| Full-FT DPO family, SACPO, SFT, RM/CM | ~14 GB weights + ~14 GB ref + activations (ZeRO-2 offload) | ~110 GB (fp32 master + Adam) |
| BFPO (down_proj only) | ~14 GB + 14 GB ref + 23 GB fp32 master/Adam for 1.4 B params | small |
| LoRA methods (MODPO, CAN, CDPO, MidPO) | ~14 GB + LoRA states | small |
| SafeRLHF / PPO-Lag | 3 backbones × 13.5 GB + LoRA states + rollout KV cache | small |
Run `bash scripts/safety_alignment/examples/smoke_h100.sh` first: it prints the peak GPU memory of every method.

