# Preference optimization methods

The public API and `hai-align train` support `ipo`, `bpo`, `tdpo`, `tis_dpo`,
`ti_dpo`, `tbpo_q`, and `tbpo_a`. These belong to Training-Time Alignment /
Preference Optimization. ARC-BPO is intentionally excluded.

```bash
python -m pip install -e ".[preference]"
# IPO delegates to TRL and also needs its extra:
python -m pip install -e ".[dpo]"
hai-align methods --category preference_optimization
hai-align train --config recipes/preference_optimization/tdpo.toml
```

Edit the recipe's model and dataset before training. The recipes are starting
configurations, **not reproductions of the Mistral benchmark table**. In
particular, training data, SFT initialization, effective batch, LoRA, optimizer,
and evaluation settings must be matched separately. No leaderboard evaluation
or published-score claim is included in this integration.

```python
from human_alignment import TDPO, PreferenceOptimizationConfig

run = TDPO(
    model="path/to/sft-policy",
    dataset="data/preference_pairs.jsonl",
    config=PreferenceOptimizationConfig(
        output_dir="outputs/tdpo", beta=0.1, alpha=0.5, tdpo2=True,
        max_length=1024, max_prompt_length=128,
    ),
).train()
run.save("outputs/tdpo-export")
```

Other public classes are `IPO`, `BPO`, `TISDPO`, `TIDPO`, `TBPOQ`, and `TBPOA`.
IPO uses `IPOConfig(loss_type="ipo")`, forwarded to TRL's DPO trainer; all other
new methods use `PreferenceOptimizationConfig` and a native Transformers trainer.
The root import remains dependency-free. Backend dependencies are imported at
training time.

## Objectives and source provenance

| ID | Implementation | Primary source |
| --- | --- | --- |
| `ipo` | Squared log-ratio margin through TRL's IPO option | [IPO paper](https://arxiv.org/abs/2310.12036) |
| `bpo` | Sequence-level Bregman ratio risk; default SBA | [BPO paper](https://arxiv.org/abs/2505.19601) |
| `tdpo` | Forward KL at each prediction state; TDPO2 detaches the chosen KL, TDPO1 does not | [TDPO reference](https://github.com/Vance0124/Token-level-Direct-Preference-Optimization/blob/master/trainers.py) |
| `tis_dpo` | Weighted token log ratios; optional weighted reverse-KL correction | [TIS-DPO reference](https://github.com/exlaw/TIS-DPO/blob/main/trainers.py) |
| `ti_dpo` | Released TDPO-based variant with gradient/Gaussian importance weights and generated-anchor triplet loss | [TI-DPO reference](https://github.com/gracefulning/TIDPO/blob/main/trainers.py) |
| `tbpo_q` | Per-token Bregman risk with learned, centered state-baseline differences | [TokenRatio trainer](https://github.com/truongnd1/TBPO/blob/main/trainers.py) |
| `tbpo_a` | Per-token Bregman risk with forward-KL advantage normalization | [TokenRatio paper](https://arxiv.org/abs/2605.12288) |

Sources inspected on 2026-09-28. Implementations are expressed in the project's
own backend; no external checkout or import of those repositories is required.

For BPO, `log R = beta * (rejected_log_ratio - chosen_log_ratio)`. Supported
generators are `logistic`, `kliep`, `lsif`, `ba`, and `sba`. Additive constants
are omitted, so some risks legitimately have negative values. The LSIF branch
uses the generator identity `h'(R) R - h(R) - h'(1/R)` (hence `R^2 - 2/R`, up to
a constant); the released TokenRatio `h_function.py` instead has `R^2 - 2R`.
This integration follows the mathematical identity for that branch and tests
the gradients against the generator definition. Log ratios are clipped to
`[-log_ratio_clip, log_ratio_clip]` before exponentiation.

TI-DPO follows the released `tidpo` branch: a TDPO2 base by default, weights from
the L1 norm of embedding gradients of the last valid position's maximum logit,
mixed with a Gaussian prior, then normalized to mean one over response tokens.
Its triplet term compares response-only policy/reference log-ratio vectors of
a sampled anchor, chosen, and rejected response using squared distances and a
hinge margin. Anchor sampling is nondifferentiable; scoring the sampled anchor
retains policy gradients. Unlike upstream exception fallbacks, invalid inputs
and nonfinite losses fail explicitly.

TBPO pairs response tokens by their index within each completion and uses the
shorter completion's length. Prompt and padding positions do not contribute.
Q baselines are linear heads over detached final policy states, centered per
pair and clipped. The baseline has its own learning-rate parameter group;
optimizer type and schedule otherwise follow the policy's HF training arguments.
This differs from the upstream separate, unscheduled baseline optimizer.

## Dataset and token-weight contract

Native methods accept JSON/JSONL lists, a Hugging Face dataset ID/directory,
or in-memory records with rendered string fields `prompt`, `chosen`, `rejected`.
Render chat templates explicitly before use. Prompt and completion are tokenized
separately; BOS is prepended if available, EOS appended, prompts left-truncated,
and completions right-truncated. All prompt/padding labels are `-100`.

TIS-DPO additionally requires either:

- `chosen_weights` and `rejected_weights`: final finite, nonnegative weights,
  one per separately tokenized completion token **before appended EOS**; EOS
  receives zero weight. Raw signed upstream `chosen_weight`/`rejected_weight`
  scores are not accepted as final weights.
- `backend_options={"positive_model": "path/to/dpo", "negative_model":
  "path/to/reverse-dpo"}`: estimate weights online using the same tokenizer.
  Rank the positive/negative log ratio on chosen tokens and its negative on
  rejected tokens, mapping ranks to `[0.7, 1.3]` per response. The input models
  must already have been trained and share the policy vocabulary/tokenizer.

Online estimation adds two frozen models to memory. It uses the released rank
transform; other upstream transforms are not exposed. Supplied weights override
online estimation. Missing weights/models raise an error rather than silently
reducing TIS-DPO to DPO.

For exact token alignment, native methods also accept `chosen_input_ids`,
`chosen_labels`, `rejected_input_ids`, `rejected_labels`. These must be unpadded,
unshifted sequences with a nonempty masked prompt followed by contiguous response
labels. In this format, weights match the full `input_ids` length; labels mask
prompt weights. Length mismatches are rejected. These pretokenized records are
specific to the native backend, not the TRL IPO/DPO API.

## Runtime, checkpoints, and validation

Use `backend_options` for `tokenizer`, `reference_model`, `model_kwargs`,
`peft_config` (a PEFT configuration object), and `dataset_split`. With a policy
object, supply a tokenizer object or path. If no reference is supplied, the
initial policy is copied/reloaded as a separate frozen reference. The fixed
reference must be the same across a resumed experiment.

Native training supports a single device and ordinary DDP. TI-DPO currently
requires one process because attribution and generated-anchor forwards access
the underlying model. DeepSpeed/FSDP are rejected by this backend. Full-vocabulary
KL computation is memory-intensive; use smaller microbatches/lengths and LoRA
where appropriate. These runtime choices differ from the upstream launch scripts.

`training.extra_args` forwards HF `TrainingArguments`, including checkpoint
frequency, mixed precision, optimizer, and gradient accumulation. Intermediate
`checkpoint-*` directories retain policy, optimizer, scheduler, and TBPO-Q
baseline state. Resume via `resume_from_checkpoint`. The final output is an HF
policy export suitable for `load_checkpoint`; TBPO-Q also exports
`baseline_head.safetensors`, and every native method writes
`preference_config.json`. Use an intermediate Trainer checkpoint, not the final
policy-only export, to resume optimizer state.

```bash
python -m unittest discover -s tests -p test_preference_optimization.py -v
```

Tests cover independent loss/gradient identities, masks, frozen references,
importance normalization, triplet gradients, all seven API/recipe entries,
tiny-model native training, online TIS weights, HF export/reload, and TBPO-Q
checkpoint resume. Trainer tests require the optional training dependencies.
Full Mistral training, distributed execution, and leaderboard results require
separate validation on the target training environment.
