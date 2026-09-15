# Codebase architecture

The package keeps the survey's four lifecycle dimensions visible while exposing a short public API. Taxonomy categories describe functional roles; they are not forced into a one-to-one mapping with executable classes.

## Quick start

The dependency-free core supports catalog inspection, pipeline composition, and custom adapters:

```bash
python -m pip install -e .
hai-align catalog validate
python examples/minimal_pipeline.py
```

Install method-specific dependencies only when needed:

```bash
python -m pip install -e ".[dpo]"
```

```python
from human_alignment import DPO, DPOConfig

run = DPO(
    model="Qwen/Qwen3-0.6B",
    dataset="trl-lib/ultrafeedback_binarized",
    config=DPOConfig(output_dir="outputs/qwen-dpo", beta=0.1),
).train()

print(run.metrics)
print(run.generate("Explain Human-AI alignment."))
```

The equivalent generic API is:

```python
from human_alignment import align

run = align(
    "dpo",
    model="Qwen/Qwen3-0.6B",
    dataset="preferences.jsonl",
    output_dir="outputs/qwen-dpo",
)
```

Pass an existing model directory as `model` to continue fine-tuning it. To
resume an interrupted Trainer job, set
`resume_from_checkpoint=True` (latest checkpoint) or provide a checkpoint path
in the method configuration.

## Lifecycle mapping

| Survey dimension | Package | Responsibility |
|---|---|---|
| Alignment Specification | `specification/` | Targets, stakeholders, constraints, context, and uncertainty |
| Alignment Supervision | `supervision/` | Feedback sources, representations, and dataset normalization |
| Alignment Mechanisms | `mechanisms/` | Training-time and inference-time behavioral interventions |
| Alignment Assurance | `assurance/` | Evaluation, evaluator reliability, robustness, preservation, and monitoring |

Two user-facing flows sit above these lifecycle components:

```text
pretrained model/checkpoint -> alignment method -> saved aligned checkpoint
saved checkpoint -> load_checkpoint() -> generation or assurance evaluation
```

A preference is a supervision representation; DPO is a mechanism that consumes preference data. A verifier model is learned by a mechanism; testing verifier reliability is assurance. These distinctions are retained in `TaxonomyTrace` and registry metadata.

Training implementations are grouped by their functional mechanism:

```text
mechanisms/
├── base.py
├── training/
│   ├── base.py
│   ├── supervised/
│   │   └── sft.py
│   ├── preference/
│   │   ├── dpo.py
│   │   ├── kto.py
│   │   └── simpo.py
│   ├── reinforcement/
│   │   ├── ppo.py
│   │   └── grpo.py
│   ├── reward_verifier.py
│   ├── distillation.py
│   └── distillation_losses.py
└── inference/
    ├── steering.py
    ├── search.py
    ├── refinement.py
    └── control.py
```

The folders make taxonomy placement discoverable to contributors, while
`human_alignment` and `mechanisms.training` continue to expose a flat public
API for users.

## Responsibility boundaries

The normal training path is:

```text
public method -> dataset normalization -> optional integration -> AlignmentRun
```

- `mechanisms/training/preference/dpo.py` provides the DPO implementation and
  lifecycle metadata.
- `mechanisms/training/__init__.py` re-exports training methods so callers do
  not depend on the internal folder layout.
- `supervision/datasets.py` validates local records and common examples.
- `integrations/trl.py` is the only module coupled to TRL trainer APIs.
- `integrations/preference_distillation.py` implements the shared Transformers trainer for VPD, PPD, DCKD, TVKD, ADPA, and CTPD.
- `integrations/distillation_data.py` executes teacher/student models to prepare
  their method-specific supervision.
- `integrations/transformers.py` owns model loading and basic generation.
- `models/checkpoint.py` is the stable public boundary for loading full
  checkpoints or PEFT adapters.
- `workflows/` composes data preparation without leaking backend details into
  the taxonomy components.
- `results.AlignmentRun` gives all methods the same save and generate interface.
- `catalog/` contains survey metadata and never initiates training.

Optional dependencies are imported lazily. Importing `human_alignment` does not require PyTorch, Transformers, TRL, datasets, or vLLM.

## Loading checkpoints

`load_checkpoint()` accepts either a Hugging Face model ID or a local directory
created by a training run. It supports causal language models, sequence
classifiers, and optional PEFT adapters:

```python
from human_alignment import CheckpointConfig, load_checkpoint

run = load_checkpoint(
    "models/base",
    config=CheckpointConfig(
        adapter_path="outputs/dpo-adapter",
        device_map="auto",
        torch_dtype="bfloat16",
    ),
)
text = run.generate("Explain preference optimization.")
report = run.evaluate(evaluator, evaluation_cases)
```

`load_model()` remains as a backward-compatible alias.

## Supported input forms

DPO accepts a Hugging Face dataset ID, a Dataset object, JSON/JSONL, dictionaries, or typed examples:

```python
from human_alignment import PreferenceExample

examples = [
    PreferenceExample(
        prompt="How should uncertainty be communicated?",
        chosen="State uncertainty and avoid unsupported claims.",
        rejected="Always answer confidently.",
    )
]
```

Local preference records require `prompt`, `chosen`, and `rejected`. SFT records require `prompt` and `completion`; KTO records require `prompt`, `completion`, and `label`; GRPO/PPO prompt datasets require `prompt`.

Preference-distillation records use either `prompt` plus two or more `responses`,
or the pairwise `chosen` and `rejected` form. See
[Preference distillation](./PREFERENCE_DISTILLATION.md) for the additional
teacher-signal columns required by each objective.

## Advanced lifecycle composition

`AlignmentPipeline` remains available when a study needs explicit acquisition, intervention, and assurance stages:

```python
pipeline = AlignmentPipeline(
    specification=specification,
    supervision=(human_feedback,),
    training=(training_method,),
    inference=inference_method,
    assurance=(behavioral, robustness, preservation),
)
result = pipeline.run(base_model, contexts=contexts, evaluation_cases=cases)
```

Stages are optional because a study may evaluate an existing model, apply only inference-time control, or compare preservation after external adaptation.

## Implementation status

The method catalog distinguishes:

- `catalog`: research is indexed but has no bundled executable code;
- `adapter`: the package exposes an interface that needs additional user components;
- `reference`: a dependency-free reference implementation is included;
- `native`: a first-class wrapper, validation, documentation, and tests are included;
- `external`: execution is delegated to an external project.

This status describes software coverage, not empirical quality or evidence of alignment.

## Adding a method

1. Add the implementation under the matching family in `mechanisms/training/`
   or under `mechanisms/inference/`.
2. Keep third-party calls in `integrations/`.
3. Return `AlignmentRun` or `InferenceResult`.
4. Add multi-label categories to `catalog/data/methods.json`.
5. Register the class in `registry.py`.
6. Add unit tests that do not require network or GPU.

Do not add a new abstraction layer until at least two implementations share meaningful logic.

## Development checks

```bash
python -m compileall -q src tests examples
python -m unittest discover -s tests -v
python -m human_alignment.cli catalog validate
```
