# Preference distillation

The temporary `Preference-Distillation` research project has been integrated as
a native training mechanism. Runtime code does not import that directory, so it
can be removed after any untracked experiment outputs have been preserved.

## Basic usage

Install the optional backend and choose either the generic class or an
objective-specific convenience class:

```bash
python -m pip install -e ".[distillation]"
```

```python
from human_alignment import VPD, PreferenceDistillationConfig

run = VPD(
    model="student-model",
    dataset=[
        {
            "prompt": "Explain the role of evaluation in alignment.",
            "responses": ["It tests behavioral claims.", "It replaces training."],
            "teacher_scores": [1.2, -0.4],
        }
    ],
    config=PreferenceDistillationConfig(
        objective="vpd",
        output_dir="outputs/student-vpd",
        beta=10.0,
    ),
).train()
```

The equivalent generic call is
`PreferenceDistillation(..., config={"objective": "vpd"})`. The available
convenience classes are `VPD`, `PPD`, `DCKD`, `TVKD`, `ADPA`, and `CTPD`.
Their no-argument objective weights preserve the corresponding migrated recipe
defaults. Runtime parameters such as the learning rate, batch size, and output
directory remain portable package defaults; an explicit configuration always
takes precedence.

For VPD and PPD, a teacher can score candidates online when the dataset does
not contain scores:

```python
run = VPD(
    model="student-model",
    teacher_model="teacher-model",
    dataset=[{"prompt": "Question: ", "responses": ["answer A", "answer B"]}],
).train()
```

Online scoring uses the student's tokenizer for both models. Precompute scores
when teacher and student tokenizers differ or when repeated teacher inference
would be too expensive.

Candidate collection is framework-neutral:

```python
from human_alignment import collect_preference_distillation_data

data = collect_preference_distillation_data(
    prompts,
    lambda prompt, count: generator(prompt, count),
    num_responses=4,
    scorer=lambda prompt, responses: teacher_scores(prompt, responses),
)
```

`integrations.vllm.VLLMGenerator.generate_many` can supply the generator when
high-throughput vLLM sampling is available. Low-level
`compress_probabilities` and `compressed_log_ratio_advantages` helpers in
`mechanisms/training/distillation_losses.py` produce the sparse DCKD/TVKD and
ADPA representations from model logits.

## End-to-end data preparation

The migrated workflow replaces `data_gen/`, preprocessing utilities, and the
shell orchestration needed by the runtime methods:

```python
from human_alignment import prepare_preference_distillation_dataset

prepared = prepare_preference_distillation_dataset(
    "HuggingFaceH4/ultrafeedback_binarized",
    objective="dckd",
    teacher_model="models/teacher_dpo",
    tokenizer="models/student_sft",
    output_dir="data/ultrafeedback-dckd",
)
```

The same workflow performs the method-specific operation selected by
`objective`:

- DCKD/TVKD: compress teacher token distributions for chosen and rejected data;
- ADPA: optionally generate on-policy rejected responses with `student_model`,
  then compute sparse teacher-reference log-ratio advantages;
- CTPD: align fast-tokenizer offset spans and compute parent-token weights;
- PPD/VPD: optionally sample multiple student candidates and score them using
  teacher sequence log-probability or order-balanced pairwise judgments.

The corresponding CLI is:

```bash
hai-align prepare distillation vpd \
  --dataset HuggingFaceH4/ultrafeedback_binarized \
  --student models/student_sft \
  --teacher models/teacher_dpo \
  --teacher-score-mode pairwise \
  --num-responses 4 \
  --output data/preference-distillation
```

Reproducible training configurations are provided under
`recipes/preference_distillation/` for teacher SFT/DPO, student SFT, and all six
distillation objectives.

## Dataset contracts

| Objective | Required teacher signal | Main loss |
|---|---|---|
| VPD | `teacher_scores`, `scores_mcq`, or a teacher model | Likelihood of the teacher-induced Plackett-Luce ranking |
| PPD | `teacher_scores`, `scores_mcq`, or a teacher model | Jensen-Shannon divergence between exact Plackett-Luce ranking distributions |
| DCKD | `teacher_chosen_probs` and `teacher_rejected_probs` | Token-mean `KL(teacher || student)` over top-k bins plus residual mass |
| TVKD | The DCKD token-probability columns | Value-based pairwise preference objective, optionally combined with token KL |
| ADPA | `rejected_margin_logp_every` | Expected teacher-reference token advantage on rejected/on-policy responses |
| CTPD | Parent lists and weights for chosen and rejected responses | Weighted parent-token preference loss against a reference model |

`chosen_compressed_probs` and `rejected_compressed_probs`, the names emitted by
the temporary preprocessing scripts, are accepted as aliases for the DCKD and
TVKD teacher columns. CTPD expects `chosen_ctpd_parent_list`,
`rejected_ctpd_parent_list`, `chosen_ctpd_weight`, and
`rejected_ctpd_weight`, and the `CTPD` constructor requires `reference_model`.

Each compressed token distribution has this form:

```python
{
    "indices": [42, 17],
    "values": [0.70, 0.20],
    "remaining_probs_sum": 0.10,
}
```

ADPA uses the same sparse shape, but `values` are teacher-reference log-probability
margins rather than probabilities.

PPD enumerates complete rankings to preserve the original probabilistic
objective. Because this grows factorially, `exact_ranking_limit` defaults to
eight responses; four responses is the practical default used by the migrated
recipes.

## Configuration mapping

The former YAML fields map to typed configuration as follows:

| Research scripts | Package configuration |
|---|---|
| `loss_type: rd` | `objective="vpd"` |
| `loss_type: pd` | `objective="ppd"` |
| `chosen_distil_weight` | `chosen_kd_weight` |
| `rejected_distil_weight` | `rejected_kd_weight` |
| `distillation_weight`, `adpa_weight`, or `ctpd_weight` | `objective_weight` |
| `sft_on_chosen` | `sft_weight` |
| `ctpd_beta` | `ctpd_beta` |
| `qadapter_softvalue_type` | `value_function` (`entropy` or `soft`) |
| `qadapter_gamma` | `value_discount` |
| `qadapter_alpha_tilde` | `value_temperature` |
| `qadapter_alpha_1` | `student_value_weight` |
| `qadapter_alpha` | `teacher_value_weight` |

Trainer-specific options such as checkpointing, precision, logging, and save
strategy belong in `extra_args`. Model loading options belong in
`backend_options["model_kwargs"]`; teacher and reference loading options have
parallel `teacher_model_kwargs` and `reference_model_kwargs` entries.

## Architectural mapping

- Typed examples and schema validation live in `supervision/datasets.py`.
- Typed hyperparameters live in `config.py`.
- Public mechanisms and taxonomy traces live in
  `mechanisms/training/distillation.py`.
- Pure differentiable objectives live in
  `mechanisms/training/distillation_losses.py`.
- Tokenization, collation, model execution, and saving live in
  `integrations/preference_distillation.py`.
- Reusable data builders live in `supervision/preference_distillation.py`, and
  model-backed preparation lives in `integrations/distillation_data.py`.
- End-to-end preparation is exposed by `workflows/preference_distillation.py`.
- Method discovery metadata lives in `catalog/data/methods.json`.

Data preparation, recipes, checkpoint loading, and execution are now available
from the main repository. Exploratory analysis, notebooks, hard-coded cluster
launchers, release helpers, and duplicated trainer forks remain excluded from
the runtime package because they do not define reusable method behavior.

ADPA+ is a two-stage recipe rather than a separate objective: train DCKD first,
then pass its resulting model to `ADPA`. This composition remains explicit and
does not introduce a redundant taxonomy category or trainer implementation.

The TVKD implementation is derived from the accompanying research project. If
you use TVKD in research, cite Minchan Kwon, Junwon Ko, Kangil Kim, and Junmo
Kim, "Preference Distillation via Value based Reinforcement Learning," arXiv
2509.16965 (2025).
