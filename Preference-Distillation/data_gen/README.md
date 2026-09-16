# On-Policy Preference Data Generation

This directory contains the PPD/VPD data-generation pipeline. It samples multiple
student responses per prompt, scores those responses with a teacher model, and
exports a Hugging Face `DatasetDict` for `scripts/run_preference_distillation.py`.

## Quick Start

From the repository root:

```bash
./run/prepare_ppd_vpd_data.sh
```

This runs:

1. `data_gen/scripts/sampling.sh`
2. `data_gen/scripts/build_ppd_vpd_dataset.sh`

The default output dataset is:

```text
data/generated/ultrafeedback/llama3.2-1b-it/pkd-dataset-teacher-llama3.1-8b-n4
```

That path is already configured in
`recipes/llama3.2-1b-deita-dpomix/PPD.yaml` and
`recipes/llama3.2-1b-deita-dpomix/VPD.yaml`.

## Configuration

Override models and datasets with environment variables:

```bash
STUDENT_MODEL=outputs/student_sft \
TEACHER_MODEL=outputs/teacher_dpo \
DATASET=HuggingFaceH4/ultrafeedback_binarized \
DATASET_SPLIT=train_prefs \
LOCAL_DATASET=0 \
./run/prepare_ppd_vpd_data.sh
```

Useful variables:

- `STUDENT_MODEL`: model used to sample candidate responses.
- `TEACHER_MODEL`: model used to score candidates.
- `DATASET`: local dataset path or Hugging Face dataset name, default `HuggingFaceH4/ultrafeedback_binarized`.
- `DATASET_SPLIT`: dataset split to sample from, default `train_prefs`.
- `LOCAL_DATASET`: set to `1` for `load_from_disk`, `0` for `load_dataset`; default `0`.
- `SEEDS`: space-separated sampling seeds, default `0 1 2 3 4`.
- `N`: number of responses kept per prompt, default `4`.
- `STUDENT_DIR`: generation directory shared by all data-generation stages.

## Requirements

Generation and scoring use `vllm`. The aggregation step also needs
`python-Levenshtein`.
