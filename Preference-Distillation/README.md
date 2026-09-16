# Preference Distillation

This codebase contains bash entry points for:

- Teacher SFT and Teacher DPO
- Student SFT initialization
- DCKD
- TVKD
- ADPA and ADPA+
- CTPD
- PPD and VPD

---

## Environment Setup

Prepare an environment with PyTorch installed for your GPU setup, then install the project dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies:

- `trl`
- `peft`
- `transformers`
- `vllm`

Recommended versions:

- `torch==2.5.1`
- `trl==0.12.0`
- `peft==0.13.0`

---

## Datasets

Default datasets and generated dataset directories:

| Stage / method | Dataset |
| --- | --- |
| Teacher SFT | `HuggingFaceH4/ultrachat_200k` |
| Student SFT | `HuggingFaceH4/ultrachat_200k` |
| Teacher DPO | `HuggingFaceH4/ultrafeedback_binarized` |
| DCKD | `data/ultrafeedback_binarized-dckd`, produced by `run/precompute.sh` |
| TVKD | `data/ultrafeedback_binarized-dckd`, produced by `run/precompute.sh` |
| ADPA | `data/ultrafeedback_binarized-adpa`, produced by `run/precompute.sh` |
| ADPA+ | `data/ultrafeedback_binarized-adpa`, preferably regenerated after DCKD |
| CTPD | `data/ultrafeedback_binarized-ctpd`, produced by `run/prepare_ctpd.sh` |
| PPD | Generated from `HuggingFaceH4/ultrafeedback_binarized` split `train_prefs` into `data/generated/ultrafeedback/llama3.2-1b-it/pkd-dataset-teacher-llama3.1-8b-n4` by `run/prepare_ppd_vpd_data.sh` |
| VPD | Generated from `HuggingFaceH4/ultrafeedback_binarized` split `train_prefs` into `data/generated/ultrafeedback/llama3.2-1b-it/pkd-dataset-teacher-llama3.1-8b-n4` by `run/prepare_ppd_vpd_data.sh` |

Default model output directories:

| Stage / method | Output |
| --- | --- |
| Teacher SFT | `models/teacher_sft` |
| Teacher DPO | `models/teacher_dpo` |
| Student SFT | `models/student_sft` |
| DCKD | `models/student_dckd` |
| TVKD | `models/tvkd` |
| ADPA | `models/student_adpa` |
| ADPA+ | `models/student_adpa_plus` |
| CTPD | `models/student_ctpd` |
| PPD | `outputs/llama3.2-1b-it-ppd` |
| VPD | `outputs/llama3.2-1b-it-vpd` |

---

## Training Procedure

### 1. Train Teacher SFT

```bash
./run/run_sft_teacher.sh
```

Recipe: `recipes/llama3.2-1b-deita-dpomix/teacher_sft.yaml`  
Dataset: `HuggingFaceH4/ultrachat_200k`  
Output: `models/teacher_sft`

### 2. Train Teacher DPO

```bash
./run/run_dpo_teacher.sh
```

Recipe: `recipes/llama3.2-1b-deita-dpomix/teacher_dpo.yaml`  
Dataset: `HuggingFaceH4/ultrafeedback_binarized`  
Input model: `models/teacher_sft`  
Output: `models/teacher_dpo`

### 3. Train Student SFT

```bash
./run/run_sft_student.sh
```

Recipe: `recipes/llama3.2-1b-deita-dpomix/student_sft_init.yaml`  
Dataset: `HuggingFaceH4/ultrachat_200k`  
Output: `models/student_sft`

### 4. Precompute Data for DCKD / TVKD / ADPA

```bash
./run/precompute.sh
```

Default input dataset: `HuggingFaceH4/ultrafeedback_binarized`

Generated outputs:

- `data/ultrafeedback_binarized-dckd` for DCKD and TVKD.
- `data/ultrafeedback_binarized-adpa-base` as the intermediate ADPA preference dataset.
- `data/ultrafeedback_binarized-adpa` for ADPA and ADPA+.

Important defaults:

- `DPO_TEACHER_MODEL=./model/dpo_teacher`
- `ADPA_GENERATOR_MODEL=models/student_sft`
- `ADPA_REF_MODEL=models/student_sft`

For ADPA+ after DCKD, regenerate the ADPA dataset with the DCKD student:

```bash
ADPA_GENERATOR_MODEL=models/student_dckd \
ADPA_REF_MODEL=models/student_dckd \
./run/precompute.sh
```

### 5. Run DCKD

```bash
./run/run_dckd.sh
```

Recipe: `recipes/llama3.2-1b-deita-dpomix/DCKD.yaml`  
Dataset: `data/ultrafeedback_binarized-dckd`  
Input model: `models/student_sft`  
Output: `models/student_dckd`

### 6. Run TVKD

```bash
./run/run_tvkd.sh
```

Recipe: `recipes/llama3.2-1b-deita-dpomix/TVKD.yaml`  
Dataset: `data/ultrafeedback_binarized-dckd`  
Input model: `models/student_dckd`  
Output: `models/tvkd`

### 7. Run ADPA

```bash
./run/run_adpa.sh
```

Recipe: `recipes/llama3.2-1b-deita-dpomix/ADPA.yaml`  
Dataset: `data/ultrafeedback_binarized-adpa`  
Input model: `models/student_sft`  
Output: `models/student_adpa`

### 8. Run ADPA+

```bash
./run/run_adpa_plus.sh
```

Recipe: `recipes/llama3.2-1b-deita-dpomix/ADPA_PLUS.yaml`  
Dataset: `data/ultrafeedback_binarized-adpa`  
Input model: `models/student_dckd`  
Output: `models/student_adpa_plus`

### 9. Prepare and Run CTPD

```bash
./run/prepare_ctpd.sh
./run/run_ctpd.sh
```

Recipe: `recipes/llama3.2-1b-deita-dpomix/CTPD.yaml`  
Dataset: `data/ultrafeedback_binarized-ctpd`  
Input model: `models/student_sft`  
Output: `models/student_ctpd`

### 10. Prepare and Run PPD / VPD

PPD and VPD share the same generated multi-response dataset. By default, it is generated from `HuggingFaceH4/ultrafeedback_binarized` split `train_prefs`.

```bash
./run/prepare_ppd_vpd_data.sh
```

Generated dataset:

```text
data/generated/ultrafeedback/llama3.2-1b-it/pkd-dataset-teacher-llama3.1-8b-n4
```

Run PPD:

```bash
./run/run_ppd.sh
```

Recipe: `recipes/llama3.2-1b-deita-dpomix/PPD.yaml`  
Dataset: `data/generated/ultrafeedback/llama3.2-1b-it/pkd-dataset-teacher-llama3.1-8b-n4`  
Loss: `loss_type: pd`  
Output: `outputs/llama3.2-1b-it-ppd`

Run VPD:

```bash
./run/run_vpd.sh
```

Recipe: `recipes/llama3.2-1b-deita-dpomix/VPD.yaml`  
Dataset: `data/generated/ultrafeedback/llama3.2-1b-it/pkd-dataset-teacher-llama3.1-8b-n4`  
Loss: `loss_type: rd`  
Output: `outputs/llama3.2-1b-it-vpd`

Common PPD/VPD data-generation overrides:

```bash
STUDENT_MODEL=outputs/student_sft \
TEACHER_MODEL=outputs/teacher_dpo \
DATASET=HuggingFaceH4/ultrafeedback_binarized \
DATASET_SPLIT=train_prefs \
LOCAL_DATASET=0 \
./run/prepare_ppd_vpd_data.sh
```

Common PPD/VPD training overrides:

```bash
PPD_CONFIG=recipes/llama3.2-1b-deita-dpomix/PPD.yaml ./run/run_ppd.sh
VPD_CONFIG=recipes/llama3.2-1b-deita-dpomix/VPD.yaml ./run/run_vpd.sh
```

---

## Method Cheat Sheet

| Method | Prepare command | Train command | Main recipe |
| --- | --- | --- | --- |
| Teacher SFT | none | `./run/run_sft_teacher.sh` | `teacher_sft.yaml` |
| Teacher DPO | none | `./run/run_dpo_teacher.sh` | `teacher_dpo.yaml` |
| Student SFT | none | `./run/run_sft_student.sh` | `student_sft_init.yaml` |
| DCKD | `./run/precompute.sh` | `./run/run_dckd.sh` | `DCKD.yaml` |
| TVKD | `./run/precompute.sh` | `./run/run_tvkd.sh` | `TVKD.yaml` |
| ADPA | `./run/precompute.sh` | `./run/run_adpa.sh` | `ADPA.yaml` |
| ADPA+ | `./run/precompute.sh` after DCKD | `./run/run_adpa_plus.sh` | `ADPA_PLUS.yaml` |
| CTPD | `./run/prepare_ctpd.sh` | `./run/run_ctpd.sh` | `CTPD.yaml` |
| PPD | `./run/prepare_ppd_vpd_data.sh` | `./run/run_ppd.sh` | `PPD.yaml` |
| VPD | `./run/prepare_ppd_vpd_data.sh` | `./run/run_vpd.sh` | `VPD.yaml` |

---
