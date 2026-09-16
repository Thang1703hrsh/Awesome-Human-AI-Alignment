#!/bin/bash
set -euo pipefail

# Stage 1: sample several on-policy responses from the student.
# Override paths/models from the shell, for example:
# STUDENT_MODEL=outputs/student_sft TEACHER_MODEL=outputs/teacher_dpo ./run/prepare_ppd_vpd_data.sh
bash data_gen/scripts/sampling.sh

# Stage 2: aggregate responses, score them with the teacher, and build the
# Hugging Face DatasetDict consumed by the PPD/VPD recipes.
bash data_gen/scripts/build_ppd_vpd_dataset.sh
