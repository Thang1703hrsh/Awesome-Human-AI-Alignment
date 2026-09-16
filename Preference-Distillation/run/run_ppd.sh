#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export ACCELERATE_LOG_LEVEL="${ACCELERATE_LOG_LEVEL:-info}"
export DS_SKIP_CUDA_CHECK="${DS_SKIP_CUDA_CHECK:-1}"

python -m accelerate.commands.launch \
  --config_file "${ACCELERATE_CONFIG:-recipes/accelerate_config/deepspeed_zero2_2gpus_p25601.yaml}" \
  scripts/run_preference_distillation.py \
  "${PPD_CONFIG:-recipes/llama3.2-1b-deita-dpomix/PPD.yaml}"
