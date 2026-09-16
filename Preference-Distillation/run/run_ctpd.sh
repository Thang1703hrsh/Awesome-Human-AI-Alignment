#!/bin/bash

python -m accelerate.commands.launch \
  --config_file recipes/accelerate_config/deepspeed_zero3.yaml \
  scripts/run_distill_dpo.py \
  recipes/llama3.2-1b-deita-dpomix/CTPD.yaml
