#!/usr/bin/env bash
# Smoke test on the H100 with the real 7B models: every method, a few optimiser steps on 64 examples.
# Verifies downloads, memory (peak GiB is printed after each run) and the full plumbing before long runs.
# Usage (from the repository root): bash scripts/safety_alignment/examples/smoke_h100.sh [method ...]
set -euo pipefail
cd "$(dirname "$0")/../../.."
OUT=output/smoke
DS=recipes/safety_alignment/deepspeed/zero2_offload.json
if ! command -v nvcc >/dev/null 2>&1; then
  echo "[smoke] nvcc not found -> DeepSpeed CPU-Adam cannot be compiled; using torch AdamW offload instead"
  DS=recipes/safety_alignment/deepspeed/zero2_offload_torch_adam.json
fi
COMMON=(data.limit=64 train.max_steps=3 train.logging_steps=1 train.deepspeed=$DS)
run() { local name=$1; shift; echo "=============== $name"; hai-align safety run "recipes/safety_alignment/methods/$name.yaml" "$@"; }
SELECTED=("$@")
sel() { [ ${#SELECTED[@]} -eq 0 ] && return 0; local m; for m in "${SELECTED[@]}"; do [ "$m" = "$1" ] && return 0; done; return 1; }

for m in dpo_helpful dpo_harmless dpo_safebetter safedpo bso; do
  sel $m && run $m "${COMMON[@]}" train.output_dir=$OUT/$m
done
sel sft && run sft "${COMMON[@]}" train.output_dir=$OUT/sft
sel reward_model && run reward_model "${COMMON[@]}" train.output_dir=$OUT/reward_model
sel cost_model && run cost_model "${COMMON[@]}" train.output_dir=$OUT/cost_model
if sel sacpo; then
  run sacpo_helpful_dpo "${COMMON[@]}" train.output_dir=$OUT/sacpo1
  run sacpo_safety_dpo "${COMMON[@]}" model.policy=$OUT/sacpo1 train.output_dir=$OUT/sacpo2
  run sacpo_safety_kto "${COMMON[@]}" model.policy=$OUT/sacpo1 train.output_dir=$OUT/sacpo_kto
  run p_sacpo train.output_dir=$OUT/p_sacpo "method_args.models=[$OUT/sacpo1, $OUT/sacpo2]"
fi
if sel modpo; then
  run modpo_margin data.limit=64 train.max_steps=3 train.output_dir=$OUT/modpo_margin
  run modpo data.limit=64 train.max_steps=3 train.output_dir=$OUT/modpo method_args.margin_adapter=$OUT/modpo_margin
fi
if sel can; then
  run can_dual train.output_dir=$OUT/can_dual method_args.num_prompts=8 method_args.num_samples=8
  run mocan data.limit=64 train.max_steps=3 train.output_dir=$OUT/mocan
  run pecan data.limit=16 train.max_steps=3 train.output_dir=$OUT/pecan \
      method_args.helpful_model=$OUT/dpo_helpful method_args.safe_model=$OUT/dpo_harmless
fi
if sel cpo; then
  run cpsft "${COMMON[@]}" train.output_dir=$OUT/cpsft
  run cdpo "${COMMON[@]}" model.policy=$OUT/cpsft train.output_dir=$OUT/cdpo
fi
sel cpsft && run cpsft "${COMMON[@]}" train.output_dir=$OUT/cpsft
sel cdpo && run cdpo "${COMMON[@]}" train.output_dir=$OUT/cdpo
sel bfpo && run bfpo data.limit=64 train.max_steps=3 train.output_dir=$OUT/bfpo method_args.buffer_limit=64
if sel midpo; then
  run midpo_safety_expert data.limit=64 train.max_steps=3 train.output_dir=$OUT/midpo_safe
  run midpo_helpfulness_expert data.limit=64 train.max_steps=3 train.output_dir=$OUT/midpo_help
  run midpo_router data.limit=64 train.max_steps=3 train.output_dir=$OUT/midpo_router \
      method_args.safety_expert=$OUT/midpo_safe method_args.helpfulness_expert=$OUT/midpo_help
fi
for m in saferlhf ppo; do
  sel $m && run $m data.limit=256 method_args.max_steps=2 method_args.ptx_limit=256 train.output_dir=$OUT/$m
done
sel morlhf && run morlhf data.limit=256 method_args.max_steps=1 train.output_dir=$OUT/morlhf
echo "[smoke] all done"

