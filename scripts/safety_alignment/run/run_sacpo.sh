#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
run_config sacpo_helpful_dpo "$@"
for beta in ${BETAS:-0.1 0.05 0.025 0.01}; do
  run_config sacpo_safety_dpo "method_args.beta=$beta" \
    "train.output_dir=output/sacpo/helpful_dpo_safety_dpo_$beta" "$@"
  run_config sacpo_safety_kto "method_args.beta=$beta" \
    "train.output_dir=output/sacpo/helpful_dpo_safety_kto_$beta" "$@"
done

