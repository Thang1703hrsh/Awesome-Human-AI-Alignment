#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
for weight in ${WEIGHTS:-0.25 0.5 0.75}; do
  complement=$(python -c "print(round(1-float('$weight'), 2))")
  run_config p_sacpo "method_args.weights=[$complement,$weight]" \
    "train.output_dir=output/sacpo/p_sacpo_$weight" "$@"
done

