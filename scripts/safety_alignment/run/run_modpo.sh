#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
run_config modpo_margin "$@"
for weight in ${WEIGHTS:-0.1 0.5 0.9}; do
  complement=$(python -c "print(round(1-float('$weight'), 1))")
  run_config modpo "method_args.modpo_w=[$weight,$complement]" "train.output_dir=output/modpo/w$weight" "$@"
done

