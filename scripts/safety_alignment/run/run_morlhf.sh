#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
for weight in ${WEIGHTS:-0.1 0.3 0.5 0.7 0.9}; do
  complement=$(python -c "print(round(1-float('$weight'), 1))")
  run_config morlhf "method_args.preference=[$weight,$complement]" "train.output_dir=output/morlhf_w$weight" "$@"
done

