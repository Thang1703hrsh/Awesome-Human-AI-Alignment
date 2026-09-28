#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
run_config can_dual "$@"
lambda=$(python -c "import json; print(json.load(open('output/can/dual/dual.json'))['lam_star'])")
run_config mocan "method_args.lam=$lambda" "train.output_dir=output/can/mocan_lam$lambda" "$@"
run_config pecan "method_args.lam=$lambda" "train.output_dir=output/can/pecan_lam$lambda" "$@"

