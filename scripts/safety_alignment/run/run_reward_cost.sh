#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
run_config reward_model "$@"
run_config cost_model "$@"

