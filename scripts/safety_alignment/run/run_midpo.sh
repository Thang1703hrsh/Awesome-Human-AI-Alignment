#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
run_config midpo_safety_expert "$@"
run_config midpo_helpfulness_expert "$@"
run_config midpo_router "$@"

