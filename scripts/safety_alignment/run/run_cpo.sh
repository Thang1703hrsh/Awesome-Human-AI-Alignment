#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
run_config cpsft "$@"
run_config cdpo model.policy=output/cpo/cpsft "$@"

