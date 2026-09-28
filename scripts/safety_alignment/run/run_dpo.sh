#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
for method in dpo_helpful dpo_harmless dpo_safebetter; do run_config "$method" "$@"; done

