#!/usr/bin/env bash
# usage: run_bso.sh [unified|qwen|llama] [dotted overrides...]
#   unified  shared Alpaca-7B backbone (default, bso.yaml)
#   qwen     authors' Qwen2.5-0.5B-Instruct run (bso_qwen2.5_0.5b.yaml)
#   llama    authors' Llama-3.2-3B-Instruct run (bso_llama3.2_3b.yaml)
source "$(dirname "$0")/_common.sh"
case "${1:-}" in
  unified) shift; run_config bso "$@" ;;
  qwen) shift; run_config bso_qwen2.5_0.5b "$@" ;;
  llama) shift; run_config bso_llama3.2_3b "$@" ;;
  *) run_config bso "$@" ;;
esac
