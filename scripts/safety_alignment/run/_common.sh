#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"
read -r -a CORE_SAFETY_CMD <<< "${CORE_SAFETY:-hai-align safety}"

run_config() {
  local config="$1"
  shift
  "${CORE_SAFETY_CMD[@]}" run "recipes/safety_alignment/methods/${config}.yaml" "$@"
}

