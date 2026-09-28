#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."

PROFILE="${1:-all}"
shift || true
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-downloads}"
read -r -a CORE_SAFETY_CMD <<< "${CORE_SAFETY:-hai-align safety}"
exec "${CORE_SAFETY_CMD[@]}" prepare "$PROFILE" --root "$DOWNLOAD_ROOT" "$@"

