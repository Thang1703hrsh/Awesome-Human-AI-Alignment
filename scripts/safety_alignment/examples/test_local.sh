#!/usr/bin/env bash
# Fast, self-contained regression suite for the integrated safety package.
# Usage: bash scripts/safety_alignment/examples/test_local.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"

"$PYTHON_BIN" -m unittest -v tests.test_safety_alignment

