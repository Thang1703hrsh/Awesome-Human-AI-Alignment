#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"

"$PYTHON_BIN" -m compileall -q src tests
if "$PYTHON_BIN" -c "import importlib.util, sys; sys.exit(importlib.util.find_spec('ruff') is None)"; then
  "$PYTHON_BIN" -m ruff check src/human_alignment/safety tests/test_safety_alignment.py
else
  echo "ruff is not installed; skipping lint (install the dev extra to enable it)"
fi
"$PYTHON_BIN" -m unittest discover -s tests -v "$@"

