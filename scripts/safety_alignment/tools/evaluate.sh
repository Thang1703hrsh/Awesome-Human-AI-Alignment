#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."

if [[ $# -lt 1 ]]; then
  echo "usage: $0 MODEL [pku|beaver|do_not_answer|wildguard|xstest] [hai-align safety generate overrides...]" >&2
  exit 2
fi
MODEL="$1"
PROFILE="${2:-pku}"
shift
[[ $# -gt 0 ]] && shift

read -r -a CORE_SAFETY_CMD <<< "${CORE_SAFETY:-hai-align safety}"
OUTPUT_ROOT="${OUTPUT_ROOT:-evaluation/$(basename "$MODEL")/$PROFILE}"
LIMIT="${LIMIT:-}"
BASELINE_MODEL="${BASELINE_MODEL:-}"
REWARD_MODEL="${REWARD_MODEL:-PKU-Alignment/beaver-7b-v1.0-reward}"
COST_MODEL="${COST_MODEL:-PKU-Alignment/beaver-7b-v1.0-cost}"

case "$PROFILE" in
  pku) DATASET=PKU-Alignment/PKU-SafeRLHF-30K; SPLIT=test; CONFIG=; COLUMN=prompt ;;
  beaver) DATASET=PKU-Alignment/BeaverTails-Evaluation; SPLIT=test; CONFIG=; COLUMN=prompt ;;
  do_not_answer) DATASET=LibrAI/do-not-answer; SPLIT=train; CONFIG=; COLUMN=question ;;
  wildguard) DATASET=allenai/wildguardmix; SPLIT=test; CONFIG=wildguardtest; COLUMN=prompt ;;
  xstest) DATASET=walledai/XSTest; SPLIT=test; CONFIG=; COLUMN=prompt ;;
  *) echo "unknown evaluation profile: $PROFILE" >&2; exit 2 ;;
esac

mkdir -p "$OUTPUT_ROOT"
GEN_ARGS=(--dataset "$DATASET" --split "$SPLIT" --prompt-column "$COLUMN" --output "$OUTPUT_ROOT/candidate.jsonl")
[[ -n "$CONFIG" ]] && GEN_ARGS+=(--dataset-config "$CONFIG")
[[ -n "$LIMIT" ]] && GEN_ARGS+=(--limit "$LIMIT")
"${CORE_SAFETY_CMD[@]}" generate "$MODEL" "${GEN_ARGS[@]}" "$@"

SCORE_INPUT="$OUTPUT_ROOT/candidate.jsonl"
if [[ -n "$BASELINE_MODEL" ]]; then
  REF_ARGS=("${GEN_ARGS[@]}")
  for ((i=0; i<${#REF_ARGS[@]}; i++)); do
    [[ "${REF_ARGS[$i]}" == "$OUTPUT_ROOT/candidate.jsonl" ]] && REF_ARGS[$i]="$OUTPUT_ROOT/reference.jsonl"
  done
  "${CORE_SAFETY_CMD[@]}" generate "$BASELINE_MODEL" "${REF_ARGS[@]}" "$@"
  "${CORE_SAFETY_CMD[@]}" pair "$OUTPUT_ROOT/candidate.jsonl" "$OUTPUT_ROOT/reference.jsonl" \
    --output "$OUTPUT_ROOT/paired.jsonl"
  SCORE_INPUT="$OUTPUT_ROOT/paired.jsonl"
fi
"${CORE_SAFETY_CMD[@]}" score "$SCORE_INPUT" --output-dir "$OUTPUT_ROOT/scored" \
  --reward-model "$REWARD_MODEL" --cost-model "$COST_MODEL"

