#!/usr/bin/env bash
# Train and evaluate BSO as in the authors' SafeBPO scripts (`script/train/bpo_safety_*_sba_flip.sh`).
#
# usage: bso_reference.sh {qwen|llama} [C=30] [LAMBDA=0.2|0.3]
#
# Training: recipes/safety_alignment/methods/bso_{qwen2.5_0.5b,llama3.2_3b}.yaml with C and λ overridden. For C > 30
# the log R clamp is raised to min(C + 10, floor(85 / (1 + λ))) so that safe-winner pairs, which start at log R = C,
# stay inside the clamp while exp((1 + λ) log R) remains finite in fp32.
# Evaluation (the source's `single_model_eval.py` and `xstest.py`): PKU-SafeRLHF-30K test prompts in the safe-rlhf
# template, sampled at T = 0.7, top-p 0.9, 512 new tokens; scored by beaver-7b-unified reward/cost at 2048 tokens,
# safe rate = share with cost <= 0. XSTest (Paul/XSTest) responses are judged by DeepSeek when DEEPSEEK_API_KEY or
# JUDGE_API_KEYS is set. Note that the source trains with the chat template but evaluates with the safe-rlhf template.
set -euo pipefail
cd "$(dirname "$0")/../../.."
read -r -a CORE_SAFETY_CMD <<< "${CORE_SAFETY:-hai-align safety}"

MODEL="${1:?usage: $0 {qwen|llama} [C] [LAMBDA]}"
case "$MODEL" in
  qwen) RECIPE=bso_qwen2.5_0.5b; DEFAULT_LAMBDA=0.2 ;;
  llama) RECIPE=bso_llama3.2_3b; DEFAULT_LAMBDA=0.3 ;;
  *) echo "unknown model $MODEL (qwen|llama)" >&2; exit 2 ;;
esac
C="${2:-30}"
LAMBDA="${3:-$DEFAULT_LAMBDA}"
read -r CLAMP_HI SUFFIX < <(python3 - "$C" "$LAMBDA" <<'PY'
import math, sys
c, lam = float(sys.argv[1]), float(sys.argv[2])
hi = 30 if c <= 30 else min(int(c + 10), int(85 / (lam + 1)))
print(hi, f"-hi{hi}" if hi > 30 else "")
PY
)
OUT="output/${RECIPE}-c${C}-lam${LAMBDA}${SUFFIX}"
EVAL="evaluation/$(basename "$OUT")"

if [[ ! -f "$OUT/config.json" ]]; then
  "${CORE_SAFETY_CMD[@]}" run "recipes/safety_alignment/methods/${RECIPE}.yaml" \
    "method_args.safety_penalty=${C}" "method_args.bso_lambda=${LAMBDA}" \
    "method_args.bso_log_r_max=${CLAMP_HI}" "train.output_dir=${OUT}"
fi

mkdir -p "$EVAL"
"${CORE_SAFETY_CMD[@]}" generate "$OUT" --dataset PKU-Alignment/PKU-SafeRLHF-30K --split test \
  --prompt-column prompt --temperature 0.7 --top-p 0.9 --max-new-tokens 512 --output "$EVAL/pku.jsonl"
"${CORE_SAFETY_CMD[@]}" score "$EVAL/pku.jsonl" --output-dir "$EVAL/pku_scored" --max-length 2048 \
  --reward-model PKU-Alignment/beaver-7b-unified-reward --cost-model PKU-Alignment/beaver-7b-unified-cost

"${CORE_SAFETY_CMD[@]}" generate "$OUT" --dataset Paul/XSTest --split train --prompt-column prompt \
  --keep-column label --temperature 0.7 --top-p 0.9 --max-new-tokens 512 --output "$EVAL/xstest.jsonl"
if [[ -n "${DEEPSEEK_API_KEY:-}${JUDGE_API_KEYS:-}" ]]; then
  "${CORE_SAFETY_CMD[@]}" xstest-judge "$EVAL/xstest.jsonl" --output-dir "$EVAL/xstest_judged"
  "${CORE_SAFETY_CMD[@]}" rubric-judge "$EVAL/pku.jsonl" --output-dir "$EVAL/pku_judged"
else
  echo "skip LLM judges: set DEEPSEEK_API_KEY or JUDGE_API_KEYS" >&2
fi
