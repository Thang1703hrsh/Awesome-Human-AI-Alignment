#!/usr/bin/env bash
# Full runs on one H100, in dependency order. Each block is independent; run the ones you need.
# Every command accepts dotted overrides, e.g.  hai-align safety run recipes/safety_alignment/methods/safedpo.yaml method_args.delta=5
set -euo pipefail
cd "$(dirname "$0")/../../.."
R() { hai-align safety run "recipes/safety_alignment/methods/$1.yaml" "${@:2}"; }

case "${1:-help}" in
  sft)      R sft ;;
  dpo)      R dpo_helpful; R dpo_harmless; R dpo_safebetter ;;
  safedpo)  R safedpo ;;
  bso)      R bso ;;
  rm_cm)    R reward_model; R cost_model ;;                       # optional: beaver-7b-v1.0-{reward,cost} exist
  saferlhf) R saferlhf ;;
  ppo)      R ppo ;;
  morlhf)   for w in 0.1 0.3 0.5 0.7 0.9; do
              R morlhf "method_args.preference=[$w, $(python -c "print(round(1-$w,1))")]" train.output_dir=output/morlhf_w$w
            done ;;
  sacpo)    R sacpo_helpful_dpo
            for b in 0.1 0.05 0.025 0.01; do
              R sacpo_safety_dpo method_args.beta=$b train.output_dir=output/sacpo/helpful_dpo_safety_dpo_$b
              R sacpo_safety_kto method_args.beta=$b train.output_dir=output/sacpo/helpful_dpo_safety_kto_$b
            done ;;
  p_sacpo)  for q in 0.25 0.5 0.75; do
              R p_sacpo "method_args.weights=[$(python -c "print(1-$q)"), $q]" train.output_dir=output/sacpo/p_sacpo_$q
            done ;;
  modpo)    R modpo_margin
            for w in 0.1 0.5 0.9; do
              R modpo "method_args.modpo_w=[$w, $(python -c "print(round(1-$w,1))")]" train.output_dir=output/modpo/w$w
            done ;;
  can)      R can_dual                                             # writes output/can/dual/dual.json (lam_star)
            LAM=$(python -c "import json;print(json.load(open('output/can/dual/dual.json'))['lam_star'])")
            R mocan method_args.lam=$LAM train.output_dir=output/can/mocan_lam$LAM
            R pecan method_args.lam=$LAM train.output_dir=output/can/pecan_lam$LAM ;;   # needs `dpo` block first
  cpo)      R cpsft; R cdpo model.policy=output/cpo/cpsft ;;
  bfpo)     R bfpo ;;
  midpo)    R midpo_safety_expert; R midpo_helpfulness_expert; R midpo_router ;;
  *) echo "usage: $0 {sft|dpo|safedpo|bso|rm_cm|saferlhf|ppo|morlhf|sacpo|p_sacpo|modpo|can|cpo|bfpo|midpo}" ;;
esac

