#!/bin/bash

set -e
set -o pipefail
trap 'echo -e "\n[ERROR] Command failed: $BASH_COMMAND\n"' ERR

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"

CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29501}"
MAX_TOKENS_PER_BATCH="${MAX_TOKENS_PER_BATCH:-2048}"
PAD_TOKEN_ID="${PAD_TOKEN_ID:-128001}"

BASE_DATASET="${BASE_DATASET:-HuggingFaceH4/ultrafeedback_binarized}"
DPO_TEACHER_MODEL="${DPO_TEACHER_MODEL:-./model/dpo_teacher}"

WORK_DIR="${WORK_DIR:-data/llama3.2-1b-deita-dpomix}"
DCKD_DATASET="${DCKD_DATASET:-data/ultrafeedback_binarized-dckd}"
ADPA_BASE_DATASET="${ADPA_BASE_DATASET:-data/ultrafeedback_binarized-adpa-base}"
ADPA_DATASET="${ADPA_DATASET:-data/ultrafeedback_binarized-adpa}"
ADPA_GENERATION_DIR="${ADPA_GENERATION_DIR:-$WORK_DIR/student_init_self_generation}"

# Defaults create an ADPA dataset from the SFT student so run_adpa.sh can run
# right after this script. For ADPA+ that follows the original DCKD-initialized
# recipe more closely, run this script again after DCKD with:
# ADPA_GENERATOR_MODEL=models/student_dckd ADPA_REF_MODEL=models/student_dckd ./run/precompute.sh
ADPA_GENERATOR_MODEL="${ADPA_GENERATOR_MODEL:-models/student_sft}"
ADPA_REF_MODEL="${ADPA_REF_MODEL:-models/student_sft}"

USER_BEGIN="${USER_BEGIN:-<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n}"
USER_END="${USER_END:-<|eot_id|>}"
ASSISTANT_BEGIN="${ASSISTANT_BEGIN:-<|start_header_id|>assistant<|end_header_id|>\n\n}"
ASSISTANT_END="${ASSISTANT_END:-<|eot_id|>}"

log_and_run () {
  local NAME="$1"
  shift
  local LOG_FILE="$LOG_DIR/${NAME}.log"

  echo -e "\n[START] $NAME"
  echo "Logging to: $LOG_FILE"
  echo "------------------------------------------------"

  {
    echo ">>> [Start: $(date)]"
    echo ">>> [Command] $@"
    echo "------------------------------------------------"
    eval "$@"
    echo ">>> [Success: $(date)]"
  } 2>&1 | tee "$LOG_FILE"

  echo -e "[DONE] $NAME\n"
}

precompute_logits () {
  local NAME="$1"
  local DATA="$2"
  local SPLIT="$3"
  local MODEL="$4"
  local CONVERSATION_KEY="$5"
  local SAVE_TO="$6"

  log_and_run "$NAME" \
"CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m accelerate.commands.launch \
  --num_processes=$NUM_PROCESSES \
  --main_process_port $MAIN_PROCESS_PORT \
  utils/precompute_logits.py \
  --data $DATA \
  --split $SPLIT \
  --model $MODEL \
  --conversation-key $CONVERSATION_KEY \
  --user-begin '$USER_BEGIN' \
  --user-end '$USER_END' \
  --assistant-begin '$ASSISTANT_BEGIN' \
  --assistant-end '$ASSISTANT_END' \
  --save-to $SAVE_TO \
  --pad-token-id $PAD_TOKEN_ID \
  --max-tokens-per-batch $MAX_TOKENS_PER_BATCH"

  log_and_run "rm_${NAME}_temp" \
"rm -f $SAVE_TO/results_rank_*.jsonl"
}

echo "=== DCKD precompute ==="

precompute_logits \
  "precompute_train_chosen" \
  "$BASE_DATASET" \
  "train_prefs" \
  "$DPO_TEACHER_MODEL" \
  "chosen" \
  "$WORK_DIR/dpomix7k-dpoteacher-chosen-logp-train"

precompute_logits \
  "precompute_train_rejected" \
  "$BASE_DATASET" \
  "train_prefs" \
  "$DPO_TEACHER_MODEL" \
  "rejected" \
  "$WORK_DIR/dpomix7k-dpoteacher-rejected-logp-train"

precompute_logits \
  "precompute_test_chosen" \
  "$BASE_DATASET" \
  "test_prefs" \
  "$DPO_TEACHER_MODEL" \
  "chosen" \
  "$WORK_DIR/dpomix7k-dpoteacher-chosen-logp-test"

precompute_logits \
  "precompute_test_rejected" \
  "$BASE_DATASET" \
  "test_prefs" \
  "$DPO_TEACHER_MODEL" \
  "rejected" \
  "$WORK_DIR/dpomix7k-dpoteacher-rejected-logp-test"

log_and_run "merge_dckd" \
"python utils/merge_logits_dckd_dataset.py \
    --input-dataset-dict          $BASE_DATASET \
    --teacher-chosen-logp-train   $WORK_DIR/dpomix7k-dpoteacher-chosen-logp-train \
    --teacher-rejected-logp-train $WORK_DIR/dpomix7k-dpoteacher-rejected-logp-train \
    --teacher-chosen-logp-test    $WORK_DIR/dpomix7k-dpoteacher-chosen-logp-test \
    --teacher-rejected-logp-test  $WORK_DIR/dpomix7k-dpoteacher-rejected-logp-test \
    --save-to                     $DCKD_DATASET"

echo "=== ADPA precompute ==="

log_and_run "adpa_generate_train_rejected" \
"CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python utils/vllm_generate.py \
    --model $ADPA_GENERATOR_MODEL \
    --data $DCKD_DATASET \
    --dataset_split train \
    --prompt_key chosen \
    --out_dir $ADPA_GENERATION_DIR \
    --apply_template true"

log_and_run "adpa_generate_test_rejected" \
"CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python utils/vllm_generate.py \
    --model $ADPA_GENERATOR_MODEL \
    --data $DCKD_DATASET \
    --dataset_split test \
    --prompt_key chosen \
    --out_dir $ADPA_GENERATION_DIR \
    --apply_template true"

log_and_run "adpa_form_preference_dataset" \
"python utils/form_preference_dataset.py \
    --original-dataset $DCKD_DATASET \
    --rejected-train $ADPA_GENERATION_DIR/ultrafeedback_binarized-dckd-train.jsonl \
    --rejected-test $ADPA_GENERATION_DIR/ultrafeedback_binarized-dckd-test.jsonl \
    --output-dir $ADPA_BASE_DATASET"

precompute_logits \
  "adpa_dpoteacher_train_student" \
  "$ADPA_BASE_DATASET" \
  "train" \
  "$DPO_TEACHER_MODEL" \
  "rejected" \
  "$WORK_DIR/dpomix7k-dpoteacher-train-student"

precompute_logits \
  "adpa_dpoteacher_test_student" \
  "$ADPA_BASE_DATASET" \
  "test" \
  "$DPO_TEACHER_MODEL" \
  "rejected" \
  "$WORK_DIR/dpomix7k-dpoteacher-test-student"

precompute_logits \
  "adpa_refteacher_train_student" \
  "$ADPA_BASE_DATASET" \
  "train" \
  "$ADPA_REF_MODEL" \
  "rejected" \
  "$WORK_DIR/dpomix7k-refteacher-train-student"

precompute_logits \
  "adpa_refteacher_test_student" \
  "$ADPA_BASE_DATASET" \
  "test" \
  "$ADPA_REF_MODEL" \
  "rejected" \
  "$WORK_DIR/dpomix7k-refteacher-test-student"

log_and_run "merge_adpa" \
"python utils/merge_logits_adpa_dataset.py \
    --input-dataset-dict $ADPA_BASE_DATASET \
    --dpo-teacher-logp-train $WORK_DIR/dpomix7k-dpoteacher-train-student \
    --ref-teacher-logp-train $WORK_DIR/dpomix7k-refteacher-train-student \
    --dpo-teacher-logp-test $WORK_DIR/dpomix7k-dpoteacher-test-student \
    --ref-teacher-logp-test $WORK_DIR/dpomix7k-refteacher-test-student \
    --save-to $ADPA_DATASET \
    --logits-key rejected_compressed_probs \
    --label-key rejected_labels \
    --output-key rejected_margin_logp_every"
