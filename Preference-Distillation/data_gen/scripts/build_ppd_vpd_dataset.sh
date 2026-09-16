#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

N="${N:-4}"
TEMPERATURE="${TEMPERATURE:-1}"
TEACHER_MODEL="${TEACHER_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
TEACHER_ID="${TEACHER_ID:-llama3.1-8b}"
STUDENT_DIR="${STUDENT_DIR:-data/generated/ultrafeedback/llama3.2-1b-it}"
TEST_SIZE="${TEST_SIZE:-0.02}"
SEED="${SEED:-42}"

python data_gen/gen/agg.py --generation_file_dir "$STUDENT_DIR" -n "$N"

python data_gen/gen/prob_sl.py \
    --model_name "$TEACHER_MODEL" \
    --temperature "$TEMPERATURE" \
    --input_file "$STUDENT_DIR/agg_outputs_n$N.json" \
    --output_file "$STUDENT_DIR/agg_outputs_n$N.prob.$TEACHER_ID.sl.json"

python data_gen/gen/prob.py \
    --model_name "$TEACHER_MODEL" \
    --temperature "$TEMPERATURE" \
    --num_options "$N" \
    --input_file "$STUDENT_DIR/agg_outputs_n$N.json" \
    --output_file "$STUDENT_DIR/agg_outputs_n$N.prob.$TEACHER_ID.json"

python data_gen/gen/generate_dataset.py \
    --prob_file "$STUDENT_DIR/agg_outputs_n$N.prob.$TEACHER_ID.json" \
    --prob_sl_file "$STUDENT_DIR/agg_outputs_n$N.prob.$TEACHER_ID.sl.json" \
    --output_dir "$STUDENT_DIR/pkd-dataset-teacher-$TEACHER_ID-n$N" \
    --test_size "$TEST_SIZE" \
    --seed "$SEED"
