#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

STUDENT_MODEL="${STUDENT_MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
DATASET="${DATASET:-HuggingFaceH4/ultrafeedback_binarized}"
DATASET_SPLIT="${DATASET_SPLIT:-train_prefs}"
OUTPUT_DIR="${OUTPUT_DIR:-data/generated/ultrafeedback/llama3.2-1b-it}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
SEEDS="${SEEDS:-0 1 2 3 4}"
LOCAL_DATASET="${LOCAL_DATASET:-0}"

LOCAL_FLAG=()
if [[ "$LOCAL_DATASET" == "1" || "$LOCAL_DATASET" == "true" ]]; then
    LOCAL_FLAG=(--local)
fi

for seed in $SEEDS; do
    python data_gen/gen/sampling.py \
        --model_name "$STUDENT_MODEL" \
        --dataset "$DATASET" \
        --dataset_split "$DATASET_SPLIT" \
        "${LOCAL_FLAG[@]}" \
        --max_tokens "$MAX_TOKENS" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        --seed "$seed" \
        --output_dir "$OUTPUT_DIR"
done
