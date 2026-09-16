#!/bin/bash

python utils/prepare_ctpd_dataset.py \
  --dataset HuggingFaceH4/ultrafeedback_binarized \
  --teacher models/teacher_dpo \
  --reference models/teacher_sft \
  --student-tokenizer models/student_sft \
  --save-to data/ultrafeedback_binarized-ctpd
