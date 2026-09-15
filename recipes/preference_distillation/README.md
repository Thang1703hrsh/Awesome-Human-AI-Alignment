# Preference-distillation recipes

These TOML files replace the executable intent of the temporary YAML and shell
recipes. Paths are deliberately relative to the repository and may be changed
without editing package code.

The optional teacher and student initialization stages are:

```bash
hai-align train --config recipes/preference_distillation/teacher_sft.toml
hai-align train --config recipes/preference_distillation/teacher_dpo.toml
hai-align train --config recipes/preference_distillation/student_sft.toml
```

Prepare a method-specific dataset first:

```bash
hai-align prepare distillation dckd \
  --dataset HuggingFaceH4/ultrafeedback_binarized \
  --teacher models/teacher_dpo \
  --tokenizer models/student_sft \
  --output data/ultrafeedback-dckd
```

Then execute its training recipe:

```bash
hai-align train --config recipes/preference_distillation/dckd.toml
```

ADPA+ is reproduced by setting the `model` in `adpa.toml` to a DCKD checkpoint.
CTPD needs a reference checkpoint during both preparation and training. PPD and
VPD preparation can use `--student` to sample several on-policy responses or
score responses that are already present in the input dataset.
