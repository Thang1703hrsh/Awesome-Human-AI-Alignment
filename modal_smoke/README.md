# Modal DCKD/TVKD smoke tests

This directory is an isolated replacement for the former
`run/run_dckd.sh` and `run/run_tvkd.sh` smoke-test use case. It does not write
to the repository, use the production recipes, or persist checkpoints.

The remote job:

1. copies only `src/` into a Modal image;
2. downloads the public `sshleifer/tiny-gpt2` checkpoint;
3. prepares DCKD/TVKD supervision for two in-memory preference rows;
4. runs one optimizer step for the selected method; and
5. writes all prepared data and checkpoints to an ephemeral remote temporary
   directory, which is removed when the function finishes.

The function requests one T4 GPU, 0.5 physical CPU cores, and 2 GiB of system
memory. This is intentionally small while retaining enough headroom for
PyTorch, Transformers, and Arrow to initialize reliably.

No Hugging Face secret or Modal Volume is required.

## Setup

From the repository root, install and authenticate the Modal client:

```powershell
python -m pip install modal
modal setup
```

On Windows PowerShell, force UTF-8 before running Modal to avoid console
encoding failures on Modal's Unicode status symbols:

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
```

## Run

Smoke-test DCKD only:

```powershell
modal run modal_smoke/dckd_tvkd.py --method dckd
```

Smoke-test TVKD only, initialized directly from the tiny base model:

```powershell
modal run modal_smoke/dckd_tvkd.py --method tvkd
```

Smoke-test the original sequential relationship, with TVKD initialized from
the one-step DCKD checkpoint:

```powershell
modal run modal_smoke/dckd_tvkd.py --method all
```

Smoke-test all preference-distillation workflows, including ADPA+, CTPD, PPD,
and VPD:

```powershell
modal run modal_smoke/all_distillation_methods.py
```

A successful result contains `checkpoint_created: true` for each requested
method. These tests validate code execution and tensor/data compatibility;
they do not measure training quality or reproduce the full research settings.

## Isolation

- Repository source is copied into the container image rather than modified.
- Training outputs use a remote `TemporaryDirectory`.
- No local `data/`, `models/`, or `outputs/` directories are created.
- No persistent Modal Volume is mounted.
- Production recipes under `recipes/preference_distillation/` are not changed.
