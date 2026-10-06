# Reproduction smoke test (2026-10-06)

Status: PASS.

## Setup

- Fresh clone of pre-release commit `ed8000697` into an empty directory. Its `FL-DNA/` tree is identical to the released one except for this report, `artifacts/priority34b/administrative_closeout.py`, `requirements-lock.txt`, small edits to `ARTIFACT_MANIFEST.tsv`, and seven internal plan snapshots (`PROJECT_*.md`) moved out of the repository and hashed in the manifest (checked with `git diff --stat`).
- Raw datasets are not in the repository; `FL-DNA/datasets` in the clone was linked to the local raw copies, whose SHA-256 values match those recorded in `artifacts/priority32_multidataset/prepared/paysim/audit.json`.
- Interpreter: the project environment `.venv-phase1` (CPU, one thread), as in the original runs.

## Checks

| Check | Result |
|---|---|
| Source files recorded in the P34A job configuration (13 files) | 13/13 SHA-256 identical to the commit |
| Source-file hash records in 135 P30–P34 freeze, manifest and receipt files | 226 match the commit; 8 belong to superseded first attempts or to reports edited after their freeze (the final attempts of P33A/P33B match); 70 refer to unmodified upstream clones listed in `external_defenses/UPSTREAM.md` |
| PaySim data preparation rerun from raw data (`priority32_multidataset.prepare_dataset`) | 11/11 prepared files byte-identical to the original |
| P34A local-BN training, PaySim, seed 321000: baseline, v1 conservative, v2 0.95/0.01 | Test and validation metrics identical; client probabilities identical; all 8 checkpoint tensors identical; per-round logs identical apart from timing fields |

Artifact hashes recorded inside the job JSON differ only because their paths contain the attempt timestamp; the file contents were compared directly.

## Scope

This is a sample, not a full replication. It covers the data pipeline and the utility training path for one dataset and seed. Reconstruction experiments were not rerun; their code identity is covered by the hash check above.
