#!/usr/bin/env bash
# Exact substantive commands used in this bounded reproduction attempt.
set -euo pipefail

# Environment (all packages installed only in external_defenses/.venv).
python3 -m venv external_defenses/.venv
external_defenses/.venv/bin/python -m pip install --upgrade pip
external_defenses/.venv/bin/pip install 'torch==2.5.1' torchvision --index-url https://download.pytorch.org/whl/cpu
external_defenses/.venv/bin/pip install matplotlib scipy scikit-image pandas

# Official repositories.
git clone https://github.com/jeremy313/Soteria.git external_defenses/soteria
git clone https://github.com/dAI-SY-Group/PRECODE.git external_defenses/precode
git clone https://github.com/mit-han-lab/dlg.git external_defenses/dlg
git clone https://github.com/gaow0007/ATSPrivacy.git external_defenses/ats
git clone https://github.com/kiddyboots216/CommEfficient.git external_defenses/fetchsgd
git -C external_defenses/soteria rev-parse HEAD
git -C external_defenses/precode rev-parse HEAD
git -C external_defenses/dlg rev-parse HEAD
git -C external_defenses/ats rev-parse HEAD
git -C external_defenses/fetchsgd rev-parse HEAD

# Original, unmodified DLG execution. This finished, but did not converge under
# current PyTorch; output is logs/dlg_original_cpu.log.
cd external_defenses/dlg
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MPLBACKEND=Agg \
  ../.venv/bin/python main.py --index 25
cd ../..

# Bounded CPU fallback checks; output is logs/cpu_qualitative_checks.log.
cd external_defenses
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venv/bin/python run_cpu_qualitative_checks.py
