"""Fingerprint immutable transform dependency code and replay environment."""
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/priority32b_bn_reconciliation'


def main():
    import numpy
    import pandas
    import sklearn
    import torch
    torch.set_num_threads(1)
    paths = [ROOT / 'dna_encoder' / name for name in
             ('transform_defense.py', 'transform_defense_v2.py', 'binary_mapper.py', 'dna_mapper.py', '__init__.py')]
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    doc = {'at': datetime.now(timezone.utc).isoformat(), 'sha256': hashes,
           'python': platform.python_version(), 'platform': platform.platform(),
           'torch': torch.__version__, 'numpy': numpy.__version__, 'pandas': pandas.__version__,
           'sklearn': sklearn.__version__, 'mps_available': torch.backends.mps.is_available(), 'torch_threads': torch.get_num_threads(),
           'disclosure': 'Supplemental dependency fingerprint collected while frozen replays run; no code changes or model forwards.'}
    path = OUT / 'transform_dependency_provenance.json'
    assert not path.exists()
    path.write_text(json.dumps(doc, indent=2) + '\n')
    print(json.dumps(doc))


if __name__ == '__main__':
    main()
