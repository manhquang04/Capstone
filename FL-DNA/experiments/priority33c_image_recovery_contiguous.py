"""Technical layout correction; original scientific runner is immutable."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
from experiments import priority33c_image_recovery as recovery

original_tensors = recovery.utility.p31.tensors


def contiguous_tensors(data, indices):
    normal, labels = original_tensors(data, indices)
    return normal.contiguous(), labels


# Executed in each spawned worker as well as the parent. No RNG operations.
recovery.utility.p31.tensors = contiguous_tensors
# Per-job source fingerprint explicitly identifies the corrected entry point.
recovery.__file__ = __file__


if __name__ == '__main__':
    recovery.main()
