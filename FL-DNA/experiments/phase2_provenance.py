"""Replay the loader's positional sampling to identify source CSV rows."""
import numpy as np
from sklearn.model_selection import train_test_split


def validation_source_rows(labels, max_rows, seed):
    ids = np.arange(len(labels))
    if max_rows < len(ids):
        _, ids = train_test_split(ids, test_size=max_rows, random_state=seed, stratify=labels)
    _, temp = train_test_split(ids, test_size=0.15 + 0.2, random_state=seed, stratify=labels[ids])
    val, _ = train_test_split(temp, test_size=0.2 / (0.15 + 0.2),
                              random_state=seed, stratify=labels[temp])
    return val
