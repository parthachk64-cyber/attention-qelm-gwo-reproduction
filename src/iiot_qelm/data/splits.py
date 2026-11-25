"""Immutable outer split, inner validation split and CV folds. Always stratified on the 15-class label so the binary and
multiclass tasks share one outer partition. The outer split seed is fixed (7) and independent of the run seeds."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterator, List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from ..errors import LeakageError

OUTER_SPLIT_SEED = 7          # upstream README: "80 / 20 stratified, seed = 7 (split is fixed across all seeds)"
OUTER_TEST_FRACTION = 0.20
INNER_VAL_FRACTION = 0.10     # implementer's choice (the paper gives none); drawn only from the outer training partition


@dataclass(frozen=True)
class OuterSplit:
    train_idx: np.ndarray
    test_idx: np.ndarray
    seed: int
    test_fraction: float

    def fingerprint(self, row_ids) -> str:
        ids = np.asarray(row_ids)
        h = hashlib.sha256()
        h.update(f"seed={self.seed};frac={self.test_fraction};".encode())
        h.update(b"train:" + "\n".join(sorted(map(str, ids[self.train_idx]))).encode())
        h.update(b"test:" + "\n".join(sorted(map(str, ids[self.test_idx]))).encode())
        return h.hexdigest()


def assert_disjoint(a, b, what: str = "partitions") -> None:
    inter = np.intersect1d(np.asarray(a), np.asarray(b))
    if inter.size:
        raise LeakageError(f"{what} overlap in {inter.size} rows")


def outer_split(y15, *, seed: int = OUTER_SPLIT_SEED, test_fraction: float = OUTER_TEST_FRACTION) -> OuterSplit:
    y15 = np.asarray(y15)
    idx = np.arange(len(y15))
    tr, te = train_test_split(idx, test_size=test_fraction, random_state=seed, stratify=y15)
    assert_disjoint(tr, te, "outer train/test")
    return OuterSplit(np.sort(tr), np.sort(te), seed, test_fraction)


def inner_validation_split(train_y, *, fraction: float = INNER_VAL_FRACTION, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Positions WITHIN the training partition: (fit, validation). Never touches the outer test rows."""
    train_y = np.asarray(train_y)
    pos = np.arange(len(train_y))
    try:
        fit, val = train_test_split(pos, test_size=fraction, random_state=seed, stratify=train_y)
    except ValueError:      # a class with <2 members: fall back to an unstratified draw (recorded by the caller)
        fit, val = train_test_split(pos, test_size=fraction, random_state=seed)
    return np.sort(fit), np.sort(val)


def cv_folds(train_y, *, n_folds: int = 5, seed: int = 0) -> List[Tuple[np.ndarray, np.ndarray]]:
    train_y = np.asarray(train_y)
    counts = np.bincount(train_y.astype(int))
    k = int(min(n_folds, counts[counts > 0].min())) if (counts > 0).any() else n_folds
    skf = StratifiedKFold(n_splits=max(k, 2), shuffle=True, random_state=seed)
    return [(np.sort(a), np.sort(b)) for a, b in skf.split(np.zeros(len(train_y)), train_y)]


def iter_batches(n: int, batch: int) -> Iterator[slice]:
    for s in range(0, n, batch):
        yield slice(s, min(s + batch, n))
