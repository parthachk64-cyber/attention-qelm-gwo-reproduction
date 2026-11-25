"""Fold-safe GWO tuning.

Every learned component (the supervised encoder, and for raw-input variants nothing) is fitted on each CV fold's FIT partition
only; the fold's validation rows are encoded but never used to train anything. Embeddings are cached under a key containing the
fold index, a hash of the fit indices, the seed and the full encoder/training configuration, so a cached embedding can never be
reused across folds or configurations. The winning configuration is frozen before the outer test set is touched.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import f1_score

from ..data.splits import cv_folds
from ..models.pipeline import AttentionQELM, PipelineConfig
from ..models.qelm import QELM, QELMConfig
from .gwo import GreyWolfOptimizer, GWOResult


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


@dataclass
class FoldEmbeddings:
    fit_idx: np.ndarray
    val_idx: np.ndarray
    emb_fit: np.ndarray
    emb_val: np.ndarray
    key: str


class FoldEmbeddingCache:
    def __init__(self):
        self._store: Dict[str, FoldEmbeddings] = {}
        self.hits = 0
        self.fits = 0

    @staticmethod
    def make_key(fold: int, fit_idx: np.ndarray, cfg: PipelineConfig, class_weights) -> str:
        return _hash({"fold": fold, "fit": hashlib.sha256(np.asarray(fit_idx, dtype=np.int64).tobytes()).hexdigest(), "seed": cfg.train.seed,
                      "encoder": cfg.encoder, "train": cfg.train, "attention": cfg.use_attention, "head": cfg.head,
                      "weights": None if class_weights is None else sorted((int(k), round(float(v), 9)) for k, v in class_weights.items())})

    def get(self, fold, fit_idx, val_idx, X, y, cfg: PipelineConfig, class_weights) -> FoldEmbeddings:
        key = self.make_key(fold, fit_idx, cfg, class_weights)
        if key in self._store:
            self.hits += 1
            return self._store[key]
        cw = None if class_weights is None else np.array([class_weights.get(c, 1.0) for c in range(cfg.n_classes)], dtype=np.float32)
        m = AttentionQELM(copy.deepcopy(cfg))
        if cfg.use_attention:
            m.fit_encoder(X[fit_idx], y[fit_idx], cw)                    # inner validation is drawn from fit_idx only
            self.fits += 1
        fe = FoldEmbeddings(fit_idx, val_idx, m.encode(X[fit_idx]), m.encode(X[val_idx]), key)
        self._store[key] = fe
        return fe


@dataclass
class TuneResult:
    gwo: GWOResult
    n_folds: int
    encoder_fits: int
    cache_hits: int
    seconds: float
    reduced_budget: bool
    fold_manifest: List[dict] = field(default_factory=list)


def tune_qelm(X, y, base: PipelineConfig, *, class_weights: Optional[dict] = None, n_folds: int = 5, n_wolves: int = 20, max_iter: int = 50,
              patience: int = 10, tol: float = 1e-4, seed: int = 0, subsample: Optional[int] = None, on_iteration=None) -> TuneResult:
    """Fitness = mean macro-F1 across folds of QELM(candidate) fitted on fold-fit embeddings and scored on fold-validation embeddings."""
    t0 = time.perf_counter()
    X, y = np.asarray(X), np.asarray(y)
    if subsample is not None and subsample < len(y):
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(y), size=subsample, replace=False))
        X, y = X[keep], y[keep]
    folds = cv_folds(y, n_folds=n_folds, seed=seed)
    cache = FoldEmbeddingCache()
    fes = [cache.get(i, a, b, X, y, base, class_weights) for i, (a, b) in enumerate(folds)]

    def fitness(cfg: dict):
        scores = []
        for fe in fes:
            q = QELM(QELMConfig(init=base.qelm.init, seed=seed, theta_mode=base.qelm.theta_mode, weight_scale=base.qelm.weight_scale, **cfg))
            q.fit(fe.emb_fit, y[fe.fit_idx], base.n_classes)
            scores.append(float(f1_score(y[fe.val_idx], q.predict(fe.emb_val), average="macro", labels=list(range(base.n_classes)), zero_division=0)))
        return float(np.mean(scores)), scores

    res = GreyWolfOptimizer(n_wolves=n_wolves, max_iter=max_iter, patience=patience, tol=tol, seed=seed).optimise(fitness, on_iteration)
    manifest = [{"fold": i, "fit_hash": hashlib.sha256(np.asarray(fe.fit_idx, dtype=np.int64).tobytes()).hexdigest()[:16], "n_fit": int(len(fe.fit_idx)),
                 "n_val": int(len(fe.val_idx)), "embedding_key": fe.key} for i, fe in enumerate(fes)]
    reduced = (n_wolves, max_iter) != (20, 50) or subsample is not None or n_folds != 5
    return TuneResult(res, len(folds), cache.fits, cache.hits, time.perf_counter() - t0, reduced, manifest)
