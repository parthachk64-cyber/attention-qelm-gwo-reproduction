"""Run orchestration: train / ablate / baselines / tune / evaluate / profile, writing ``runs/<run-id>/``.

Layout::

    runs/<run-id>/run.json                       provenance (kind, git sha, env, dataset manifest hash, split fingerprint, seeds, budget flags)
    runs/<run-id>/seed<S>/<model>/scores_test.npz  y_test + decision scores (everything metrics are recomputed from)
    runs/<run-id>/seed<S>/<model>/{metrics.json, history.json, manifest.json, bundle/}
    runs/<run-id>/gwo/<task>.json                candidate history, fold scores, selected values, actual stopping iteration
    runs/<run-id>/profile.json                   efficiency measurements

``kind`` is ``synthetic_fixture`` (code-path check, never a result) or ``benchmark``. Metrics are always recomputed from the saved scores.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import sklearn
import torch

from ..baselines import ALL_BASELINES, make_baseline
from ..data import schema as S
from ..data.prepare import Prepared, load_prepared
from ..errors import ProvenanceError
from ..evaluation import metrics as M
from ..models.pipeline import AttentionQELM, TrainConfig
from ..models.qelm import QELMConfig
from ..models.variants import ABLATIONS, build_config
from ..optim.tuning import tune_qelm

PAPER_SEEDS = (7, 13, 21, 42, 99)
FIXTURE, BENCHMARK = "synthetic_fixture", "benchmark"


def git_sha(cwd=None) -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def env_metadata() -> dict:
    return {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__, "scikit_learn": sklearn.__version__, "torch": torch.__version__,
            "cuda": bool(torch.cuda.is_available()), "paper_reported": {"python": "3.9", "tensorflow": "2.11", "scikit_learn": "1.2", "numpy": "1.24", "pandas": "1.5.3",
                                                                          "hardware": "NVIDIA RTX 3080 10 GB, Intel i7-10700, 32 GB RAM"}}


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def kind_of(prepared: Prepared) -> str:
    return FIXTURE if prepared.manifest.get("source", {}).get("fixture") else BENCHMARK


def init_run(run_dir, prepared: Prepared, *, task: str, seeds: Sequence[int], extra: Optional[dict] = None) -> Path:
    d = Path(run_dir)
    d.mkdir(parents=True, exist_ok=True)
    rec = {"run_id": d.name, "kind": kind_of(prepared), "task": task, "seeds": list(seeds), "git_sha": git_sha(Path(__file__).parent), "env": env_metadata(),
           "dataset": {"source": prepared.manifest.get("source"), "manifest_sha256": hashlib.sha256(json.dumps(prepared.manifest, sort_keys=True, default=str).encode()).hexdigest(),
                       "split_fingerprint": prepared.manifest.get("split_fingerprint"), "counts": prepared.manifest.get("counts")},
           "created_unix": time.time()}
    if rec["kind"] == FIXTURE:
        rec["banner"] = "SYNTHETIC FIXTURE: exercises the code path only; NOT Edge-IIoTset and NOT a reproduction of any published result."
    rec.update(extra or {})
    p = d / "run.json"
    if p.is_file():                                       # merge, never discard earlier flags
        old = json.loads(p.read_text(encoding="utf-8"))
        old.update({k: v for k, v in rec.items() if k not in ("created_unix",)})
        rec = old
    p.write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    return d


def _weights_vec(prepared: Prepared, k: int) -> np.ndarray:
    return np.array([prepared.class_weights.get(c, 1.0) for c in range(k)], dtype=np.float32)


def _write_model_outputs(out: Path, model_name: str, model, prepared: Prepared, task: str, seed: int, fit_seconds: float, extra: dict) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    names = S.class_names_for(task)
    scores = model.decision_scores(prepared.X_test)
    np.savez_compressed(out / "scores_test.npz", y=prepared.y_test, scores=scores.astype(np.float32))
    train_acc = float((model.decision_scores(prepared.X_train).argmax(1) == prepared.y_train).mean())
    m = M.summarise(prepared.y_test, scores, task=task, class_names=names)
    m["train_accuracy"], m["train_accuracy_recomputable_from_saved_artifacts"] = train_acc, False
    (out / "metrics.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    (out / "history.json").write_text(json.dumps(getattr(model, "history", {}) or {}), encoding="utf-8")
    man = {"model": model_name, "seed": seed, "task": task, "fit_seconds": fit_seconds, "scores_sha256": sha256_file(out / "scores_test.npz"),
           "n_test": int(len(prepared.y_test)), "split_fingerprint": prepared.manifest.get("split_fingerprint")}
    man.update(extra)
    (out / "manifest.json").write_text(json.dumps(man, indent=1, default=str), encoding="utf-8")
    return m


def run_variant(variant: str, prepared: Prepared, task: str, seed: int, run_dir, *, train: Optional[TrainConfig] = None, qelm_overrides: Optional[dict] = None,
                save_bundle: bool = True, monitor_test: bool = False) -> dict:
    k = len(S.class_names_for(task))
    cfg = build_config(variant, k, input_dim=prepared.X_train.shape[1], qelm_overrides=qelm_overrides, train=train, seed=seed)
    cw = _weights_vec(prepared, k) if cfg.train.loss.use_class_weights else None
    t0 = time.perf_counter()
    model = AttentionQELM(cfg)
    model.fit(prepared.X_train, prepared.y_train, cw, extra_eval={"test": (prepared.X_test, prepared.y_test)} if monitor_test else None)
    secs = time.perf_counter() - t0
    out = Path(run_dir) / f"seed{seed}" / variant
    extra = {"parameters": model.parameter_report(), "timings": model.timings, "meta": model.meta, "qelm_config": cfg.qelm.__dict__ if cfg.head == "qelm" else None,
             "test_monitoring": "test metrics were LOGGED in history.json but never used for selection" if monitor_test else "off"}
    m = _write_model_outputs(out, variant, model, prepared, task, seed, secs, extra)
    if save_bundle:
        extra["artifact_id"] = model.save(out / "bundle", extra_manifest={"split_fingerprint": prepared.manifest.get("split_fingerprint"), "feature_order": prepared.manifest.get("feature_order")})
    return m


def run_baseline(name: str, prepared: Prepared, task: str, seed: int, run_dir, *, train: Optional[TrainConfig] = None, weighted: bool = True) -> dict:
    k = len(S.class_names_for(task))
    model = make_baseline(name, k, seed, weighted=weighted, **({"train": train} if train is not None else {}))
    cw = {int(c): float(w) for c, w in prepared.class_weights.items()}
    t0 = time.perf_counter()
    model.fit(prepared.X_train, prepared.y_train, cw)
    secs = time.perf_counter() - t0
    out = Path(run_dir) / f"seed{seed}" / name
    return _write_model_outputs(out, name, model, prepared, task, seed, secs, {"params": model.n_params(), "weighted": weighted})


def run_models(models: Iterable[str], prepared: Prepared, task: str, seeds: Sequence[int], run_dir, *, train: Optional[TrainConfig] = None,
               qelm_overrides: Optional[dict] = None, progress=None) -> List[dict]:
    done = []
    for seed in seeds:
        for name in models:
            out = Path(run_dir) / f"seed{seed}" / name
            if (out / "manifest.json").is_file():                      # resumable: finished (model, seed) pairs are skipped
                if progress:
                    progress(f"skip {name} seed {seed} (already complete)")
                continue
            if progress:
                progress(f"{name} seed {seed}")
            fn = run_variant if name in ABLATIONS else run_baseline
            if name not in ABLATIONS and name not in ALL_BASELINES:
                raise KeyError(f"unknown model {name!r}; variants {sorted(ABLATIONS)}, baselines {list(ALL_BASELINES)}")
            kw = {"train": train, **({"qelm_overrides": qelm_overrides} if name in ABLATIONS else {})}
            done.append({"model": name, "seed": seed, "macro_f1": fn(name, prepared, task, seed, run_dir, **kw)["macro_f1"]})
    return done


def run_tuning(prepared: Prepared, task: str, run_dir, *, seed: int = 0, train: Optional[TrainConfig] = None, variant: str = "full", n_folds: int = 5, n_wolves: int = 20,
               max_iter: int = 50, patience: int = 10, subsample: Optional[int] = None, progress=None) -> dict:
    k = len(S.class_names_for(task))
    cfg = build_config(variant, k, input_dim=prepared.X_train.shape[1], train=train, seed=seed)
    res = tune_qelm(prepared.X_train, prepared.y_train, cfg, class_weights=prepared.class_weights, n_folds=n_folds, n_wolves=n_wolves, max_iter=max_iter, patience=patience, seed=seed,
                    subsample=subsample, on_iteration=(lambda it, d: progress(f"gwo iter {it}: {d}")) if progress else None)
    g = res.gwo
    rec = {"task": task, "variant": variant, "seed": seed, "reduced_budget": res.reduced_budget, "settings": g.settings, "n_folds": res.n_folds, "encoder_fits": res.encoder_fits,
           "embedding_cache_hits": res.cache_hits, "seconds": res.seconds, "best_config": g.best_config, "best_fitness": g.best_fitness, "iterations_run": g.iterations_run,
           "stopped_early": g.stopped_early, "stop_reason": g.stop_reason, "n_candidate_evaluations": g.n_candidate_evaluations,
           "n_fold_fits": g.n_candidate_evaluations * res.n_folds - g.cache_hits * res.n_folds, "memoised_candidates": g.cache_hits, "best_history": g.best_history,
           "candidates": g.candidates, "fold_manifest": res.fold_manifest, "frozen_before_outer_test_evaluation": True, "subsample": subsample}
    out = Path(run_dir) / "gwo"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{task}.json").write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    p = Path(run_dir) / "run.json"
    if p.is_file():
        r = json.loads(p.read_text(encoding="utf-8"))
        r["reduced_budget"] = bool(r.get("reduced_budget") or res.reduced_budget)
        p.write_text(json.dumps(r, indent=1, default=str), encoding="utf-8")
    return rec


def load_scores(model_dir) -> dict:
    z = np.load(Path(model_dir) / "scores_test.npz", allow_pickle=False)
    return {"y": z["y"], "scores": z["scores"].astype(float)}


def recompute_metrics(model_dir, task: str) -> dict:
    """Recompute every metric from the saved scores and verify the stored hash. Raises if the artifact was altered."""
    d = Path(model_dir)
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    if sha256_file(d / "scores_test.npz") != man["scores_sha256"]:
        raise ProvenanceError(f"{d / 'scores_test.npz'} does not match the hash recorded when it was produced")
    z = load_scores(d)
    return M.summarise(z["y"], z["scores"], task=task, class_names=S.class_names_for(task))
