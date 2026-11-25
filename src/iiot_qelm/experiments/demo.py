"""Offline CPU demo on an INVENTED fixture: prepare -> fit a small instance of the real encoder + QELM -> save -> reload -> predict
held-out rows -> report and figures. Nothing here is Edge-IIoTset and nothing is a reproduction of a published number."""
from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np

from ..data import schema as S
from ..data.prepare import PrepareConfig, prepare_frame, save_prepared
from ..data.synthetic import FIXTURE_MARK, make_fixture
from ..evaluation import plots
from ..models.pipeline import AttentionQELM, TrainConfig
from .report import build_report
from .runner import init_run, run_baseline, run_variant, run_tuning


def run_demo(out_dir="runs/demo-fixture", *, n: int = 1800, seed: int = 0, epochs: int = 8, gwo: bool = False, with_baselines: bool = True, task: str = "multiclass",
             log=print) -> dict:
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    log(f"*** {FIXTURE_MARK}")
    t0 = time.perf_counter()
    raw = make_fixture(n, seed)
    prepared = prepare_frame(raw, PrepareConfig(task=task), source={"fixture": True, "description": FIXTURE_MARK, "rows": n, "seed": seed})
    save_prepared(prepared, out / "data")
    log(f"prepared: train {prepared.X_train.shape}, test {prepared.X_test.shape}, features {len(prepared.manifest['feature_order'])}, split fingerprint {prepared.manifest['split_fingerprint'][:12]}")
    init_run(out, prepared, task=task, seeds=[seed], extra={"reduced_budget": True, "note": "demo: small encoder training budget and a narrower QELM"})
    train = TrainConfig(epochs=epochs, patience=4, batch_size=64)
    small = {"width": 128, "n_layers": 2}                     # reduced QELM width/depth for the demo only
    m = run_variant("full", prepared, task, seed, out, train=train, qelm_overrides=small, monitor_test=True)
    log(f"proposed (small instance): test accuracy {m['accuracy']:.3f}, macro-F1 {m['macro_f1']:.3f}  [FIXTURE]")
    # round trip: load the saved bundle and compare scores on held-out rows
    bundle = out / f"seed{seed}" / "full" / "bundle"
    reloaded = AttentionQELM.load(bundle)
    a = np.load(out / f"seed{seed}" / "full" / "scores_test.npz")["scores"]
    b = reloaded.decision_scores(prepared.X_test)
    parity = float(np.max(np.abs(a - b)))
    log(f"save/reload parity: max |score difference| = {parity:.2e}")
    if with_baselines:
        for name in ("logistic_regression", "decision_tree"):
            r = run_baseline(name, prepared, task, seed, out, train=train)
            log(f"{name}: test accuracy {r['accuracy']:.3f}, macro-F1 {r['macro_f1']:.3f}  [FIXTURE]")
    if gwo:
        g = run_tuning(prepared, task, out, seed=seed, train=TrainConfig(epochs=3, patience=2, batch_size=64), n_folds=3, n_wolves=6, max_iter=4, patience=2, subsample=600,
                       progress=lambda s: log("  " + s))
        log(f"FAST-BUDGET GWO (reduced): best {g['best_config']} cv macro-F1 {g['best_fitness']:.3f}, iterations {g['iterations_run']}, candidates {g['n_candidate_evaluations']}")
        plots.plot_gwo(g["best_history"], out / "figures" / "gwo_convergence.png", banner="FIXTURE, reduced budget")
    names = S.class_names_for(task)
    z = np.load(out / f"seed{seed}" / "full" / "scores_test.npz")
    cm = np.array(m["confusion_matrix"])
    banner = "SYNTHETIC FIXTURE"
    import json
    hist = json.loads((out / f"seed{seed}" / "full" / "history.json").read_text())
    plots.plot_curves(hist, out / "figures" / "phase1_curves.png", banner=banner)
    plots.plot_confusion(cm, names, out / "figures" / "confusion_full_test.png", banner=banner)
    plots.plot_confusion(cm, names, out / "figures" / "confusion_full_test_normalised.png", normalize=True, banner=banner)
    plots.plot_roc(z["y"], z["scores"], names, out / "figures" / "roc_ovr.png", banner=banner)
    build_report(out, task, out / "report.md")
    log(f"report: {out / 'report.md'}   figures: {out / 'figures'}   bundle: {bundle}   ({time.perf_counter() - t0:.1f}s)")
    return {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "parity": parity, "out": str(out)}
