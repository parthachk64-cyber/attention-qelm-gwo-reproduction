"""Console entry point ``iiot-qelm``. Full-scale commands fail with the exact missing input; the offline demo, audit and docs never need data."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import PAPER_DOI, UPSTREAM_URL, __version__
from .errors import IIoTError, MissingInputError


def _prepared(args):
    from .data.prepare import load_prepared
    return load_prepared(args.data)


def _cfg(args):
    from .experiments.config import load_config
    return load_config(args.config) if getattr(args, "config", None) else None


def cmd_demo(a):
    from .experiments.demo import run_demo
    run_demo(a.out, n=a.rows, epochs=a.epochs, gwo=a.gwo, with_baselines=not a.no_baselines)


def cmd_audit(a):
    from .evaluation.audit import run_all, to_markdown
    res = run_all()
    text = to_markdown(res)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text, encoding="utf-8")
        Path(a.out).with_suffix(".json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    for r in res:
        print(f"{r['id']:6s} {r['severity']:6s} {r['status']:32s} {r['title']}")
    bad = [r for r in res if r["status"] == "inconsistent"]
    print(f"\n{len(res)} checks, {len(bad)} inconsistent. Outcome label: recomputed_from_printed_aggregates (nothing here is a new measurement).")
    return 1 if (a.strict and bad) else 0


def cmd_docs(a):
    from .docsgen import build_docs
    print(json.dumps(build_docs(a.out), indent=1))


def cmd_data(a):
    from .data import schema as S
    from .data.prepare import PrepareConfig, load_raw, prepare_csv, save_prepared, sha256_file, validate_raw
    if a.action == "fixture":
        from .data.synthetic import make_fixture
        make_fixture(a.rows, 0).to_csv(a.out, index=False)
        print(f"wrote INVENTED fixture CSV {a.out}")
        return
    if a.action == "download":
        import shutil
        import subprocess
        if shutil.which("kaggle") is None:
            raise MissingInputError("the kaggle CLI / credentials", "accept the dataset terms at the provider page (docs/data.md), `pip install kaggle`, place your own token at ~/.kaggle/kaggle.json, rerun; or download manually and use --csv")
        subprocess.run(["kaggle", "datasets", "download", "-d", "mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot", "-p", a.out, "--unzip"], check=True)
        return
    if a.action == "check":
        df = load_raw(a.csv)
        canon = validate_raw(df, a.label_col)
        print(json.dumps({"file": Path(a.csv).name, "bytes": Path(a.csv).stat().st_size, "sha256": sha256_file(a.csv), "rows": len(df), "columns": df.shape[1],
                          "label_column": a.label_col, "class_histogram": {k: int(v) for k, v in canon.value_counts().items()},
                          "classes_missing_vs_paper": [c for c in S.CLASSES if c not in set(canon)]}, indent=1))
        return
    if a.action == "prepare":
        fl = json.loads(Path(a.feature_list).read_text(encoding="utf-8")) if a.feature_list else None
        cfg = PrepareConfig(task=a.task, label_col=a.label_col, feature_list=fl, expected_features=None if a.expected_features == 0 else a.expected_features, conflict_policy=a.conflicts)
        p = prepare_csv(a.csv, cfg)
        save_prepared(p, a.out)
        print(json.dumps(p.manifest["counts"], indent=1), "\nsplit fingerprint", p.manifest["split_fingerprint"])


def _train_cfg(a, exp):
    from .models.pipeline import TrainConfig
    t = exp.train if exp else TrainConfig()
    if getattr(a, "epochs", None):
        t.epochs = a.epochs
    return t


def cmd_train(a):
    from .experiments.runner import PAPER_SEEDS, init_run, run_models
    exp = _cfg(a)
    p = _prepared(a)
    task = exp.task if exp else a.task
    seeds = tuple(a.seeds) if a.seeds else (exp.seeds if exp else PAPER_SEEDS)
    init_run(a.run_dir, p, task=task, seeds=seeds)
    qo = dict(exp.qelm_overrides) if exp else {}
    if a.gwo_config:
        qo.update(json.loads(Path(a.gwo_config).read_text(encoding="utf-8"))["best_config"])
    run_models(["full"], p, task, seeds, a.run_dir, train=_train_cfg(a, exp), qelm_overrides=qo, progress=print)


def cmd_tune(a):
    from .experiments.config import GWOSettings
    from .experiments.runner import init_run, run_tuning
    exp = _cfg(a)
    p = _prepared(a)
    task = exp.task if exp else a.task
    g = exp.gwo if exp and exp.gwo else GWOSettings()
    if a.fast:
        g = GWOSettings(n_wolves=6, max_iter=4, patience=2, n_folds=3, subsample=a.subsample or 2000)
        print("FAST-BUDGET GWO (reduced): not the paper's 20 wolves / 50 iterations / 5 folds")
    init_run(a.run_dir, p, task=task, seeds=[g.seed])
    from .models.pipeline import TrainConfig
    tc = _train_cfg(a, exp)
    if a.fast:
        tc = TrainConfig(epochs=a.epochs or 3, patience=2, batch_size=64)
    rec = run_tuning(p, task, a.run_dir, seed=g.seed, train=tc, n_folds=g.n_folds, n_wolves=g.n_wolves, max_iter=g.max_iter, patience=g.patience, subsample=g.subsample or a.subsample,
                     progress=print)
    print(json.dumps({k: rec[k] for k in ("best_config", "best_fitness", "iterations_run", "stop_reason", "n_candidate_evaluations", "n_fold_fits", "reduced_budget")}, indent=1))


def cmd_baselines(a):
    from .baselines import ALL_BASELINES
    from .experiments.runner import PAPER_SEEDS, init_run, run_models
    exp = _cfg(a)
    p = _prepared(a)
    task = exp.task if exp else a.task
    seeds = tuple(a.seeds) if a.seeds else PAPER_SEEDS
    init_run(a.run_dir, p, task=task, seeds=seeds)
    run_models(a.models or list(ALL_BASELINES), p, task, seeds, a.run_dir, train=_train_cfg(a, exp), progress=print)


def cmd_ablate(a):
    from .experiments.runner import PAPER_SEEDS, init_run, run_models
    from .models.variants import ABLATIONS
    exp = _cfg(a)
    p = _prepared(a)
    task = exp.task if exp else a.task
    seeds = tuple(a.seeds) if a.seeds else PAPER_SEEDS
    init_run(a.run_dir, p, task=task, seeds=seeds)
    qo = {}
    if a.gwo_config:
        qo = json.loads(Path(a.gwo_config).read_text(encoding="utf-8"))["best_config"]
    run_models(a.variants or list(ABLATIONS), p, task, seeds, a.run_dir, train=_train_cfg(a, exp), qelm_overrides=qo or None, progress=print)


def cmd_evaluate(a):
    from .experiments.runner import recompute_metrics
    for md in sorted(Path(a.run_dir).glob("seed*/*")):
        if (md / "scores_test.npz").is_file():
            m = recompute_metrics(md, a.task)
            print(f"{md.parent.name}/{md.name:22s} acc {m['accuracy']:.4f}  macro-F1 {m['macro_f1']:.4f}" + (f"  AUC {m['auc_roc']:.4f}" if m.get("auc_roc") else ""))


def cmd_profile(a):
    from .evaluation.efficiency import encoder_macs, hardware_metadata, qelm_macs, time_inference
    from .models.pipeline import AttentionQELM
    p = _prepared(a)
    bundle = Path(a.run_dir) / f"seed{a.seed}" / "full" / "bundle"
    m = AttentionQELM.load(bundle)
    prof = {"hardware": hardware_metadata(), "bundle": str(bundle), "parameters": m.parameter_report(), "macs_per_sample": {"encoder": encoder_macs(m.cfg.encoder), "qelm": qelm_macs(m.cfg.qelm, m.cfg.encoder.out_dim, m.cfg.n_classes)},
            "mac_convention": "1 MAC = 2 FLOPs; linear layers and attention matmuls", "timings_s": m.timings,
            "latency_cpu_batch1": time_inference(m.decision_scores, p.X_test, batch=1, warmup=a.warmup, timed=a.timed), "latency_cpu_batch_large": time_inference(m.decision_scores, p.X_test, batch=min(256, len(p.X_test)), warmup=3, timed=20),
            "note": "workstation timing; says nothing about embedded-gateway latency"}
    if hardware_metadata()["cuda_available"]:
        prof["note_gpu"] = "CUDA present but this profile times the CPU path; run with a CUDA model to profile the GPU"
    Path(a.run_dir, "profile.json").write_text(json.dumps(prof, indent=1, default=str), encoding="utf-8")
    print(json.dumps(prof["latency_cpu_batch1"], indent=1))


def cmd_report(a):
    from .data import schema as S
    from .evaluation import plots
    from .experiments.report import PROPOSED, build_report
    out = a.out or str(Path(a.run_dir) / f"report_{a.task}.md")
    build_report(a.run_dir, a.task, out)
    print("wrote", out)
    if a.figures:
        names = S.class_names_for(a.task)
        for md in sorted(Path(a.run_dir).glob(f"seed*/{PROPOSED}")):
            z = np.load(md / "scores_test.npz")
            from .evaluation import metrics as M
            cm = np.array(M.summarise(z["y"], z["scores"], task=a.task, class_names=names)["confusion_matrix"])
            fig = Path(a.run_dir) / "figures" / md.parent.name
            run = json.loads((Path(a.run_dir) / "run.json").read_text(encoding="utf-8"))
            ban = "SYNTHETIC FIXTURE" if run["kind"] == "synthetic_fixture" else "measured"
            plots.plot_confusion(cm, names, fig / "confusion_counts.png", banner=ban)
            plots.plot_confusion(cm, names, fig / "confusion_normalised.png", normalize=True, banner=ban)
            plots.plot_roc(z["y"], z["scores"], names, fig / "roc_ovr.png", banner=ban)
            plots.plot_curves(json.loads((md / "history.json").read_text(encoding="utf-8")), fig / "curves.png", banner=ban)
        g = Path(a.run_dir) / "gwo" / f"{a.task}.json"
        if g.is_file():
            plots.plot_gwo(json.loads(g.read_text(encoding="utf-8"))["best_history"], Path(a.run_dir) / "figures" / "gwo_convergence.png")
        print("figures under", Path(a.run_dir) / "figures")


def cmd_predict(a):
    import pandas as pd
    from .data.prepare import Preprocessor
    from .models.pipeline import AttentionQELM
    pre = Preprocessor.from_json(json.loads((Path(a.data) / "preprocessor.json").read_text(encoding="utf-8")))
    model = AttentionQELM.load(a.bundle)
    man = json.loads((Path(a.bundle) / "manifest.json").read_text(encoding="utf-8"))
    df = pd.read_csv(a.csv, low_memory=False)
    X = pre.transform(df)
    sc = model.decision_scores(X)
    out = pd.DataFrame({"pred_index": sc.argmax(1)})
    from .data import schema as S
    names = S.class_names_for("binary" if model.cfg.n_classes == 2 else "multiclass")
    out["pred_label"] = [names[i] for i in out["pred_index"]]
    for i, n in enumerate(names):
        out[f"score_{n}"] = sc[:, i]
    out["artifact_id"] = man["artifact_id"]
    out.to_csv(a.out, index=False)
    print(f"wrote {len(out)} predictions to {a.out} (artifact {man['artifact_id']}). Load only bundles you trust.")


def cmd_estimate(a):
    import torch
    from .models.pipeline import PipelineConfig, TrainConfig, train_phase1
    from .models.encoder import Phase1Network
    net = Phase1Network(a.classes)
    X = np.random.default_rng(0).random((640, 48)).astype(np.float32)
    y = np.random.default_rng(1).integers(0, a.classes, 640)
    t0 = time.perf_counter()
    train_phase1(net, X, y, X[:64], y[:64], TrainConfig(epochs=1, batch_size=32), None)
    steps = 640 / 32
    sps = steps / (time.perf_counter() - t0)
    per_epoch_h = (a.rows * 0.9 / 32) / sps / 3600
    print(json.dumps({"measured_steps_per_second_batch32_cpu": round(sps, 1), "train_rows": a.rows, "phase1_hours_per_epoch_cpu": round(per_epoch_h, 2),
                      "phase1_hours_at_100_epochs_upper_bound": round(per_epoch_h * 100, 1),
                      "note": "rough linear extrapolation from a 640-row micro-benchmark on THIS machine; early stopping (patience 15) usually ends sooner; GWO adds 5 encoder fits per search plus candidates x folds QELM solves",
                      "paid_compute": "this tool never launches paid compute"}, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="iiot-qelm", description=f"Independent companion to doi:{PAPER_DOI}. Upstream: {UPSTREAM_URL}")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("demo", help="offline CPU demo on an invented fixture")
    s.add_argument("--out", default="runs/demo-fixture")
    s.add_argument("--rows", type=int, default=1800)
    s.add_argument("--epochs", type=int, default=8)
    s.add_argument("--gwo", action="store_true", help="also run the FAST-BUDGET GWO demo")
    s.add_argument("--no-baselines", action="store_true")
    s.set_defaults(fn=cmd_demo)

    s = sub.add_parser("audit", help="recompute derived numbers of the published article (needs no data)")
    s.add_argument("--out", default=None)
    s.add_argument("--strict", action="store_true")
    s.set_defaults(fn=cmd_audit)

    s = sub.add_parser("docs", help="regenerate generated docs (architecture figure, GWO contract, coverage table)")
    s.add_argument("--out", default="docs")
    s.set_defaults(fn=cmd_docs)

    s = sub.add_parser("data", help="check / prepare / download local benchmark data")
    s.add_argument("action", choices=("check", "prepare", "fixture", "download"))
    s.add_argument("--csv")
    s.add_argument("--out", default="data/prepared")
    s.add_argument("--task", default="multiclass", choices=("binary", "multiclass"))
    s.add_argument("--label-col", default="Attack_type")
    s.add_argument("--feature-list")
    s.add_argument("--expected-features", type=int, default=48, help="0 accepts the rule-derived set (recorded in the manifest)")
    s.add_argument("--conflicts", default="drop", choices=("drop", "keep_first", "error"))
    s.add_argument("--rows", type=int, default=2000)
    s.set_defaults(fn=cmd_data)

    def common(p, runs=True):
        p.add_argument("--data", default="data/prepared")
        p.add_argument("--task", default="multiclass", choices=("binary", "multiclass"))
        p.add_argument("--config")
        p.add_argument("--run-dir", default="runs/run")
        p.add_argument("--epochs", type=int)
        p.add_argument("--seeds", type=int, nargs="*")

    s = sub.add_parser("train", help="train the proposed model with fixed (published or GWO-selected) hyperparameters")
    common(s)
    s.add_argument("--gwo-config", help="gwo/<task>.json from `tune`")
    s.set_defaults(fn=cmd_train)
    s = sub.add_parser("tune", help="GWO search (full budget by default; --fast for the reduced demo budget)")
    common(s)
    s.add_argument("--fast", action="store_true")
    s.add_argument("--subsample", type=int)
    s.set_defaults(fn=cmd_tune)
    s = sub.add_parser("baselines", help="train the ten baselines on the same split")
    common(s)
    s.add_argument("--models", nargs="*")
    s.set_defaults(fn=cmd_baselines)
    s = sub.add_parser("ablate", help="the seven Table-11 ablations (plus the full model)")
    common(s)
    s.add_argument("--variants", nargs="*")
    s.add_argument("--gwo-config")
    s.set_defaults(fn=cmd_ablate)
    s = sub.add_parser("evaluate", help="recompute metrics from saved scores (hash-checked)")
    s.add_argument("--run-dir", required=True)
    s.add_argument("--task", default="multiclass")
    s.set_defaults(fn=cmd_evaluate)
    s = sub.add_parser("profile", help="parameters, MACs, CPU latency for a saved bundle")
    s.add_argument("--run-dir", required=True)
    s.add_argument("--data", default="data/prepared")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--warmup", type=int, default=20)
    s.add_argument("--timed", type=int, default=200)
    s.set_defaults(fn=cmd_profile)
    s = sub.add_parser("report", help="build the table analogues (and optional figures) from a run directory")
    s.add_argument("--run-dir", required=True)
    s.add_argument("--task", default="multiclass")
    s.add_argument("--out")
    s.add_argument("--figures", action="store_true")
    s.set_defaults(fn=cmd_report)
    s = sub.add_parser("predict", help="label a local CSV with a saved bundle (trusted bundles only)")
    s.add_argument("--bundle", required=True)
    s.add_argument("--data", required=True, help="prepared-data dir holding preprocessor.json")
    s.add_argument("--csv", required=True)
    s.add_argument("--out", default="predictions.csv")
    s.set_defaults(fn=cmd_predict)
    s = sub.add_parser("estimate", help="transparent runtime estimate for large jobs")
    s.add_argument("--rows", type=int, default=1_527_736)
    s.add_argument("--classes", type=int, default=15)
    s.set_defaults(fn=cmd_estimate)

    a = ap.parse_args(argv)
    try:
        return a.fn(a) or 0
    except IIoTError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
