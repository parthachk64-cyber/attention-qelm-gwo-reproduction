import json
import re
from pathlib import Path

import numpy as np
import pytest

from iiot_qelm import reported as R
from iiot_qelm.cli import main
from iiot_qelm.docsgen import COVERAGE, build_docs
from iiot_qelm.errors import MissingInputError, ProvenanceError
from iiot_qelm.evaluation import audit as A
from iiot_qelm.experiments import runner as RUN
from iiot_qelm.experiments.config import load_config
from iiot_qelm.experiments.report import build_report
from iiot_qelm.models.pipeline import TrainConfig

ROOT = Path(__file__).resolve().parents[1]
RES = {r["id"]: r for r in A.run_all()}


# ---------------------------------------------------------------- audit and reported references
def test_reported_files_are_attributed_and_labelled():
    for t in ("table1", "table2", "table3", "table5", "table6", "table7", "table8", "table9", "table10", "table11", "table12", "text_claims"):
        d = R.load(t)
        assert d["label"] == "published_reference" and d["source"]["doi"] == "10.1007/s44163-026-01704-3" and d["source"]["page"]
    with pytest.raises(MissingInputError):
        import os
        old = os.environ["IIOT_QELM_REFERENCES"]
        os.environ["IIOT_QELM_REFERENCES"] = "/nonexistent"
        try:
            from unittest import mock
            with mock.patch("pathlib.Path.cwd", return_value=Path("/nonexistent")), mock.patch.object(R, "__file__", "/nonexistent/x/y/reported.py"):
                R.references_dir()
        finally:
            os.environ["IIOT_QELM_REFERENCES"] = old


def test_checkpoint_values_transcribed_correctly():
    t6 = {r["model"]: r for r in R.rows("table6")}["Proposed Model"]
    assert (t6["test_acc"], t6["f1"], t6["fpr"], t6["auc_roc"]) == (0.9954, 0.9942, 0.0043, 0.9997)
    t7 = {r["model"]: r for r in R.rows("table7")}["Proposed Model"]
    assert (t7["test_acc"], t7["macro_f1"], t7["fpr"]) == (0.9894, 0.9859, 0.0079)
    t8 = {r["class"]: r for r in R.rows("table8")}
    assert (t8["MITM"]["proposed_f1"], t8["MITM"]["densenet_f1"], t8["MITM"]["diff_vs_best_baseline_pp"]) == (0.9421, 0.8764, 6.57)
    assert (t8["Ransomware"]["proposed_f1"], t8["OS_Fingerprinting"]["proposed_f1"], t8["Backdoor"]["proposed_f1"]) == (0.9612, 0.9673, 0.9714)
    assert (t8["DDoS_HTTP"]["proposed_f1"], t8["DDoS_HTTP"]["densenet_f1"], t8["DDoS_HTTP"]["diff_vs_best_baseline_pp"]) == (0.9891, 0.9914, -0.23)
    c = {r["id"]: r for r in (dict(zip(R.load("text_claims")["columns"], x)) for x in R.load("text_claims")["rows"])}["table12_text"]["value"]
    assert (c["train_min"], c["gwo_min"], c["fit_min"], c["gpu_ms"], c["cpu_ms"], c["gpu_samples_per_s"], c["peak_gpu_mb"], c["params_M"], c["ops_M"]) == (67, 42, 25, 0.31, 4.12, 3226, 487, 0.21, 2.83)


def test_inference_and_training_code_never_import_the_reported_values():
    for p in (ROOT / "src" / "iiot_qelm").rglob("*.py"):
        rel = p.relative_to(ROOT / "src" / "iiot_qelm").as_posix()
        if rel.split("/")[0] in ("models", "data", "optim", "baselines") or rel in ("experiments/runner.py", "experiments/demo.py"):
            assert "reported" not in p.read_text(encoding="utf-8").replace("# ", "").split("import")[0] or True
            assert not re.search(r"from\s+\.+\s*import\s+reported|import\s+reported|from\s+\.+reported", p.read_text(encoding="utf-8")), rel


def test_audit_table1_class_sum_and_missing_19000():
    a = RES["A-01"]
    assert a["status"] == "inconsistent" and a["data"]["class_sum"] == 1890671 and a["data"]["printed_total"] == 1909671 and a["data"]["difference"] == 19000
    assert RES["A-04"]["data"]["attack_sum"] == 522712 and RES["A-04"]["data"]["printed"] == 541712


def test_audit_preserves_the_conflicting_udp_shares():
    d = RES["A-03"]["detail"]
    assert "5.31" in d and "6.31" in d and "Origin unresolved" in d and RES["A-03"]["status"] == "inconsistent"


def test_audit_table8_mean_vs_table7_macro_f1():
    a = RES["A-08"]
    assert np.isclose(a["data"]["mean"], 0.98152, atol=1e-5) and a["data"]["printed"] == 0.9859 and a["status"] == "inconsistent" and "per-class precision/recall" in a["detail"]


def test_audit_friedman_and_table10_reproduction():
    assert np.isclose(RES["A-11"]["data"]["chi2"], 50.0) and RES["A-11"]["data"]["max_attainable_chi2"] == 50.0 and "54.98" in RES["A-11"]["detail"]
    assert RES["A-12"]["status"] == "reproducible_from_printed_runs" and "10/10 rows" in RES["A-12"]["detail"]
    for row in RES["A-12"]["data"]["rows"]:
        assert row["d_pooled"] == pytest.approx(row["printed_d"], abs=0.006)
        assert row["paired_t_p_holm"] == pytest.approx(row["printed_p_holm"], rel=0.006)


def test_audit_binary_internal_inconsistency_and_gap_claim():
    assert RES["A-06"]["status"] == "inconsistent" and RES["A-14"]["status"] == "inconsistent" and "BiLSTM" in RES["A-14"]["detail"]
    assert RES["A-07"]["status"] == "consistent" and RES["A-10"]["status"] == "consistent" and RES["A-13"]["status"] == "consistent" and RES["A-15"]["status"] == "consistent"


def test_every_audit_line_is_labelled_not_measured():
    assert all(r["outcome"] == "recomputed_from_printed_aggregates" for r in RES.values())
    md = A.to_markdown(list(RES.values()))
    assert "A-01" in md and "recomputed_from_printed_aggregates" in md


def test_coverage_table_references_every_published_item():
    items = {c[0] for c in COVERAGE}
    assert {f"Table {i}" for i in range(1, 13)} <= items and {"Figure 1", "Figure 2", "Figure 3", "Figure 4", "Figures 5-6", "Figure 7"} <= items
    assert all(all(field.strip() for field in c) for c in COVERAGE)


def test_docs_generation(tmp_path):
    out = build_docs(tmp_path)
    assert Path(out["architecture_png"]).stat().st_size > 5000 and "gelu" in Path(out["gwo_contract"]).read_text() and "Table 12" in Path(out["coverage"]).read_text()


def test_configs_load_and_pin_published_settings():
    cfgs = {p.stem: load_config(p) for p in (ROOT / "configs").glob("*.yaml")}
    assert {"binary", "multiclass", "published_fixed", "gwo_full", "cpu_smoke"} <= set(cfgs)
    assert cfgs["binary"].task == "binary" and cfgs["multiclass"].task == "multiclass"
    g = cfgs["gwo_full"].gwo
    assert (g.n_wolves, g.max_iter, g.patience, g.tol, g.n_folds) == (20, 50, 10, 1e-4, 5)
    assert cfgs["published_fixed"].seeds == (7, 13, 21, 42, 99) and cfgs["published_fixed"].train.epochs == 100 and cfgs["cpu_smoke"].fixture_only


# ---------------------------------------------------------------- runner, provenance, report
@pytest.fixture(scope="module")
def run_dir(tmp_path_factory, prepared):
    d = tmp_path_factory.mktemp("run") / "r"
    RUN.init_run(d, prepared, task="multiclass", seeds=[1, 2])
    tc = TrainConfig(epochs=1, patience=1, batch_size=64)
    RUN.run_models(["full", "logistic_regression", "decision_tree"], prepared, "multiclass", [1, 2], d, train=tc, qelm_overrides={"width": 16})
    return d


def test_run_layout_and_fixture_flag(run_dir):
    run = json.loads((run_dir / "run.json").read_text())
    assert run["kind"] == "synthetic_fixture" and "SYNTHETIC FIXTURE" in run["banner"] and run["dataset"]["split_fingerprint"] and run["env"]["paper_reported"]["tensorflow"] == "2.11"
    for sd in (1, 2):
        for m in ("full", "logistic_regression", "decision_tree"):
            for f in ("scores_test.npz", "metrics.json", "history.json", "manifest.json"):
                assert (run_dir / f"seed{sd}" / m / f).is_file()
    assert (run_dir / "seed1" / "full" / "bundle" / "manifest.json").is_file()


def test_metrics_are_recomputed_from_scores_and_hash_checked(run_dir, tmp_path):
    md = run_dir / "seed1" / "full"
    m = RUN.recompute_metrics(md, "multiclass")
    stored = json.loads((md / "metrics.json").read_text())
    assert m["macro_f1"] == pytest.approx(stored["macro_f1"]) and stored["train_accuracy_recomputable_from_saved_artifacts"] is False
    import shutil
    cp = tmp_path / "c"
    shutil.copytree(md, cp)
    z = np.load(cp / "scores_test.npz")
    np.savez_compressed(cp / "scores_test.npz", y=z["y"], scores=z["scores"][::-1])
    with pytest.raises(ProvenanceError):
        RUN.recompute_metrics(cp, "multiclass")


def test_runs_are_resumable(run_dir, prepared):
    msgs = []
    RUN.run_models(["full"], prepared, "multiclass", [1], run_dir, progress=msgs.append)
    assert msgs and "skip" in msgs[0]
    with pytest.raises(KeyError):
        RUN.run_models(["nonsense"], prepared, "multiclass", [3], run_dir)


def test_report_watermarks_fixture_and_labels_outcome(run_dir):
    text = build_report(run_dir, "multiclass")
    assert "SYNTHETIC FIXTURE" in text and "fixture (code path only)" in text and "Baselines not run here" in text
    assert "Macro-F1 by seed" in text and "Friedman" in text and "Planned paired t-tests" in text and "weighted-F1" in text
    assert "Per-class F1" in text and "Proposed vs" in text


def test_tuning_record_is_complete_and_flags_reduced_budget(prepared, tmp_path):
    d = tmp_path / "t"
    RUN.init_run(d, prepared, task="multiclass", seeds=[0])
    rec = RUN.run_tuning(prepared, "multiclass", d, variant="no_attention", n_folds=3, n_wolves=4, max_iter=2, patience=2, subsample=400)
    for k in ("best_config", "n_candidate_evaluations", "n_fold_fits", "iterations_run", "stop_reason", "candidates", "fold_manifest", "reduced_budget", "frozen_before_outer_test_evaluation"):
        assert k in rec
    assert rec["reduced_budget"] and json.loads((d / "run.json").read_text())["reduced_budget"] is True and (d / "gwo" / "multiclass.json").is_file()
    assert rec["n_fold_fits"] <= rec["n_candidate_evaluations"] * 3


def test_binary_task_run_and_report(prepared_binary, tmp_path):
    d = tmp_path / "b"
    RUN.init_run(d, prepared_binary, task="binary", seeds=[1, 2])
    RUN.run_models(["full", "naive_bayes", "decision_tree"], prepared_binary, "binary", [1, 2], d, train=TrainConfig(epochs=1, patience=1, batch_size=64), qelm_overrides={"width": 16})
    t = build_report(d, "binary")
    assert "Binary (Table 6 analogue" in t and "FPR = FP/(FP+TN)" in t
    m = RUN.recompute_metrics(d / "seed1" / "full", "binary")
    assert 0 <= m["fpr"] <= 1 and m["auc_roc"] is not None


# ---------------------------------------------------------------- CLI
def test_cli_audit_runs_without_data(tmp_path, capsys):
    assert main(["audit", "--out", str(tmp_path / "a.md")]) == 0
    out = capsys.readouterr().out
    assert "A-01" in out and "recomputed_from_printed_aggregates" in out and (tmp_path / "a.md").is_file() and json.loads((tmp_path / "a.json").read_text())
    assert main(["audit", "--strict"]) == 1


def test_cli_full_scale_command_fails_with_exact_missing_input(tmp_path, capsys):
    assert main(["train", "--data", str(tmp_path / "nope"), "--run-dir", str(tmp_path / "r")]) == 2
    err = capsys.readouterr().err
    assert "missing: prepared data" in err and "iiot-qelm data prepare" in err
    assert main(["data", "check", "--csv", str(tmp_path / "none.csv")]) == 2
    assert "download Edge-IIoTset" in capsys.readouterr().err


def test_cli_data_fixture_check_prepare_train_report_predict(tmp_path, capsys, monkeypatch):
    csv, prep, run = tmp_path / "f.csv", tmp_path / "prep", tmp_path / "run"
    assert main(["data", "fixture", "--out", str(csv), "--rows", "900"]) == 0
    assert main(["data", "check", "--csv", str(csv)]) == 0
    chk = json.loads(capsys.readouterr().out.split("wrote INVENTED fixture CSV")[1].split("\n", 1)[1])
    assert chk["rows"] == 900 and len(chk["sha256"]) == 64 and chk["classes_missing_vs_paper"] == []
    assert main(["data", "prepare", "--csv", str(csv), "--out", str(prep)]) == 0
    capsys.readouterr()
    assert main(["data", "prepare", "--csv", str(csv), "--out", str(tmp_path / "p2"), "--expected-features", "47"]) == 2           # count mismatch is refused
    assert "will not drop arbitrary columns" in capsys.readouterr().err
    assert main(["train", "--data", str(prep), "--run-dir", str(run), "--seeds", "1", "--epochs", "1", "--config", str(ROOT / "configs" / "cpu_smoke.yaml")]) == 0
    assert main(["evaluate", "--run-dir", str(run)]) == 0
    assert "macro-F1" in capsys.readouterr().out
    assert main(["report", "--run-dir", str(run), "--figures"]) == 0
    assert (run / "figures" / "seed1" / "confusion_counts.png").is_file() and (run / "figures" / "seed1" / "roc_ovr.png").is_file() and (run / "figures" / "seed1" / "curves.png").is_file()
    assert main(["profile", "--run-dir", str(run), "--data", str(prep), "--seed", "1", "--warmup", "2", "--timed", "5"]) == 0
    prof = json.loads((run / "profile.json").read_text())
    assert prof["latency_cpu_batch1"]["batch"] == 1 and "embedded-gateway" in prof["note"] and prof["parameters"]["deployed_total"] > 0
    bundle = run / "seed1" / "full" / "bundle"
    assert main(["predict", "--bundle", str(bundle), "--data", str(prep), "--csv", str(csv), "--out", str(tmp_path / "pred.csv")]) == 0
    import pandas as pd
    pr = pd.read_csv(tmp_path / "pred.csv")
    assert len(pr) == 900 and set(pr["pred_label"]) <= set(__import__("iiot_qelm").data.schema.CLASSES) and pr["artifact_id"].nunique() == 1


def test_cli_ablate_and_baselines_small(tmp_path, prepared):
    from iiot_qelm.data.prepare import save_prepared
    save_prepared(prepared, tmp_path / "prep")
    r = tmp_path / "r"
    assert main(["baselines", "--data", str(tmp_path / "prep"), "--run-dir", str(r), "--seeds", "1", "--models", "naive_bayes", "decision_tree"]) == 0
    assert main(["ablate", "--data", str(tmp_path / "prep"), "--run-dir", str(r), "--seeds", "1", "--epochs", "1", "--variants", "no_attention"]) == 0
    assert (r / "seed1" / "no_attention" / "scores_test.npz").is_file() and (r / "seed1" / "naive_bayes" / "scores_test.npz").is_file()


def test_cli_estimate_and_download_without_credentials(capsys, monkeypatch):
    assert main(["estimate", "--rows", "1000"]) == 0
    assert "never launches paid compute" in capsys.readouterr().out
    monkeypatch.setattr("shutil.which", lambda _: None)
    assert main(["data", "download", "--out", "x"]) == 2
    assert "kaggle" in capsys.readouterr().err


def test_cli_demo_end_to_end(tmp_path, capsys):
    assert main(["demo", "--out", str(tmp_path / "d"), "--rows", "700", "--epochs", "2", "--no-baselines"]) == 0
    out = capsys.readouterr().out
    assert "SYNTHETIC FIXTURE" in out and "save/reload parity" in out
    assert (tmp_path / "d" / "report.md").is_file() and (tmp_path / "d" / "figures" / "confusion_full_test.png").is_file()
    parity = float(re.search(r"max \|score difference\| = ([0-9.e+-]+)", out).group(1))
    assert parity < 1e-4
