import numpy as np
import pytest
import torch

from iiot_qelm.errors import IIoTError
from iiot_qelm.models import focal as focal_mod
from iiot_qelm.models import pipeline as pipe_mod
from iiot_qelm.models.pipeline import AttentionQELM, TrainConfig
from iiot_qelm.models.variants import ABLATIONS, DEFAULT_NO_GWO, PUBLISHED_FIXED, build_config, exercises

K = 15


def _cw(prepared):
    return np.array([prepared.class_weights.get(c, 1.0) for c in range(K)], dtype=np.float32)


@pytest.fixture
def subset(prepared):
    return prepared.X_train[:240], prepared.y_train[:240]


def test_seven_ablations_plus_full_are_registered():
    assert set(ABLATIONS) == {"full", "no_attention", "no_qelm", "no_quantum_init", "no_gwo", "no_focal", "no_gwo_no_focal", "encoder_mlp"} and len(ABLATIONS) == 8


def test_published_fixed_hyperparameters():
    c = build_config("full", 15).qelm
    assert (c.n_layers, c.width, c.activation, c.lambda1, c.lambda2, c.theta, c.rho) == (3, 512, "gelu", 3.2e-4, 1.7e-3, 1.84, 0.63)
    t = build_config("full", 15).train
    assert (t.lr, t.betas, t.batch_size, t.epochs, t.patience) == (1e-3, (0.9, 0.999), 32, 100, 15)
    assert (t.loss.kind, t.loss.gamma, t.loss.alpha, t.loss.use_class_weights) == ("focal", 2.0, 0.25, True)


def test_no_gwo_defaults_and_lambda_mapping():
    q = build_config("no_gwo", 15).qelm
    assert (q.n_layers, q.width, q.activation, q.rho) == (2, 256, "relu", 0.5) and np.isclose(q.theta, np.pi)
    assert np.isclose(q.lambda1 + q.lambda2, 1e-3)                    # a single lambda = 1e-3 maps to the summed ridge penalty
    assert build_config("no_gwo_no_focal", 15).qelm.width == 256 and build_config("no_gwo", 15, qelm_overrides={"width": 99}).qelm.width == 99


def test_exercises_table_per_variant():
    e = {n: exercises(n) for n in ABLATIONS}
    assert e["full"] == {"encoder_trained": True, "qelm_head": True, "mlp_head": False, "quantum_init": True, "classical_init": False, "focal_gradient_path": True,
                         "ce_gradient_path": False, "gwo_selected_hyperparameters": True}
    assert not e["no_attention"]["encoder_trained"] and not e["no_attention"]["focal_gradient_path"] and e["no_attention"]["qelm_head"]       # nothing for focal to train
    assert e["no_qelm"]["mlp_head"] and not e["no_qelm"]["qelm_head"] and e["no_qelm"]["focal_gradient_path"]
    assert e["no_quantum_init"]["classical_init"] and not e["no_quantum_init"]["quantum_init"]
    assert not e["no_gwo"]["gwo_selected_hyperparameters"] and e["no_gwo"]["focal_gradient_path"]
    assert e["no_focal"]["ce_gradient_path"] and not e["no_focal"]["focal_gradient_path"] and e["no_focal"]["gwo_selected_hyperparameters"]
    assert e["no_gwo_no_focal"]["ce_gradient_path"] and not e["no_gwo_no_focal"]["gwo_selected_hyperparameters"]
    assert e["encoder_mlp"]["mlp_head"] and e["encoder_mlp"]["ce_gradient_path"] and not e["encoder_mlp"]["gwo_selected_hyperparameters"]


@pytest.mark.parametrize("name", sorted(ABLATIONS))
def test_every_variant_runs_and_exercises_exactly_what_its_config_says(name, subset, prepared, monkeypatch):
    X, y = subset
    calls = {"focal": 0, "ce": 0}
    real_f, real_c = focal_mod.focal_loss_from_logits, focal_mod.cross_entropy_from_logits

    def spy_f(*a, **k):
        calls["focal"] += 1
        return real_f(*a, **k)

    def spy_c(*a, **k):
        calls["ce"] += 1
        return real_c(*a, **k)

    monkeypatch.setattr(focal_mod, "focal_loss_from_logits", spy_f)
    monkeypatch.setattr(focal_mod, "cross_entropy_from_logits", spy_c)
    cfg = build_config(name, K, train=TrainConfig(epochs=1, patience=1, batch_size=64), qelm_overrides={"width": 32})
    m = AttentionQELM(cfg).fit(X, y, _cw(prepared))
    ex = exercises(name)
    sc = m.decision_scores(prepared.X_test[:20])
    assert sc.shape == (20, K) and np.isfinite(sc).all()
    assert (calls["focal"] > 0) == ex["focal_gradient_path"] and (calls["ce"] > 0) == ex["ce_gradient_path"]
    assert (m.net is not None) == ex["encoder_trained"]
    assert (m.qelm is not None) == ex["qelm_head"]
    if ex["qelm_head"]:
        assert (m.qelm.cfg.init == "quantum") == ex["quantum_init"]
    assert m.encode(X[:3]).shape == (3, 256 if ex["encoder_trained"] else 48)


def test_no_attention_variant_trains_no_phase1_and_reports_it(subset, prepared):
    X, y = subset
    m = AttentionQELM(build_config("no_attention", K, qelm_overrides={"width": 16})).fit(X, y)
    assert m.net is None and m.history == {} and m.parameter_report()["encoder_trainable"] == 0


def test_mlp_head_variants_have_no_qelm_phase(subset):
    X, y = subset
    m = AttentionQELM(build_config("no_qelm", K, train=TrainConfig(epochs=1, patience=1, batch_size=64))).fit(X, y)
    assert m.qelm is None
    with pytest.raises(IIoTError):
        m.fit_qelm(X, y)


def test_phase1_history_uses_original_metric_names_and_actual_epochs(subset, prepared):
    X, y = subset
    m = AttentionQELM(build_config("full", K, train=TrainConfig(epochs=3, patience=2, batch_size=64), qelm_overrides={"width": 16}))
    m.fit(X, y, _cw(prepared), extra_eval={"test": (prepared.X_test[:50], prepared.y_test[:50])})
    h = m.history
    assert set(h) == {"loss", "accuracy", "val_loss", "val_accuracy", "test_loss", "test_accuracy"}
    assert len(h["loss"]) == m.meta["epochs_run"] <= 3 and m.meta["selection_split"].startswith("validation drawn from the training partition")
    assert m.meta["class_weights_applied_once"] is True


def test_early_stopping_restores_best_epoch(prepared):
    X, y = prepared.X_train[:200], prepared.y_train[:200]
    cfg = build_config("full", K, train=TrainConfig(epochs=30, patience=2, batch_size=64, lr=0.5), qelm_overrides={"width": 16})   # huge lr => validation loss wanders
    m = AttentionQELM(cfg).fit(X, y)
    assert m.meta["epochs_run"] < 30 and m.meta["best_epoch"] < m.meta["epochs_run"]


def test_validation_for_early_stopping_comes_only_from_training_rows(prepared, monkeypatch):
    seen = {}
    real = pipe_mod.train_phase1

    def spy(net, X, y, Xv, yv, *a, **k):
        seen.update(Xv=Xv, X=X)
        return real(net, X, y, Xv, yv, *a, **k)

    monkeypatch.setattr(pipe_mod, "train_phase1", spy)
    AttentionQELM(build_config("full", K, train=TrainConfig(epochs=1, patience=1, batch_size=64), qelm_overrides={"width": 16})).fit(prepared.X_train[:200], prepared.y_train[:200])
    train_rows = {r.tobytes() for r in prepared.X_train[:200]}
    test_rows = {r.tobytes() for r in prepared.X_test}
    assert all(r.tobytes() in train_rows for r in seen["Xv"]) and not any(r.tobytes() in test_rows for r in seen["Xv"])


def test_bundle_round_trip_gives_identical_scores_and_artifact_id(subset, prepared, tmp_path):
    X, y = subset
    m = AttentionQELM(build_config("full", K, train=TrainConfig(epochs=1, patience=1, batch_size=64), qelm_overrides={"width": 24})).fit(X, y, _cw(prepared))
    aid = m.save(tmp_path / "b", extra_manifest={"note": "t"})
    r = AttentionQELM.load(tmp_path / "b")
    Xt = prepared.X_test[:40]
    assert np.allclose(m.decision_scores(Xt), r.decision_scores(Xt), atol=1e-6) and (m.predict(Xt) == r.predict(Xt)).all()
    assert len(aid) == 16 and AttentionQELM.load(tmp_path / "b").meta == m.meta
    assert np.allclose(r.predict_proba(Xt).sum(1), 1)
    rep = r.parameter_report()
    assert rep["encoder_trainable"] > 0 and rep["qelm_solved"] == 24 * K and rep["deployed_total"] == rep["encoder_trainable"] + rep["qelm_frozen_random"] + rep["qelm_solved"]


def test_unfitted_pipeline_errors():
    m = AttentionQELM(build_config("full", K))
    with pytest.raises(RuntimeError):
        m.decision_scores(np.zeros((1, 48), dtype=np.float32))
    with pytest.raises(RuntimeError):
        m.encode(np.zeros((1, 48), dtype=np.float32))
    with pytest.raises(KeyError):
        build_config("nonsense", K)


def test_inference_never_runs_gwo_or_phase1(subset, prepared, monkeypatch):
    X, y = subset
    m = AttentionQELM(build_config("full", K, train=TrainConfig(epochs=1, patience=1, batch_size=64), qelm_overrides={"width": 16})).fit(X, y)
    import iiot_qelm.optim.gwo as g
    monkeypatch.setattr(g.GreyWolfOptimizer, "optimise", lambda *a, **k: (_ for _ in ()).throw(AssertionError("GWO called at inference")))
    monkeypatch.setattr(pipe_mod, "train_phase1", lambda *a, **k: (_ for _ in ()).throw(AssertionError("training at inference")))
    assert m.predict(prepared.X_test[:10]).shape == (10,)
