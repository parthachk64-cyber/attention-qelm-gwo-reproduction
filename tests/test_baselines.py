import numpy as np
import pytest

from iiot_qelm.baselines import ALL_BASELINES, CLASSICAL, LABELS, NEURAL, make_baseline
from iiot_qelm.baselines.neural import build_net
from iiot_qelm.models.pipeline import TrainConfig


def test_ten_baselines_registered_with_labels():
    assert len(ALL_BASELINES) == 10 and set(ALL_BASELINES) == set(LABELS)
    assert set(CLASSICAL) | set(NEURAL) == set(ALL_BASELINES)
    with pytest.raises(KeyError):
        make_baseline("svm", 3)


@pytest.mark.parametrize("name", CLASSICAL)
def test_classical_baseline_fits_and_scores_with_correct_shape(name, prepared):
    X, y = prepared.X_train[:300], prepared.y_train[:300]
    b = make_baseline(name, 15, seed=1, n_jobs=1, **({"weighted": True}))
    b.fit(X, y, prepared.class_weights)
    s = b.decision_scores(prepared.X_test[:25])
    assert s.shape == (25, 15) and np.isfinite(s).all() and b.predict(prepared.X_test[:25]).shape == (25,)
    assert np.allclose(s.sum(1), 1, atol=1e-5)


def test_classical_settings_match_the_published_ones():
    assert make_baseline("decision_tree", 3).model.max_depth == 18
    assert make_baseline("random_forest", 3).model.n_estimators == 100
    x = make_baseline("xgboost", 3).model
    assert x.n_estimators == 200 and x.max_depth == 8
    assert make_baseline("logistic_regression", 3).model.class_weight == "balanced" and make_baseline("logistic_regression", 3, weighted=False).model.class_weight is None


def test_classical_param_counts_and_weight_handling(prepared):
    X, y = prepared.X_train[:200], prepared.y_train[:200]
    lr = make_baseline("logistic_regression", 15).fit(X, y)
    assert lr.n_params() == lr.model.coef_.size + lr.model.intercept_.size
    nb = make_baseline("naive_bayes", 15).fit(X, y)
    assert nb.n_params() > 0 and nb.use_sample_weight is True
    assert make_baseline("random_forest", 15).fit(X, y).n_params() is None and make_baseline("naive_bayes", 15, weighted=False).use_sample_weight is False


def test_absent_training_class_is_padded_not_crashing(prepared):
    X, y = prepared.X_train[:200], prepared.y_train[:200]
    keep = y != 14
    b = make_baseline("decision_tree", 15).fit(X[keep], y[keep])
    assert b.decision_scores(X[:5]).shape == (5, 15) and (b.decision_scores(X[:5])[:, 14] == 0).all()


@pytest.mark.parametrize("name", NEURAL)
def test_neural_baseline_architecture_fit_and_shapes(name, prepared):
    net = build_net(name, 15)
    import torch
    assert net(torch.rand(2, 48)).shape == (2, 15) and net.eval()(torch.rand(1, 48)).shape == (1, 15)
    b = make_baseline(name, 15, seed=2, train=TrainConfig(epochs=1, patience=1, batch_size=64))
    b.fit(prepared.X_train[:160], prepared.y_train[:160], prepared.class_weights)
    s = b.decision_scores(prepared.X_test[:12])
    assert s.shape == (12, 15) and np.allclose(s.sum(1), 1, atol=1e-5) and b.n_params() > 0
    assert set(b.history) >= {"loss", "val_loss"} and b.train_cfg.loss.kind == "ce"


def test_neural_architecture_details():
    from iiot_qelm.baselines.neural import CNN1D, DenseNet1D, RNN
    cnn = build_net("cnn1d", 3)
    assert isinstance(cnn, CNN1D) and [m.out_channels for m in cnn.features if hasattr(m, "out_channels")] == [64, 128, 128]
    for n, bi in (("lstm", False), ("gru", False), ("bilstm", True)):
        r = build_net(n, 3)
        assert isinstance(r, RNN) and r.r1.hidden_size == r.r2.hidden_size == 128 and r.r1.bidirectional == bi
    dn = build_net("densenet1d", 3)
    assert isinstance(dn, DenseNet1D) and sum(1 for m in dn.modules() if m.__class__.__name__ == "_DenseLayer") == 16       # 4 blocks x 4 layers
    with pytest.raises(KeyError):
        build_net("mamba", 3)
