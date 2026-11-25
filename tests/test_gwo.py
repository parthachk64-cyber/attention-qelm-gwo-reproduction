import numpy as np
import pytest

from iiot_qelm.models.pipeline import PipelineConfig, TrainConfig
from iiot_qelm.optim.gwo import BOUNDS, GreyWolfOptimizer, decode, decode_key
from iiot_qelm.optim.tuning import FoldEmbeddingCache, tune_qelm


def test_bounds_and_decode_rules():
    assert BOUNDS.shape == (7, 2)
    lo, hi = decode(BOUNDS[:, 0]), decode(BOUNDS[:, 1])
    assert (lo["n_layers"], lo["width"], lo["activation"], lo["rho"]) == (1, 64, "sigmoid", 0.0)
    assert (hi["n_layers"], hi["width"], hi["activation"], hi["rho"]) == (5, 1024, "gelu", 1.0)         # x = 4.0 -> index 3
    assert np.isclose(lo["lambda1"], 1e-6) and np.isclose(hi["lambda2"], 1e-1) and np.isclose(hi["theta"], 2 * np.pi)
    assert decode([2.5, 100.5, 1.99, -3, -3, 1, 0.5])["n_layers"] == 3 and decode([2.5, 100.5, 1.99, -3, -3, 1, 0.5])["width"] == 101      # half up
    assert decode([1, 64, 1.99, -3, -3, 1, 0.5])["activation"] == "tanh" and decode([1, 64, 2.0, -3, -3, 1, 0.5])["activation"] == "relu"
    assert decode([99, 99999, 99, 5, 5, 99, 9])["width"] == 1024                                         # out of range is clipped


def test_positions_stay_inside_bounds_and_evaluation_count_is_exact():
    seen = []

    def fit(cfg):
        seen.append(cfg)
        return float(np.random.default_rng(len(seen)).random())

    g = GreyWolfOptimizer(n_wolves=6, max_iter=5, patience=99, seed=1).optimise(fit)
    pos = np.array([c["position"] for c in g.candidates])
    assert (pos >= BOUNDS[:, 0] - 1e-12).all() and (pos <= BOUNDS[:, 1] + 1e-12).all()
    assert g.n_candidate_evaluations == 6 * (5 + 1) == len(g.candidates)
    assert g.settings["nominal_budget_N_x_T"] == 30 and g.settings["max_candidate_evaluations_N_x_Tplus1"] == 36
    assert g.iterations_run == 5 and g.stop_reason == "max_iter" and not g.stopped_early


def test_default_budget_contract():
    g = GreyWolfOptimizer()
    assert (g.N, g.T, g.patience, g.tol) == (20, 50, 10, 1e-4)
    assert g.N * g.T == 1000 and g.N * (g.T + 1) == 1020


def test_stops_after_ten_stagnant_iterations_with_constant_fitness():
    g = GreyWolfOptimizer(n_wolves=5, max_iter=50, patience=10, tol=1e-4, seed=0).optimise(lambda cfg: 0.5)
    assert g.stopped_early and g.stop_reason == "stagnation" and g.iterations_run == 10            # stagnates from iteration 1


def test_stagnation_tolerance_ignores_tiny_improvements():
    counter = {"n": 0}

    def fit(cfg):
        counter["n"] += 1
        return 1e-6 * counter["n"] / 100.0                                                         # gains far below tol

    g = GreyWolfOptimizer(n_wolves=4, max_iter=50, patience=3, tol=1e-4, seed=0).optimise(fit)
    assert g.stopped_early and g.iterations_run <= 4


def test_seeded_determinism_and_seed_sensitivity():
    f = lambda cfg: -abs(cfg["rho"] - 0.3) - abs(cfg["theta"] - 2.0) / 7
    a = GreyWolfOptimizer(n_wolves=6, max_iter=4, seed=3).optimise(f)
    b = GreyWolfOptimizer(n_wolves=6, max_iter=4, seed=3).optimise(f)
    c = GreyWolfOptimizer(n_wolves=6, max_iter=4, seed=4).optimise(f)
    assert a.best_history == b.best_history and np.array_equal(a.best_position, b.best_position) and a.best_history != c.best_history


def test_optimises_a_known_landscape_and_is_monotone():
    f = lambda cfg: -abs(cfg["rho"] - 0.7) - abs(cfg["theta"] - 3.0) / 6.3
    g = GreyWolfOptimizer(n_wolves=12, max_iter=25, patience=99, seed=0).optimise(f)
    assert all(b >= a for a, b in zip(g.best_history, g.best_history[1:]))
    assert g.best_fitness > -0.1 and abs(g.best_config["rho"] - 0.7) < 0.12


def test_memoisation_counts_hits_and_does_not_refit_identical_decoded_configs():
    calls = []

    def fit(cfg):
        calls.append(decode_key(cfg))
        return 0.1

    g = GreyWolfOptimizer(n_wolves=4, max_iter=3, patience=99, seed=0).optimise(fit)
    assert len(calls) == len(set(calls)) and g.cache_hits == g.n_candidate_evaluations - len(calls)


def test_fitness_may_return_fold_scores_and_leaders_are_distinct():
    g = GreyWolfOptimizer(n_wolves=5, max_iter=2, seed=0).optimise(lambda cfg: (0.4, [0.3, 0.5]))
    assert g.candidates[0]["fold_scores"] == [0.3, 0.5]
    with pytest.raises(ValueError):
        GreyWolfOptimizer(n_wolves=2)


# ---------------------------------------------------------------- fold-safe tuning
def test_tuning_fits_encoder_per_fold_on_fit_rows_only_and_keys_never_collide(prepared):
    Xs, ys = prepared.X_train[:300], prepared.y_train[:300]
    cfg = PipelineConfig(n_classes=15, train=TrainConfig(epochs=1, patience=1, batch_size=64))
    cache = FoldEmbeddingCache()
    cw = prepared.class_weights
    from iiot_qelm.data.splits import cv_folds
    folds = cv_folds(ys, n_folds=3, seed=0)
    fes = [cache.get(i, a, b, Xs, ys, cfg, cw) for i, (a, b) in enumerate(folds)]
    assert len({fe.key for fe in fes}) == 3 and cache.fits == 3
    for fe in fes:
        assert not set(fe.fit_idx) & set(fe.val_idx) and fe.emb_fit.shape == (len(fe.fit_idx), 256) and fe.emb_val.shape == (len(fe.val_idx), 256)
    cache.get(0, folds[0][0], folds[0][1], Xs, ys, cfg, cw)
    assert cache.hits == 1 and cache.fits == 3                           # same key -> reuse
    cfg2 = PipelineConfig(n_classes=15, train=TrainConfig(epochs=1, patience=1, batch_size=64, seed=9))
    assert cache.make_key(0, folds[0][0], cfg2, cw) != fes[0].key         # seed is part of the key
    assert cache.make_key(1, folds[0][0], cfg, cw) != fes[0].key          # fold index is part of the key
    assert cache.make_key(0, folds[1][0], cfg, cw) != fes[0].key          # fit indices are part of the key


def test_tune_qelm_end_to_end_reduced_budget_is_flagged(prepared):
    cfg = PipelineConfig(n_classes=15, use_attention=False)            # raw-input variant: no encoder to fit, fast
    r = tune_qelm(prepared.X_train, prepared.y_train, cfg, class_weights=prepared.class_weights, n_folds=3, n_wolves=4, max_iter=2, patience=2, seed=0, subsample=400)
    assert r.reduced_budget and r.encoder_fits == 0 and r.gwo.n_candidate_evaluations == 12
    assert len(r.fold_manifest) == 3 and all(len(c["fold_scores"]) == 3 for c in r.gwo.candidates if not c["memoised"])
