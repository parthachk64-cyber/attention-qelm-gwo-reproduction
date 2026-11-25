import math

import numpy as np
import pytest

from iiot_qelm.evaluation import efficiency as EF
from iiot_qelm.evaluation import metrics as M
from iiot_qelm.evaluation import stats as ST
from iiot_qelm.models.encoder import EncoderConfig
from iiot_qelm.models.qelm import QELMConfig


def _scores(y, pred, k):
    s = np.full((len(y), k), 0.01)
    s[np.arange(len(y)), pred] = 0.9
    return s


def test_macro_f1_is_the_unweighted_mean_and_differs_from_weighted():
    y = np.array([0] * 90 + [1] * 10)
    pred = np.array([0] * 90 + [0] * 10)
    r = M.summarise(y, _scores(y, pred, 2), task="multiclass", class_names=["a", "b"])
    assert r["per_class"]["b"]["f1"] == 0.0 and np.isclose(r["macro_f1"], np.mean([r["per_class"]["a"]["f1"], 0.0]))
    assert r["weighted_f1"] > r["macro_f1"] + 0.3


def test_macro_f1_from_scores_matches_per_class_mean():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 300)
    s = rng.random((300, 4)) + np.eye(4)[y]
    r = M.summarise(y, s, task="multiclass", class_names=list("abcd"))
    assert np.isclose(M.macro_f1_from_scores(y, s, 4), r["macro_f1"]) and np.isclose(r["macro_f1"], np.mean([v["f1"] for v in r["per_class"].values()]))


def test_absent_class_is_nan_not_zero_and_excluded_from_macro():
    y = np.array([0, 0, 1, 1])
    r = M.summarise(y, _scores(y, y, 3), task="multiclass", class_names=["a", "b", "c"])
    assert r["absent_classes"] == ["c"] and r["per_class"]["c"]["f1"] is None and r["macro_f1"] == 1.0
    assert r["auc_ovr_per_class"]["c"] is None and r["auc_ovr_macro"] == 1.0


def test_binary_fpr_definition_and_auc():
    y = np.array([0] * 8 + [1] * 4)
    pred = np.array([0] * 6 + [1] * 2 + [1] * 3 + [0])
    s = np.stack([1 - pred * 0.9, pred * 0.9 + 0.05], 1)
    r = M.summarise(y, s, task="binary", class_names=["Normal", "Attack"])
    assert (r["fp"], r["tn"], r["tp"], r["fn"]) == (2, 6, 3, 1)
    assert np.isclose(r["fpr"], 2 / 8) and np.isclose(r["precision"], 3 / 5) and np.isclose(r["recall"], 3 / 4)
    assert 0.5 < r["auc_roc"] <= 1.0 and r["auc_note"].startswith("from the continuous")


def test_confusion_matrix_and_roc_points():
    cm = M.confusion_matrix([0, 1, 1, 2], [0, 1, 2, 2], 3)
    assert cm.tolist() == [[1, 0, 0], [0, 1, 1], [0, 0, 1]]
    f, t = M.roc_points(np.array([0, 1, 0, 1]), np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6]]), 1)
    assert f[0] == 0 and t[-1] == 1 and len(f) == len(t)


def test_top_comparator_delta_in_percentage_points():
    d = M.top_comparator_delta({"proposed": {"A": 0.99, "B": 0.95}, "x": {"A": 0.98, "B": 0.96}, "y": {"A": 0.97, "B": None}})
    assert np.isclose(d["A"]["delta_pp"], 1.0) and d["A"]["best_comparator"] == "x" and np.isclose(d["B"]["delta_pp"], -1.0)


def test_mean_sd_ci_uses_sample_sd_and_t_quantile():
    s = ST.mean_sd_ci([0.9847, 0.9866, 0.9854, 0.9871, 0.9858])
    assert math.isclose(s["mean"], 0.98592, abs_tol=1e-9) and math.isclose(s["sd"], 0.00095, abs_tol=2e-5) and s["df"] == 4
    assert math.isclose(s["ci_low"], 0.9847, abs_tol=2e-4) and math.isclose(s["ci_high"], 0.9871, abs_tol=2e-4)
    assert math.isnan(ST.mean_sd_ci([1.0])["sd"])


def test_friedman_maximum_for_identical_ranks_and_holm():
    m = {f"m{i}": [10 - i + 0.1 * s for s in range(5)] for i in range(11)}
    f = ST.friedman(m)
    assert math.isclose(f["chi2"], 50.0, abs_tol=1e-9) and f["df"] == 10 and f["max_attainable_chi2"] == 50.0 and f["mean_rank"]["m0"] == 1.0
    assert f["p"] < 1e-6
    assert ST.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert ST.holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])
    with pytest.raises(ValueError):
        ST.friedman({"a": [1, 2], "b": [2, 3]})


def test_paired_comparison_variants_and_effect_sizes():
    a, b = [0.986, 0.987, 0.985, 0.988, 0.986], [0.979, 0.981, 0.979, 0.981, 0.979]
    r = ST.paired_comparison(a, b)
    assert r["test"] == "paired_t" and r["p_raw"] < 1e-3 and r["d_z"] != r["d_pooled"] and r["min_attainable_p_wilcoxon_two_sided"] == 0.0625
    assert ST.paired_comparison(a, b, "welch_t")["test"] == "welch_t"
    assert ST.paired_comparison(a, b, "wilcoxon")["p_raw"] >= 0.0625 - 1e-12             # a rank test cannot reach 1e-3 with n = 5
    with pytest.raises(ValueError):
        ST.paired_comparison(a, b, "anova")
    rows = ST.planned_vs_reference({"p": a, "x": b, "y": [0.9] * 5 + []}, "p")
    assert all(r["p_holm"] >= r["p_raw"] for r in rows) and rows[0]["significant_0.05"]


def test_efficiency_counts_and_latency_timer():
    macs = EF.encoder_macs(EncoderConfig())
    assert macs == 6144 + 8192 + 16384 + 8 * (3 * 32 * 256 + 256 * 32 + 2 * 32 * 128) + 2 * 8 * 8 * 8 * 32
    assert EF.qelm_macs(QELMConfig(n_layers=3, width=512), 256, 15) == 256 * 512 + 2 * 512 * 512 + 512 * 15
    t = EF.time_inference(lambda x: x.sum(), np.random.rand(50, 4), batch=5, warmup=2, timed=10)
    assert t["batch"] == 5 and t["ms_per_sample_mean"] > 0 and t["samples_per_s"] > 0 and "excludes data loading" in t["timer_includes"] and t["peak_gpu_mb"] is None
    with pytest.raises(ValueError):
        EF.time_inference(lambda x: x, np.zeros((2, 3)), batch=5)
    assert "torch" in EF.hardware_metadata()
