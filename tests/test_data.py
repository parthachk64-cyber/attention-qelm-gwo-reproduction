import json

import numpy as np
import pandas as pd
import pytest

from iiot_qelm.data import schema as S
from iiot_qelm.data.prepare import (PrepareConfig, Preprocessor, inverse_frequency_weights, load_prepared, load_raw, prepare_csv, prepare_frame, save_prepared, sha256_file,
                                    validate_feature_set, validate_raw)
from iiot_qelm.data.splits import (OUTER_SPLIT_SEED, assert_disjoint, cv_folds, inner_validation_split, iter_batches, outer_split)
from iiot_qelm.data.synthetic import make_fixture
from iiot_qelm.errors import LeakageError, MissingInputError, SchemaError


# ---------------------------------------------------------------- classes, aliases, labels
def test_class_list_matches_paper():
    assert len(S.CLASSES) == 15 and S.CLASSES[0] == "Normal" and S.CLASSES[-1] == "MITM"


@pytest.mark.parametrize("alias,canon", [("DDos_HTTP", "DDoS_HTTP"), ("SQL_injection", "SQL_Injection"), ("Fingerprinting", "OS_Fingerprinting"), ("Vulnerability_scanner", "Vulnerability_Scanner"),
                                          ("MTTM", "MITM"), ("man-in-the-middle", "MITM"), ("  normal ", "Normal"), ("ddos_udp", "DDoS_UDP")])
def test_aliases_resolve(alias, canon):
    assert S.canonical_label(alias) == canon


def test_unknown_label_is_an_error():
    with pytest.raises(SchemaError, match="unknown class label"):
        S.canonical_label("Teardrop")


def test_binary_and_multiclass_targets():
    lab = ["Normal", "MITM", "DDoS_UDP"]
    assert S.binary_target(lab) == [0, 1, 1] and S.multiclass_target(lab) == [0, 14, 1]
    assert S.class_names_for("binary") == ["Normal", "Attack"] and len(S.class_names_for("multiclass")) == 15
    with pytest.raises(SchemaError):
        S.target_for("ternary", lab)


def test_label_columns_never_enter_features():
    with pytest.raises(LeakageError):
        S.assert_no_label_leak(["a", "Attack_label"])
    with pytest.raises(LeakageError):
        S.assert_no_label_leak(["attack_type"])
    S.assert_no_label_leak(["a", "b"])


def test_prepared_features_exclude_both_label_columns(prepared, prepared_binary):
    for p in (prepared, prepared_binary):
        assert not set(p.manifest["feature_order"]) & set(S.LABEL_COLUMNS)


def test_exact_metadata_match_does_not_overreach():
    # upstream matches substrings: 'mqtt.msg' would also remove 'mqtt.msgtype'. Here the match is exact.
    df = make_fixture(300, 1)
    df["mqtt.msgtype"] = np.arange(len(df)) % 5
    df["mqtt.msg"] = "x"
    p = prepare_frame(df, PrepareConfig(expected_features=None))
    assert "mqtt.msgtype" in p.manifest["feature_order"] and "mqtt.msg" not in p.manifest["feature_order"]


# ---------------------------------------------------------------- raw validation
def test_validate_raw_rejects_missing_label_and_unknown_class(raw_df):
    with pytest.raises(SchemaError, match="label column"):
        validate_raw(raw_df.drop(columns=["Attack_type"]), "Attack_type")
    bad = raw_df.copy()
    bad.loc[0, "Attack_type"] = "Nonsense"
    with pytest.raises(SchemaError):
        validate_raw(bad, "Attack_type")


def test_missing_raw_file_names_the_fix(tmp_path):
    with pytest.raises(MissingInputError, match="download Edge-IIoTset"):
        load_raw(tmp_path / "nope.csv")


# ---------------------------------------------------------------- preparation / leakage controls
def test_identifier_constant_duplicate_and_missing_columns_removed(prepared):
    fo = set(prepared.manifest["feature_order"])
    assert not fo & {"frame.time", "ip.src_host", "fx.constant", "fx.dup_of_f00", "fx.mostly_missing"}
    d = prepared.manifest["dropped_by_fit"]
    assert d["constant"] == ["fx.constant"] and d["duplicate"] == ["fx.dup_of_f00"] and d["high_missing"] == ["fx.mostly_missing"]
    assert "frame.time" in prepared.manifest["schema_dropped_columns"]


def test_48_features_and_stable_order(prepared):
    assert prepared.X_train.shape[1] == prepared.X_test.shape[1] == 48 == len(prepared.manifest["feature_order"])
    assert prepared.preprocessor.feature_order == prepared.manifest["feature_order"]


def test_feature_count_mismatch_refuses_to_trim(raw_df):
    df = raw_df.copy()
    df["extra_signal"] = np.random.default_rng(0).normal(size=len(df))
    with pytest.raises(SchemaError, match="will not drop arbitrary columns"):
        prepare_frame(df, PrepareConfig())
    ok = prepare_frame(df, PrepareConfig(expected_features=None))
    assert ok.X_train.shape[1] == 49 and ok.manifest["feature_validation"]["expected"] is None


def test_authoritative_feature_list_is_enforced(raw_df, prepared):
    names = list(prepared.manifest["feature_order"])
    assert validate_feature_set(names, PrepareConfig(feature_list=names))["order_source"] == "authoritative list"
    with pytest.raises(SchemaError, match="differ from the authoritative list"):
        validate_feature_set(names[:-1], PrepareConfig(feature_list=names))
    p = prepare_frame(raw_df, PrepareConfig(feature_list=names))
    assert p.manifest["feature_validation"]["authoritative_list"] is True


def test_scaler_and_statistics_are_fit_on_training_rows_only(raw_df):
    p = prepare_frame(raw_df, PrepareConfig())
    changed = raw_df.copy()
    test_ids = set(p.ids_test)
    from iiot_qelm.data.prepare import row_hashes
    # perturb ONLY rows that landed in the test partition, refit, and compare the fitted statistics
    feats = changed.drop(columns=[c for c in changed.columns if c in S.LABEL_COLUMNS or c in S.METADATA_EXACT])
    mask = np.isin(row_hashes(feats), list(test_ids))
    changed.loc[mask, "fx.f05"] = 1e6
    p2 = prepare_frame(changed, PrepareConfig())
    assert p2.preprocessor.hi["fx.f05"] == p.preprocessor.hi["fx.f05"]
    assert p2.preprocessor.medians["fx.f05"] == p.preprocessor.medians["fx.f05"]


def test_test_values_outside_training_range_are_clipped(prepared):
    pre = prepared.preprocessor
    row = pd.DataFrame([{c: (pre.hi[c] + 100.0 if c in pre.numeric else pre.categories[c][0]) for c in pre.feature_order}])
    assert np.all((pre.transform(row) >= 0) & (pre.transform(row) <= 1))


def test_unseen_category_gets_reserved_code_and_stays_finite(prepared):
    pre = prepared.preprocessor
    base = {c: (pre.medians[c] if c in pre.numeric else pre.categories[c][0]) for c in pre.feature_order}
    row = pd.DataFrame([dict(base, **{"fx.categorical": "never_seen_protocol"})])
    x = pre.transform(row)
    assert np.isfinite(x).all()
    assert pre._encode(row)["fx.categorical"].iloc[0] == len(pre.categories["fx.categorical"])


def test_all_missing_column_is_dropped_and_missing_values_imputed():
    df = make_fixture(400, 2)
    df["fx.allnan"] = np.nan
    p = prepare_frame(df, PrepareConfig(expected_features=None))
    assert "fx.allnan" in p.manifest["dropped_by_fit"]["high_missing"]
    assert np.isfinite(p.X_train).all() and np.isfinite(p.X_test).all()


def test_infinity_replaced_by_finite_training_maximum(prepared):
    pre = prepared.preprocessor
    base = {c: (pre.medians[c] if c in pre.numeric else pre.categories[c][0]) for c in pre.feature_order}
    row = pd.DataFrame([dict(base, **{"fx.f01": np.inf})])
    assert pre._encode(row)["fx.f01"].iloc[0] == pre.finite_max["fx.f01"]


def test_missing_fitted_column_at_inference_is_an_error(prepared):
    with pytest.raises(SchemaError, match="missing fitted feature"):
        prepared.preprocessor.transform(pd.DataFrame({"fx.f00": [0.1]}))


def test_duplicates_are_removed_and_never_cross_partitions(raw_df):
    dup = pd.concat([raw_df, raw_df.sample(200, random_state=1)], ignore_index=True)
    p = prepare_frame(dup, PrepareConfig())
    assert p.manifest["counts"]["duplicates_removed"] >= 200
    assert not set(p.ids_train) & set(p.ids_test)
    assert len(set(p.ids_train)) == len(p.ids_train)


def test_conflicting_label_groups_are_audited_and_dropped(raw_df):
    df = raw_df.copy()
    twin = df.iloc[[0]].copy()
    twin["Attack_type"] = "MITM" if df.iloc[0]["Attack_type"] != "MITM" else "XSS"
    p = prepare_frame(pd.concat([df, twin], ignore_index=True), PrepareConfig())
    a = p.manifest["conflicting_label_audit"]
    assert a["groups"] == 1 and a["rows"] == 2 and a["policy"] == "drop"
    with pytest.raises(LeakageError):
        prepare_frame(pd.concat([df, twin], ignore_index=True), PrepareConfig(conflict_policy="error"))


def test_class_weights_are_inverse_frequency_with_mean_one():
    w = inverse_frequency_weights([0] * 90 + [1] * 9 + [2])
    assert w[2] > w[1] > w[0] and np.isclose(np.mean(list(w.values())), 1.0)


def test_manifest_records_required_provenance(prepared):
    m = prepared.manifest
    for k in ("counts", "feature_order", "split_fingerprint", "class_histogram_train", "class_histogram_test", "conflicting_label_audit", "schema_dropped_columns", "outer_split_seed"):
        assert k in m
    assert m["outer_split_seed"] == OUTER_SPLIT_SEED == 7


def test_csv_pipeline_records_file_hash_and_round_trips(tmp_path, raw_df):
    csv = tmp_path / "flows.csv"
    raw_df.to_csv(csv, index=False)
    p = prepare_csv(csv, PrepareConfig())
    assert p.manifest["source"]["sha256"] == sha256_file(csv) and p.manifest["source"]["bytes"] == csv.stat().st_size and p.manifest["source"]["filename"] == "flows.csv"
    save_prepared(p, tmp_path / "prep")
    q = load_prepared(tmp_path / "prep")
    assert np.array_equal(q.X_test, p.X_test) and q.class_weights == p.class_weights and q.preprocessor.feature_order == p.preprocessor.feature_order
    json.loads((tmp_path / "prep" / "manifest.json").read_text())
    with pytest.raises(MissingInputError):
        load_prepared(tmp_path / "absent")


# ---------------------------------------------------------------- splits
def test_outer_split_is_shared_by_both_tasks(prepared, prepared_binary):
    assert np.array_equal(prepared.ids_train, prepared_binary.ids_train) and np.array_equal(prepared.ids_test, prepared_binary.ids_test)
    assert prepared.manifest["split_fingerprint"] == prepared_binary.manifest["split_fingerprint"]


def test_outer_split_properties():
    y = np.repeat(np.arange(5), 40)
    s = outer_split(y)
    assert len(s.test_idx) == 40 and s.seed == 7
    assert np.array_equal(outer_split(y).test_idx, s.test_idx)
    assert not set(s.train_idx) & set(s.test_idx)
    assert s.fingerprint(np.arange(200).astype(str)) == outer_split(y).fingerprint(np.arange(200).astype(str))
    assert outer_split(y, seed=8).fingerprint(np.arange(200).astype(str)) != s.fingerprint(np.arange(200).astype(str))
    with pytest.raises(LeakageError):
        assert_disjoint([1, 2], [2, 3])


def test_inner_validation_and_cv_folds_use_only_training_positions():
    y = np.repeat(np.arange(4), 50)
    fit, val = inner_validation_split(y, fraction=0.1, seed=1)
    assert not set(fit) & set(val) and len(fit) + len(val) == 200
    folds = cv_folds(y, n_folds=5, seed=0)
    allv = np.concatenate([v for _, v in folds])
    assert sorted(allv) == list(range(200))
    for f, v in folds:
        assert not set(f) & set(v)
    rare = np.array([0] * 30 + [1] * 2)
    assert len(cv_folds(rare, n_folds=5)) == 2          # fewer folds than classes allow: reduced, never an error
    fit2, _ = inner_validation_split(np.array([0] * 20 + [1]), fraction=0.2, seed=0)   # singleton class: unstratified fallback
    assert len(fit2) > 0
    assert [s for s in iter_batches(5, 2)] == [slice(0, 2), slice(2, 4), slice(4, 5)]
