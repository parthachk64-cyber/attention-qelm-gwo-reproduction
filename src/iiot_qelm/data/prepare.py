"""Raw-file validation and the Sec. 2.2 preprocessing pipeline, with training-only fitting.

Order of operations (each step records row/column counts in the manifest):
  1. validate the raw CSV (label column present, no unknown classes, label columns never become features);
  2. schema-level column removal (labels, identifier/metadata columns: exact names, no data statistics used);
  3. hash-deduplicate on the retained feature columns + label; audit and resolve identical-feature/different-label groups;
  4. stratified outer 80/20 split (seed 7);
  5. FIT on the training partition only: drop constant/duplicate/>40%-missing columns, finite maxima, medians, modes,
     categorical codes (unseen -> reserved code), min-max scaling, class weights;
  6. validate the retained feature set against an authoritative list and the expected count (48).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..errors import LeakageError, MissingInputError, SchemaError
from . import schema as S
from .splits import OUTER_SPLIT_SEED, OUTER_TEST_FRACTION, assert_disjoint, outer_split

EXPECTED_FEATURES = 48
MISSING_THRESHOLD = 0.40


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def row_hashes(df: pd.DataFrame) -> np.ndarray:
    """Stable 16-hex row ids from the retained feature values + label (used as immutable row IDs)."""
    h = pd.util.hash_pandas_object(df, index=False).to_numpy(dtype="uint64")
    return np.array([f"{int(x):016x}" for x in h])


@dataclass
class PrepareConfig:
    task: str = "multiclass"
    label_col: str = "Attack_type"
    test_fraction: float = OUTER_TEST_FRACTION
    split_seed: int = OUTER_SPLIT_SEED
    missing_threshold: float = MISSING_THRESHOLD
    expected_features: Optional[int] = EXPECTED_FEATURES
    feature_list: Optional[Sequence[str]] = None
    metadata_columns: Sequence[str] = S.METADATA_EXACT
    conflict_policy: str = "drop"          # drop | keep_first | error


# ------------------------------------------------------------------ raw validation
def validate_raw(df: pd.DataFrame, label_col: str) -> pd.Series:
    if label_col not in df.columns:
        raise SchemaError(f"label column {label_col!r} not in file; columns present: {list(df.columns)[:12]}...")
    return pd.Series(S.canonicalise(df[label_col].astype(str)), index=df.index, name="canon")


def load_raw(path) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        raise MissingInputError(f"raw CSV {p}", "download Edge-IIoTset (docs/data.md), then pass --csv <path to the ML-ready flow CSV>")
    return pd.read_csv(p, low_memory=False)


# ------------------------------------------------------------------ the fitted transformer
@dataclass
class Preprocessor:
    feature_order: List[str] = field(default_factory=list)
    numeric: List[str] = field(default_factory=list)
    categorical: List[str] = field(default_factory=list)
    finite_max: Dict[str, float] = field(default_factory=dict)
    finite_min: Dict[str, float] = field(default_factory=dict)
    medians: Dict[str, float] = field(default_factory=dict)
    modes: Dict[str, str] = field(default_factory=dict)
    categories: Dict[str, List[str]] = field(default_factory=dict)
    lo: Dict[str, float] = field(default_factory=dict)
    hi: Dict[str, float] = field(default_factory=dict)
    dropped: Dict[str, List[str]] = field(default_factory=dict)

    # ---- fit (training rows only)
    def fit(self, train: pd.DataFrame, *, missing_threshold: float = MISSING_THRESHOLD) -> "Preprocessor":
        cols = list(train.columns)
        miss = train.isna().mean()
        high_missing = [c for c in cols if miss[c] > missing_threshold]            # includes all-missing columns
        rest = [c for c in cols if c not in high_missing]
        constant = [c for c in rest if train[c].nunique(dropna=False) <= 1]
        rest = [c for c in rest if c not in constant]
        seen, dup = {}, []
        for c in rest:
            key = pd.util.hash_pandas_object(train[c].astype(str), index=False).to_numpy().tobytes()
            if key in seen and train[c].astype(str).equals(train[seen[key]].astype(str)):
                dup.append(c)
            else:
                seen[key] = c
        rest = [c for c in rest if c not in dup]
        self.dropped = {"high_missing": high_missing, "constant": constant, "duplicate": dup}
        self.feature_order = rest
        self.numeric = [c for c in rest if pd.api.types.is_numeric_dtype(train[c])]
        self.categorical = [c for c in rest if c not in self.numeric]
        for c in self.numeric:
            col = train[c].replace([np.inf, -np.inf], np.nan)
            fin = col.dropna()
            self.finite_max[c] = float(fin.max()) if len(fin) else 0.0
            self.finite_min[c] = float(fin.min()) if len(fin) else 0.0
            self.medians[c] = float(fin.median()) if len(fin) else 0.0
        for c in self.categorical:
            s = train[c].astype(str).where(train[c].notna())
            m = s.mode(dropna=True)
            self.modes[c] = str(m.iloc[0]) if len(m) else "__missing__"
            self.categories[c] = sorted(set(s.fillna(self.modes[c]).astype(str)))
        enc = self._encode(train)
        self.lo = {c: float(enc[c].min()) for c in self.feature_order}
        self.hi = {c: float(enc[c].max()) for c in self.feature_order}
        return self

    # ---- transform (never refits)
    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.feature_order if c not in df.columns]
        if missing:
            raise SchemaError(f"input is missing fitted feature column(s): {missing}")
        out = pd.DataFrame(index=df.index)
        for c in self.numeric:
            col = pd.to_numeric(df[c], errors="coerce")
            col = col.where(col != np.inf, self.finite_max[c]).where(col != -np.inf, self.finite_min[c])
            out[c] = col.fillna(self.medians[c]).astype(float)
        for c in self.categorical:
            s = df[c].astype(str).where(df[c].notna(), self.modes[c])
            idx = {k: i for i, k in enumerate(self.categories[c])}
            out[c] = s.map(lambda v: idx.get(v, len(idx))).astype(float)           # unseen -> reserved code len(classes)
        return out[self.feature_order]

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        enc = self._encode(df)
        lo = np.array([self.lo[c] for c in self.feature_order])
        hi = np.array([self.hi[c] for c in self.feature_order])
        span = np.where(hi - lo > 0, hi - lo, 1.0)
        return np.clip((enc.to_numpy(dtype=np.float64) - lo) / span, 0.0, 1.0).astype(np.float32)

    # ---- persistence
    def to_json(self) -> dict:
        return {k: getattr(self, k) for k in ("feature_order", "numeric", "categorical", "finite_max", "finite_min", "medians", "modes",
                                              "categories", "lo", "hi", "dropped")}

    @classmethod
    def from_json(cls, d: dict) -> "Preprocessor":
        return cls(**d)


def inverse_frequency_weights(y) -> Dict[int, float]:
    """Inverse-frequency class weights normalised to mean 1 over the classes present in training."""
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    raw = counts.sum() / (len(classes) * counts)
    raw = raw / raw.mean()
    return {int(c): float(w) for c, w in zip(classes, raw)}


# ------------------------------------------------------------------ feature-set validation
def validate_feature_set(retained: Sequence[str], cfg: PrepareConfig) -> dict:
    S.assert_no_label_leak(retained)
    status = {"retained": len(retained), "expected": cfg.expected_features, "authoritative_list": cfg.feature_list is not None}
    if cfg.feature_list is not None:
        want, got = list(cfg.feature_list), list(retained)
        if set(want) != set(got):
            raise SchemaError(f"retained features differ from the authoritative list: missing {sorted(set(want) - set(got))}, extra {sorted(set(got) - set(want))}")
        status["order_source"] = "authoritative list"
        return status
    if cfg.expected_features is not None and len(retained) != cfg.expected_features:
        raise SchemaError(
            f"{len(retained)} features retained but the paper reports {cfg.expected_features}. No authoritative 48-column list is published "
            f"with the article or the upstream repository at the pinned revision, so this tool will not drop arbitrary columns to reach 48. "
            f"Supply --feature-list <json list of the 48 column names> (or --expected-features 0 to accept the rule-derived set, recorded in the manifest).")
    status["order_source"] = "source-file column order"
    return status


# ------------------------------------------------------------------ the pipeline
@dataclass
class Prepared:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    y15_train: np.ndarray
    y15_test: np.ndarray
    ids_train: np.ndarray
    ids_test: np.ndarray
    preprocessor: Preprocessor
    class_weights: Dict[int, float]
    manifest: dict


def prepare_frame(df: pd.DataFrame, cfg: PrepareConfig, *, source: Optional[dict] = None) -> Prepared:
    manifest: dict = {"source": source or {}, "task": cfg.task, "config": {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in cfg.__dict__.items() if k != "feature_list"},
                      "counts": {"raw_rows": int(len(df)), "raw_columns": int(df.shape[1])}}
    canon = validate_raw(df, cfg.label_col)
    manifest["raw_class_histogram"] = {k: int(v) for k, v in canon.value_counts().items()}
    drop_now = [c for c in df.columns if c in cfg.metadata_columns or c in S.LABEL_COLUMNS or c == cfg.label_col]
    feats = df.drop(columns=drop_now)
    manifest["schema_dropped_columns"] = sorted(drop_now)
    frame = feats.copy()
    frame["__canon"] = canon.to_numpy()

    keycols = list(feats.columns) + ["__canon"]
    n0 = len(frame)
    frame = frame.drop_duplicates(subset=keycols, keep="first")
    manifest["counts"]["after_exact_dedup"] = int(len(frame))
    dup_feat = frame.duplicated(subset=list(feats.columns), keep=False)
    n_conflict_rows = int(dup_feat.sum())
    n_conflict_groups = int(frame[dup_feat].groupby(list(feats.columns), dropna=False).ngroups) if n_conflict_rows else 0
    manifest["conflicting_label_audit"] = {"groups": n_conflict_groups, "rows": n_conflict_rows, "policy": cfg.conflict_policy}
    if n_conflict_rows:
        if cfg.conflict_policy == "error":
            raise LeakageError(f"{n_conflict_groups} identical-feature groups carry different labels")
        frame = frame[~dup_feat] if cfg.conflict_policy == "drop" else frame.drop_duplicates(subset=list(feats.columns), keep="first")
    manifest["counts"]["after_conflict_resolution"] = int(len(frame))
    manifest["counts"]["duplicates_removed"] = int(n0 - len(frame))

    canon_arr = frame.pop("__canon").to_numpy()
    y15 = np.array(S.multiclass_target(canon_arr))
    y = np.array(S.target_for(cfg.task, canon_arr))
    ids = row_hashes(frame)
    if len(set(ids)) != len(ids):
        raise LeakageError("row ids are not unique after de-duplication")
    split = outer_split(y15, seed=cfg.split_seed, test_fraction=cfg.test_fraction)
    assert_disjoint(ids[split.train_idx], ids[split.test_idx], "train/test row ids")
    tr, te = frame.iloc[split.train_idx], frame.iloc[split.test_idx]
    pre = Preprocessor().fit(tr, missing_threshold=cfg.missing_threshold)
    manifest["feature_validation"] = validate_feature_set(pre.feature_order, cfg)
    manifest["feature_order"] = list(pre.feature_order)
    manifest["dropped_by_fit"] = pre.dropped
    manifest["counts"].update({"train_rows": int(len(tr)), "test_rows": int(len(te)), "features": len(pre.feature_order)})
    manifest["class_histogram_train"] = {k: int(v) for k, v in zip(*np.unique(y15[split.train_idx], return_counts=True))}
    manifest["class_histogram_test"] = {k: int(v) for k, v in zip(*np.unique(y15[split.test_idx], return_counts=True))}
    manifest["class_histogram_train"] = {S.CLASSES[int(k)]: v for k, v in manifest["class_histogram_train"].items()}
    manifest["class_histogram_test"] = {S.CLASSES[int(k)]: v for k, v in manifest["class_histogram_test"].items()}
    manifest["split_fingerprint"] = split.fingerprint(ids)
    manifest["outer_split_seed"] = split.seed
    return Prepared(pre.transform(tr), pre.transform(te), y[split.train_idx], y[split.test_idx], y15[split.train_idx], y15[split.test_idx],
                    ids[split.train_idx], ids[split.test_idx], pre, inverse_frequency_weights(y[split.train_idx]), manifest)


def prepare_csv(csv_path, cfg: PrepareConfig) -> Prepared:
    df = load_raw(csv_path)
    p = Path(csv_path)
    src = {"filename": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)}
    return prepare_frame(df, cfg, source=src)


def save_prepared(p: Prepared, out_dir) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "arrays.npz", X_train=p.X_train, X_test=p.X_test, y_train=p.y_train, y_test=p.y_test, y15_train=p.y15_train,
                        y15_test=p.y15_test, ids_train=p.ids_train, ids_test=p.ids_test)
    (out / "preprocessor.json").write_text(json.dumps(p.preprocessor.to_json(), indent=1), encoding="utf-8")
    m = dict(p.manifest, class_weights={str(k): v for k, v in p.class_weights.items()})
    (out / "manifest.json").write_text(json.dumps(m, indent=1, default=str), encoding="utf-8")
    return out


def load_prepared(out_dir) -> Prepared:
    out = Path(out_dir)
    if not (out / "arrays.npz").is_file():
        raise MissingInputError(f"prepared data under {out}", "run `iiot-qelm data prepare --csv <raw.csv> --out <dir>` (or `iiot-qelm demo` for the fixture)")
    z = np.load(out / "arrays.npz", allow_pickle=False)
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    pre = Preprocessor.from_json(json.loads((out / "preprocessor.json").read_text(encoding="utf-8")))
    cw = {int(k): float(v) for k, v in m.pop("class_weights").items()}
    return Prepared(z["X_train"], z["X_test"], z["y_train"], z["y_test"], z["y15_train"], z["y15_test"], z["ids_train"], z["ids_test"], pre, cw, m)
