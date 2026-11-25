"""INVENTED numeric fixtures for the offline demo and tests. Not Edge-IIoTset; not a benchmark.

The generator writes a raw-CSV-shaped frame (identifier/metadata columns, constants, duplicates, missing values, an
infinity, a categorical column) so the full preparation path is exercised. Class geometry is class-conditional Gaussian
clusters with class-dependent feature subsets, deliberately noisy so no model is perfect."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema as S

FIXTURE_MARK = "SYNTHETIC FIXTURE: invented numbers that exercise the code path; NOT Edge-IIoTset and NOT a reproduction of any published result."
N_FIXTURE_FEATURES = 47          # + 1 categorical column = 48 retained features
RAW_ALIAS_STYLE = {"DDoS_HTTP": "DDos_HTTP", "SQL_Injection": "SQL_injection", "OS_Fingerprinting": "Fingerprinting", "Vulnerability_Scanner": "Vulnerability_scanner"}


def fixture_feature_names(n: int = N_FIXTURE_FEATURES):
    return [f"fx.f{i:02d}" for i in range(n)]


def make_fixture(n: int = 1500, seed: int = 0, *, imbalance: bool = True, n_features: int = N_FIXTURE_FEATURES, noise: float = 2.8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    k = len(S.CLASSES)
    probs = np.array([0.40] + [0.60 / (k - 1)] * (k - 1)) if imbalance else np.full(k, 1.0 / k)
    probs[-1] = max(probs[-1] * 0.5, 0.01)
    probs = probs / probs.sum()
    y = rng.choice(k, size=n, p=probs)
    y[:k] = np.arange(k)                                                          # guarantee every class occurs
    centres = rng.normal(0, 1.6, size=(k, n_features))
    mask = rng.random((k, n_features)) < 0.35
    X = rng.normal(0, noise, size=(n, n_features)) + centres[y] * mask[y]
    names = fixture_feature_names(n_features)
    df = pd.DataFrame(X, columns=names)
    df["fx.categorical"] = np.array(["tcp", "udp", "icmp", "http", "mqtt", "arp"])[np.clip(y % 6, 0, 5)]
    df["fx.constant"] = 1.0
    df["fx.dup_of_f00"] = df["fx.f00"]
    df["fx.mostly_missing"] = np.where(rng.random(n) < 0.7, np.nan, 1.0)
    df["frame.time"] = np.arange(n)                                               # identifier-like, removed by name
    df["ip.src_host"] = [f"10.0.{i % 255}.{i % 7}" for i in range(n)]
    df.loc[3, "fx.f01"] = np.inf
    df.loc[rng.choice(n, size=max(n // 50, 1), replace=False), "fx.f02"] = np.nan
    canon = np.array(S.CLASSES)[y]
    df["Attack_type"] = [RAW_ALIAS_STYLE.get(c, c) for c in canon]
    df["Attack_label"] = (canon != "Normal").astype(int)
    return df
