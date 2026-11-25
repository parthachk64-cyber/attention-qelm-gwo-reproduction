"""Seed-level statistics: mean, sample SD, Student-t CI, Friedman, planned paired comparisons with Holm correction.

Statistical notes (also in docs/paper-audit.md):
  * Five seeds on ONE dataset and ONE outer split measure training variability, not generalisation across datasets/splits.
  * Friedman: models are the treatments, seeds the blocks. With 5 blocks and 11 treatments the largest attainable statistic is
    chi2 = 50.00 (identical rank order in every seed); p = chi2.sf(50, 10).
  * Planned comparisons: ``paired_t`` (two-sided paired t-test on seed differences, df = n-1) and ``welch_t`` (independent samples) are both
    provided; the paired test is the repository default. Effect sizes: Cohen's d with the pooled SD of the two models
    (``d_pooled``) and the paired ``d_z`` (mean difference / SD of differences) are reported SEPARATELY, because the article does not name
    its formula. Holm correction is applied over the family of planned comparisons against the proposed model.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np
from scipy import stats as sps


def mean_sd_ci(values: Sequence[float], conf: float = 0.95) -> dict:
    v = np.asarray(values, float)
    n = len(v)
    m = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else float("nan")
    if n > 1:
        h = float(sps.t.ppf(0.5 + conf / 2, n - 1) * sd / math.sqrt(n))
    else:
        h = float("nan")
    return {"n": n, "mean": m, "sd": sd, "ci_low": m - h, "ci_high": m + h, "df": n - 1}


def friedman(matrix: Dict[str, Sequence[float]]) -> dict:
    """``matrix[model] = values over seeds``. Returns chi2, df, p, mean ranks (rank 1 = highest value)."""
    names = list(matrix)
    A = np.array([matrix[m] for m in names], float)             # (k models, n blocks)
    k, n = A.shape
    if k < 3 or n < 2:
        raise ValueError("Friedman needs >= 3 models and >= 2 seeds")
    ranks = np.array([sps.rankdata(-A[:, j]) for j in range(n)]).T          # rank 1 = best
    mean_ranks = ranks.mean(axis=1)
    chi2, p = sps.friedmanchisquare(*[A[i] for i in range(k)])
    return {"chi2": float(chi2), "df": k - 1, "p": float(p), "n_blocks": n, "n_models": k, "mean_rank": dict(zip(names, map(float, mean_ranks))),
            "max_attainable_chi2": float(12 * n * sum(j * j for j in range(1, k + 1)) / (k * (k + 1)) - 3 * n * (k + 1))}


def holm(pvals: Sequence[float]) -> List[float]:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def paired_comparison(a: Sequence[float], b: Sequence[float], test: str = "paired_t") -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    sd_d = float(d.std(ddof=1))
    pooled = math.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2.0)
    if test == "paired_t":
        t, p = sps.ttest_rel(a, b)
    elif test == "welch_t":
        t, p = sps.ttest_ind(a, b, equal_var=False)
    elif test == "wilcoxon":
        res = sps.wilcoxon(a, b, alternative="two-sided", method="exact")
        t, p = float("nan"), float(res.pvalue)
    else:
        raise ValueError("test must be paired_t, welch_t or wilcoxon")
    return {"mean_diff": float(d.mean()), "t": float(t), "p_raw": float(p), "test": test, "n": n,
            "d_pooled": float(d.mean() / pooled) if pooled > 0 else float("inf"), "d_z": float(d.mean() / sd_d) if sd_d > 0 else float("inf"),
            "min_attainable_p_wilcoxon_two_sided": float(2.0 / 2 ** n)}


def planned_vs_reference(scores: Dict[str, Sequence[float]], reference: str, test: str = "paired_t") -> List[dict]:
    rows = [dict(comparator=m, **paired_comparison(scores[reference], scores[m], test)) for m in scores if m != reference]
    adj = holm([r["p_raw"] for r in rows])
    for r, a in zip(rows, adj):
        r["p_holm"], r["significant_0.05"] = a, a < 0.05
    return rows
