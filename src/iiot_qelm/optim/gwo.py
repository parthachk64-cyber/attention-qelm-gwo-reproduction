"""Grey Wolf Optimizer over the seven-dimensional QELM space (Sec. 2.5).

Search contract (also rendered into docs by ``iiot-qelm docs``):
  dim 0 layers L      continuous in [1, 5]        -> round half up, clipped
  dim 1 width n_h     continuous in [64, 1024]    -> round half up, clipped
  dim 2 activation    continuous in [0, 4]        -> floor, x == 4 -> index 3; index into (sigmoid, tanh, relu, gelu)
  dim 3 log10 lambda1 continuous in [-6, -1]      -> 10 ** x
  dim 4 log10 lambda2 continuous in [-6, -1]      -> 10 ** x
  dim 5 theta         continuous in [0, 2 pi]
  dim 6 rho           continuous in [0, 1]
Positions stay continuous; decoding (rounding/indexing) happens only when a candidate is evaluated. Positions are clipped to the bounds
after every update. Maximisation. Leaders: alpha/beta/delta are the three best DISTINCT positions seen so far (stable tie-break by
first occurrence). Stopping: ``patience`` consecutive iterations whose best fitness does not beat the running best by more than ``tol``.
Candidate evaluations: N for the initial population plus N per iteration, i.e. at most N (T + 1) = 1,020 for N=20, T=50 (the article's
nominal 1,000 = N T); each candidate evaluation costs ``n_folds`` fold fits. Candidates whose DECODED configuration was already evaluated
reuse the memoised fitness (counted in ``cache_hits``) because the fitness is deterministic for a fixed seed (common random numbers).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from ..models.qelm import ACTIVATIONS

BOUNDS = np.array([[1.0, 5.0], [64.0, 1024.0], [0.0, 4.0], [-6.0, -1.0], [-6.0, -1.0], [0.0, 2.0 * math.pi], [0.0, 1.0]])
DIM_NAMES = ("n_layers", "width", "activation", "log10_lambda1", "log10_lambda2", "theta", "rho")


def _round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def decode(pos) -> dict:
    p = np.clip(np.asarray(pos, dtype=float), BOUNDS[:, 0], BOUNDS[:, 1])
    return {"n_layers": int(np.clip(_round_half_up(p[0]), 1, 5)), "width": int(np.clip(_round_half_up(p[1]), 64, 1024)),
            "activation": ACTIVATIONS[min(int(math.floor(p[2])), 3)], "lambda1": float(10.0 ** p[3]), "lambda2": float(10.0 ** p[4]),
            "theta": float(p[5]), "rho": float(p[6])}


def decode_key(cfg: dict) -> tuple:
    return (cfg["n_layers"], cfg["width"], cfg["activation"], round(cfg["lambda1"], 12), round(cfg["lambda2"], 12), round(cfg["theta"], 9), round(cfg["rho"], 9))


@dataclass
class GWOResult:
    best_position: np.ndarray
    best_config: dict
    best_fitness: float
    iterations_run: int
    stopped_early: bool
    stop_reason: str
    best_history: List[float]
    candidates: List[dict]
    n_candidate_evaluations: int
    cache_hits: int
    settings: dict = field(default_factory=dict)


class GreyWolfOptimizer:
    def __init__(self, *, n_wolves: int = 20, max_iter: int = 50, patience: int = 10, tol: float = 1e-4, seed: int = 0, bounds=BOUNDS):
        if n_wolves < 3:
            raise ValueError("GWO needs at least three wolves (alpha, beta, delta)")
        self.N, self.T, self.patience, self.tol = n_wolves, max_iter, patience, tol
        self.bounds = np.asarray(bounds, float)
        self.rng = np.random.default_rng(seed)
        self.seed = seed

    def _clip(self, x):
        return np.clip(x, self.bounds[:, 0], self.bounds[:, 1])

    def optimise(self, fitness: Callable[[dict], "float | tuple"], on_iteration: Optional[Callable[[int, dict], None]] = None) -> GWOResult:
        """``fitness(decoded_config) -> float`` or ``(float, fold_scores)``. Higher is better."""
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        pop = self.rng.uniform(lo, hi, size=(self.N, len(lo)))
        memo: Dict[tuple, tuple] = {}
        cands: List[dict] = []
        hits = 0

        def evaluate(it: int, i: int, pos) -> float:
            nonlocal hits
            cfg = decode(pos)
            key = decode_key(cfg)
            cached = key in memo
            if cached:
                hits += 1
                f, folds = memo[key]
            else:
                out = fitness(cfg)
                f, folds = (float(out[0]), list(out[1])) if isinstance(out, tuple) else (float(out), [])
                memo[key] = (f, folds)
            cands.append({"iteration": it, "wolf": i, "position": [float(v) for v in pos], "config": cfg, "fitness": f, "fold_scores": folds, "memoised": cached})
            return f

        fit = np.array([evaluate(0, i, p) for i, p in enumerate(pop)])
        pool_pos, pool_fit = pop.copy(), fit.copy()

        def leaders(P, F):
            order = np.argsort(-F, kind="stable")
            chosen, seen = [], set()
            for j in order:
                k = tuple(np.round(P[j], 12))
                if k not in seen:
                    seen.add(k)
                    chosen.append(j)
                if len(chosen) == 3:
                    break
            while len(chosen) < 3:                         # degenerate pool: reuse the best
                chosen.append(order[0])
            return P[chosen].copy(), F[chosen].copy()

        (LP, LF) = leaders(pool_pos, pool_fit)
        best_hist = [float(LF[0])]
        running_best, stagnant, stop, it = float(LF[0]), 0, "max_iter", 0
        for it in range(1, self.T + 1):
            a = 2.0 * (1.0 - it / self.T)
            new = np.empty_like(pop)
            for i in range(self.N):
                acc = np.zeros(len(lo))
                for lead in LP:
                    r1, r2 = self.rng.random(len(lo)), self.rng.random(len(lo))
                    A, C = 2.0 * a * r1 - a, 2.0 * r2
                    acc += lead - A * np.abs(C * lead - pop[i])
                new[i] = self._clip(acc / 3.0)
            pop = new
            fit = np.array([evaluate(it, i, p) for i, p in enumerate(pop)])
            pool_pos, pool_fit = np.vstack([LP, pop]), np.concatenate([LF, fit])
            LP, LF = leaders(pool_pos, pool_fit)
            best_hist.append(float(LF[0]))
            if LF[0] > running_best + self.tol:
                running_best, stagnant = float(LF[0]), 0
            else:
                stagnant += 1
            if on_iteration:
                on_iteration(it, {"alpha": float(LF[0]), "beta": float(LF[1]), "delta": float(LF[2]), "stagnant": stagnant})
            if stagnant >= self.patience:
                stop = "stagnation"
                break
        return GWOResult(LP[0], decode(LP[0]), float(LF[0]), it, stop == "stagnation", stop, best_hist, cands, len(cands), hits,
                         {"n_wolves": self.N, "max_iter": self.T, "patience": self.patience, "tol": self.tol, "seed": self.seed,
                          "nominal_budget_N_x_T": self.N * self.T, "max_candidate_evaluations_N_x_Tplus1": self.N * (self.T + 1)})
