"""Quantum-inspired extreme learning machine (classical simulation; no quantum hardware).

Hidden layers (fixed, seeded, never trained):   h_l = f(h_{l-1} W_l + b_l)
Quantum-inspired weights:                       W_ij = rho cos(phi_ij) + (1 - rho) sin(phi_ij),  phi_ij ~ U(0, 2 pi)
Output head (closed form, both penalties ridge): beta = (H^T H + lambda1 I + lambda2 I)^-1 H^T T      (printed equation)

Underspecified details and what this code does (all recorded in the saved bundle):
  * global angle theta: the article lists theta as a tuned hyperparameter but does not say how it enters. The upstream code
    adds it to every sampled angle (``phase_shift``). Because phi ~ U(0, 2 pi) is invariant to a constant shift modulo 2 pi,
    theta then changes WHICH weights are drawn for a given seed but not their DISTRIBUTION, so it is distributionally inert;
    ``theta_inertness`` measures this. ``theta_mode='none'`` ignores theta.
  * weight scale / bias: not given in the article; upstream scales by sqrt(2 / fan_in) and draws biases from 0.1 * U(-1, 1).
  * lambda1 and lambda2 are both L2 (identity) penalties here because the printed equation adds both times I. Calling
    lambda1 an "L1" coefficient would be wrong: an identity penalty is not sparsity-inducing.
  * stacked layers: the output of layer l-1 feeds layer l (upstream convention).
  * scores -> probabilities: regression outputs of one-hot targets are unbounded; ``predict_proba`` applies softmax (upstream),
    which is a monotone-per-row convenience, not a calibrated probability.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve, lstsq
from scipy.special import erf

ACTIVATIONS = ("sigmoid", "tanh", "relu", "gelu")


def activate(name: str, x: np.ndarray) -> np.ndarray:
    if name == "sigmoid":
        return 0.5 * (1.0 + np.tanh(0.5 * x))                      # overflow-free logistic
    if name == "tanh":
        return np.tanh(x)
    if name == "relu":
        return np.maximum(x, 0.0)
    if name == "gelu":
        return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))             # exact GELU (upstream uses the tanh approximation)
    raise ValueError(f"unknown activation {name!r}; choose from {ACTIVATIONS}")


@dataclass
class QELMConfig:
    n_layers: int = 3
    width: int = 512
    activation: str = "gelu"
    lambda1: float = 3.2e-4
    lambda2: float = 1.7e-3
    theta: float = 1.84
    rho: float = 0.63
    init: str = "quantum"             # quantum | classical
    theta_mode: str = "phase_shift"   # phase_shift (upstream) | none
    weight_scale: str = "he"          # he | none
    bias_scale: float = 0.1
    seed: int = 0
    solve_batch: int = 8192

    def validate(self):
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation {self.activation!r} not in {ACTIVATIONS}")
        if not 0.0 <= self.rho <= 1.0:
            raise ValueError("rho must lie in [0, 1]")
        if self.n_layers < 1 or self.width < 1:
            raise ValueError("n_layers and width must be positive")
        if self.lambda1 < 0 or self.lambda2 < 0:
            raise ValueError("regularisers must be non-negative")
        if self.init not in ("quantum", "classical") or self.theta_mode not in ("phase_shift", "none") or self.weight_scale not in ("he", "none"):
            raise ValueError("bad init/theta_mode/weight_scale")


def quantum_weights(rng: np.random.Generator, fan_in: int, fan_out: int, rho: float, theta: float = 0.0, theta_mode: str = "phase_shift",
                    scale: bool = True) -> np.ndarray:
    phi = rng.uniform(0.0, 2.0 * np.pi, size=(fan_in, fan_out))
    if theta_mode == "phase_shift":
        phi = phi + theta
    w = rho * np.cos(phi) + (1.0 - rho) * np.sin(phi)
    return w * np.sqrt(2.0 / fan_in) if scale else w


def theta_inertness(rho: float = 0.63, thetas=(0.0, 1.84), n: int = 400_000, seed: int = 0) -> dict:
    """Empirical (mean, std) of the sampled weights for different theta: equal up to Monte-Carlo error => theta is inert."""
    out = {}
    for t in thetas:
        w = quantum_weights(np.random.default_rng(seed + 1), n, 1, rho, t, "phase_shift", scale=False).ravel()
        out[float(t)] = {"mean": float(w.mean()), "std": float(w.std())}
    return out


class QELM:
    def __init__(self, cfg: Optional[QELMConfig] = None):
        self.cfg = cfg or QELMConfig()
        self.cfg.validate()
        self.W: List[np.ndarray] = []
        self.b: List[np.ndarray] = []
        self.beta: Optional[np.ndarray] = None
        self.n_classes: Optional[int] = None
        self.solver_info: dict = {}

    # ------------------------------------------------------------ fixed random hidden layers
    def _build(self, input_dim: int) -> None:
        c = self.cfg
        rng = np.random.default_rng(c.seed)
        self.W, self.b, fan = [], [], input_dim
        for _ in range(c.n_layers):
            if c.init == "quantum":
                W = quantum_weights(rng, fan, c.width, c.rho, c.theta, c.theta_mode, scale=c.weight_scale == "he")
            else:
                W = rng.standard_normal((fan, c.width)) * (np.sqrt(2.0 / fan) if c.weight_scale == "he" else 1.0)
            self.W.append(W.astype(np.float64))
            self.b.append(rng.uniform(-1.0, 1.0, size=c.width) * c.bias_scale)
            fan = c.width

    def hidden(self, X: np.ndarray) -> np.ndarray:
        if not self.W:
            raise RuntimeError("QELM hidden layers are not built; call fit first")
        h = np.asarray(X, dtype=np.float64)
        for W, b in zip(self.W, self.b):
            h = activate(self.cfg.activation, h @ W + b)
        return h

    # ------------------------------------------------------------ fitting
    def fit_batches(self, batches: Iterable[Tuple[np.ndarray, np.ndarray]], n_classes: int, input_dim: int) -> "QELM":
        """Accumulate H^T H and H^T T over batches (float64) and solve once. Memory is O(width^2), independent of n."""
        self._build(input_dim)
        self.n_classes = n_classes
        width = self.cfg.width
        A = np.zeros((width, width))
        B = np.zeros((width, n_classes))
        n = 0
        for Xb, yb in batches:
            H = self.hidden(Xb)
            T = np.zeros((len(yb), n_classes))
            T[np.arange(len(yb)), np.asarray(yb, dtype=int)] = 1.0
            A += H.T @ H
            B += H.T @ T
            n += len(yb)
        if n == 0:
            raise ValueError("no training rows")
        self.beta = self._solve(A, B)
        self.solver_info["n_rows"] = n
        return self

    def fit(self, X: np.ndarray, y: np.ndarray, n_classes: Optional[int] = None) -> "QELM":
        X, y = np.asarray(X), np.asarray(y).astype(int)
        k = int(n_classes if n_classes is not None else y.max() + 1)
        bs = self.cfg.solve_batch
        gen = ((X[s:s + bs], y[s:s + bs]) for s in range(0, len(y), bs))
        return self.fit_batches(gen, k, X.shape[1])

    def _solve(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """(A + (lambda1 + lambda2) I) beta = B. Cholesky; on failure raise the jitter, then fall back to least squares."""
        lam = self.cfg.lambda1 + self.cfg.lambda2
        eye = np.eye(A.shape[0])
        jitter = 0.0
        for attempt in range(7):
            try:
                M = A + (lam + jitter) * eye
                beta = cho_solve(cho_factor(M, lower=True), B)
                if np.isfinite(beta).all():
                    self.solver_info = {"method": "cholesky", "jitter": jitter, "attempts": attempt + 1, "lambda_total": lam}
                    return beta
            except (LinAlgError, ValueError):
                pass
            jitter = max(jitter * 10.0, 1e-10 * max(np.trace(A) / A.shape[0], 1e-12))
        beta = lstsq(A + lam * eye, B)[0]
        self.solver_info = {"method": "lstsq", "jitter": jitter, "attempts": 7, "lambda_total": lam}
        return beta

    # ------------------------------------------------------------ inference
    def decision_scores(self, X: np.ndarray) -> np.ndarray:
        if self.beta is None:
            raise RuntimeError("QELM is not fitted")
        bs = self.cfg.solve_batch
        X = np.asarray(X)
        return np.concatenate([self.hidden(X[s:s + bs]) @ self.beta for s in range(0, len(X), bs)], axis=0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        s = self.decision_scores(X)
        s = s - s.max(axis=1, keepdims=True)
        e = np.exp(s)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.decision_scores(X).argmax(axis=1)

    # ------------------------------------------------------------ bundle
    def n_params(self) -> dict:
        return {"frozen_random": int(sum(w.size for w in self.W) + sum(b.size for b in self.b)), "solved": int(self.beta.size) if self.beta is not None else 0}

    def save(self, path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        arrs = {f"W{i}": w for i, w in enumerate(self.W)} | {f"b{i}": b for i, b in enumerate(self.b)}
        np.savez_compressed(p, beta=self.beta, n_classes=self.n_classes, config=json.dumps(asdict(self.cfg)), solver=json.dumps(self.solver_info), **arrs)

    @classmethod
    def load(cls, path) -> "QELM":
        z = np.load(path, allow_pickle=False)
        q = cls(QELMConfig(**json.loads(str(z["config"]))))
        q.W = [z[f"W{i}"] for i in range(q.cfg.n_layers)]
        q.b = [z[f"b{i}"] for i in range(q.cfg.n_layers)]
        q.beta, q.n_classes, q.solver_info = z["beta"], int(z["n_classes"]), json.loads(str(z["solver"]))
        return q
