"""Two-phase pipeline: Phase 1 trains the attention encoder under (focal | CE) loss with class weights; Phase 2 freezes it
and solves the QELM output layer in closed form. Also the matched ablation variants (see ``variants.py``).

A saved bundle contains everything inference needs (preprocessor JSON, encoder weights, fixed random QELM weights, solved
head, manifest with an artifact id). GWO never runs at inference. Bundles are pickled-free (``torch.load(weights_only)`` +
``npz`` with ``allow_pickle=False``); still, load only bundles you trust."""
from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from ..errors import IIoTError
from .encoder import EncoderConfig, Phase1Network
from .focal import LossConfig
from .qelm import QELM, QELMConfig


@dataclass
class TrainConfig:
    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    batch_size: int = 32
    epochs: int = 100
    patience: int = 15
    val_fraction: float = 0.10          # implementer's choice, drawn from the outer TRAINING partition only
    loss: LossConfig = field(default_factory=LossConfig)
    seed: int = 0
    device: str = "cpu"


@dataclass
class PipelineConfig:
    n_classes: int = 15
    use_attention: bool = True
    head: str = "qelm"                  # qelm | mlp
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    qelm: QELMConfig = field(default_factory=QELMConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def input_dim(self) -> int:
        return self.encoder.input_dim

    def to_dict(self) -> dict:
        return json.loads(json.dumps(asdict(self), default=list))


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _batches(n: int, bs: int, rng: np.random.Generator):
    perm = rng.permutation(n)
    for s in range(0, n, bs):
        yield perm[s:s + bs]


def train_phase1(net: Phase1Network, X, y, Xv, yv, tcfg: TrainConfig, class_weights, *, extra_eval: Optional[Dict[str, tuple]] = None) -> Tuple[dict, int]:
    """Adam, early stopping on validation loss, restore best weights. ``extra_eval`` (e.g. {'test': (X, y)}) is LOGGED ONLY and
    never used for selection. Returns (history with Keras-style metric names, best epoch index)."""
    dev = torch.device(tcfg.device)
    net.to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=tcfg.lr, betas=tcfg.betas)
    cw = None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32, device=dev)
    Xt, yt = torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long)
    Xvt, yvt = torch.as_tensor(Xv, dtype=torch.float32), torch.as_tensor(yv, dtype=torch.long)
    rng = np.random.default_rng(tcfg.seed)
    hist: Dict[str, list] = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    for name in (extra_eval or {}):
        hist[f"{name}_loss"], hist[f"{name}_accuracy"] = [], []

    def evaluate(Xe, ye):
        net.eval()
        with torch.no_grad():
            lg = torch.cat([net(Xe[i:i + 4096].to(dev)) for i in range(0, len(Xe), 4096)])
            return float(tcfg.loss(lg, ye.to(dev), cw)), float((lg.argmax(1).cpu() == ye).float().mean())

    best, best_state, best_ep, stale = float("inf"), copy.deepcopy(net.state_dict()), 0, 0
    for ep in range(tcfg.epochs):
        net.train()
        for idx in _batches(len(yt), tcfg.batch_size, rng):
            xb, yb = Xt[idx].to(dev), yt[idx].to(dev)
            loss = tcfg.loss(net(xb), yb, cw)
            opt.zero_grad()
            loss.backward()
            opt.step()
        l, a = evaluate(Xt, yt)
        vl, va = evaluate(Xvt, yvt)
        hist["loss"].append(l), hist["accuracy"].append(a), hist["val_loss"].append(vl), hist["val_accuracy"].append(va)
        for name, (Xe, ye) in (extra_eval or {}).items():
            el, ea = evaluate(torch.as_tensor(Xe, dtype=torch.float32), torch.as_tensor(ye, dtype=torch.long))
            hist[f"{name}_loss"].append(el), hist[f"{name}_accuracy"].append(ea)
        if vl < best - 1e-12:
            best, best_state, best_ep, stale = vl, copy.deepcopy(net.state_dict()), ep, 0
        else:
            stale += 1
            if stale >= tcfg.patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    return hist, best_ep


class AttentionQELM:
    def __init__(self, cfg: Optional[PipelineConfig] = None):
        self.cfg = cfg or PipelineConfig()
        self.net: Optional[Phase1Network] = None
        self.qelm: Optional[QELM] = None
        self.history: dict = {}
        self.timings: dict = {}
        self.meta: dict = {}

    # --------------------------------------------------------------- encoder
    def _to_tensor(self, X):
        return torch.as_tensor(np.asarray(X, dtype=np.float32), device=torch.device(self.cfg.train.device))

    def fit_encoder(self, X, y, class_weights=None, *, extra_eval=None, val: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> "AttentionQELM":
        from ..data.splits import inner_validation_split
        c = self.cfg
        set_seeds(c.train.seed)
        if val is None:
            fi, vi = inner_validation_split(np.asarray(y), fraction=c.train.val_fraction, seed=c.train.seed)
            (X, y), (Xv, yv) = (np.asarray(X)[fi], np.asarray(y)[fi]), (np.asarray(X)[vi], np.asarray(y)[vi])
            cw_src = None
        else:
            Xv, yv = val
        self.net = Phase1Network(c.n_classes, c.encoder, head=c.head)
        t0 = time.perf_counter()
        self.history, best_ep = train_phase1(self.net, X, y, Xv, yv, c.train, class_weights, extra_eval=extra_eval)
        self.timings["phase1_s"] = time.perf_counter() - t0
        self.meta.update({"best_epoch": best_ep, "epochs_run": len(self.history["loss"]), "selection_split": "validation drawn from the training partition",
                          "class_weights_applied_once": bool(c.train.loss.use_class_weights and class_weights is not None)})
        return self

    def encode(self, X) -> np.ndarray:
        if not self.cfg.use_attention:
            return np.asarray(X, dtype=np.float64)
        if self.net is None:
            raise RuntimeError("encoder is not trained")
        self.net.eval()
        with torch.no_grad():
            out = [self.net.encoder(self._to_tensor(np.asarray(X)[i:i + 4096])).cpu().numpy() for i in range(0, len(X), 4096)]
        return np.concatenate(out, axis=0).astype(np.float64)

    # --------------------------------------------------------------- phase 2
    def fit_qelm(self, X, y, qelm_cfg: Optional[QELMConfig] = None, *, embeddings=None) -> "AttentionQELM":
        if self.cfg.head != "qelm":
            raise IIoTError("this variant has an MLP head and no QELM phase")
        emb = self.encode(X) if embeddings is None else embeddings
        q = copy.deepcopy(qelm_cfg or self.cfg.qelm)
        t0 = time.perf_counter()
        self.qelm = QELM(q).fit(emb, np.asarray(y), self.cfg.n_classes)
        self.timings["phase2_s"] = time.perf_counter() - t0
        return self

    def fit(self, X, y, class_weights=None, *, extra_eval=None) -> "AttentionQELM":
        if self.cfg.use_attention:
            self.fit_encoder(X, y, class_weights, extra_eval=extra_eval)
        if self.cfg.head == "qelm":
            self.fit_qelm(X, y)
        return self

    # --------------------------------------------------------------- inference
    def decision_scores(self, X) -> np.ndarray:
        if self.cfg.head == "mlp":
            if self.net is None:
                raise RuntimeError("not fitted")
            self.net.eval()
            with torch.no_grad():
                return np.concatenate([self.net(self._to_tensor(np.asarray(X)[i:i + 4096])).cpu().numpy() for i in range(0, len(X), 4096)])
        if self.qelm is None:
            raise RuntimeError("not fitted")
        return self.qelm.decision_scores(self.encode(X))

    def predict_proba(self, X) -> np.ndarray:
        s = self.decision_scores(X)
        s = s - s.max(axis=1, keepdims=True)
        e = np.exp(s)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        return self.decision_scores(X).argmax(axis=1)

    # --------------------------------------------------------------- reporting
    def parameter_report(self) -> dict:
        """Conventions: trainable = encoder (+ aux/MLP head while training); frozen = QELM random hidden layers; solved = QELM output
        weights. 'deployed' counts what inference actually uses (aux head of the QELM variants is discarded)."""
        enc = sum(p.numel() for p in self.net.encoder.parameters()) if self.net is not None else 0
        head = sum(p.numel() for p in self.net.head.parameters()) if self.net is not None else 0
        q = self.qelm.n_params() if self.qelm is not None else {"frozen_random": 0, "solved": 0}
        deployed_trainable = enc + (head if self.cfg.head == "mlp" else 0)
        return {"encoder_trainable": enc, "phase1_head_trainable": head, "qelm_frozen_random": q["frozen_random"], "qelm_solved": q["solved"],
                "deployed_trainable": deployed_trainable, "deployed_total": deployed_trainable + q["frozen_random"] + q["solved"]}

    # --------------------------------------------------------------- bundle
    def save(self, directory, *, extra_manifest: Optional[dict] = None) -> str:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        if self.net is not None:
            torch.save(self.net.state_dict(), d / "phase1.pt")
        if self.qelm is not None:
            self.qelm.save(d / "qelm.npz")
        (d / "history.json").write_text(json.dumps(self.history), encoding="utf-8")
        manifest = {"config": self.cfg.to_dict(), "meta": self.meta, "timings": self.timings, "parameters": self.parameter_report()}
        manifest.update(extra_manifest or {})
        h = hashlib.sha256()
        for f in sorted(d.glob("*")):
            if f.name != "manifest.json":
                h.update(f.name.encode() + f.read_bytes())
        manifest["artifact_id"] = h.hexdigest()[:16]
        (d / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
        return manifest["artifact_id"]

    @classmethod
    def load(cls, directory) -> "AttentionQELM":
        d = Path(directory)
        m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        c = m["config"]
        t = c["train"]
        tcfg = TrainConfig(**{**t, "betas": tuple(t["betas"]), "loss": LossConfig(**t["loss"])})
        cfg = PipelineConfig(n_classes=c["n_classes"], use_attention=c["use_attention"], head=c["head"], encoder=EncoderConfig(**c["encoder"]),
                             qelm=QELMConfig(**c["qelm"]), train=tcfg)
        obj = cls(cfg)
        if (d / "phase1.pt").is_file():
            obj.net = Phase1Network(cfg.n_classes, cfg.encoder, head=cfg.head)
            obj.net.load_state_dict(torch.load(d / "phase1.pt", map_location="cpu", weights_only=True))
            obj.net.eval()
        if (d / "qelm.npz").is_file():
            obj.qelm = QELM.load(d / "qelm.npz")
        obj.history, obj.meta, obj.timings = json.loads((d / "history.json").read_text(encoding="utf-8")), m.get("meta", {}), m.get("timings", {})
        return obj
