"""Figures drawn ONLY from saved run artifacts (epoch logs, score matrices, GWO histories) or from the code's own shape contract.
Nothing is traced from the article's images and no point is invented. Every figure states its provenance in the title/footer."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from . import metrics as M  # noqa: E402

_LABEL = {"loss": "train", "accuracy": "train", "val_loss": "validation", "val_accuracy": "validation", "test_loss": "test (logged only; never used for selection)",
          "test_accuracy": "test (logged only; never used for selection)"}


def _save(fig, path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def plot_curves(history: Dict[str, Sequence[float]], path, *, banner: str = "") -> Optional[Path]:
    if not history or not history.get("loss"):
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, metric in zip(axes, ("accuracy", "loss")):
        for key, label in _LABEL.items():
            if key.endswith(metric) and history.get(key):
                ax.plot(range(1, len(history[key]) + 1), history[key], label=label)
        ax.set_xlabel("epoch (actual epochs run)")
        ax.set_ylabel(metric)
        ax.legend(fontsize=7)
    fig.suptitle(f"Phase-1 training curves from the saved epoch log. {banner}", fontsize=8)
    return _save(fig, path)


def plot_roc(y, scores, class_names: Sequence[str], path, *, banner: str = "") -> Path:
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    y = np.asarray(y)
    for c, name in enumerate(class_names):
        if (y == c).any() and (y != c).any():
            f, t = M.roc_points(y, scores, c)
            ax.plot(f, t, lw=1, label=name)
    ax.plot([0, 1], [0, 1], "k:", lw=0.7)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.legend(fontsize=5, ncol=2)
    ax.set_title(f"One-vs-rest ROC from saved scores. {banner}", fontsize=7)
    return _save(fig, path)


def plot_confusion(cm, class_names: Sequence[str], path, *, normalize: bool = False, banner: str = "") -> Path:
    cm = np.asarray(cm, float)
    shown = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) if normalize else cm
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(shown, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=70, ha="right", fontsize=6)
    ax.set_yticks(range(len(class_names)), class_names, fontsize=6)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    if len(class_names) <= 16:
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                v = shown[i, j]
                if v > 0:
                    ax.text(j, i, f"{v:.2f}" if normalize else f"{int(v)}", ha="center", va="center", fontsize=4.5, color="white" if v > shown.max() * 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title(f"{'Row-normalised' if normalize else 'Counts'}, FULL test partition (n={int(cm.sum())}). {banner}", fontsize=7)
    return _save(fig, path)


def plot_gwo(best_history: Sequence[float], path, *, banner: str = "") -> Path:
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.plot(range(len(best_history)), best_history, marker="o", ms=3)
    ax.set_xlabel("iteration (0 = initial population)")
    ax.set_ylabel("best CV macro-F1 (alpha wolf)")
    ax.set_title(f"GWO convergence from the saved search history. {banner}", fontsize=7)
    return _save(fig, path)


def plot_architecture(path, cfg=None, qcfg=None, n_classes: int = 15) -> Path:
    """Original diagram generated from the code's shape contract (training path above, inference path below)."""
    from ..models.encoder import EncoderConfig
    from ..models.qelm import QELMConfig
    e, q = cfg or EncoderConfig(), qcfg or QELMConfig()
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.axis("off")

    def box(x, y, w, h, text, fc="#e8f0fe"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec="#334", lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.5)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=0.8))

    top = [f"saved flow\n(B, {e.input_dim})", f"input proj\n(B, {e.model_dim})", f"bottleneck GELU\n(B, {e.bottleneck_dim})",
           f"token proj + index emb.\n(B, {e.num_tokens}, {e.token_dim})", f"MHSA x{e.num_heads} (distinct Q,K,V)\n+ residual + LN\n(B, {e.num_tokens}, {e.token_dim})",
           f"FFN x{e.ffn_expansion} + residual + LN\n(B, {e.num_tokens}, {e.token_dim})", f"flatten\n(B, {e.out_dim})"]
    for i, t in enumerate(top):
        box(i * 1.75, 3.2, 1.55, 0.9, t)
        if i:
            arrow(i * 1.75 - 0.2, 3.65, i * 1.75, 3.65)
    box(0, 4.45, 7.2, 0.4, "PHASE 1 (training): Adam 1e-3, batch 32, focal(gamma=2, alpha=0.25) x inverse-frequency class weights, aux head (discarded)", "#fff3cd")
    ax.text(0, 2.75, "PHASE 2 (training) and INFERENCE: frozen encoder -> fixed random QELM layers -> closed-form output head (no GWO at inference)", fontsize=7, weight="bold")
    qs = [f"embedding\n(B, {e.out_dim})"] + [f"QELM layer {i + 1}\n{q.activation}, (B, {q.width})" for i in range(q.n_layers)] + [f"beta (solved)\n({q.width}, {n_classes})", f"scores (B, {n_classes})\nargmax / softmax"]
    for i, t in enumerate(qs):
        box(i * 1.75, 1.6, 1.55, 0.9, t, "#e6f4ea")
        if i:
            arrow(i * 1.75 - 0.2, 2.05, i * 1.75, 2.05)
    box(0, 0.35, 12.0, 0.7, "GWO (tuning only, offline): 7-D search over (L, n_h, f, lambda1, lambda2, theta, rho); fitness = 5-fold CV macro-F1; encoder fitted per fold on the fit partition only", "#fde7e9")
    ax.set_xlim(-0.2, 12.6)
    ax.set_ylim(0, 5.1)
    ax.set_title("Attention-QELM-GWO (this repository's reconstruction): tensor shapes, training vs inference paths", fontsize=8)
    return _save(fig, path)
