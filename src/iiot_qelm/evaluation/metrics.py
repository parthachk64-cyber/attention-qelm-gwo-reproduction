"""Metrics computed from saved score matrices (so every reported number is recomputable).

Conventions (documented because the article does not state them):
  * binary: attack = positive class 1; FPR = FP / (FP + TN) on Normal rows; AUC from the continuous attack score.
  * multiclass: precision/recall/F1 per class from the confusion matrix; MACRO means average ONLY over classes with support > 0 in y_true
    (an absent class has per-class precision/recall/F1 = NaN and is listed in ``absent_classes``); weighted F1 is reported separately and
    never substituted for macro F1; multiclass FPR is the unweighted mean of one-vs-rest FPR over supported classes (one convention among
    several; the article does not define it); AUC is one-vs-rest, macro over classes having both positives and negatives.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def confusion_matrix(y, pred, k: int) -> np.ndarray:
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (np.asarray(y, dtype=int), np.asarray(pred, dtype=int)), 1)
    return cm


def _safe(a, b):
    return float(a / b) if b > 0 else float("nan")


def per_class(cm: np.ndarray) -> Dict[str, np.ndarray]:
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)
    predicted = cm.sum(axis=0).astype(float)
    fp = predicted - tp
    fn = support - tp
    tn = cm.sum() - tp - fp - fn
    prec = np.array([_safe(t, p) for t, p in zip(tp, predicted)])
    rec = np.array([_safe(t, s) for t, s in zip(tp, support)])
    f1 = np.array([_safe(2 * p * r, p + r) if np.isfinite(p) and np.isfinite(r) else float("nan") for p, r in zip(prec, rec)])
    # F1 with precision undefined (never predicted) but class present: F1 = 0 (all support missed)
    f1 = np.where(np.isnan(f1) & (support > 0), 0.0, f1)
    prec = np.where(np.isnan(prec) & (support > 0), 0.0, prec)
    fpr = np.array([_safe(f, f + t) for f, t in zip(fp, tn)])
    return {"precision": prec, "recall": rec, "f1": f1, "fpr": fpr, "support": support, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def macro_f1_from_scores(y, scores, k: int) -> float:
    cm = confusion_matrix(y, np.asarray(scores).argmax(1), k)
    pc = per_class(cm)
    return float(np.nanmean(pc["f1"][pc["support"] > 0]))


def summarise(y, scores, *, task: str, class_names: Sequence[str], y_train=None, train_scores=None) -> dict:
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    k = len(class_names)
    pred = scores.argmax(1)
    cm = confusion_matrix(y, pred, k)
    pc = per_class(cm)
    present = pc["support"] > 0
    out: dict = {"task": task, "n": int(len(y)), "accuracy": float((pred == y).mean()), "class_names": list(class_names),
                 "absent_classes": [class_names[i] for i in range(k) if not present[i]],
                 "per_class": {class_names[i]: {m: (None if not np.isfinite(pc[m][i]) else float(pc[m][i])) for m in ("precision", "recall", "f1", "fpr", "support")} for i in range(k)},
                 "macro_precision": float(np.nanmean(pc["precision"][present])), "macro_recall": float(np.nanmean(pc["recall"][present])),
                 "macro_f1": float(np.nanmean(pc["f1"][present])),
                 "weighted_f1": float(np.nansum(np.nan_to_num(pc["f1"]) * pc["support"]) / pc["support"].sum()),
                 "confusion_matrix": cm.tolist()}
    if task == "binary":
        tn, fp = pc["tn"][1], pc["fp"][1]                        # attack-as-positive: FP are Normal rows predicted attack
        out.update(precision=float(pc["precision"][1]), recall=float(pc["recall"][1]), f1=float(pc["f1"][1]),
                   fpr=_safe(float(fp), float(fp + tn)), tp=int(pc["tp"][1]), fp=int(fp), fn=int(pc["fn"][1]), tn=int(tn))
        out["auc_roc"] = float(roc_auc_score(y, scores[:, 1] - scores[:, 0])) if len(np.unique(y)) == 2 else None
        out["auc_note"] = "from the continuous attack-minus-normal score"
    else:
        out["fpr_macro_ovr"] = float(np.nanmean(pc["fpr"][present]))
        aucs: List[Optional[float]] = []
        for c in range(k):
            pos = y == c
            aucs.append(float(roc_auc_score(pos, scores[:, c])) if pos.any() and (~pos).any() else None)
        valid = [a for a in aucs if a is not None]
        out["auc_ovr_per_class"] = {class_names[i]: aucs[i] for i in range(k)}
        out["auc_ovr_macro"] = float(np.mean(valid)) if valid else None
    if y_train is not None and train_scores is not None:
        out["train_accuracy"] = float((np.asarray(train_scores).argmax(1) == np.asarray(y_train)).mean())
    return out


def roc_points(y, scores, cls: int, max_points: int = 2000):
    """ROC coordinates computed from saved scores; thinned for plotting (never invented)."""
    pos = np.asarray(y) == cls
    fpr, tpr, _ = roc_curve(pos, np.asarray(scores)[:, cls])
    if len(fpr) > max_points:
        idx = np.unique(np.linspace(0, len(fpr) - 1, max_points).astype(int))
        fpr, tpr = fpr[idx], tpr[idx]
    return fpr, tpr


def top_comparator_delta(per_class_f1: Dict[str, Dict[str, float]], proposed: str = "proposed") -> Dict[str, dict]:
    """For each class: difference (percentage points) between the proposed F1 and the strongest comparator among the supplied models."""
    classes = per_class_f1[proposed].keys()
    out = {}
    for c in classes:
        others = {m: v[c] for m, v in per_class_f1.items() if m != proposed and v.get(c) is not None}
        if not others or per_class_f1[proposed][c] is None:
            out[c] = {"best_comparator": None, "delta_pp": None}
            continue
        best = max(others, key=others.get)
        out[c] = {"best_comparator": best, "best_f1": others[best], "delta_pp": 100.0 * (per_class_f1[proposed][c] - others[best])}
    return out
