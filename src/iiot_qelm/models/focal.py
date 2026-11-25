"""Numerically stable focal loss on logits.

FL = -alpha_t * (1 - p_t)^gamma * log p_t, computed from ``log_softmax`` (never from clipped probabilities).

alpha semantics (the paper gives one scalar, alpha = 0.25, and no per-class rule):
  * ``uniform``  : alpha_t = alpha for every class. This is what the upstream implementation does; it rescales the loss by a
                   constant and does NOT rebalance classes.
  * ``binary``   : Lin et al. convention for two classes, alpha_t = alpha for the positive (attack) class and 1 - alpha for the
                   negative (Normal) class.
  * ``per_class``: ``alpha`` is a vector with one entry per class.
Inverse-frequency class weights are a SEPARATE factor (``class_weights``): each sample is multiplied by w[y]. The trainer
passes them exactly once (``FocalConfig.class_weights_applied`` is recorded); with ``gamma = 0`` and alpha = 1 the loss equals
weighted cross-entropy, which the tests check.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F


def focal_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, *, gamma: float = 2.0, alpha: Union[float, Sequence[float]] = 0.25,
                           alpha_mode: str = "uniform", class_weights: Optional[torch.Tensor] = None, reduction: str = "mean") -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError("logits must be (batch, classes)")
    logp = F.log_softmax(logits.float(), dim=-1)
    logpt = logp.gather(1, target.view(-1, 1)).squeeze(1)
    one_minus_pt = (-torch.expm1(logpt)).clamp_min(0.0)               # 1 - p_t without cancellation
    mod = one_minus_pt.pow(gamma) if gamma != 0 else torch.ones_like(one_minus_pt)
    if alpha_mode == "uniform":
        a_t = torch.full_like(logpt, float(alpha))
    elif alpha_mode == "binary":
        if logits.shape[1] != 2:
            raise ValueError("alpha_mode='binary' needs exactly two classes")
        a_t = torch.where(target == 1, torch.full_like(logpt, float(alpha)), torch.full_like(logpt, 1.0 - float(alpha)))
    elif alpha_mode == "per_class":
        a_t = torch.as_tensor(alpha, dtype=logpt.dtype, device=logpt.device)[target]
    else:
        raise ValueError(f"unknown alpha_mode {alpha_mode!r}")
    loss = -a_t * mod * logpt
    if class_weights is not None:
        loss = loss * class_weights.to(loss.device, loss.dtype)[target]
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def cross_entropy_from_logits(logits, target, class_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Per-sample weighted cross-entropy, same reduction as the focal loss (mean of w[y] * CE)."""
    ce = F.cross_entropy(logits.float(), target, reduction="none")
    if class_weights is not None:
        ce = ce * class_weights.to(ce.device, ce.dtype)[target]
    return ce.mean()


@dataclass(frozen=True)
class LossConfig:
    kind: str = "focal"            # focal | ce
    gamma: float = 2.0
    alpha: float = 0.25
    alpha_mode: str = "uniform"
    use_class_weights: bool = True

    def __call__(self, logits, target, class_weights=None):
        w = class_weights if self.use_class_weights else None
        if self.kind == "ce":
            return cross_entropy_from_logits(logits, target, w)
        if self.kind == "focal":
            return focal_loss_from_logits(logits, target, gamma=self.gamma, alpha=self.alpha, alpha_mode=self.alpha_mode, class_weights=w)
        raise ValueError(f"unknown loss kind {self.kind!r}")
