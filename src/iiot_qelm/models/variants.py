"""The proposed model and its seven Table-11 ablations as declarative configurations, with the components each one exercises.

``exercises`` is asserted by tests against the objects actually built, so a configuration cannot silently exercise a different
path than its name says."""
from __future__ import annotations

import copy
from typing import Dict, Optional

from .focal import LossConfig
from .pipeline import PipelineConfig, TrainConfig
from .qelm import QELMConfig

PUBLISHED_FIXED = dict(n_layers=3, width=512, activation="gelu", lambda1=3.2e-4, lambda2=1.7e-3, theta=1.84, rho=0.63)
# Table 11 'w/o GWO': L=2, n_h=256, ReLU, lambda=1e-3, theta=pi, rho=0.5. A single lambda is mapped to the two-penalty solver by
# setting lambda1 = lambda2 = 5e-4 so that lambda1 + lambda2 (the only quantity the solver sees) equals 1e-3. Reconstruction.
DEFAULT_NO_GWO = dict(n_layers=2, width=256, activation="relu", lambda1=5e-4, lambda2=5e-4, theta=3.141592653589793, rho=0.5)

# name -> (description, overrides, exercises)
ABLATIONS: Dict[str, dict] = {
    "full": dict(label="Full proposed", attention=True, head="qelm", init="quantum", loss="focal", gwo=True),
    "no_attention": dict(label="w/o multi-head attention", attention=False, head="qelm", init="quantum", loss="focal", gwo=True),
    "no_qelm": dict(label="w/o QELM", attention=True, head="mlp", init="quantum", loss="focal", gwo=False),
    "no_quantum_init": dict(label="w/o quantum initialization", attention=True, head="qelm", init="classical", loss="focal", gwo=True),
    "no_gwo": dict(label="w/o GWO", attention=True, head="qelm", init="quantum", loss="focal", gwo=False),
    "no_focal": dict(label="w/o focal loss", attention=True, head="qelm", init="quantum", loss="ce", gwo=True),
    "no_gwo_no_focal": dict(label="w/o GWO and focal loss", attention=True, head="qelm", init="quantum", loss="ce", gwo=False),
    "encoder_mlp": dict(label="Encoder + MLP baseline", attention=True, head="mlp", init="quantum", loss="ce", gwo=False),
}


def build_config(name: str, n_classes: int, input_dim: int = 48, *, qelm_overrides: Optional[dict] = None, train: Optional[TrainConfig] = None,
                 seed: int = 0) -> PipelineConfig:
    if name not in ABLATIONS:
        raise KeyError(f"unknown variant {name!r}; choose from {sorted(ABLATIONS)}")
    spec = ABLATIONS[name]
    base = dict(PUBLISHED_FIXED)
    if not spec["gwo"] and spec["head"] == "qelm":
        base = dict(DEFAULT_NO_GWO)                 # defaults replace the GWO-selected values
    base.update(qelm_overrides or {})
    from .encoder import EncoderConfig
    tcfg = copy.deepcopy(train) if train is not None else TrainConfig()
    tcfg.seed = seed
    tcfg.loss = LossConfig(kind=spec["loss"], gamma=tcfg.loss.gamma, alpha=tcfg.loss.alpha, alpha_mode=tcfg.loss.alpha_mode, use_class_weights=tcfg.loss.use_class_weights)
    return PipelineConfig(n_classes=n_classes, use_attention=spec["attention"], head=spec["head"], encoder=EncoderConfig(input_dim=input_dim),
                          qelm=QELMConfig(init=spec["init"], seed=seed, **base), train=tcfg)


def exercises(name: str) -> dict:
    """Which components a variant uses. 'focal_gradient_path' is False whenever there is no trainable encoder/head receiving a focal
    loss (for 'no_attention' the loss is never evaluated: the QELM consumes the raw 48 features and is solved in closed form)."""
    s = ABLATIONS[name]
    trainable = s["attention"]
    return {"encoder_trained": trainable, "qelm_head": s["head"] == "qelm", "mlp_head": s["head"] == "mlp", "quantum_init": s["head"] == "qelm" and s["init"] == "quantum",
            "classical_init": s["head"] == "qelm" and s["init"] == "classical", "focal_gradient_path": trainable and s["loss"] == "focal",
            "ce_gradient_path": trainable and s["loss"] == "ce", "gwo_selected_hyperparameters": s["gwo"] and s["head"] == "qelm"}
