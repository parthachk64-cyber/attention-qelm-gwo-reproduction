"""YAML experiment configs -> typed objects. Shipped configs: configs/{binary,multiclass,published_fixed,gwo_full,cpu_smoke}.yaml."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from ..models.focal import LossConfig
from ..models.pipeline import TrainConfig

PAPER_SEEDS = (7, 13, 21, 42, 99)


@dataclass
class GWOSettings:
    n_wolves: int = 20
    max_iter: int = 50
    patience: int = 10
    tol: float = 1e-4
    n_folds: int = 5
    subsample: Optional[int] = None
    seed: int = 0


@dataclass
class ExperimentConfig:
    name: str = "multiclass_published_fixed"
    task: str = "multiclass"
    seeds: tuple = PAPER_SEEDS
    train: TrainConfig = field(default_factory=TrainConfig)
    qelm_overrides: dict = field(default_factory=dict)
    gwo: Optional[GWOSettings] = None
    fixture_only: bool = False
    notes: str = ""


def load_config(path) -> ExperimentConfig:
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    t = d.get("train", {})
    loss = LossConfig(**d.get("loss", {}))
    train = TrainConfig(lr=float(t.get("lr", 1e-3)), batch_size=int(t.get("batch_size", 32)), epochs=int(t.get("epochs", 100)), patience=int(t.get("patience", 15)),
                        val_fraction=float(t.get("val_fraction", 0.10)), loss=loss, device=t.get("device", "cpu"))
    g = d.get("gwo")
    return ExperimentConfig(name=d.get("name", Path(path).stem), task=d.get("task", "multiclass"), seeds=tuple(d.get("seeds", PAPER_SEEDS)), train=train,
                            qelm_overrides=d.get("qelm", {}), gwo=GWOSettings(**g) if g else None, fixture_only=bool(d.get("fixture_only", False)), notes=d.get("notes", ""))
