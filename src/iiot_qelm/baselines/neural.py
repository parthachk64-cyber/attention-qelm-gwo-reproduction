"""Neural baselines (PyTorch re-implementations of the upstream Keras models; the published numbers used TensorFlow 2.11).

Architectures follow the upstream code at fc486ac:
  CNN-1D   : 3 conv blocks (64, 128, 128 filters, k=3, ReLU, MaxPool 2, dropout 0.2) -> GAP -> Dense 64 -> classes
  LSTM/GRU : 2 stacked recurrent layers of 128 units (dropout 0.2) -> Dense 64 -> classes; BiLSTM wraps both layers bidirectionally
  DenseNet-1D: stem conv 64, 4 dense blocks (4 layers, growth 32, BN-ReLU-conv3-dropout 0.1), transitions (BN-ReLU-conv1 x0.5, avgpool 2)
Input (B, F) is treated as a length-F sequence with one channel. Training uses the same loop, optimiser (Adam 1e-3, batch 32),
early stopping and inverse-frequency class weights (cross-entropy) as the proposed model's Phase 1."""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch
from torch import nn

from ..data.splits import inner_validation_split
from ..models.focal import LossConfig
from ..models.pipeline import TrainConfig, set_seeds, train_phase1

NEURAL = ("cnn1d", "lstm", "gru", "bilstm", "densenet1d")


class CNN1D(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        blocks, c = [], 1
        for f in (64, 128, 128):
            blocks += [nn.Conv1d(c, f, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2, ceil_mode=True), nn.Dropout(0.2)]
            c = f
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Linear(c, 64), nn.ReLU(), nn.Linear(64, n_classes))

    def forward(self, x):
        return self.head(self.features(x.unsqueeze(1)).mean(dim=2))


class RNN(nn.Module):
    def __init__(self, n_classes: int, cell: str = "lstm", bidirectional: bool = False):
        super().__init__()
        C = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.r1 = C(1, 128, batch_first=True, bidirectional=bidirectional)
        self.r2 = C(128 * (2 if bidirectional else 1), 128, batch_first=True, bidirectional=bidirectional)
        self.drop = nn.Dropout(0.2)
        self.head = nn.Sequential(nn.Linear(128 * (2 if bidirectional else 1), 64), nn.ReLU(), nn.Linear(64, n_classes))

    def forward(self, x):
        o, _ = self.r1(x.unsqueeze(-1))
        o, _ = self.r2(self.drop(o))
        return self.head(self.drop(o[:, -1, :]))


class _DenseLayer(nn.Module):
    def __init__(self, c_in: int, growth: int):
        super().__init__()
        self.f = nn.Sequential(nn.BatchNorm1d(c_in), nn.ReLU(), nn.Conv1d(c_in, growth, 3, padding=1), nn.Dropout(0.1))

    def forward(self, x):
        return torch.cat([x, self.f(x)], dim=1)


class DenseNet1D(nn.Module):
    def __init__(self, n_classes: int, blocks=(4, 4, 4, 4), growth: int = 32):
        super().__init__()
        layers, c = [nn.Conv1d(1, 64, 3, padding=1)], 64
        for nl in blocks:
            for _ in range(nl):
                layers.append(_DenseLayer(c, growth))
                c += growth
            c2 = int(c * 0.5)
            layers += [nn.BatchNorm1d(c), nn.ReLU(), nn.Conv1d(c, c2, 1), nn.AvgPool1d(2, ceil_mode=True)]
            c = c2
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(c, 128), nn.ReLU(), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.head(self.features(x.unsqueeze(1)).mean(dim=2))


def build_net(name: str, n_classes: int) -> nn.Module:
    if name == "cnn1d":
        return CNN1D(n_classes)
    if name == "lstm":
        return RNN(n_classes, "lstm")
    if name == "gru":
        return RNN(n_classes, "gru")
    if name == "bilstm":
        return RNN(n_classes, "lstm", True)
    if name == "densenet1d":
        return DenseNet1D(n_classes)
    raise KeyError(f"unknown neural baseline {name!r}; choose from {NEURAL}")


class TorchAdapter:
    kind = "neural"

    def __init__(self, name: str, n_classes: int, seed: int = 0, weighted: bool = True, train: Optional[TrainConfig] = None):
        self.name, self.n_classes, self.seed = name, n_classes, seed
        self.train_cfg = train or TrainConfig()
        self.train_cfg.seed = seed
        self.train_cfg.loss = LossConfig(kind="ce", use_class_weights=weighted)
        set_seeds(seed)
        self.net = build_net(name, n_classes)
        self.history: dict = {}
        self.fit_seconds = 0.0

    def fit(self, X, y, class_weights=None, **_):
        c = self.train_cfg
        set_seeds(c.seed)
        fi, vi = inner_validation_split(np.asarray(y), fraction=c.val_fraction, seed=c.seed)
        cw = None if class_weights is None else np.array([class_weights.get(k, 1.0) for k in range(self.n_classes)], dtype=np.float32)
        t0 = time.perf_counter()
        self.history, self.best_epoch = train_phase1(self.net, np.asarray(X)[fi], np.asarray(y)[fi], np.asarray(X)[vi], np.asarray(y)[vi], c, cw)
        self.fit_seconds = time.perf_counter() - t0
        return self

    def decision_scores(self, X) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            lg = torch.cat([self.net(torch.as_tensor(np.asarray(X)[i:i + 4096], dtype=torch.float32)) for i in range(0, len(X), 4096)])
        return torch.softmax(lg, dim=1).numpy()

    def predict(self, X):
        return self.decision_scores(X).argmax(axis=1)

    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.net.parameters()))
