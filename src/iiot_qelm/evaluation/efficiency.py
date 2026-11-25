"""Efficiency profile (Table 12 analogue). A workstation measurement cannot establish embedded-gateway latency.

Conventions, all recorded in the output: parameters split into trainable / frozen-random / solved (see AttentionQELM.parameter_report);
``macs_per_sample`` counts multiply-accumulates of linear layers and attention matmuls (1 MAC = 2 FLOPs; the article says only
"operations"); latency is wall-clock for a forward pass over the stated batch, including host->device copy (CPU only here), with the
stated warm-up and timed iterations; peak GPU memory is recorded only when CUDA is present."""
from __future__ import annotations

import platform
import time
from typing import Callable, Optional

import numpy as np
import torch

from ..models.encoder import EncoderConfig
from ..models.qelm import QELMConfig


def encoder_macs(c: EncoderConfig) -> int:
    lin = c.input_dim * c.model_dim + c.model_dim * c.bottleneck_dim + c.bottleneck_dim * c.num_tokens * c.token_dim
    inner = c.num_heads * c.head_dim
    per_token = 3 * c.token_dim * inner + inner * c.token_dim + 2 * c.token_dim * c.ffn_expansion * c.token_dim
    att = 2 * c.num_heads * c.num_tokens * c.num_tokens * c.head_dim
    return int(lin + c.num_tokens * per_token + att)


def qelm_macs(q: QELMConfig, in_dim: int, n_classes: int) -> int:
    macs, fan = 0, in_dim
    for _ in range(q.n_layers):
        macs += fan * q.width
        fan = q.width
    return int(macs + fan * n_classes)


def hardware_metadata() -> dict:
    return {"platform": platform.platform(), "processor": platform.processor(), "python": platform.python_version(), "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()), "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "threads": torch.get_num_threads()}


def time_inference(fn: Callable[[np.ndarray], object], X: np.ndarray, *, batch: int = 1, warmup: int = 20, timed: int = 200, device: str = "cpu") -> dict:
    """Median/mean latency per sample over ``timed`` calls on batches of ``batch`` rows, after ``warmup`` untimed calls."""
    X = np.asarray(X, dtype=np.float32)
    if len(X) < batch:
        raise ValueError("not enough rows for the requested batch size")
    idx = np.arange(warmup + timed) % max(len(X) - batch + 1, 1)
    sync = (lambda: torch.cuda.synchronize()) if device == "cuda" and torch.cuda.is_available() else (lambda: None)
    for i in idx[:warmup]:
        fn(X[i:i + batch])
    sync()
    ts = []
    for i in idx[warmup:]:
        t0 = time.perf_counter()
        fn(X[i:i + batch])
        sync()
        ts.append(time.perf_counter() - t0)
    per = np.asarray(ts) / batch * 1000.0
    out = {"device": device, "batch": batch, "warmup": warmup, "timed": timed, "ms_per_sample_median": float(np.median(per)), "ms_per_sample_mean": float(per.mean()),
           "ms_per_sample_p95": float(np.percentile(per, 95)), "samples_per_s": float(1000.0 / per.mean()),
           "timer_includes": "model forward over the batch (encoder + QELM or baseline predict); excludes data loading and preprocessing",
           "peak_gpu_mb": None}
    if device == "cuda" and torch.cuda.is_available():
        out["peak_gpu_mb"] = float(torch.cuda.max_memory_allocated() / 2**20)
    return out
