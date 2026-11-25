"""Attention encoder (PyTorch), Sec. 2.4 of the article.

Tensor shapes, batch size B (this repository's reconstruction; see docs/architecture.md):

  x                      (B, 48)
  z = W_e x + b_e        (B, 128)        input projection
  h = GELU(W_b z + b_b)  (B, 64)         bottleneck
  t = W_t h + b_t        (B, 8*32=256) -> reshape (B, 8, 32)    # RECONSTRUCTION: a 64-vector cannot be reshaped to 8 x 32
  t = t + E_idx          (B, 8, 32)      learned token-index embedding, shape (8, 32)
  Q,K,V = per-head projections 32 -> 8 heads x 32 = 256 (distinct W_Q, W_K, W_V)
  A = softmax(Q K^T / sqrt(32)) V        (B, 8 heads, 8, 32) -> concat (B, 8, 256)
  A = W_O A              (B, 8, 32)      RECONSTRUCTION: output projection 256 -> 32 per token (the paper prints 256 -> 128)
  a' = LayerNorm(t + A)  (B, 8, 32)
  e  = LayerNorm(a' + FFN(a')),  FFN = Linear(32,128) GELU Dropout Linear(128,32)
  flatten                (B, 256)        input of the QELM

Upstream (``src/models/attention_encoder.py`` at fc486ac) reshapes the 64-wide bottleneck to (8, 32) behind a config check
``bottleneck_dim == num_tokens * key_dim`` that FAILS for its own defaults (64 != 256); see docs/paper-audit.md.
The "feature-index" embedding here indexes the 8 TOKENS (derived from a learned projection), not the 48 raw fields.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class EncoderConfig:
    input_dim: int = 48
    model_dim: int = 128
    bottleneck_dim: int = 64
    num_tokens: int = 8
    token_dim: int = 32
    num_heads: int = 8
    head_dim: int = 32
    ffn_expansion: int = 4
    dropout: float = 0.1

    @property
    def out_dim(self) -> int:
        return self.num_tokens * self.token_dim

    def to_dict(self) -> dict:
        return asdict(self)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.h, self.dh = cfg.num_heads, cfg.head_dim
        inner = self.h * self.dh
        self.q = nn.Linear(cfg.token_dim, inner)
        self.k = nn.Linear(cfg.token_dim, inner)
        self.v = nn.Linear(cfg.token_dim, inner)
        self.o = nn.Linear(inner, cfg.token_dim)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        b, n, _ = t.shape
        split = lambda x: x.view(b, n, self.h, self.dh).transpose(1, 2)           # (B, h, n, dh)
        q, k, v = split(self.q(t)), split(self.k(t)), split(self.v(t))
        att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.dh), dim=-1)
        out = self.drop(att) @ v
        return self.o(out.transpose(1, 2).reshape(b, n, self.h * self.dh))


class AttentionEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig = EncoderConfig()):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.input_dim, cfg.model_dim)
        self.bottleneck = nn.Linear(cfg.model_dim, cfg.bottleneck_dim)
        self.token_proj = nn.Linear(cfg.bottleneck_dim, cfg.num_tokens * cfg.token_dim)
        self.index_embedding = nn.Parameter(torch.empty(cfg.num_tokens, cfg.token_dim))
        nn.init.trunc_normal_(self.index_embedding, std=0.02)
        self.attn = MultiHeadSelfAttention(cfg)
        self.norm1 = nn.LayerNorm(cfg.token_dim, eps=1e-6)
        self.ffn = nn.Sequential(nn.Linear(cfg.token_dim, cfg.ffn_expansion * cfg.token_dim), nn.GELU(), nn.Dropout(cfg.dropout),
                                 nn.Linear(cfg.ffn_expansion * cfg.token_dim, cfg.token_dim))
        self.norm2 = nn.LayerNorm(cfg.token_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.cfg.input_dim:
            raise ValueError(f"expected (B, {self.cfg.input_dim}), got {tuple(x.shape)}")
        h = nn.functional.gelu(self.bottleneck(self.input_proj(x)))
        t = self.token_proj(h).view(x.shape[0], self.cfg.num_tokens, self.cfg.token_dim) + self.index_embedding
        a = self.norm1(t + self.attn(t))
        e = self.norm2(a + self.ffn(a))
        return e.reshape(x.shape[0], -1)


class Phase1Network(nn.Module):
    """Encoder + auxiliary classifier head used ONLY to supervise Phase 1; the head is discarded before Phase 2
    (``head='qelm'``) or kept as the classifier (``head='mlp'``, the 'w/o QELM' ablations)."""

    def __init__(self, n_classes: int, cfg: EncoderConfig = EncoderConfig(), head: str = "qelm", dropout: float = 0.1):
        super().__init__()
        self.encoder = AttentionEncoder(cfg)
        d = cfg.out_dim
        if head == "qelm":
            self.head = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, n_classes))
        elif head == "mlp":                       # three hidden layers of width 256 (the paper does not give the widths)
            layers, w = [], d
            for _ in range(3):
                layers += [nn.Linear(w, 256), nn.GELU(), nn.Dropout(dropout)]
                w = 256
            layers.append(nn.Linear(w, n_classes))
            self.head = nn.Sequential(*layers)
        else:
            raise ValueError(f"head must be 'qelm' or 'mlp', got {head!r}")

    def forward(self, x):
        return self.head(self.encoder(x))
