"""
src/models/transformer.py
Transformer encoder with CLS token and learned positional/time embeddings.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.models.heads import StudentTHead, GaussianHead, ClassificationHead, ExpectedRHead


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        n_classes: int = 3,
        use_student_t: bool = True,
        max_len: int = 512,
    ):
        super().__init__()
        self.use_student_t = use_student_t
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN (more stable)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.cls_head = ClassificationHead(d_model, n_classes)
        self.exp_r_head = ExpectedRHead(d_model)
        if use_student_t:
            self.reg_head = StudentTHead(d_model)
        else:
            self.reg_head = GaussianHead(d_model)

    def forward(self, x: torch.Tensor):
        # x: [B, L, F]
        B = x.size(0)
        x = self.input_proj(x)       # [B, L, d]
        x = self.pos_enc(x)          # [B, L, d]
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, d]
        x = torch.cat([cls, x], dim=1)          # [B, L+1, d]
        x = self.encoder(x)
        h = self.norm(x[:, 0, :])   # CLS token output
        logits = self.cls_head(h)
        exp_r = self.exp_r_head(h)
        reg_out = self.reg_head(h)
        return logits, reg_out, exp_r
