"""
src/models/lstm.py
2-layer LSTM/GRU model with dual heads.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.heads import StudentTHead, GaussianHead, ClassificationHead, ExpectedRHead


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_gru: bool = False,
        n_classes: int = 3,
        use_student_t: bool = True,
    ):
        super().__init__()
        self.use_student_t = use_student_t
        rnn_cls = nn.GRU if use_gru else nn.LSTM
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.cls_head = ClassificationHead(hidden_size, n_classes)
        self.exp_r_head = ExpectedRHead(hidden_size)
        if use_student_t:
            self.reg_head = StudentTHead(hidden_size)
        else:
            self.reg_head = GaussianHead(hidden_size)

    def forward(self, x: torch.Tensor):
        # x: [B, L, F]
        out, _ = self.rnn(x)
        h = self.norm(out[:, -1, :])   # last hidden state
        logits = self.cls_head(h)
        exp_r = self.exp_r_head(h)
        reg_out = self.reg_head(h)
        return logits, reg_out, exp_r
