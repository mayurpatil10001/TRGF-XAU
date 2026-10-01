"""
src/models/gated_fusion.py
Gated Dual-Branch Fusion model (TEGF — the proposed core).

Architecture:
  deep branch:  Conv1D(F->64,k=3) -> GELU -> Conv1D(64->64,k=3) -> GELU
                -> LSTM(64->128) -> last hidden h_deep [128]
  skip branch:  Flatten(L*F) -> Dense(128) -> GELU -> h_skip [128]
  gate:         g = sigmoid(Dense(concat[h_deep, h_skip]) -> 128)
                fused = g*h_deep + (1-g)*h_skip
  heads on fused

Also logs mean gate value per forward pass (stored for fold/regime analysis).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F_func

from src.models.heads import StudentTHead, GaussianHead, ClassificationHead, ExpectedRHead


class GatedFusion(nn.Module):
    def __init__(
        self,
        input_dim: int,           # F (features per bar)
        lookback: int = 60,       # L (sequence length)
        conv_channels: int = 64,
        lstm_hidden: int = 128,
        dropout: float = 0.1,
        n_classes: int = 3,
        use_student_t: bool = True,
    ):
        super().__init__()
        self.use_student_t = use_student_t
        self.lookback = lookback

        # ── Deep branch ──────────────────────────────────────────────────────
        self.conv1 = nn.Conv1d(input_dim, conv_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.deep_norm = nn.LayerNorm(lstm_hidden)
        self.deep_drop = nn.Dropout(dropout)

        # ── Skip branch ──────────────────────────────────────────────────────
        skip_in = input_dim * lookback
        self.skip_proj = nn.Linear(skip_in, lstm_hidden)
        self.skip_norm = nn.LayerNorm(lstm_hidden)
        self.skip_drop = nn.Dropout(dropout)

        # ── Gate ─────────────────────────────────────────────────────────────
        self.gate_proj = nn.Linear(lstm_hidden * 2, lstm_hidden)

        # ── Heads ─────────────────────────────────────────────────────────────
        self.cls_head = ClassificationHead(lstm_hidden, n_classes)
        self.exp_r_head = ExpectedRHead(lstm_hidden)
        if use_student_t:
            self.reg_head = StudentTHead(lstm_hidden)
        else:
            self.reg_head = GaussianHead(lstm_hidden)

        # Tracking: gate values for interpretability
        self._last_gate_mean: float = 0.5

    def forward(self, x: torch.Tensor):
        """
        x: [B, L, F]
        Returns: logits, reg_out, exp_r, gate_mean
        """
        B, L, F_in = x.shape

        # ── Deep branch ──────────────────────────────────────────────────────
        # Conv1D expects [B, C, L]
        xc = x.permute(0, 2, 1)              # [B, F_in, L]
        xc = F_func.gelu(self.conv1(xc))     # [B, 64, L]
        xc = F_func.gelu(self.conv2(xc))     # [B, 64, L]
        xc = xc.permute(0, 2, 1)             # [B, L, 64]
        _, (h_n, _) = self.lstm(xc)
        h_deep = self.deep_drop(self.deep_norm(h_n[-1]))  # [B, 128]

        # ── Skip branch ──────────────────────────────────────────────────────
        x_flat = x.flatten(1)                # [B, L*F_in]
        h_skip = self.skip_drop(
            F_func.gelu(self.skip_norm(self.skip_proj(x_flat)))
        )                                    # [B, 128]

        # ── Gate ─────────────────────────────────────────────────────────────
        gate_in = torch.cat([h_deep, h_skip], dim=-1)  # [B, 256]
        g = torch.sigmoid(self.gate_proj(gate_in))     # [B, 128]
        fused = g * h_deep + (1 - g) * h_skip         # [B, 128]

        # Log gate mean
        gate_mean = g.mean().item()
        self._last_gate_mean = gate_mean

        # ── Heads ─────────────────────────────────────────────────────────────
        logits = self.cls_head(fused)
        exp_r = self.exp_r_head(fused)
        reg_out = self.reg_head(fused)

        return logits, reg_out, exp_r, gate_mean
