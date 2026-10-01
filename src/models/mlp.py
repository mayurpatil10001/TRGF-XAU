"""
src/models/mlp.py
MLP model: flatten [L*F] -> 512 -> 256 -> 128, GELU, LayerNorm, dropout.
Dual heads: Student's t regression + 3-class classifier + optional expected-R.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.heads import StudentTHead, GaussianHead, ClassificationHead, ExpectedRHead


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,           # L * F
        hidden_dims=(512, 256, 128),
        dropout: float = 0.1,
        n_classes: int = 3,
        use_student_t: bool = True,
    ):
        super().__init__()
        self.use_student_t = use_student_t
        layers = []
        in_d = input_dim
        for h_d in hidden_dims:
            layers += [nn.Linear(in_d, h_d), nn.LayerNorm(h_d), nn.GELU(), nn.Dropout(dropout)]
            in_d = h_d
        self.backbone = nn.Sequential(*layers)
        self.cls_head = ClassificationHead(in_d, n_classes)
        self.exp_r_head = ExpectedRHead(in_d)
        if use_student_t:
            self.reg_head = StudentTHead(in_d)
        else:
            self.reg_head = GaussianHead(in_d)

    def forward(self, x: torch.Tensor):
        # x: [B, L, F] or [B, L*F]
        if x.dim() == 3:
            x = x.flatten(1)
        h = self.backbone(x)
        logits = self.cls_head(h)
        exp_r = self.exp_r_head(h)
        reg_out = self.reg_head(h)
        return logits, reg_out, exp_r
