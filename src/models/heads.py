"""
src/models/heads.py
Shared output heads: Student's t regression + 3-class classifier + expected-R.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StudentTHead(nn.Module):
    """Maps a hidden vector to Student's t parameters (loc, scale, df)."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.loc_proj = nn.Linear(in_dim, 1)
        self.scale_proj = nn.Linear(in_dim, 1)
        self.df_proj = nn.Linear(in_dim, 1)

    def forward(self, h: torch.Tensor):
        loc = self.loc_proj(h).squeeze(-1)
        scale = F.softplus(self.scale_proj(h)).squeeze(-1) + 1e-6
        df = F.softplus(self.df_proj(h)).squeeze(-1) + 2.0
        return loc, scale, df


class GaussianHead(nn.Module):
    """Maps a hidden vector to Gaussian parameters (loc, scale)."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.loc_proj = nn.Linear(in_dim, 1)
        self.scale_proj = nn.Linear(in_dim, 1)

    def forward(self, h: torch.Tensor):
        loc = self.loc_proj(h).squeeze(-1)
        scale = F.softplus(self.scale_proj(h)).squeeze(-1) + 1e-6
        return loc, scale


class ClassificationHead(nn.Module):
    """3-class logits (NONE=0, LONG=1, SHORT=2)."""

    def __init__(self, in_dim: int, n_classes: int = 3):
        super().__init__()
        self.proj = nn.Linear(in_dim, n_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h)


class ExpectedRHead(nn.Module):
    """Scalar expected R-multiple head."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h).squeeze(-1)
