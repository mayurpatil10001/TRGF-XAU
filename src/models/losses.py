"""
src/models/losses.py
Loss functions for the dual-head models.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StudentTNLL(nn.Module):
    """Negative log-likelihood of Student's t distribution."""

    def forward(self, loc: torch.Tensor, scale: torch.Tensor, df: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        # scale > 0 (ensured by softplus), df > 2 (ensured by 2 + softplus)
        z = (target - loc) / scale
        nll = (
            torch.lgamma((df + 1) / 2)
            - torch.lgamma(df / 2)
            - 0.5 * torch.log(df * torch.pi)
            - torch.log(scale)
            + ((df + 1) / 2) * torch.log(1 + z ** 2 / df)
        )
        return nll.mean()


class GaussianNLL(nn.Module):
    """Negative log-likelihood of Gaussian for ablation."""

    def forward(self, loc: torch.Tensor, scale: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        return F.gaussian_nll_loss(loc, target, scale ** 2, reduction="mean")


class FocalLoss(nn.Module):
    """Focal loss for imbalanced classification (never resamples across time)."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()
