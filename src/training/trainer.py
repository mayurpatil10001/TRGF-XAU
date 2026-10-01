"""
src/training/trainer.py
PyTorch training loop with:
  - AdamW optimizer
  - Cosine LR schedule or ReduceLROnPlateau
  - Gradient clipping
  - Early stopping with best-checkpoint restore
  - Mixed precision on GPU
  - Dual-loss: NLL_t + lambda * CE
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset

from src.common.config import Config
from src.common.logging import get_logger
from src.models.losses import StudentTNLL, GaussianNLL, FocalLoss

logger = get_logger("training.trainer")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        logger.info("Using CUDA GPU.")
        return torch.device("cuda")
    logger.info("Using CPU.")
    return torch.device("cpu")


def make_dataset(
    X: np.ndarray,        # [N, L, F] or [N, F]
    y_cls: np.ndarray,    # [N] int labels
    y_reg: np.ndarray,    # [N] regression target
) -> TensorDataset:
    X_t = torch.tensor(X, dtype=torch.float32)
    y_cls_t = torch.tensor(y_cls, dtype=torch.long)
    y_reg_t = torch.tensor(y_reg, dtype=torch.float32)
    return TensorDataset(X_t, y_cls_t, y_reg_t)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    nll_loss_fn,
    ce_loss_fn: nn.Module,
    lambda_nll_ce: float,
    grad_clip: float,
    device: torch.device,
    use_student_t: bool,
) -> Dict[str, float]:
    model.train()
    total_loss = total_nll = total_ce = 0.0
    n = 0

    for batch in loader:
        Xb, y_cls_b, y_reg_b = [t.to(device) for t in batch]
        optimizer.zero_grad()

        with autocast(enabled=scaler is not None):
            out = model(Xb)
            # Handle gated fusion returning 4 values
            if len(out) == 4:
                logits, reg_out, exp_r, _ = out
            else:
                logits, reg_out, exp_r = out

            if use_student_t:
                loc, scale, df = reg_out
                nll = nll_loss_fn(loc, scale, df, y_reg_b)
            else:
                loc, scale = reg_out
                nll = nll_loss_fn(loc, scale, y_reg_b)

            ce = ce_loss_fn(logits, y_cls_b)
            loss = nll + lambda_nll_ce * ce

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = Xb.size(0)
        total_loss += loss.item() * bs
        total_nll += nll.item() * bs
        total_ce += ce.item() * bs
        n += bs

    return {"loss": total_loss / n, "nll": total_nll / n, "ce": total_ce / n}


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    nll_loss_fn,
    ce_loss_fn: nn.Module,
    lambda_nll_ce: float,
    device: torch.device,
    use_student_t: bool,
) -> Dict[str, float]:
    model.eval()
    total_loss = total_nll = total_ce = 0.0
    n = 0
    for batch in loader:
        Xb, y_cls_b, y_reg_b = [t.to(device) for t in batch]
        out = model(Xb)
        if len(out) == 4:
            logits, reg_out, exp_r, _ = out
        else:
            logits, reg_out, exp_r = out

        if use_student_t:
            loc, scale, df = reg_out
            nll = nll_loss_fn(loc, scale, df, y_reg_b)
        else:
            loc, scale = reg_out
            nll = nll_loss_fn(loc, scale, y_reg_b)

        ce = ce_loss_fn(logits, y_cls_b)
        loss = nll + lambda_nll_ce * ce

        bs = Xb.size(0)
        total_loss += loss.item() * bs
        total_nll += nll.item() * bs
        total_ce += ce.item() * bs
        n += bs

    return {"loss": total_loss / n, "nll": total_nll / n, "ce": total_ce / n}


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_cls_train: np.ndarray,
    y_reg_train: np.ndarray,
    X_val: np.ndarray,
    y_cls_val: np.ndarray,
    y_reg_val: np.ndarray,
    cfg: Config,
    lr: float = 1e-3,
    dropout: float = 0.1,
    weight_decay: float = 1e-4,
    save_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Full training loop.
    Returns dict with: best_val_loss, train_curves, best_epoch, model (loaded from best ckpt)
    """
    device = get_device()
    model = model.to(device)
    use_student_t = getattr(model, "use_student_t", True)

    # Class weights from train set
    class_counts = np.bincount(y_cls_train, minlength=3)
    total = class_counts.sum()
    class_weights = torch.tensor(
        total / (3 * (class_counts + 1)), dtype=torch.float32, device=device
    )

    nll_fn = StudentTNLL() if use_student_t else GaussianNLL()
    ce_fn = FocalLoss(gamma=2.0, weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.model.epochs_max, eta_min=lr * 0.01
    )
    use_amp = device.type == "cuda"
    scaler = GradScaler() if use_amp else None

    train_ds = make_dataset(X_train, y_cls_train, y_reg_train)
    val_ds = make_dataset(X_val, y_cls_val, y_reg_val)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.model.batch_size, shuffle=True, drop_last=True,
        num_workers=0, pin_memory=use_amp,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.model.batch_size * 2, shuffle=False, num_workers=0)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0
    train_curves = {"train_loss": [], "val_loss": []}

    for epoch in range(cfg.model.epochs_max):
        tr = train_epoch(
            model, train_loader, optimizer, scaler,
            nll_fn, ce_fn, cfg.training.lambda_nll_ce,
            cfg.training.grad_clip, device, use_student_t,
        )
        vl = eval_epoch(
            model, val_loader, nll_fn, ce_fn,
            cfg.training.lambda_nll_ce, device, use_student_t,
        )
        scheduler.step()

        train_curves["train_loss"].append(tr["loss"])
        train_curves["val_loss"].append(vl["loss"])

        if vl["loss"] < best_val_loss:
            best_val_loss = vl["loss"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
            if save_path is not None:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, save_path)
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"  Epoch {epoch+1}: train={tr['loss']:.4f}, val={vl['loss']:.4f}, "
                f"best={best_val_loss:.4f} @ ep{best_epoch}"
            )

        if patience_counter >= cfg.model.early_stop_patience:
            logger.info(f"  Early stop at epoch {epoch+1}.")
            break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "train_curves": train_curves,
        "model": model,
    }


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    """Return class probabilities [N, 3]."""
    model.eval()
    model = model.to(device)
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=2048, shuffle=False, num_workers=0)
    probs_list = []
    for (Xb,) in loader:
        Xb = Xb.to(device)
        out = model(Xb)
        if len(out) == 4:
            logits = out[0]
        else:
            logits = out[0]
        probs = torch.softmax(logits, dim=-1)
        probs_list.append(probs.cpu().numpy())
    return np.concatenate(probs_list, axis=0)
