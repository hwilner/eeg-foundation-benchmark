"""Masked-reconstruction (channel-masked autoencoding) SSL pretraining."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .models import EEGTransformerEncoder, MaskedReconstructionModel


def pretrain_mae(
    encoder: EEGTransformerEncoder,
    dataset,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 5e-3,
    mask_ratio: float = 0.4,
    seed: int = 0,
) -> list[float]:
    """Pretrain `encoder` in place with channel-masked reconstruction.

    `mask_ratio` is the fraction of *channels* masked per window.
    Returns the per-epoch mean reconstruction (MSE) loss history.
    """
    torch.manual_seed(seed)
    model = MaskedReconstructionModel(encoder)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history: list[float] = []
    model.train()
    for _ in range(epochs):
        total, count = 0.0, 0
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            recon, target, mask = model(x, channel_mask_ratio=mask_ratio)
            loss = F.mse_loss(recon[mask], target[mask])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
            count += len(x)
        sched.step()
        history.append(total / max(count, 1))
    encoder.eval()
    return history


def train_test_split(X: np.ndarray, y: np.ndarray, test_frac: float = 0.3, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_frac)
    te, tr = idx[:n_test], idx[n_test:]
    return X[tr], X[te], y[tr], y[te]
