"""Linear-probe and fine-tune evaluation of a (pretrained) encoder."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import EEGWindowDataset
from .models import EEGTransformerEncoder, LinearProbe


def evaluate_linear_probe(
    encoder: EEGTransformerEncoder,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    freeze_encoder: bool = True,
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 3e-3,
    seed: int = 0,
) -> np.ndarray:
    """Train a linear head on top of the encoder; return test probabilities.

    With `freeze_encoder=True` this is a linear probe: the encoder is
    frozen (eval mode, no gradient) and its channel-pooled features are
    standardized with train-set statistics before the linear head — this
    makes the small-sample probe optimization well-conditioned. With
    `freeze_encoder=False` the whole network is fine-tuned end-to-end.

    Args:
        encoder: Encoder to evaluate; a `LinearProbe` head is built on its
            channel-pooled features.
        X_train: Training windows of shape (n_train, n_channels, n_times).
        y_train: Training labels of shape (n_train,).
        X_test: Test windows of shape (n_test, n_channels, n_times).
        y_test: Test labels of shape (n_test,) (unused; kept for a uniform
            evaluation signature).
        freeze_encoder: If True, freeze the encoder and train only the
            linear head (linear probe); if False, fine-tune end-to-end.
        epochs: Number of training passes over the labeled split.
        batch_size: Windows per optimization step.
        lr: Adam learning rate.
        seed: Torch RNG seed, set before the probe head is built.

    Returns:
        Array of shape (n_test,) with the predicted probability of class 1
        for each test window.
    """
    torch.manual_seed(seed)
    probe = LinearProbe(encoder, n_classes=2)

    feat_stats = None
    if freeze_encoder:
        for p in probe.encoder.parameters():
            p.requires_grad_(False)
        probe.encoder.eval()
        with torch.no_grad():
            feats = probe.encoder.forward_channel_features(
                torch.as_tensor(X_train, dtype=torch.float32)
            )
        feat_stats = (feats.mean(dim=0, keepdim=True), feats.std(dim=0, keepdim=True) + 1e-6)

    def head_logits(x: torch.Tensor) -> torch.Tensor:
        f = probe.encoder.forward_channel_features(x)
        if feat_stats is not None:
            f = (f - feat_stats[0]) / feat_stats[1]
        return probe.head(f)

    params = [p for p in probe.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(EEGWindowDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    probe.train()
    if freeze_encoder:
        probe.encoder.eval()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(head_logits(xb), yb).backward()
            opt.step()

    probe.eval()
    with torch.no_grad():
        logits = head_logits(torch.as_tensor(X_test, dtype=torch.float32))
    return torch.softmax(logits, dim=-1)[:, 1].numpy()
