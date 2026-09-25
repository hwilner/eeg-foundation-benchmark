"""Synthetic EEG window generator with planted class-discriminative patterns.

Used for data-free development and CI. The synthetic task encodes the
label in the *phase coupling* between two channel groups sharing the
same ~10 Hz oscillation:

- Class 0: posterior channels oscillate in phase with anterior channels.
- Class 1: posterior channels oscillate in anti-phase (pi shift).

Because every window contains the same rhythm at the same per-channel
power, per-channel spectral features are at chance: the discriminative
structure is purely cross-channel, which is exactly what channel-masked
reconstruction pretraining must learn. This mirrors why supervised
from-scratch models struggle on small labeled sets while a self-
supervised encoder, pretrained on abundant unlabeled windows, exposes
the pattern to a linear probe.
"""

from __future__ import annotations

import numpy as np


def generate_synthetic_eeg(
    n_samples: int = 200,
    n_channels: int = 8,
    n_times: int = 256,
    sfreq: float = 128.0,
    base_freq: float = 10.0,
    amplitude: float = 1.5,
    noise: float = 0.6,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate labeled synthetic EEG windows with a phase-coupling label.

    Returns:
    -------
    X : (n_samples, n_channels, n_times) float32
    y : (n_samples,) int64 labels in {0, 1}
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_times) / sfreq
    half_c = n_channels // 2

    X = np.zeros((n_samples, n_channels, n_times), dtype=np.float64)
    y = (np.arange(n_samples) % 2).astype(np.int64)
    rng.shuffle(y)

    for i in range(n_samples):
        phase = rng.uniform(0, 2 * np.pi)
        shift = 0.0 if y[i] == 0 else np.pi
        shared = 0.1 * np.convolve(rng.standard_normal(n_times), np.ones(8) / 8, mode="same")
        for c in range(n_channels):
            osc = amplitude * np.sin(
                2 * np.pi * base_freq * t
                + phase
                + (shift if c >= half_c else 0.0)
                + 0.05 * rng.standard_normal()
            )
            bg = noise * np.convolve(rng.standard_normal(n_times), np.ones(8) / 8, mode="same")
            X[i, c] = osc + bg + shared

    # per-channel standardization
    X = (X - X.mean(axis=-1, keepdims=True)) / (X.std(axis=-1, keepdims=True) + 1e-6)
    return X.astype(np.float32), y
