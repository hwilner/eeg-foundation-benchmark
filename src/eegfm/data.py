"""EEG window dataset interface and channel-set standardization helpers.

The pipeline is corpus-agnostic: `EEGWindowDataset` wraps any
(n_samples, n_channels, n_times) array, and `standardize_channels`
harmonizes recordings to a canonical montage (e.g., the TUH 10-20
subset) by reordering and zero-filling missing channels.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

# Canonical 19-channel 10-20 montage used across the TUH EEG family.
CANONICAL_CHANNELS: tuple[str, ...] = (
    "FP1", "FP2", "F7", "F8", "F3", "F4", "T3", "T4", "C3", "C4",
    "T5", "T6", "P3", "P4", "O1", "O2", "FZ", "CZ", "PZ",
)


def standardize_channels(
    x: np.ndarray,
    source_channels: Sequence[str],
    target_channels: Sequence[str] = CANONICAL_CHANNELS,
) -> np.ndarray:
    """Reproject a window batch onto a canonical channel set.

    Channels present in `source_channels` are reordered into the target
    layout; channels missing from the source are zero-filled. This is
    deterministic and preserves known signals (used by synthetic tests).

    Parameters
    ----------
    x : (n_samples, n_source_channels, n_times)
    source_channels : names matching axis 1 of `x`
    target_channels : desired output montage

    Returns:
    -------
    (n_samples, len(target_channels), n_times) float array
    """
    src_index = {ch.upper(): i for i, ch in enumerate(source_channels)}
    out = np.zeros((x.shape[0], len(target_channels), x.shape[2]), dtype=x.dtype)
    for j, ch in enumerate(target_channels):
        i = src_index.get(ch.upper())
        if i is not None:
            out[:, j] = x[:, i]
    return out


class EEGWindowDataset(Dataset):
    """Windowed EEG dataset.

    Wraps pre-loaded windows of shape (n_samples, n_channels, n_times)
    with optional integer labels (unlabeled for SSL pretraining).
    """

    def __init__(
        self,
        windows: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> None:
        """Initialize the instance.

        Args:
        windows (np.ndarray): windows.
        labels (np.ndarray | None): labels.
        """
        if windows.ndim != 3:
            raise ValueError("windows must be (n_samples, n_channels, n_times)")
        if labels is not None and len(labels) != len(windows):
            raise ValueError("labels length must match n_samples")
        self.windows = torch.as_tensor(windows, dtype=torch.float32)
        self.labels = None if labels is None else torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        """Return the number of items.

        Returns:
        int: the result.
        """
        return len(self.windows)

    def __getitem__(self, idx: int):
        """Getitem.

        Args:
        idx (int): idx.

        Returns:
        The result.
        """
        if self.labels is None:
            return self.windows[idx]
        return self.windows[idx], self.labels[idx]
