#!/usr/bin/env python
"""Pretrain the SSL encoder on staged windows (one command).

    python scripts/prepare_tueg.py --public-demo --out data/demo_windows
    python scripts/run_pretrain.py --windows data/demo_windows/windows.npz

Saves ``encoder.pt`` (state dict + config) and ``pretrain_history.csv``
to the output directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from eegfm.data import EEGWindowDataset
from eegfm.models import EEGTransformerEncoder
from eegfm.pretrain import pretrain_mae


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--windows", type=Path, required=True,
                   help="Path to windows.npz produced by scripts/prepare_tueg.py.")
    p.add_argument("--out", type=Path, default=Path("checkpoints"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--mask-ratio", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    data = np.load(args.windows)
    windows = data["windows"]
    print(f"Loaded {windows.shape[0]} windows of shape {windows.shape[1:]}.")

    encoder = EEGTransformerEncoder(
        n_channels=windows.shape[1], n_times=windows.shape[2]
    )
    history = pretrain_mae(
        encoder, EEGWindowDataset(windows), epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, mask_ratio=args.mask_ratio,
        seed=args.seed,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": encoder.state_dict(),
            "config": {"n_channels": windows.shape[1], "n_times": windows.shape[2]},
            "source": str(args.windows),
        },
        args.out / "encoder.pt",
    )
    pd.DataFrame({"epoch": np.arange(1, len(history) + 1), "mse": history}).to_csv(
        args.out / "pretrain_history.csv", index=False
    )
    print(f"Pretraining done: mse {history[0]:.4f} -> {history[-1]:.4f}. "
          f"Saved {args.out / 'encoder.pt'} and pretrain_history.csv.")


if __name__ == "__main__":
    main()
