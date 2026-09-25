#!/usr/bin/env python
"""Run the SSL-pretrained vs from-scratch benchmark on staged windows.

Two modes:

- ``--windows`` + ``--labels``: a windows.npz from scripts/prepare_tueg.py
  and a CSV with one label per window (column ``label``).
- ``--synthetic``: data-free validation using eegfm.simulate (default task).

Example (public-data validation, eyes-open vs eyes-closed):

    python scripts/prepare_tueg.py --public-demo --out data/demo_windows
    python scripts/run_benchmark.py --public-demo-task

The public-demo task labels each window by its source recording
(baseline eyes-open vs eyes-closed from eegmmidb runs R01/R02).

Writes the results table to ``reports/benchmark_results.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from eegfm import tuh
from eegfm.benchmark import run_benchmark
from eegfm.simulate import generate_synthetic_eeg


def _load_public_demo_task(windows_path: Path):
    """Eyes-open (R01) vs eyes-closed (R02) task from eegmmidb staging."""
    data = np.load(windows_path)
    windows, rec_ids = data["windows"], data["recording_id"]
    manifest = pd.read_csv(windows_path.parent / "manifest.csv")
    ok = manifest[manifest["status"] == "ok"].reset_index(drop=True)
    # R01 = baseline eyes open (label 0), R02 = eyes closed (label 1)
    rec_label = {
        i: (1 if "R02" in f else 0)
        for i, f in enumerate(ok["file"])
        if "R01" in f or "R02" in f
    }
    keep = np.array([r in rec_label for r in rec_ids])
    X = windows[keep]
    y = np.array([rec_label[r] for r in rec_ids[keep]], dtype=np.int64)
    if len(np.unique(y)) < 2:
        raise SystemExit("Public-demo task needs both R01 and R02 recordings staged.")
    return X, y


def main() -> None:
    """Main."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--windows", type=Path,
                     help="windows.npz from scripts/prepare_tueg.py.")
    src.add_argument("--synthetic", action="store_true",
                     help="Run on the built-in synthetic generator.")
    src.add_argument("--public-demo-task", action="store_true",
                     help="Eyes-open vs eyes-closed benchmark on the staged "
                          "eegmmidb public demo (data/demo_windows by default).")
    p.add_argument("--labels", type=Path, default=None,
                   help="CSV with a 'label' column, one row per window.")
    p.add_argument("--demo-dir", type=Path, default=Path("data/demo_windows"))
    p.add_argument("--out", type=Path, default=Path("reports/benchmark_results.csv"))
    p.add_argument("--pretrain-epochs", type=int, default=40)
    p.add_argument("--probe-epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.synthetic:
        X, y = generate_synthetic_eeg(seed=args.seed)
    elif args.public_demo_task:
        X, y = _load_public_demo_task(args.demo_dir / "windows.npz")
    else:
        data = np.load(args.windows)
        X = data["windows"]
        if args.labels is None:
            raise SystemExit("--windows requires --labels CSV for the benchmark.")
        y = pd.read_csv(args.labels)["label"].to_numpy(dtype=np.int64)

    print(f"Benchmark on {X.shape[0]} windows of shape {X.shape[1:]} "
          f"(class balance {np.bincount(y).tolist()}).")
    results = run_benchmark(
        X, y, pretrain_epochs=args.pretrain_epochs,
        probe_epochs=args.probe_epochs, seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False)
    print(results.to_string(index=False))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
