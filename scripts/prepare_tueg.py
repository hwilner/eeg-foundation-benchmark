#!/usr/bin/env python
"""Stage a TUH-style EDF corpus (TUEG/TUAB/TUSZ) into windowed training format.

Usage
-----
TUH corpus (requires signed DUA — see docs/DATA_ACCESS.md):

    python scripts/prepare_tueg.py --raw-root /path/to/tuh_eeg --out data/tueg_windows

Credential-free public validation set (PhysioNet eegmmidb, open access):

    python scripts/prepare_tueg.py --public-demo --out data/demo_windows

Outputs: ``windows.npz`` (windows, recording_id, channels, target_sfreq)
and ``manifest.csv`` in the output directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from eegfm import tuh


def main() -> None:
    """Main."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--raw-root", type=Path, default=None,
                   help="Root of the raw TUH EDF tree (rsync'ed after DUA approval).")
    p.add_argument("--out", type=Path, default=Path("data/windows"),
                   help="Output directory for windows.npz and manifest.csv.")
    p.add_argument("--sfreq", type=float, default=128.0, help="Target sampling rate (Hz).")
    p.add_argument("--window-sec", type=float, default=2.0, help="Window length (s).")
    p.add_argument("--step-sec", type=float, default=None,
                   help="Window step (s); defaults to non-overlapping windows.")
    p.add_argument("--max-recordings", type=int, default=None,
                   help="Cap the number of staged recordings (smoke runs).")
    p.add_argument("--public-demo", action="store_true",
                   help="Download the openly licensed PhysioNet eegmmidb sample "
                        "(no credentials) and stage it instead of a TUH tree.")
    args = p.parse_args()

    if args.public_demo:
        raw_root = Path("data/public_eegmmidb")
        print(f"Downloading public EEG (eegmmidb, PhysioNet open access) to {raw_root} ...")
        paths = tuh.download_eegmmidb(raw_root)
        print(f"Downloaded {len(paths)} recordings.")
    else:
        raw_root = args.raw_root
        if raw_root is None:
            raise SystemExit(
                "Provide --raw-root pointing at a TUH EDF tree (requires a signed "
                "DUA; see docs/DATA_ACCESS.md), or use --public-demo to stage the "
                "credential-free PhysioNet eegmmidb sample."
            )

    manifest = tuh.prepare_corpus(
        raw_root, args.out, target_sfreq=args.sfreq,
        window_sec=args.window_sec, step_sec=args.step_sec,
        max_recordings=args.max_recordings,
    )
    ok = manifest[manifest["status"] == "ok"]
    print(f"Staged {int(ok['n_windows'].sum())} windows from {len(ok)} recordings "
          f"({len(manifest) - len(ok)} failed integrity checks).")
    print(f"Wrote {args.out / 'windows.npz'} and {args.out / 'manifest.csv'}")


if __name__ == "__main__":
    main()
