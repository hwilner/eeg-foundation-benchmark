"""TUH EEG Corpus (TUEG/TUAB/TUSZ) staging: EDF loading and windowing.

This module turns a raw TUH-style EDF tree into the windowed training
format used by the rest of the package:

- EDF reading via `pyedflib` (preferred) or `mne` (optional fallback);
- channel-label normalization from TUH labels (e.g. ``EEG FP1-REF``,
``FP1-LE``) to the canonical 19-channel 10-20 montage
(:data:`eegfm.data.CANONICAL_CHANNELS`);
- resampling to a common sampling rate (scipy polyphase);
- fixed-length windowing into (n_windows, n_channels, n_times) arrays;
- integrity checks and a per-recording manifest (pandas DataFrame /
CSV).

The TUH EEG Corpus is **not publicly downloadable** — it requires a
signed DUA with Temple University. All staging entry points raise
:class:`TUHDataNotFoundError` with a pointer to ``docs/DATA_ACCESS.md``
when no EDF files are present.

A credential-free validation path is provided via PhysioNet's openly
licensed EEG Motor Movement/Imagery Dataset (eegmmidb), which uses the
same EDF format and a superset of the 10-20 montage:
:func:`download_eegmmidb` fetches recordings over plain HTTPS (no
account required) so the full pretrain/benchmark pipeline can be
exercised on real EEG before TUH access is granted.
"""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .data import CANONICAL_CHANNELS, standardize_channels

DATA_ACCESS_DOC = "docs/DATA_ACCESS.md"

# Openly licensed public EEG used to validate the pipeline end-to-end
# without TUH credentials (PhysioNet eegmmidb, ODC-BY license).
EEGMMIDB_BASE_URL = "https://physionet.org/files/eegmmidb/1.0.0"
EEGMMIDB_DEFAULT_FILES = (
    "S001/S001R01.edf",  # baseline, eyes open
    "S001/S001R02.edf",  # baseline, eyes closed
    "S001/S001R03.edf",  # motor imagery task 1
    "S002/S002R01.edf",
    "S002/S002R02.edf",
    "S002/S002R03.edf",
)


class TUHDataNotFoundError(FileNotFoundError):
    """Raised when no EDF corpus is staged at the expected location."""


# Map raw EDF channel labels (TUH "EEG FP1-REF", "FP1-LE", plain "Fp1",
# eegmmidb "Fc5.", etc.) to canonical 10-20 names.
_EARLOBE = {"A1", "A2", "M1", "M2"}

# 10-10 equivalents folded onto the canonical 10-20 montage (e.g. the
# eegmmidb/BCI2000 cap uses the 10-10 names T7/T8/P7/P8).
_ALIASES = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}


def normalize_channel_name(label: str) -> str | None:
    """Normalize a raw EDF channel label to a canonical 10-20 name.

    Handles TUH conventions (``EEG FP1-REF``, ``EEG T3-REF``, ``FP1-LE``)
    and dotted/BDF-style labels (``Fc5.``). Returns None for non-scalp
    channels (ECG, EMG, annotations, earlobe references, ...).
    """
    lab = label.strip().upper()
    lab = re.sub(r"^EEG\s+", "", lab)
    lab = re.sub(r"[-\s]?(REF|LE|AR|AVG)$", "", lab)
    lab = lab.rstrip(".")
    if lab in _EARLOBE or not re.fullmatch(r"[A-Z]{1,2}\d{0,2}Z?", lab):
        return None
    lab = _ALIASES.get(lab, lab)
    return lab if lab in CANONICAL_CHANNELS else None


def _read_edf_pyedflib(path: Path):
    import pyedflib

    with pyedflib.EdfReader(str(path)) as f:
        labels = f.getSignalLabels()
        sfreqs = [f.getSampleFrequency(i) for i in range(f.signals_in_file)]
        n = f.getNSamples()
        data = np.stack(
            [f.readSignal(i) for i in range(f.signals_in_file)], axis=0
        ).astype(np.float64)
    if len(set(np.round(sfreqs, 6))) != 1:
        raise ValueError(f"{path}: heterogeneous sampling rates {set(sfreqs)}")
    if len({n_i for n_i in n}) != 1:
        # truncate to shortest channel
        m = min(n)
        data = data[:, :m]
    return data, float(sfreqs[0]), labels


def _read_edf_mne(path: Path):
    import mne

    raw = mne.io.read_raw_edf(str(path), preload=True, verbose="ERROR")
    return raw.get_data(), float(raw.info["sfreq"]), list(raw.ch_names)


def read_edf(path: str | Path):
    """Read an EDF recording.

    Returns:
    -------
    data : (n_channels, n_samples) float array (microvolts as stored)
    sfreq : float sampling rate in Hz
    channel_names : list of raw channel labels

    Raises ImportError with install guidance if neither `pyedflib` nor
    `mne` is available.
    """
    path = Path(path)
    try:
        import pyedflib  # noqa: F401

        return _read_edf_pyedflib(path)
    except ImportError:
        pass
    try:
        import mne  # noqa: F401

        return _read_edf_mne(path)
    except ImportError as exc:
        raise ImportError(
            "Reading EDF files requires `pyedflib` (pip install pyedflib) "
            "or `mne` (pip install mne)."
        ) from exc


def resample_signal(data: np.ndarray, sfreq: float, target_sfreq: float) -> np.ndarray:
    """Polyphase-resample (n_channels, n_samples) to `target_sfreq` Hz."""
    if np.isclose(sfreq, target_sfreq):
        return data
    from scipy.signal import resample_poly

    from fractions import Fraction

    ratio = Fraction(target_sfreq / sfreq).limit_denominator(1000)
    return resample_poly(data, ratio.numerator, ratio.denominator, axis=1)


def window_recording(
    path: str | Path,
    target_sfreq: float = 128.0,
    window_sec: float = 2.0,
    step_sec: float | None = None,
    target_channels: tuple[str, ...] = CANONICAL_CHANNELS,
    min_matched_channels: int = 4,
) -> tuple[np.ndarray, dict]:
    """Load one EDF recording and return standardized windows.

    Steps: read EDF -> map labels onto the canonical montage -> resample
    -> per-window/channel standardization -> fixed-length windows.

    Returns:
    -------
    windows : (n_windows, len(target_channels), n_times) float32 array
    info : dict with recording metadata (sfreq, duration_sec,
    matched_channels, n_windows) for the manifest.
    """
    data, sfreq, labels = read_edf(path)
    norm = [normalize_channel_name(c) for c in labels]
    keep = [i for i, c in enumerate(norm) if c is not None]
    matched = [norm[i] for i in keep]
    # Deduplicate (first occurrence wins).
    seen, keep2, matched2 = set(), [], []
    for i, c in zip(keep, matched):
        if c not in seen:
            seen.add(c)
            keep2.append(i)
            matched2.append(c)
    if len(keep2) < min_matched_channels:
        raise ValueError(
            f"{path}: only {len(keep2)} canonical channels matched "
            f"(need >= {min_matched_channels})"
        )
    data = data[keep2]
    data = resample_signal(data, sfreq, target_sfreq)

    n_times = int(round(window_sec * target_sfreq))
    step = int(round((step_sec or window_sec) * target_sfreq))
    if data.shape[1] < n_times:
        raise ValueError(f"{path}: recording shorter than one window")
    starts = np.arange(0, data.shape[1] - n_times + 1, step)
    wins = np.stack([data[:, s : s + n_times] for s in starts], axis=0)
    wins = standardize_channels(wins, matched2, target_channels)
    # Per-window, per-channel standardization (matches simulate.py).
    wins = (wins - wins.mean(axis=-1, keepdims=True)) / (
        wins.std(axis=-1, keepdims=True) + 1e-6
    )
    info = {
        "sfreq": sfreq,
        "duration_sec": data.shape[1] / target_sfreq,
        "n_raw_channels": len(labels),
        "n_matched_channels": len(matched2),
        "matched_channels": ",".join(matched2),
        "n_windows": len(starts),
    }
    return wins.astype(np.float32), info


def iter_edf_files(root: str | Path) -> list[Path]:
    """Recursively list .edf files under `root` (sorted)."""
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.suffix.lower() == ".edf")


def build_manifest(
    root: str | Path,
    target_sfreq: float = 128.0,
    window_sec: float = 2.0,
    step_sec: float | None = None,
    target_channels: tuple[str, ...] = CANONICAL_CHANNELS,
) -> pd.DataFrame:
    """Scan an EDF tree and return a per-recording integrity manifest.

    Each readable recording yields one row: relative path, sfreq,
    duration, matched channel count, number of extracted windows, and
    a status flag ("ok" or the error message). Raises
    :class:`TUHDataNotFoundError` if no EDF files are found.
    """
    root = Path(root)
    files = iter_edf_files(root)
    if not files:
        raise TUHDataNotFoundError(
            f"No EDF files found under {root}. The TUH EEG Corpus requires "
            f"a signed DUA with Temple University — see {DATA_ACCESS_DOC} "
            "for access steps, or use `download_eegmmidb` / "
            "`scripts/prepare_tueg.py --public-demo` for a credential-free "
            "public EEG validation set."
        )
    rows = []
    for p in files:
        rel = str(p.relative_to(root))
        try:
            _, info = window_recording(
                p, target_sfreq=target_sfreq, window_sec=window_sec,
                step_sec=step_sec, target_channels=target_channels,
            )
            rows.append({"file": rel, "status": "ok", **info})
        except Exception as exc:  # corrupt/unsupported recording
            rows.append({"file": rel, "status": f"error: {exc}"})
    return pd.DataFrame(rows)


def prepare_corpus(
    root: str | Path,
    out_dir: str | Path,
    target_sfreq: float = 128.0,
    window_sec: float = 2.0,
    step_sec: float | None = None,
    target_channels: tuple[str, ...] = CANONICAL_CHANNELS,
    max_recordings: int | None = None,
) -> pd.DataFrame:
    """Stage an EDF corpus into the windowed training format.

    Writes ``windows.npz`` (windows + recording index) and
    ``manifest.csv`` to `out_dir`. Returns the manifest DataFrame.
    Raises :class:`TUHDataNotFoundError` if the corpus is absent.
    """
    root, out_dir = Path(root), Path(out_dir)
    manifest = build_manifest(
        root, target_sfreq=target_sfreq, window_sec=window_sec,
        step_sec=step_sec, target_channels=target_channels,
    )
    ok = manifest[manifest["status"] == "ok"]
    if max_recordings is not None:
        ok = ok.head(max_recordings)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_wins, rec_ids = [], []
    for i, row in enumerate(ok.itertuples()):
        wins, _ = window_recording(
            root / row.file, target_sfreq=target_sfreq, window_sec=window_sec,
            step_sec=step_sec, target_channels=target_channels,
        )
        all_wins.append(wins)
        rec_ids.extend([i] * len(wins))
    if not all_wins:
        raise TUHDataNotFoundError(
            f"EDF files were found under {root} but none passed integrity "
            f"checks; see manifest. Access guidance: {DATA_ACCESS_DOC}."
        )
    windows = np.concatenate(all_wins, axis=0)
    np.savez_compressed(
        out_dir / "windows.npz",
        windows=windows,
        recording_id=np.asarray(rec_ids, dtype=np.int64),
        target_sfreq=target_sfreq,
        channels=np.asarray(target_channels),
    )
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    return manifest


def download_eegmmidb(
    out_dir: str | Path,
    files: tuple[str, ...] = EEGMMIDB_DEFAULT_FILES,
    base_url: str = EEGMMIDB_BASE_URL,
) -> list[Path]:
    """Download openly licensed public EEG (PhysioNet eegmmidb) as EDF.

    The EEG Motor Movement/Imagery Dataset is open-access (ODC-BY); no
    account or DUA is required. It is used to validate the full
    pretrain/benchmark pipeline on real EEG while TUH access (signed
    DUA, see docs/DATA_ACCESS.md) is pending.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for rel in files:
        dest = out_dir / rel.replace("/", "_")
        if not dest.exists():
            urllib.request.urlretrieve(f"{base_url}/{rel}", dest)
        paths.append(dest)
    return paths
