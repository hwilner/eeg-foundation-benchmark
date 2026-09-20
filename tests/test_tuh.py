"""Tests for eegfm.tuh staging utilities (synthetic EDF fixtures; no corpus needed).

EDF-reading tests require pyedflib (used to *write* a tiny synthetic
fixture and read it back); they are skipped if it is not installed.
"""

import numpy as np
import pytest

from eegfm import tuh
from eegfm.data import CANONICAL_CHANNELS

pyedflib = pytest.importorskip("pyedflib")


def test_normalize_channel_name():
    assert tuh.normalize_channel_name("EEG FP1-REF") == "FP1"
    assert tuh.normalize_channel_name("EEG T3-REF") == "T3"
    assert tuh.normalize_channel_name("FP1-LE") == "FP1"
    assert tuh.normalize_channel_name("Fp2.") == "FP2"
    assert tuh.normalize_channel_name("T7") == "T3"  # 10-10 alias
    assert tuh.normalize_channel_name("P8") == "T6"  # 10-10 alias
    assert tuh.normalize_channel_name("ECG") is None
    assert tuh.normalize_channel_name("A1") is None
    assert tuh.normalize_channel_name("EEG EKG-REF") is None


def test_missing_corpus_raises_with_access_pointer(tmp_path):
    with pytest.raises(tuh.TUHDataNotFoundError) as excinfo:
        tuh.build_manifest(tmp_path / "empty")
    assert "docs/DATA_ACCESS.md" in str(excinfo.value)


def _write_synthetic_edf(path, channels, sfreq=128.0, seconds=6.0):
    n = int(sfreq * seconds)
    t = np.arange(n) / sfreq
    writer = pyedflib.EdfWriter(str(path), len(channels))
    headers = []
    for i, ch in enumerate(channels):
        headers.append(
            {
                "label": ch,
                "dimension": "uV",
                "sample_frequency": sfreq,
                "physical_min": -200.0,
                "physical_max": 200.0,
                "digital_min": -32768,
                "digital_max": 32767,
                "prefilter": "",
                "transducer": "",
            }
        )
        writer.setSignalHeader(i, headers[-1])
    data = np.stack(
        [50.0 * np.sin(2 * np.pi * 10 * t + i) for i in range(len(channels))]
    )
    writer.writeSamples(data)
    writer.close()


def test_window_recording_and_manifest(tmp_path):
    # TUH-style labels with -REF suffixes.
    channels = [f"EEG {c}-REF" for c in CANONICAL_CHANNELS[:8]]
    edf = tmp_path / "sub-01" / "rec1.edf"
    edf.parent.mkdir(parents=True)
    _write_synthetic_edf(edf, channels)

    wins, info = tuh.window_recording(edf, target_sfreq=128.0, window_sec=1.0)
    assert wins.shape == (6, len(CANONICAL_CHANNELS), 128)
    assert wins.dtype == np.float32
    assert info["n_matched_channels"] == 8
    # Known channels are preserved (per-window standardized => non-zero),
    # unmeasured canonical channels are zero-filled.
    ch_std = wins.std(axis=(0, 2))
    assert (ch_std[:8] > 0.5).all()
    assert (ch_std[8:] == 0).all()

    manifest = tuh.build_manifest(tmp_path, target_sfreq=128.0, window_sec=1.0)
    assert len(manifest) == 1
    assert manifest.iloc[0]["status"] == "ok"
    assert manifest.iloc[0]["n_windows"] == 6

    out = tmp_path / "staged"
    tuh.prepare_corpus(tmp_path, out, target_sfreq=128.0, window_sec=1.0)
    data = np.load(out / "windows.npz")
    assert data["windows"].shape == (6, len(CANONICAL_CHANNELS), 128)
    assert (out / "manifest.csv").exists()


def test_resample_signal():
    x = np.random.default_rng(0).standard_normal((3, 256))
    y = tuh.resample_signal(x, 256.0, 128.0)
    assert y.shape == (3, 128)
    z = tuh.resample_signal(x, 128.0, 128.0)
    assert np.array_equal(z, x)
