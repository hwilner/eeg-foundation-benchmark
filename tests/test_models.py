import numpy as np
import torch

from eegfm.data import EEGWindowDataset, standardize_channels
from eegfm.models import EEGTransformerEncoder, LinearProbe, MaskedReconstructionModel


def test_standardize_channels_reorders_and_zerofills():
    x = np.arange(2 * 3 * 5, dtype=np.float32).reshape(2, 3, 5)
    out = standardize_channels(x, ["C3", "C4", "FZ"], ["FZ", "C3", "C4", "PZ"])
    assert out.shape == (2, 4, 5)
    np.testing.assert_array_equal(out[:, 0], x[:, 2])  # FZ
    np.testing.assert_array_equal(out[:, 1], x[:, 0])  # C3
    assert np.all(out[:, 3] == 0)  # PZ missing -> zeros


def test_standardize_channels_preserves_known_signal():
    rng = np.random.default_rng(0)
    t = np.arange(64) / 64.0
    sig = np.sin(2 * np.pi * 8 * t)[None, None, :].repeat(2, 0)
    x = np.concatenate([sig, rng.normal(size=(2, 2, 64))], axis=1).astype(np.float32)
    out = standardize_channels(x, ["C3", "C4", "XX1", "XX2"], ["C3", "C4"])
    np.testing.assert_allclose(out[:, 0], sig[:, 0], atol=1e-6)


def test_dataset_interface():
    X = np.zeros((10, 4, 64), dtype=np.float32)
    y = np.arange(10) % 2
    ds = EEGWindowDataset(X, y)
    assert len(ds) == 10
    xb, yb = ds[0]
    assert xb.shape == (4, 64)
    unlabeled = EEGWindowDataset(X)
    assert not isinstance(unlabeled[0], tuple)


def test_encoder_forward_shapes():
    enc = EEGTransformerEncoder(
        n_channels=6, n_times=128, patch_len=16, d_model=32, n_heads=4, n_layers=1
    )
    x = torch.randn(5, 6, 128)
    assert enc(x).shape == (5, 32)
    tokens = enc.forward_tokens(x)
    assert tokens.shape == (5, 6 * 8, 32)
    feats = enc.forward_channel_features(x)
    assert feats.shape == (5, 6 * 32)


def test_linear_probe_shape():
    enc = EEGTransformerEncoder(n_channels=4, n_times=64, patch_len=16, d_model=16, n_layers=1)
    probe = LinearProbe(enc, n_classes=2)
    assert probe.head.in_features == 4 * 16
    assert probe(torch.randn(3, 4, 64)).shape == (3, 2)


def test_channel_masked_reconstruction_shapes():
    enc = EEGTransformerEncoder(n_channels=4, n_times=64, patch_len=16, d_model=16, n_layers=1)
    model = MaskedReconstructionModel(enc)
    recon, target, mask = model(torch.randn(3, 4, 64), channel_mask_ratio=0.5)
    assert recon.shape == target.shape == (3, 16, 16)
    assert mask.shape == (3, 16)
    # whole channels masked: mask pattern repeats per channel
    mask_c = mask.view(3, 4, 4)
    assert (mask_c.all(-1) | ~mask_c.any(-1)).all()
    assert mask.any() and (~mask).any()
