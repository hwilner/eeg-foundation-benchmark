import numpy as np

from eegfm.data import EEGWindowDataset
from eegfm.models import EEGTransformerEncoder
from eegfm.pretrain import pretrain_mae
from eegfm.simulate import generate_synthetic_eeg


def test_synthetic_generator_shapes_and_labels():
    X, y = generate_synthetic_eeg(n_samples=40, n_channels=8, n_times=128, seed=1)
    assert X.shape == (40, 8, 128)
    assert X.dtype == np.float32
    assert set(np.unique(y)) == {0, 1}


def test_synthetic_label_is_phase_coupling_not_power():
    # per-channel band power must be uninformative: mean power is
    # identical across classes in both channel groups
    X, y = generate_synthetic_eeg(n_samples=200, n_channels=8, n_times=128, seed=2)
    power = (X**2).mean(axis=-1)  # (n, C)
    diff = np.abs(power[y == 0].mean(axis=0) - power[y == 1].mean(axis=0))
    assert diff.max() < 0.15


def test_pretraining_reduces_reconstruction_loss():
    X, _ = generate_synthetic_eeg(n_samples=160, n_channels=8, n_times=128, seed=0)
    enc = EEGTransformerEncoder(
        n_channels=8, n_times=128, patch_len=16, d_model=32, n_heads=4, n_layers=1
    )
    history = pretrain_mae(enc, EEGWindowDataset(X), epochs=20, batch_size=32, seed=0)
    assert len(history) == 20
    assert history[-1] < history[0]
