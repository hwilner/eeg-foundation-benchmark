"""eegfm: self-supervised EEG foundation model benchmarking (Paper 1)."""

from .data import CANONICAL_CHANNELS, EEGWindowDataset, standardize_channels
from .models import EEGTransformerEncoder, LinearProbe, MaskedReconstructionModel
from .simulate import generate_synthetic_eeg

__all__ = [
    "CANONICAL_CHANNELS",
    "EEGWindowDataset",
    "standardize_channels",
    "EEGTransformerEncoder",
    "LinearProbe",
    "MaskedReconstructionModel",
    "generate_synthetic_eeg",
]

__version__ = "0.1.0"
