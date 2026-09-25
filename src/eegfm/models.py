"""Compact transformer encoder for multichannel EEG windows.

Patches span (channel, time) tiles: each patch is a contiguous time
slice of a single channel. A learned linear projection embeds each
patch; learned positional encodings index (channel, time-position).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EEGTransformerEncoder(nn.Module):
    """Transformer encoder over (channel, time-patch) tokens."""

    def __init__(
        self,
        n_channels: int = 8,
        n_times: int = 256,
        patch_len: int = 32,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the instance.

        Args:
        n_channels (int): n channels.
        n_times (int): n times.
        patch_len (int): patch len.
        d_model (int): d model.
        n_heads (int): n heads.
        n_layers (int): n layers.
        dropout (float): dropout.
        """
        super().__init__()
        if n_times % patch_len != 0:
            raise ValueError("n_times must be divisible by patch_len")
        self.n_channels = n_channels
        self.n_times = n_times
        self.patch_len = patch_len
        self.n_time_patches = n_times // patch_len
        self.n_patches = n_channels * self.n_time_patches

        self.patch_embed = nn.Linear(patch_len, d_model)
        # one positional vector per (channel, time-patch) token
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def to_patches(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T) -> (B, C * T//P, P)."""
        B, C, T = x.shape
        x = x.view(B, C, self.n_time_patches, self.patch_len)
        return x.reshape(B, self.n_patches, self.patch_len)

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Embed and encode; returns token sequence (B, n_patches, d_model)."""
        tokens = self.patch_embed(self.to_patches(x)) + self.pos_embed
        return self.norm(self.encoder(tokens))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode windows; returns mean-pooled embedding (B, d_model)."""
        return self.forward_tokens(x).mean(dim=1)

    def forward_channel_features(self, x: torch.Tensor) -> torch.Tensor:
        """Per-channel mean-pooled features, flattened: (B, C * d_model).

        Preserves which channel carries which pattern — important for
        linear readout of spatial/cross-channel structure.
        """
        tokens = self.forward_tokens(x)
        B = tokens.shape[0]
        tokens = tokens.view(B, self.n_channels, self.n_time_patches, self.d_model)
        return tokens.mean(dim=2).reshape(B, self.n_channels * self.d_model)


class LinearProbe(nn.Module):
    """Linear classification head on channel-pooled encoder features."""

    def __init__(self, encoder: EEGTransformerEncoder, n_classes: int = 2) -> None:
        """Initialize the instance.

        Args:
        encoder (EEGTransformerEncoder): encoder.
        n_classes (int): n classes.
        """
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.n_channels * encoder.d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
        x (torch.Tensor): x.

        Returns:
        torch.Tensor: the result.
        """
        return self.head(self.encoder.forward_channel_features(x))


class MaskedReconstructionModel(nn.Module):
    """Channel-masked autoencoder wrapper for SSL pretraining.

    Entire channels are masked (replaced with a learned mask token) and
    the decoder must reconstruct their per-patch-standardized waveform
    from the visible channels. Block masking prevents the trivial
    temporal-interpolation shortcut of random patch masking and forces
    the encoder to learn cross-channel structure.
    """

    def __init__(self, encoder: EEGTransformerEncoder) -> None:
        """Initialize the instance.

        Args:
        encoder (EEGTransformerEncoder): encoder.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = nn.Sequential(
            nn.Linear(encoder.d_model, encoder.d_model),
            nn.GELU(),
            nn.Linear(encoder.d_model, encoder.patch_len),
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, encoder.d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(
        self, x: torch.Tensor, channel_mask_ratio: float = 0.4
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, normalized_target_patches, mask).

        reconstruction/target: (B, n_patches, patch_len); mask: (B, n_patches)
        bool, True where the patch belongs to a masked-out channel.
        """
        enc = self.encoder
        B = x.shape[0]
        device = x.device
        tokens = enc.patch_embed(enc.to_patches(x))

        n_masked = max(1, int(enc.n_channels * channel_mask_ratio))
        ids = torch.rand(B, enc.n_channels, device=device).argsort(dim=1)[:, :n_masked]
        chan_mask = torch.zeros(B, enc.n_channels, dtype=torch.bool, device=device)
        chan_mask.scatter_(1, ids, True)
        mask = chan_mask.repeat_interleave(enc.n_time_patches, dim=1)

        visible = torch.where(
            mask.unsqueeze(-1), self.mask_token.expand_as(tokens), tokens
        )
        encoded = enc.norm(enc.encoder(visible + enc.pos_embed))
        recon = self.decoder(encoded)

        target = enc.to_patches(x)
        target = (target - target.mean(dim=-1, keepdim=True)) / (
            target.std(dim=-1, keepdim=True) + 1e-5
        )
        return recon, target, mask
