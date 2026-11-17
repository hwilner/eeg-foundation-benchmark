"""Compact transformer encoder for multichannel EEG windows.

Patches span (channel, time) tiles: each patch is a contiguous time
slice of a single channel. A learned linear projection embeds each
patch; learned positional encodings index (channel, time-position).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EEGTransformerEncoder(nn.Module):
    """Transformer encoder over (channel, time-patch) tokens.

    Attributes:
        n_channels: Number of EEG channels per window.
        n_times: Number of time samples per window.
        patch_len: Number of time samples per (channel, time) patch.
        n_time_patches: Number of time patches per channel
            (``n_times // patch_len``).
        n_patches: Total number of tokens per window
            (``n_channels * n_time_patches``).
        d_model: Embedding width of each patch token.
    """

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
        """Build the encoder stack for a fixed window geometry.

        Args:
            n_channels: Number of EEG channels per window.
            n_times: Number of time samples per window; must be divisible
                by `patch_len`.
            patch_len: Number of time samples per (channel, time) patch.
            d_model: Embedding width of each patch token.
            n_heads: Number of attention heads per encoder layer.
            n_layers: Number of transformer encoder layers.
            dropout: Dropout probability inside the encoder layers.

        Raises:
            ValueError: If `n_times` is not divisible by `patch_len`.
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
        """Reshape a window batch into (channel, time-patch) tokens.

        Args:
            x: Window batch of shape (B, C, T).

        Returns:
            Tensor of shape (B, C * T // patch_len, patch_len) holding one
            contiguous time slice of a single channel per token.
        """
        B, C, T = x.shape
        x = x.view(B, C, self.n_time_patches, self.patch_len)
        return x.reshape(B, self.n_patches, self.patch_len)

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Embed, position-encode, and encode a window batch.

        Args:
            x: Window batch of shape (B, C, T).

        Returns:
            Encoded token sequence of shape (B, n_patches, d_model).
        """
        tokens = self.patch_embed(self.to_patches(x)) + self.pos_embed
        return self.norm(self.encoder(tokens))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode windows into a single mean-pooled embedding each.

        Args:
            x: Window batch of shape (B, C, T).

        Returns:
            Mean-pooled embeddings of shape (B, d_model).
        """
        return self.forward_tokens(x).mean(dim=1)

    def forward_channel_features(self, x: torch.Tensor) -> torch.Tensor:
        """Encode windows into per-channel features, flattened.

        Preserves which channel carries which pattern — important for
        linear readout of spatial/cross-channel structure.

        Args:
            x: Window batch of shape (B, C, T).

        Returns:
            Tensor of shape (B, C * d_model): each channel's tokens are
            mean-pooled over time patches, then concatenated across channels.
        """
        tokens = self.forward_tokens(x)
        B = tokens.shape[0]
        tokens = tokens.view(B, self.n_channels, self.n_time_patches, self.d_model)
        return tokens.mean(dim=2).reshape(B, self.n_channels * self.d_model)


class LinearProbe(nn.Module):
    """Linear classification head on channel-pooled encoder features."""

    def __init__(self, encoder: EEGTransformerEncoder, n_classes: int = 2) -> None:
        """Attach a linear head to an encoder's channel-pooled features.

        Args:
            encoder: Encoder supplying `forward_channel_features`; its
                channel count and embedding width set the head's input size.
            n_classes: Number of output classes (logits per window).
        """
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.n_channels * encoder.d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify a window batch.

        Args:
            x: Window batch of shape (B, C, T).

        Returns:
            Class logits of shape (B, n_classes).
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
        """Wrap an encoder with a patch-reconstruction decoder.

        Args:
            encoder: Encoder to pretrain; a learned mask token and a small
                decoder mapping `d_model` back to `patch_len` are created
                here.
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
        """Mask whole channels and reconstruct their normalized waveforms.

        A random subset of channels (at least one) is replaced with the
        learned mask token per window; the decoder reconstructs the
        per-patch-standardized waveform of every patch from the visible
        context.

        Args:
            x: Window batch of shape (B, C, T).
            channel_mask_ratio: Fraction of channels to mask per window.

        Returns:
            A tuple ``(reconstruction, target, mask)``: ``reconstruction``
            and ``target`` have shape (B, n_patches, patch_len), with the
            target patches standardized per patch; ``mask`` is a boolean
            tensor of shape (B, n_patches) that is True where the patch
            belongs to a masked-out channel.
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
