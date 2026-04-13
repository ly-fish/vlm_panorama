"""Distortion encoder for panoramic projection parameters.

Encodes theta = (lat, lon, FoV) into a dense geometric prior vector z_theta via
sinusoidal positional encoding followed by a lightweight MLP:

    z_theta = MLP_dist([phi(lat), phi(lon), phi(FoV)])

where phi(.) is a multi-frequency sinusoidal embedding.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ProjectionParams:
    """Projection parameters for a perspective view into an ERP panorama."""

    lat: float   # latitude of view centre in degrees  [-90, 90]
    lon: float   # longitude of view centre in degrees [-180, 180]
    fov: float   # horizontal field-of-view in degrees [30, 150]

    def to_tensor(self, device: str | torch.device = "cpu") -> torch.Tensor:
        """Return a [1, 3] float32 tensor with values normalised to [-1, 1]."""
        lat_n = self.lat / 90.0
        lon_n = self.lon / 180.0
        fov_n = (self.fov - 90.0) / 90.0   # centres 90 deg -> 0
        return torch.tensor([[lat_n, lon_n, fov_n]], dtype=torch.float32, device=device)

    @staticmethod
    def from_box(
        box: list[float],
        img_w: int,
        img_h: int,
        fov: float | None = None,
        margin: float = 1.5,
    ) -> "ProjectionParams":
        """Compute projection params from a bounding box in ERP pixel coordinates.

        Args:
            box:    [x1, y1, x2, y2] in pixel coordinates.
            img_w:  ERP image width in pixels.
            img_h:  ERP image height in pixels.
            fov:    Desired horizontal FoV in degrees.  If *None* (default),
                    the FoV is computed adaptively from the box width (longitude
                    span × ``margin``), clamped to [30°, 150°].
            margin: Multiplicative safety margin applied to the box angular span
                    when computing the adaptive FoV (default 1.5).

        Returns:
            ProjectionParams with lat/lon pointing at the box centre.
        """
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        lon = (cx / img_w - 0.5) * 360.0
        lat = (0.5 - cy / img_h) * 180.0
        if fov is None:
            box_w = box[2] - box[0]
            box_h = box[3] - box[1]
            # Angular span of box in ERP space (degrees)
            fov_from_w = (box_w / img_w) * 360.0 * margin
            fov_from_h = (box_h / img_h) * 180.0 * margin
            fov = float(max(30.0, min(150.0, max(fov_from_w, fov_from_h))))
        return ProjectionParams(lat=lat, lon=lon, fov=fov)


class SinusoidalEmbedding(nn.Module):
    """Multi-frequency sinusoidal embedding for a scalar batch.

    For each scalar x the output is:
        [sin(2^0 * pi * x), cos(2^0 * pi * x), ..., sin(2^(L-1) * pi * x), cos(2^(L-1) * pi * x)]
    producing a 2L-dimensional vector per element.
    """

    def __init__(self, num_frequencies: int = 8) -> None:
        super().__init__()
        self.num_frequencies = num_frequencies
        freqs = (2.0 ** torch.arange(num_frequencies, dtype=torch.float32)) * math.pi
        self.register_buffer("freqs", freqs)

    @property
    def output_dim(self) -> int:
        return 2 * self.num_frequencies

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B] -> [B, 2 * num_frequencies]"""
        angles = x.unsqueeze(-1) * self.freqs          # [B, L]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class DistortionEncoder(nn.Module):
    """Encodes (lat, lon, FoV) projection parameters into geometric prior z_theta.

    Architecture
    ------------
    1. Sinusoidal embedding applied independently to each scalar (lat, lon, fov).
    2. Concatenated embeddings fed through a 3-layer MLP with GELU activations.

    Args:
        hidden_dim:      Width of MLP hidden layers.
        out_dim:         Dimensionality of output z_theta (condition vector).
        num_frequencies: Number of frequency bands per sinusoidal embedding.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        out_dim: int = 512,
        num_frequencies: int = 8,
    ) -> None:
        super().__init__()
        self.sin_embed = SinusoidalEmbedding(num_frequencies)
        in_dim = 3 * self.sin_embed.output_dim   # lat + lon + fov each embedded

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    @property
    def out_dim(self) -> int:
        return self.mlp[-1].out_features

    def forward(self, params: "ProjectionParams | torch.Tensor") -> torch.Tensor:
        """
        Args:
            params: :class:`ProjectionParams` or float tensor of shape [B, 3]
                    with (lat_norm, lon_norm, fov_norm) normalised to [-1, 1].

        Returns:
            z_theta: tensor of shape [B, out_dim].
        """
        if isinstance(params, ProjectionParams):
            x = params.to_tensor(next(self.parameters()).device)   # [1, 3]
        else:
            x = params.to(next(self.parameters()).device)          # [B, 3]

        lat_emb = self.sin_embed(x[:, 0])   # [B, 2L]
        lon_emb = self.sin_embed(x[:, 1])   # [B, 2L]
        fov_emb = self.sin_embed(x[:, 2])   # [B, 2L]

        combined = torch.cat([lat_emb, lon_emb, fov_emb], dim=-1)  # [B, 6L]
        return self.mlp(combined)                                    # [B, out_dim]
