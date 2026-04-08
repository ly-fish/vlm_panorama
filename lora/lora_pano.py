"""LoRA-Pano: panoramic geometric adaptation branch.

Maintains a collection of ConditionalLoRALayer modules, one per target linear
layer in the backbone.  Each delta is conditioned on the distortion encoding
z_theta produced by the DistortionEncoder.

    delta_W_pano = B_pano . A_pano(z_theta)

The DistortionEncoder lives here so LoRA-Pano is a self-contained module.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from lora.distortion_encoder import DistortionEncoder, ProjectionParams
from lora.lora_layer import ConditionalLoRALayer


class LoRAPano(nn.Module):
    """Panoramic geometric LoRA branch.

    Wraps a dictionary of ConditionalLoRALayer adapters (keyed by layer name)
    and a shared DistortionEncoder.

    Args:
        layer_configs:  Dict mapping layer name -> dict with keys
                        ``base_layer`` (nn.Linear), ``rank``, ``alpha``.
        encoder_hidden: Hidden dim of DistortionEncoder MLP.
        encoder_out:    Output dim of DistortionEncoder (= cond_dim for FiLM).
        num_frequencies: Sinusoidal frequencies in the distortion encoder.
        dropout:        Dropout in LoRA layers.
    """

    def __init__(
        self,
        layer_configs: dict[str, dict[str, Any]],
        encoder_hidden: int = 256,
        encoder_out: int = 512,
        num_frequencies: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.distortion_encoder = DistortionEncoder(
            hidden_dim=encoder_hidden,
            out_dim=encoder_out,
            num_frequencies=num_frequencies,
        )
        self.adapters = nn.ModuleDict()
        for name, cfg in layer_configs.items():
            self.adapters[name] = ConditionalLoRALayer(
                base_layer=cfg["base_layer"],
                rank=cfg.get("rank", 8),
                cond_dim=encoder_out,
                alpha=cfg.get("alpha"),
                dropout=dropout,
            )

    # ------------------------------------------------------------------
    # Convenience: encode projection parameters
    # ------------------------------------------------------------------
    def encode_projection(
        self, params: ProjectionParams | torch.Tensor
    ) -> torch.Tensor:
        """Return z_theta for the given projection parameters."""
        return self.distortion_encoder(params)

    # ------------------------------------------------------------------
    # Forward: apply a single named adapter
    # ------------------------------------------------------------------
    def forward(
        self,
        layer_name: str,
        x: torch.Tensor,
        projection_params: ProjectionParams | torch.Tensor,
    ) -> torch.Tensor:
        """Apply the LoRA-Pano adapter for *layer_name* to input *x*.

        Args:
            layer_name:        Key in self.adapters.
            x:                 Input tensor [B, *, d_in].
            projection_params: Projection parameters (ProjectionParams or [B, 3] tensor).

        Returns:
            Adapted output [B, *, d_out].
        """
        z_theta = self.encode_projection(projection_params)     # [B, enc_out]
        return self.adapters[layer_name](x, z=z_theta)

    def compute_delta_w(
        self, layer_name: str, projection_params: ProjectionParams | torch.Tensor
    ) -> torch.Tensor:
        """Compute delta_W_pano for a named layer (used by gating / fusion)."""
        z_theta = self.encode_projection(projection_params)
        return self.adapters[layer_name].compute_delta_w(z_theta)

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return only the trainable parameters (adapters + encoder)."""
        return list(self.parameters())
