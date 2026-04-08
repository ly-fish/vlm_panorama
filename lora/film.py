"""FiLM (Feature-wise Linear Modulation) conditioning.

FiLM affinely transforms a feature tensor conditioned on an external signal:
    FiLM(h; z) = (1 + gamma(z)) * h + beta(z)

Used to inject the distortion condition z_theta or AESG condition z_G
into the LoRA down-projection matrix A.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    """Projects a condition vector to (gamma, beta) for element-wise modulation.

    Args:
        cond_dim:    Dimensionality of the condition vector z.
        feature_dim: Dimensionality of the feature h to be modulated.
        hidden_dim:  Width of the intermediate projection layer.
    """

    def __init__(
        self,
        cond_dim: int,
        feature_dim: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or max(cond_dim, feature_dim)
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * feature_dim),  # predicts [gamma | beta]
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        # Initialise the final layer so gamma=0, beta=0 (identity at init)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: feature tensor of shape [..., feature_dim].
            z: condition tensor of shape [B, cond_dim].

        Returns:
            Modulated tensor of the same shape as h.
        """
        params = self.net(z)                               # [B, 2 * feature_dim]
        gamma, beta = params.chunk(2, dim=-1)              # each [B, feature_dim]

        # Broadcast over any leading spatial dimensions in h
        for _ in range(h.dim() - gamma.dim()):
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)

        return (1.0 + gamma) * h + beta
