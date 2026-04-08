"""Conditional LoRA layer with FiLM modulation.

Wraps an existing frozen nn.Linear and adds a FiLM-conditioned low-rank delta:

    output = W x  +  B · A(z) · x
           = W x  +  delta_W(z) · x

where A(z) = FiLM(A_base; z) is the condition-modulated down-projection.

This is the building block shared by both LoRA-Pano and LoRA-AESG branches.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lora.film import FiLMLayer


class ConditionalLoRALayer(nn.Module):
    """Conditional low-rank adapter for a single frozen linear layer.

    Args:
        base_layer:  The original frozen nn.Linear (d_in -> d_out).
        rank:        LoRA rank r.
        cond_dim:    Dimensionality of the conditioning vector z.
        alpha:       LoRA scaling factor (delta_W is scaled by alpha / rank).
        dropout:     Dropout probability applied to the LoRA path.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        cond_dim: int = 512,
        alpha: float | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad_(False)

        d_out, d_in = base_layer.weight.shape
        self.rank = rank
        self.scale = (alpha if alpha is not None else float(rank)) / rank

        # A: down-projection [r, d_in] — modulated by FiLM
        # B: up-projection   [d_out, r] — static learnable
        self.lora_A = nn.Parameter(torch.empty(rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        self.film = FiLMLayer(cond_dim=cond_dim, feature_dim=d_in)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self._init_lora_weights()

    def _init_lora_weights(self) -> None:
        # Kaiming uniform for A (as in the original LoRA paper)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B initialised to zero so delta_W = 0 at the start of training
        nn.init.zeros_(self.lora_B)

    def compute_delta_w(self, z: torch.Tensor) -> torch.Tensor:
        """Compute the conditioned weight delta delta_W(z).

        Args:
            z: condition vector [B, cond_dim].

        Returns:
            delta_W: [d_out, d_in]  (averaged over batch for a single forward pass).
        """
        # A_base: [r, d_in]; after FiLM: z modulates each row
        A_mod = self.film(self.lora_A.unsqueeze(0), z)  # [B, r, d_in]
        A_mod = A_mod.mean(dim=0)                        # [r, d_in]  (avg over batch)
        return self.scale * (self.lora_B @ A_mod)        # [d_out, d_in]

    def forward(self, x: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape [B, *, d_in].
            z: condition vector [B, cond_dim]; if None the base layer is used unchanged.

        Returns:
            Output tensor of shape [B, *, d_out].
        """
        base_out = self.base_layer(x)

        if z is None:
            return base_out

        # A_base: [r, d_in]; FiLM modulate per sample
        # We need per-sample modulation: for each sample b, A_mod[b] = film(A; z[b])
        B_sz = z.shape[0]
        A_mod = self.film(
            self.lora_A.unsqueeze(0).expand(B_sz, -1, -1),  # [B, r, d_in]
            z,
        )  # [B, r, d_in]

        # x: [B, seq, d_in] or [B, d_in]
        x_drop = self.dropout(x)
        if x_drop.dim() == 3:
            # [B, seq, d_in] @ [B, d_in, r] -> [B, seq, r]
            lora_mid = torch.bmm(x_drop, A_mod.transpose(1, 2))
            # [B, seq, r] @ [r, d_out]^T -> [B, seq, d_out]
            lora_out = lora_mid @ self.lora_B.t()
        else:
            lora_mid = torch.einsum("bi,bri->br", x_drop, A_mod)   # [B, r]
            lora_out = lora_mid @ self.lora_B.t()                   # [B, d_out]

        return base_out + self.scale * lora_out

    def merge_weights(self) -> nn.Linear:
        """Return a new nn.Linear with LoRA merged into W (inference only)."""
        with torch.no_grad():
            merged = nn.Linear(
                self.base_layer.in_features,
                self.base_layer.out_features,
                bias=self.base_layer.bias is not None,
            )
            # Use mean-condition delta_W as a static approximation for merged weights
            # For dynamic inference use forward() directly instead.
            merged.weight.copy_(self.base_layer.weight + self.scale * self.lora_B @ self.lora_A)
            if self.base_layer.bias is not None:
                merged.bias.copy_(self.base_layer.bias)
        return merged
