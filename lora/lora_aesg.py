"""LoRA-AESG: structural-semantic adaptation branch.

Aggregates the four AESG token branches (anchor, object, context, relation)
into a single condition vector z_G via learned weighted sum, then injects it
into FiLM-conditioned LoRA adapters:

    z_G_hat = alpha_a * z_a + alpha_c * z_c + alpha_x * z_x + alpha_r * z_r
    delta_W_aesg = B_aesg . A_aesg(z_G_hat)

The alpha weights are task-type-conditioned (Stage 2 only).
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from lora.lora_layer import ConditionalLoRALayer


_TASK_TYPES = ["reconstruct", "inpaint"]
_TASK_TO_IDX = {t: i for i, t in enumerate(_TASK_TYPES)}


class AESGConditionAggregator(nn.Module):
    """Aggregates four AESG token branches into a single condition vector.

    The token tensors (anchor, object, context, relation) produced by
    :class:`aesg.encoder.AESGEncoder` have shape [B, N_branch, hidden_size].
    We first mean-pool each branch, then compute a weighted sum whose
    weights alpha are predicted by a small task-type-conditioned MLP.

    Args:
        hidden_size: Dimensionality of AESG encoder tokens (default 3584).
        out_dim:     Dimensionality of the aggregated condition vector.
        num_tasks:   Number of task type embeddings.
    """

    def __init__(
        self,
        hidden_size: int = 3584,
        out_dim: int = 512,
        num_tasks: int = len(_TASK_TYPES),
    ) -> None:
        super().__init__()
        self.out_dim = out_dim

        # Project each branch from hidden_size -> out_dim
        self.proj_anchor = nn.Linear(hidden_size, out_dim)
        self.proj_object = nn.Linear(hidden_size, out_dim)
        self.proj_context = nn.Linear(hidden_size, out_dim)
        self.proj_relation = nn.Linear(hidden_size, out_dim)

        # Task embedding: maps task_idx -> small vector for alpha prediction
        self.task_embed = nn.Embedding(num_tasks, 64)

        # Alpha predictor: [out_dim * 4 + 64] -> 4 weights (softmax)
        self.alpha_net = nn.Sequential(
            nn.Linear(out_dim * 4 + 64, 256),
            nn.GELU(),
            nn.Linear(256, 4),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        # Initialise final alpha layer uniformly
        nn.init.zeros_(self.alpha_net[-1].weight)
        nn.init.zeros_(self.alpha_net[-1].bias)

    def forward(
        self,
        aesg_condition: dict[str, Any],
        task_type: str | int = "reconstruct",
    ) -> torch.Tensor:
        """
        Args:
            aesg_condition: dict from :class:`aesg.encoder.AESGEncoder` with
                            keys ``anchor_tokens``, ``object_tokens``,
                            ``context_tokens``, ``relation_tokens``, each [B, N, H].
            task_type:      "reconstruct" | "inpaint"  or integer index.

        Returns:
            z_G_hat: [B, out_dim] aggregated condition vector.
        """
        device = next(self.parameters()).device

        def _pool(key: str, proj: nn.Linear) -> torch.Tensor:
            tokens = aesg_condition.get(key)
            if tokens is None:
                return torch.zeros(1, self.out_dim, device=device)
            t = tokens.to(device=device, dtype=proj.weight.dtype)
            return proj(t.mean(dim=1))      # [B, out_dim]

        z_a = _pool("anchor_tokens", self.proj_anchor)    # [B, out_dim]
        z_c = _pool("object_tokens", self.proj_object)
        z_x = _pool("context_tokens", self.proj_context)
        z_r = _pool("relation_tokens", self.proj_relation)

        # Task embedding
        if isinstance(task_type, str):
            task_idx = _TASK_TO_IDX.get(task_type, 0)
        else:
            task_idx = int(task_type)
        t_emb = self.task_embed(
            torch.tensor([task_idx], device=device)
        ).expand(z_a.shape[0], -1)                        # [B, 64]

        # Alpha weights
        cat = torch.cat([z_a, z_c, z_x, z_r, t_emb], dim=-1)  # [B, 4*out+64]
        alpha = torch.softmax(self.alpha_net(cat), dim=-1)      # [B, 4]

        z_G = (
            alpha[:, 0:1] * z_a
            + alpha[:, 1:2] * z_c
            + alpha[:, 2:3] * z_x
            + alpha[:, 3:4] * z_r
        )   # [B, out_dim]
        return z_G


class LoRAAESG(nn.Module):
    """Structural-semantic LoRA branch.

    Args:
        layer_configs:  Dict mapping layer name -> dict with keys
                        ``base_layer`` (nn.Linear), ``rank``, ``alpha``.
        hidden_size:    AESG encoder hidden size.
        cond_dim:       Condition vector dim (= aggregator out_dim).
        dropout:        Dropout in LoRA layers.
    """

    def __init__(
        self,
        layer_configs: dict[str, dict[str, Any]],
        hidden_size: int = 3584,
        cond_dim: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.aggregator = AESGConditionAggregator(
            hidden_size=hidden_size,
            out_dim=cond_dim,
        )
        self.adapters = nn.ModuleDict()
        for name, cfg in layer_configs.items():
            self.adapters[name] = ConditionalLoRALayer(
                base_layer=cfg["base_layer"],
                rank=cfg.get("rank", 8),
                cond_dim=cond_dim,
                alpha=cfg.get("alpha"),
                dropout=dropout,
            )

    def encode_aesg(
        self,
        aesg_condition: dict[str, Any],
        task_type: str | int = "reconstruct",
    ) -> torch.Tensor:
        """Return z_G_hat for the given AESG condition."""
        return self.aggregator(aesg_condition, task_type=task_type)

    def forward(
        self,
        layer_name: str,
        x: torch.Tensor,
        aesg_condition: dict[str, Any],
        task_type: str | int = "reconstruct",
    ) -> torch.Tensor:
        """Apply the LoRA-AESG adapter for *layer_name* to input *x*."""
        z_G = self.encode_aesg(aesg_condition, task_type=task_type)
        return self.adapters[layer_name](x, z=z_G)

    def compute_delta_w(
        self,
        layer_name: str,
        aesg_condition: dict[str, Any],
        task_type: str | int = "reconstruct",
    ) -> torch.Tensor:
        """Compute delta_W_aesg for a named layer."""
        z_G = self.encode_aesg(aesg_condition, task_type=task_type)
        return self.adapters[layer_name].compute_delta_w(z_G)
