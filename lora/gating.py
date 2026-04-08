"""Adaptive gating network for Dual-LoRA fusion.

Predicts per-layer, per-sample gate values (gamma_p, gamma_s) that balance
the geometric and semantic LoRA branches:

    [gamma_p, gamma_s] = sigmoid(MLP_gate([z_theta; z_G_hat; e_task]))

Sigmoid (not softmax) is used because the two branches encode complementary
information and should be able to activate independently.
"""
from __future__ import annotations

import torch
import torch.nn as nn


_TASK_TYPES = ["reconstruct", "inpaint"]
_TASK_TO_IDX = {t: i for i, t in enumerate(_TASK_TYPES)}


class AdaptiveGatingNetwork(nn.Module):
    """Predicts (gamma_p, gamma_s) gate scalars from dual-branch conditions.

    Args:
        pano_cond_dim:  Dimensionality of z_theta (DistortionEncoder output).
        aesg_cond_dim:  Dimensionality of z_G_hat (AESGConditionAggregator output).
        hidden_dim:     Width of the gating MLP.
        num_tasks:      Number of task type embeddings.
        init_gate:      Initial gate value (bias initialisation); 0.5 = balanced.
    """

    def __init__(
        self,
        pano_cond_dim: int = 512,
        aesg_cond_dim: int = 512,
        hidden_dim: int = 256,
        num_tasks: int = len(_TASK_TYPES),
        init_gate: float = 0.5,
    ) -> None:
        super().__init__()
        self.task_embed = nn.Embedding(num_tasks, 64)
        in_dim = pano_cond_dim + aesg_cond_dim + 64

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),   # -> [logit_p, logit_s]
        )
        self._init_weights(init_gate)

    def _init_weights(self, init_gate: float) -> None:
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        # Bias the final layer so sigmoid output starts near init_gate
        import math
        init_logit = math.log(init_gate / (1.0 - init_gate + 1e-6))
        nn.init.constant_(self.mlp[-1].bias, init_logit)

    def forward(
        self,
        z_theta: torch.Tensor,
        z_G: torch.Tensor,
        task_type: str | int = "reconstruct",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_theta:   [B, pano_cond_dim] distortion condition.
            z_G:       [B, aesg_cond_dim] aggregated AESG condition.
            task_type: "reconstruct" | "inpaint" or integer index.

        Returns:
            (gamma_p, gamma_s): each of shape [B, 1], values in (0, 1).
        """
        if isinstance(task_type, str):
            task_idx = _TASK_TO_IDX.get(task_type, 0)
        else:
            task_idx = int(task_type)

        device = z_theta.device
        B = z_theta.shape[0]
        t_emb = self.task_embed(torch.tensor([task_idx], device=device)).expand(B, -1)

        combined = torch.cat([z_theta, z_G, t_emb], dim=-1)   # [B, D]
        logits = self.mlp(combined)                            # [B, 2]
        gates = torch.sigmoid(logits)                          # [B, 2]  in (0,1)
        return gates[:, 0:1], gates[:, 1:2]                   # gamma_p, gamma_s
