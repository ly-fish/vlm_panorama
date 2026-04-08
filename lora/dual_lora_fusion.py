"""Dual-LoRA Fusion: combines LoRA-Pano and LoRA-AESG with adaptive gating.

The adapted weight matrix per layer is:
    W' = W + gamma_p * delta_W_pano(theta) + gamma_s * delta_W_aesg(Z_G)

This module provides:
  - DualLoRAFusion: orchestrates both branches + gating for a named layer.
  - patch_model_with_dual_lora: finds target linear layers in a model and
    wraps them with DualLoRAFusion in-place.
  - save_lora_weights / load_lora_weights: checkpoint helpers.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from lora.distortion_encoder import DistortionEncoder, ProjectionParams
from lora.lora_pano import LoRAPano
from lora.lora_aesg import LoRAAESG
from lora.gating import AdaptiveGatingNetwork


# ---------------------------------------------------------------------------
# DualLoRAFusion
# ---------------------------------------------------------------------------

class DualLoRAFusion(nn.Module):
    """Single-layer dual-LoRA adapter that owns both branches and the gate.

    Used as a drop-in replacement for a frozen nn.Linear.  During a forward
    pass the caller must supply ``projection_params`` (geometric) and
    optionally ``aesg_condition`` (semantic) keyword arguments.

    Args:
        base_layer:        The original frozen nn.Linear.
        rank:              LoRA rank r (applied to both branches).
        pano_cond_dim:     Dim of z_theta from the shared DistortionEncoder.
        aesg_cond_dim:     Dim of z_G from the shared AESGConditionAggregator.
        alpha:             LoRA scaling factor.
        dropout:           Dropout on LoRA paths.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        pano_cond_dim: int = 512,
        aesg_cond_dim: int = 512,
        alpha: float | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # Keep a reference to the frozen base; do not wrap it in submodules
        # to avoid double-registration of its parameters.
        self._base_layer = base_layer
        for p in base_layer.parameters():
            p.requires_grad_(False)

        from lora.lora_layer import ConditionalLoRALayer
        self.lora_pano_adapter = ConditionalLoRALayer(
            base_layer=base_layer,
            rank=rank,
            cond_dim=pano_cond_dim,
            alpha=alpha,
            dropout=dropout,
        )
        self.lora_aesg_adapter = ConditionalLoRALayer(
            base_layer=base_layer,
            rank=rank,
            cond_dim=aesg_cond_dim,
            alpha=alpha,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        z_theta: torch.Tensor | None = None,
        z_G: torch.Tensor | None = None,
        gamma_p: torch.Tensor | None = None,
        gamma_s: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:       Input [B, *, d_in].
            z_theta: Distortion condition [B, pano_cond_dim].
            z_G:     AESG condition [B, aesg_cond_dim].
            gamma_p: Geometric gate [B, 1]; if None, pano branch is skipped.
            gamma_s: Semantic gate [B, 1]; if None, AESG branch is skipped.

        Returns:
            W' x of shape [B, *, d_out].
        """
        base_out = self._base_layer(x)

        lora_delta = torch.zeros_like(base_out)

        if z_theta is not None:
            pano_out = self.lora_pano_adapter(x, z=z_theta) - base_out   # delta only
            gp = gamma_p if gamma_p is not None else torch.ones(x.shape[0], 1, device=x.device)
            if x.dim() == 3:
                gp = gp.unsqueeze(1)
            lora_delta = lora_delta + gp * pano_out

        if z_G is not None:
            aesg_out = self.lora_aesg_adapter(x, z=z_G) - base_out       # delta only
            gs = gamma_s if gamma_s is not None else torch.ones(x.shape[0], 1, device=x.device)
            if x.dim() == 3:
                gs = gs.unsqueeze(1)
            lora_delta = lora_delta + gs * aesg_out

        return base_out + lora_delta


# ---------------------------------------------------------------------------
# Model-level orchestrator
# ---------------------------------------------------------------------------

class DualLoRAModel(nn.Module):
    """Orchestrates the complete Dual-LoRA pass over a backbone model.

    Holds the shared encoders and gating network, and manages a registry of
    DualLoRAFusion layers that have been patched into the backbone.

    Args:
        backbone:          The frozen backbone model (e.g. Qwen-Image-Edit).
        target_patterns:   List of regex patterns for linear layer names to patch.
        rank:              LoRA rank.
        pano_cond_dim:     Distortion encoder output dim.
        aesg_cond_dim:     AESG aggregator output dim.
        aesg_hidden_size:  AESG encoder token dim.
        encoder_hidden:    DistortionEncoder MLP hidden dim.
        num_frequencies:   Sinusoidal frequencies in distortion encoder.
        alpha:             LoRA alpha scaling.
        dropout:           Dropout on LoRA paths.
    """

    def __init__(
        self,
        backbone: nn.Module,
        target_patterns: list[str] | None = None,
        rank: int = 8,
        pano_cond_dim: int = 512,
        aesg_cond_dim: int = 512,
        aesg_hidden_size: int = 3584,
        encoder_hidden: int = 256,
        num_frequencies: int = 8,
        alpha: float | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        # Shared encoders
        self.distortion_encoder = DistortionEncoder(
            hidden_dim=encoder_hidden,
            out_dim=pano_cond_dim,
            num_frequencies=num_frequencies,
        )
        from lora.lora_aesg import AESGConditionAggregator
        self.aesg_aggregator = AESGConditionAggregator(
            hidden_size=aesg_hidden_size,
            out_dim=aesg_cond_dim,
        )
        self.gating = AdaptiveGatingNetwork(
            pano_cond_dim=pano_cond_dim,
            aesg_cond_dim=aesg_cond_dim,
        )

        # Patch target layers
        patterns = target_patterns or [
            r".*attn.*\.q_proj$",
            r".*attn.*\.k_proj$",
            r".*attn.*\.v_proj$",
            r".*attn.*\.out_proj$",
        ]
        self.fusion_layers: dict[str, DualLoRAFusion] = {}
        self._patch_backbone(backbone, patterns, rank, pano_cond_dim, aesg_cond_dim, alpha, dropout)
        # Register patched layers so their parameters are tracked
        self.lora_modules = nn.ModuleDict(self.fusion_layers)

        self._stage: int = 1   # 1 = pano only, 2 = joint

    def _patch_backbone(
        self,
        model: nn.Module,
        patterns: list[str],
        rank: int,
        pano_cond_dim: int,
        aesg_cond_dim: int,
        alpha: float | None,
        dropout: float,
    ) -> None:
        """Walk the model tree, replacing matching nn.Linear layers in-place."""
        for full_name, module in list(model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            if not any(re.fullmatch(pat, full_name) for pat in patterns):
                continue

            fusion = DualLoRAFusion(
                base_layer=module,
                rank=rank,
                pano_cond_dim=pano_cond_dim,
                aesg_cond_dim=aesg_cond_dim,
                alpha=alpha,
                dropout=dropout,
            )
            # Replace the module in its parent
            parts = full_name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], fusion)
            # Use safe dict key
            self.fusion_layers[full_name.replace(".", "__")] = fusion

    def set_stage(self, stage: int) -> None:
        """Switch training stage (1 = pano only, 2 = joint)."""
        self._stage = stage
        for name, param in self.named_parameters():
            if "backbone" in name:
                param.requires_grad_(False)
                continue
            if stage == 1:
                # Train only pano adapters + distortion encoder
                trainable = (
                    "distortion_encoder" in name
                    or "lora_pano_adapter" in name
                )
            else:
                # Train AESG aggregator + gating + AESG adapters;
                # pano adapters get 10x lower LR via optimiser param groups.
                trainable = True
            param.requires_grad_(trainable)

    def get_param_groups(self, base_lr: float = 1e-4) -> list[dict]:
        """Return optimiser param groups for stage-aware learning rates."""
        pano_params, other_params = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "distortion_encoder" in name or "lora_pano_adapter" in name:
                pano_params.append(param)
            else:
                other_params.append(param)
        if self._stage == 1:
            return [{"params": pano_params, "lr": base_lr}]
        # Stage 2: pano at 10x lower LR
        groups = [{"params": other_params, "lr": base_lr}]
        if pano_params:
            groups.append({"params": pano_params, "lr": base_lr / 10.0})
        return groups

    def encode_conditions(
        self,
        projection_params: ProjectionParams | torch.Tensor,
        aesg_condition: dict[str, Any] | None = None,
        task_type: str | int = "reconstruct",
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Encode all conditions and compute gate values.

        Returns:
            (z_theta, z_G, gamma_p, gamma_s)
        """
        z_theta = self.distortion_encoder(projection_params)

        if aesg_condition is not None and self._stage == 2:
            z_G = self.aesg_aggregator(aesg_condition, task_type=task_type)
            gamma_p, gamma_s = self.gating(z_theta, z_G, task_type=task_type)
        else:
            z_G = None
            gamma_p = torch.ones(z_theta.shape[0], 1, device=z_theta.device)
            gamma_s = None

        return z_theta, z_G, gamma_p, gamma_s


# ---------------------------------------------------------------------------
# Utility: patch a plain model without DualLoRAModel wrapper
# ---------------------------------------------------------------------------

def patch_model_with_dual_lora(
    model: nn.Module,
    target_patterns: list[str] | None = None,
    rank: int = 8,
    pano_cond_dim: int = 512,
    aesg_cond_dim: int = 512,
    alpha: float | None = None,
    dropout: float = 0.0,
) -> dict[str, DualLoRAFusion]:
    """Replace matching nn.Linear layers in *model* with DualLoRAFusion in-place.

    Returns a dict mapping safe_name -> DualLoRAFusion for all patched layers.
    """
    patterns = target_patterns or [
        r".*attn.*\.q_proj$",
        r".*attn.*\.k_proj$",
        r".*attn.*\.v_proj$",
        r".*attn.*\.out_proj$",
    ]
    patched: dict[str, DualLoRAFusion] = {}
    for full_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(re.fullmatch(pat, full_name) for pat in patterns):
            continue
        fusion = DualLoRAFusion(
            base_layer=module,
            rank=rank,
            pano_cond_dim=pano_cond_dim,
            aesg_cond_dim=aesg_cond_dim,
            alpha=alpha,
            dropout=dropout,
        )
        parts = full_name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], fusion)
        patched[full_name] = fusion
    return patched


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_lora_weights(model: DualLoRAModel, path: str | Path) -> None:
    """Save only the trainable LoRA + encoder + gating weights."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        k: v for k, v in model.state_dict().items()
        if "backbone" not in k
    }
    torch.save(state, path)
    print(f"[DualLoRA] Saved {len(state)} tensors to {path}")


def load_lora_weights(model: DualLoRAModel, path: str | Path, strict: bool = False) -> None:
    """Load LoRA weights into *model*, ignoring backbone keys."""
    path = Path(path)
    state = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if missing:
        print(f"[DualLoRA] Missing keys ({len(missing)}): {missing[:5]} ...")
    if unexpected:
        print(f"[DualLoRA] Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")
    print(f"[DualLoRA] Loaded weights from {path}")
