"""Full loss functions for Dual-LoRA training.

Stage 1:
    L_stage1 = L_recon(I_p_hat, I_p_GT) + lambda_pano * L_pano(reproject(I_p_hat, theta), P)

    L_recon = MSE + lambda_perc * L_perceptual + lambda_ssim * L_ssim

Stage 2:
    L_stage2 = lambda_1 L_recon + lambda_2 L_rel + lambda_3 L_aff
             + lambda_4 L_ctx  + lambda_5 L_seam + lambda_6 L_pano
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Perceptual loss (VGG-based)
# ---------------------------------------------------------------------------

class VGGPerceptualLoss(nn.Module):
    """VGG-16 perceptual loss using relu1_2, relu2_2, relu3_3 features."""

    def __init__(self) -> None:
        super().__init__()
        try:
            from torchvision import models
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        except Exception:
            from torchvision import models
            vgg = models.vgg16(pretrained=True)

        features = vgg.features
        self.slice1 = nn.Sequential(*list(features.children())[:4])    # relu1_2
        self.slice2 = nn.Sequential(*list(features.children())[4:9])   # relu2_2
        self.slice3 = nn.Sequential(*list(features.children())[9:16])  # relu3_3

        for p in self.parameters():
            p.requires_grad_(False)

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _normalise(self, x: torch.Tensor) -> torch.Tensor:
        """x in [-1, 1] -> ImageNet normalised."""
        x = (x + 1.0) / 2.0
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_n   = self._normalise(pred)
        target_n = self._normalise(target)
        loss = 0.0
        for slc in (self.slice1, self.slice2, self.slice3):
            pred_n   = slc(pred_n)
            target_n = slc(target_n)
            loss = loss + F.l1_loss(pred_n, target_n)
        return loss


# ---------------------------------------------------------------------------
# SSIM loss
# ---------------------------------------------------------------------------

def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)


class SSIMLoss(nn.Module):
    """1 - SSIM loss for image tensors in [-1, 1]."""

    def __init__(self, window_size: int = 11, sigma: float = 1.5) -> None:
        super().__init__()
        kernel = _gaussian_kernel(window_size, sigma)
        self.register_buffer("kernel", kernel)
        self.window_size = window_size
        self.C1 = 0.01**2
        self.C2 = 0.03**2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Normalise to [0, 1]
        pred   = (pred + 1.0) / 2.0
        target = (target + 1.0) / 2.0

        C, H, W = pred.shape[1], pred.shape[2], pred.shape[3]
        kernel = self.kernel.expand(C, 1, -1, -1).to(pred.device)
        pad = self.window_size // 2

        mu_p  = F.conv2d(pred,   kernel, padding=pad, groups=C)
        mu_t  = F.conv2d(target, kernel, padding=pad, groups=C)
        mu_pp = mu_p * mu_p
        mu_tt = mu_t * mu_t
        mu_pt = mu_p * mu_t

        sigma_pp = F.conv2d(pred * pred,     kernel, padding=pad, groups=C) - mu_pp
        sigma_tt = F.conv2d(target * target, kernel, padding=pad, groups=C) - mu_tt
        sigma_pt = F.conv2d(pred * target,   kernel, padding=pad, groups=C) - mu_pt

        ssim_map = (
            (2 * mu_pt + self.C1) * (2 * sigma_pt + self.C2)
        ) / (
            (mu_pp + mu_tt + self.C1) * (sigma_pp + sigma_tt + self.C2)
        )
        return 1.0 - ssim_map.mean()


# ---------------------------------------------------------------------------
# Reconstruction loss
# ---------------------------------------------------------------------------

class ReconstructionLoss(nn.Module):
    """L_recon = MSE + lambda_perc * L_perc + lambda_ssim * L_ssim"""

    def __init__(
        self,
        lambda_perc: float = 0.1,
        lambda_ssim: float = 0.1,
        use_perceptual: bool = True,
    ) -> None:
        super().__init__()
        self.lambda_perc = lambda_perc
        self.lambda_ssim = lambda_ssim
        self.ssim = SSIMLoss()
        self.perceptual: VGGPerceptualLoss | None = None
        if use_perceptual:
            try:
                self.perceptual = VGGPerceptualLoss()
            except Exception:
                self.perceptual = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = F.mse_loss(pred, target)
        loss = loss + self.lambda_ssim * self.ssim(pred, target)
        if self.perceptual is not None:
            loss = loss + self.lambda_perc * self.perceptual(pred, target)
        return loss


# ---------------------------------------------------------------------------
# Panoramic reprojection consistency loss (L_pano)
# ---------------------------------------------------------------------------

def reprojection_consistency_loss(
    pred_persp: torch.Tensor,
    gt_erp_local: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Measure consistency of reprojected prediction against original ERP.

    In practice we compare the predicted perspective patch against the
    ground-truth ERP pixels that correspond to the same region (already
    cropped to the perspective view size and supplied as gt_erp_local).

    Strategy
    --------
    * If *mask* is provided and non-empty, compute L1 inside the masked
      region (the edited area) **plus** an unweighted boundary term on the
      full patch so the loss is never zero.
    * If mask is empty or not provided, fall back to full-patch L1 —
      this avoids the degenerate L_pano = 0 when the target object has no
      pixels inside the perspective crop.

    Args:
        pred_persp:    [B, 3, H_p, W_p] predicted perspective image.
        gt_erp_local:  [B, 3, H_p, W_p] GT ERP pixels (perspective-projected).
        mask:          Optional [B, 1, H_p, W_p] mask (1 = edited region).

    Returns:
        Scalar loss.
    """
    diff = (pred_persp - gt_erp_local).abs()

    mask_sum = mask.sum() if mask is not None else 0
    if mask is not None and mask_sum > 0:
        # Masked-region term (inpainting quality in ERP context)
        masked_loss = (diff * mask).sum() / (mask_sum * 3 + 1e-6)
        # Full-patch boundary term (prevents L_pano collapsing to 0)
        full_loss = diff.mean()
        return 0.7 * masked_loss + 0.3 * full_loss

    # Fallback: full-patch L1 — always non-zero, gives geometric signal
    return diff.mean()


# ---------------------------------------------------------------------------
# Seam continuity loss (L_seam)
# ---------------------------------------------------------------------------

def seam_loss(pred_erp: torch.Tensor) -> torch.Tensor:
    """Penalise discontinuity at the left-right ERP boundary.

    Args:
        pred_erp: [B, 3, H, W] predicted full ERP image.

    Returns:
        Scalar loss.
    """
    left  = pred_erp[:, :, :, :1]    # [B, 3, H, 1]
    right = pred_erp[:, :, :, -1:]   # [B, 3, H, 1]
    return F.l1_loss(left, right)


# ---------------------------------------------------------------------------
# Spatial relation / affiliation / context losses (AESG-based)
# ---------------------------------------------------------------------------

def spatial_relation_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    relation_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """L_rel: reconstructed objects should maintain AESG-specified relative positions.

    Approximated as pixel-level L1 weighted by a relation-relevant region mask.
    """
    diff = F.l1_loss(pred, target, reduction="none")
    if relation_mask is not None:
        diff = diff * relation_mask
        return diff.sum() / (relation_mask.sum() * 3 + 1e-6)
    return diff.mean()


def affiliation_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    affiliation_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """L_aff: all affiliated objects must be reconstructed together."""
    return spatial_relation_loss(pred, target, affiliation_mask)


def context_consistency_loss(
    pred: torch.Tensor,
    context_region: torch.Tensor,
    context_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """L_ctx: style consistency between reconstructed region and surrounding panorama.

    Measures gram-matrix style difference in the context region.
    """
    if context_mask is not None:
        pred_ctx    = pred * (1.0 - context_mask)
        target_ctx  = context_region * (1.0 - context_mask)
    else:
        pred_ctx   = pred
        target_ctx = context_region

    # Gram matrix style loss
    B, C, H, W = pred_ctx.shape
    p_flat = pred_ctx.view(B, C, -1)
    t_flat = target_ctx.view(B, C, -1)
    G_p = torch.bmm(p_flat, p_flat.transpose(1, 2)) / (C * H * W)
    G_t = torch.bmm(t_flat, t_flat.transpose(1, 2)) / (C * H * W)
    return F.mse_loss(G_p, G_t)


# ---------------------------------------------------------------------------
# Combined stage losses
# ---------------------------------------------------------------------------

def compute_stage1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    recon_loss_fn: ReconstructionLoss | None = None,
    lambda_pano: float = 0.3,
    config: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute Stage 1 loss: L_recon + lambda_pano * L_pano.

    Args:
        pred:            [B, 3, H_p, W_p] predicted perspective patch.
        target:          [B, 3, H_p, W_p] ground-truth perspective patch.
        mask:            [B, 1, H_p, W_p] object mask in perspective space.
        recon_loss_fn:   ReconstructionLoss instance (created if None).
        lambda_pano:     Weight for panoramic reprojection consistency.
        config:          Optional config dict with lambda overrides.

    Returns:
        Dict with "L_recon", "L_pano", "total" scalar tensors.
    """
    cfg = config or {}
    lambda_pano = float(cfg.get("lambda_pano", lambda_pano))

    if recon_loss_fn is None:
        recon_loss_fn = ReconstructionLoss()

    L_recon = recon_loss_fn(pred, target)
    L_pano  = reprojection_consistency_loss(pred, target, mask=mask)

    total = L_recon + lambda_pano * L_pano
    return {"L_recon": L_recon, "L_pano": L_pano, "total": total}


def compute_stage2_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    recon_loss_fn: ReconstructionLoss | None = None,
    relation_mask: torch.Tensor | None = None,
    affiliation_mask: torch.Tensor | None = None,
    context_region: torch.Tensor | None = None,
    context_mask: torch.Tensor | None = None,
    pred_erp: torch.Tensor | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute Stage 2 multi-objective loss.

    L_stage2 = lambda_1 L_recon + lambda_2 L_rel + lambda_3 L_aff
             + lambda_4 L_ctx  + lambda_5 L_seam + lambda_6 L_pano

    Args:
        pred:             [B, 3, H_p, W_p] predicted perspective patch.
        target:           [B, 3, H_p, W_p] ground-truth perspective patch.
        mask:             [B, 1, H_p, W_p] object mask.
        recon_loss_fn:    ReconstructionLoss instance.
        relation_mask:    Optional mask for spatial relation regions.
        affiliation_mask: Optional mask for affiliated objects.
        context_region:   Optional [B, 3, H_p, W_p] surrounding context.
        context_mask:     Optional [B, 1, H_p, W_p] context region mask.
        pred_erp:         Optional [B, 3, H, W] full predicted ERP for seam loss.
        config:           Optional config dict with lambda overrides.

    Returns:
        Dict with individual and "total" losses.
    """
    cfg = config or {}
    lam = {
        "recon": float(cfg.get("lambda_recon", 1.0)),
        "rel":   float(cfg.get("lambda_rel",   0.25)),
        "aff":   float(cfg.get("lambda_aff",   0.25)),
        "ctx":   float(cfg.get("lambda_ctx",   0.25)),
        "seam":  float(cfg.get("lambda_seam",  0.1)),
        "pano":  float(cfg.get("lambda_pano",  0.3)),
    }

    if recon_loss_fn is None:
        recon_loss_fn = ReconstructionLoss()

    L_recon = recon_loss_fn(pred, target)
    L_pano  = reprojection_consistency_loss(pred, target, mask=mask)
    L_rel   = spatial_relation_loss(pred, target, relation_mask)
    L_aff   = affiliation_loss(pred, target, affiliation_mask)

    if context_region is not None:
        L_ctx = context_consistency_loss(pred, context_region, context_mask)
    else:
        L_ctx = torch.tensor(0.0, device=pred.device)

    L_seam = seam_loss(pred_erp) if pred_erp is not None else torch.tensor(0.0, device=pred.device)

    total = (
        lam["recon"] * L_recon
        + lam["rel"]  * L_rel
        + lam["aff"]  * L_aff
        + lam["ctx"]  * L_ctx
        + lam["seam"] * L_seam
        + lam["pano"] * L_pano
    )

    return {
        "L_recon": L_recon,
        "L_pano":  L_pano,
        "L_rel":   L_rel,
        "L_aff":   L_aff,
        "L_ctx":   L_ctx,
        "L_seam":  L_seam,
        "total":   total,
    }
