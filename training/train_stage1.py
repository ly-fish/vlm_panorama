"""Stage 1 training: panoramic geometric adaptation (LoRA-Pano only).

Objective: Learn ERP-aware geometric priors via self-supervised reconstruction,
without any AESG involvement.

    L_stage1 = L_recon(I_p_hat, I_p_GT) + lambda_pano * L_pano

Usage:
    python -m training.train_stage1 \\
        --data_root /users/2522553y/liangyue_ws/vlm_panorama/data \\
        --output_dir ./checkpoints/stage1 \\
        --epochs 30 \\
        --batch_size 4 \\
        --lr 1e-4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.panorama_dataset import PanoramaDataset, collate_fn
from lora.distortion_encoder import DistortionEncoder
from lora.dual_lora_fusion import DualLoRAModel, save_lora_weights
from training.losses import ReconstructionLoss, compute_stage1_loss


# ---------------------------------------------------------------------------
# Minimal backbone stub for training without the full Qwen model
# ---------------------------------------------------------------------------

class _PatchReconNet(nn.Module):
    """Lightweight UNet-like reconstruction network used when the full backbone
    is unavailable.  Replace with the real Qwen backbone in production.

    Input:  [B, 4, H, W]  (degraded RGB + mask concatenated)
    Output: [B, 3, H, W]  reconstructed RGB
    """

    def __init__(self, base_ch: int = 64) -> None:
        super().__init__()

        def _block(in_c: int, out_c: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.GroupNorm(8, out_c),
                nn.GELU(),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.GroupNorm(8, out_c),
                nn.GELU(),
            )

        self.enc1 = _block(4, base_ch)
        self.enc2 = _block(base_ch, base_ch * 2)
        self.enc3 = _block(base_ch * 2, base_ch * 4)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = _block(base_ch * 4, base_ch * 4)

        # Conditioning injection: z_theta projected to channel-wise scale/shift
        self.cond_proj = nn.Linear(512, base_ch * 4 * 2)

        # Decoder: 3 upsampling levels to mirror 3 encoder pools
        # up3: H/8 -> H/4; cat e3 (base_ch*4) -> base_ch*2 + base_ch*4 = base_ch*6
        self.up3  = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec3 = _block(base_ch * 2 + base_ch * 4, base_ch * 2)
        # up2: H/4 -> H/2; cat e2 (base_ch*2) -> base_ch + base_ch*2 = base_ch*3
        self.up2  = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec2 = _block(base_ch + base_ch * 2, base_ch)
        # up1: H/2 -> H;   cat e1 (base_ch)   -> base_ch//2 + base_ch = 3*base_ch//2
        self.up1  = nn.ConvTranspose2d(base_ch, base_ch // 2, 2, stride=2)
        self.dec1 = _block(base_ch // 2 + base_ch, base_ch // 2)

        self.head = nn.Conv2d(base_ch // 2, 3, 1)

    def forward(
        self,
        x: torch.Tensor,
        z_theta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:       [B, 4, H, W] degraded image + mask.
            z_theta: [B, 512] distortion condition.

        Returns:
            [B, 3, H, W] reconstructed image.
        """
        # Encoder (3 pooling steps)
        e1 = self.enc1(x)               # [B, C,   H,   W  ]
        e2 = self.enc2(self.pool(e1))   # [B, 2C,  H/2, W/2]
        e3 = self.enc3(self.pool(e2))   # [B, 4C,  H/4, W/4]

        # Bottleneck at H/8 + FiLM conditioning
        b = self.bottleneck(self.pool(e3))              # [B, 4C, H/8, W/8]
        cond = self.cond_proj(z_theta)                  # [B, 4C*2]
        gamma, beta = cond.chunk(2, dim=-1)             # each [B, 4C]
        gamma = gamma.view(-1, b.shape[1], 1, 1)
        beta  = beta.view(-1, b.shape[1], 1, 1)
        b = (1.0 + gamma) * b + beta

        # Decoder (3 upsampling steps)
        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))  # H/4
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # H/2
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # H

        return torch.tanh(self.head(d1))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_stage1(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Stage 1] Device: {device}")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    train_ds = PanoramaDataset(
        data_root=args.data_root,
        split="train",
        perspective_size=(args.img_size, args.img_size),
        fov_range=(60.0, 120.0),
        fov_steps=4,
        num_views_per_obj=4,
        logit_threshold=args.logit_threshold,
        degrade_strategy="random",
        build_aesg=False,
        max_scenes=args.max_scenes,
    )
    val_ds = PanoramaDataset(
        data_root=args.data_root,
        split="test",
        perspective_size=(args.img_size, args.img_size),
        fov_range=(60.0, 120.0),
        fov_steps=4,
        num_views_per_obj=2,
        logit_threshold=args.logit_threshold,
        degrade_strategy="gray",
        build_aesg=False,
        max_scenes=args.max_val_scenes,
    )
    print(f"[Stage 1] Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # ------------------------------------------------------------------
    # Model: DistortionEncoder + reconstruction backbone
    # If real backbone path is given, attempt to load and patch it.
    # Otherwise fall back to the lightweight stub.
    # ------------------------------------------------------------------
    distortion_enc = DistortionEncoder(
        hidden_dim=256,
        out_dim=512,
        num_frequencies=8,
    ).to(device)

    if args.backbone_path and Path(args.backbone_path).exists():
        print(f"[Stage 1] Loading backbone from {args.backbone_path}")
        try:
            from panorama_editing.qwen_image_editing.pipeline_qwenimage_edit_plus import (
                QwenImageEditPlusPipeline,
            )
            pipeline = QwenImageEditPlusPipeline.from_pretrained(
                args.backbone_path, torch_dtype=torch.float32
            )
            backbone = pipeline.transformer.to(device)

            from lora.dual_lora_fusion import patch_model_with_dual_lora
            patched = patch_model_with_dual_lora(
                backbone,
                target_patterns=args.lora_patterns,
                rank=args.rank,
                pano_cond_dim=512,
                aesg_cond_dim=512,
                alpha=args.alpha,
                dropout=args.dropout,
            )
            print(f"[Stage 1] Patched {len(patched)} layers with DualLoRA")

            # Freeze AESG adapters (Stage 1 trains pano only)
            for layer in patched.values():
                for p in layer.lora_aesg_adapter.parameters():
                    p.requires_grad_(False)

            use_stub = False
        except Exception as exc:
            print(f"[Stage 1] WARNING: Could not load backbone ({exc}), using stub.")
            use_stub = True
    else:
        print("[Stage 1] No backbone path provided. Using lightweight stub network.")
        use_stub = True

    if use_stub:
        backbone = _PatchReconNet(base_ch=64).to(device)
        patched = {}

    # ------------------------------------------------------------------
    # Loss and optimiser
    # ------------------------------------------------------------------
    recon_loss_fn = ReconstructionLoss(
        lambda_perc=args.lambda_perc,
        lambda_ssim=args.lambda_ssim,
        use_perceptual=args.use_perceptual,
    ).to(device)

    trainable_params = (
        list(distortion_enc.parameters())
        + list(backbone.parameters())
    )
    # Add LoRA-Pano adapter params if backbone was patched
    for layer in patched.values():
        trainable_params.extend(layer.lora_pano_adapter.parameters())

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)

    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr * 0.01
    )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        distortion_enc.train()
        backbone.train()

        epoch_losses: dict[str, float] = {}
        for step, batch in enumerate(train_loader):
            I_p_deg = batch["I_p_deg"].to(device)   # [B, 3, H, W]
            I_p_GT  = batch["I_p_GT"].to(device)    # [B, 3, H, W]
            theta   = batch["theta"].to(device)      # [B, 3]
            mask    = batch["mask"].to(device)       # [B, 1, H, W]

            # Encode projection
            z_theta = distortion_enc(theta)           # [B, 512]

            # Forward through backbone
            if use_stub:
                inp = torch.cat([I_p_deg, mask], dim=1)  # [B, 4, H, W]
                pred = backbone(inp, z_theta)
            else:
                # For the real backbone, the z_theta is injected via patched LoRA layers
                # The pipeline handles the forward pass; here we approximate:
                pred = backbone(I_p_deg, encoder_hidden_states=None)
                if hasattr(pred, "sample"):
                    pred = pred.sample

            losses = compute_stage1_loss(
                pred=pred,
                target=I_p_GT,
                mask=mask,
                recon_loss_fn=recon_loss_fn,
                lambda_pano=args.lambda_pano,
            )

            optimizer.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()

            if step % args.log_every == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"  [E{epoch:03d}|S{step:04d}] "
                    f"total={losses['total'].item():.4f}  "
                    f"recon={losses['L_recon'].item():.4f}  "
                    f"pano={losses['L_pano'].item():.4f}  "
                    f"lr={lr_now:.2e}"
                )

        # Average epoch losses
        n = len(train_loader)
        avg = {k: v / n for k, v in epoch_losses.items()}

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------
        distortion_enc.eval()
        backbone.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                I_p_deg = batch["I_p_deg"].to(device)
                I_p_GT  = batch["I_p_GT"].to(device)
                theta   = batch["theta"].to(device)
                mask    = batch["mask"].to(device)

                z_theta = distortion_enc(theta)
                if use_stub:
                    inp = torch.cat([I_p_deg, mask], dim=1)
                    pred = backbone(inp, z_theta)
                else:
                    pred = backbone(I_p_deg)
                    if hasattr(pred, "sample"):
                        pred = pred.sample

                losses = compute_stage1_loss(
                    pred=pred, target=I_p_GT, mask=mask,
                    recon_loss_fn=recon_loss_fn,
                    lambda_pano=args.lambda_pano,
                )
                val_loss += losses["total"].item()

        val_loss /= max(len(val_loader), 1)
        avg["val_total"] = val_loss
        history.append({"epoch": epoch, **avg})

        print(
            f"[Epoch {epoch:03d}] "
            f"train={avg['total']:.4f}  val={val_loss:.4f}"
        )

        # Save checkpoint
        ckpt = {
            "epoch": epoch,
            "distortion_encoder": distortion_enc.state_dict(),
            "backbone": backbone.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_loss,
        }
        # Add patched LoRA-Pano weights
        for name, layer in patched.items():
            ckpt[f"lora_pano__{name}"] = layer.lora_pano_adapter.state_dict()

        torch.save(ckpt, output_dir / f"checkpoint_epoch{epoch:03d}.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, output_dir / "best_checkpoint.pt")
            print(f"  [Stage 1] New best val_loss={best_val_loss:.4f}")

    # Save training history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"[Stage 1] Done. Best val_loss={best_val_loss:.4f}")
    print(f"[Stage 1] Checkpoints saved to {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1: LoRA-Pano training")
    parser.add_argument("--data_root", type=str,
                        default="/users/2522553y/liangyue_ws/vlm_panorama/data")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/stage1")
    parser.add_argument("--backbone_path", type=str, default="",
                        help="Path to Qwen-Image-Edit-2511 model or '' for stub")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lambda_pano", type=float, default=0.3)
    parser.add_argument("--lambda_perc", type=float, default=0.1)
    parser.add_argument("--lambda_ssim", type=float, default=0.1)
    parser.add_argument("--use_perceptual", action="store_true", default=True)
    parser.add_argument("--logit_threshold", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--max_scenes", type=int, default=None)
    parser.add_argument("--max_val_scenes", type=int, default=None)
    parser.add_argument(
        "--lora_patterns", nargs="+",
        default=[".*attn.*\\.q_proj$", ".*attn.*\\.k_proj$",
                 ".*attn.*\\.v_proj$", ".*attn.*\\.out_proj$"],
        help="Regex patterns for linear layers to patch with LoRA",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train_stage1(parse_args())
