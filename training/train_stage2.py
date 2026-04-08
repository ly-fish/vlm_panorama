"""Stage 2 training: joint Dual-LoRA fusion (LoRA-Pano + LoRA-AESG + gating).

Objective: Jointly train LoRA-AESG and adaptive gating network.
LoRA-Pano is frozen or trained at 10x lower learning rate.

    L_stage2 = lambda_1 L_recon + lambda_2 L_rel + lambda_3 L_aff
             + lambda_4 L_ctx  + lambda_5 L_seam + lambda_6 L_pano

Usage:
    python -m training.train_stage2 \\
        --data_root /users/2522553y/liangyue_ws/vlm_panorama/data \\
        --stage1_ckpt ./checkpoints/stage1/best_checkpoint.pt \\
        --output_dir  ./checkpoints/stage2 \\
        --epochs 30 \\
        --batch_size 4 \\
        --lr 1e-4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.panorama_dataset import PanoramaDataset, collate_fn
from lora.distortion_encoder import DistortionEncoder
from lora.lora_aesg import AESGConditionAggregator
from lora.gating import AdaptiveGatingNetwork
from training.losses import ReconstructionLoss, compute_stage2_loss
from training.train_stage1 import _PatchReconNet   # reuse stub if needed


# ---------------------------------------------------------------------------
# Stage-2 network wrapper
# ---------------------------------------------------------------------------

class DualLoRAStage2Net(nn.Module):
    """Wraps backbone + all dual-LoRA components for Stage 2 training.

    When using the stub backbone (no real Qwen model), the AESG condition is
    injected via a lightweight cross-attention adapter appended after the
    reconstruction network.
    """

    def __init__(
        self,
        backbone: nn.Module,
        distortion_enc: DistortionEncoder,
        aesg_aggregator: AESGConditionAggregator,
        gating: AdaptiveGatingNetwork,
        use_stub: bool = True,
    ) -> None:
        super().__init__()
        self.backbone         = backbone
        self.distortion_enc   = distortion_enc
        self.aesg_aggregator  = aesg_aggregator
        self.gating           = gating
        self.use_stub         = use_stub

        if use_stub:
            # Simple cross-attention to inject AESG into stub features
            d_model = 3   # channels in stub output
            self.aesg_cross_attn = nn.MultiheadAttention(
                embed_dim=512,
                num_heads=8,
                batch_first=True,
            )
            # Project image features [B, 3, H, W] -> [B, HW, 512]
            self.img_feat_proj = nn.Conv2d(3, 512, 1)
            self.img_feat_unproj = nn.Conv2d(512, 3, 1)

    def forward(
        self,
        I_p_deg: torch.Tensor,
        mask: torch.Tensor,
        theta: torch.Tensor,
        aesg_condition: dict | None = None,
        task_type: str = "reconstruct",
    ) -> torch.Tensor:
        z_theta = self.distortion_enc(theta)             # [B, 512]

        if aesg_condition is not None:
            z_G = self.aesg_aggregator(aesg_condition, task_type=task_type)  # [B, 512]
            gamma_p, gamma_s = self.gating(z_theta, z_G, task_type=task_type)
        else:
            z_G = torch.zeros_like(z_theta)
            gamma_p = torch.ones(theta.shape[0], 1, device=theta.device)
            gamma_s = torch.zeros(theta.shape[0], 1, device=theta.device)

        if self.use_stub:
            inp  = torch.cat([I_p_deg, mask], dim=1)    # [B, 4, H, W]
            pred = self.backbone(inp, z_theta)            # [B, 3, H, W]

            if aesg_condition is not None and gamma_s.mean() > 0.05:
                # Inject AESG via cross-attention on image features
                B, C, H, W = pred.shape
                feat = self.img_feat_proj(pred)           # [B, 512, H, W]
                feat = feat.flatten(2).permute(0, 2, 1)  # [B, HW, 512]
                z_G_seq = z_G.unsqueeze(1)                # [B, 1, 512]
                feat_out, _ = self.aesg_cross_attn(feat, z_G_seq, z_G_seq)
                feat_out = feat_out.permute(0, 2, 1).view(B, 512, H, W)
                delta = self.img_feat_unproj(feat_out)    # [B, 3, H, W]
                gs = gamma_s.unsqueeze(-1).unsqueeze(-1)
                pred = pred + gs * delta
        else:
            # Real backbone: z_theta and z_G are injected via patched LoRA layers
            pred = self.backbone(I_p_deg)
            if hasattr(pred, "sample"):
                pred = pred.sample

        return torch.tanh(pred) if not torch.is_floating_point(pred) else pred.clamp(-1, 1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_stage2(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Stage 2] Device: {device}")

    # ------------------------------------------------------------------
    # Dataset (with AESG)
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
        build_aesg=True,
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
        build_aesg=True,
        max_scenes=args.max_val_scenes,
    )
    print(f"[Stage 2] Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

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
    # Build model components
    # ------------------------------------------------------------------
    distortion_enc = DistortionEncoder(hidden_dim=256, out_dim=512, num_frequencies=8).to(device)
    aesg_aggregator = AESGConditionAggregator(hidden_size=3584, out_dim=512).to(device)
    gating = AdaptiveGatingNetwork(pano_cond_dim=512, aesg_cond_dim=512).to(device)

    use_stub = True
    backbone = _PatchReconNet(base_ch=64).to(device)
    patched: dict = {}

    if args.backbone_path and Path(args.backbone_path).exists():
        try:
            from panorama_editing.qwen_image_editing.pipeline_qwenimage_edit_plus import (
                QwenImageEditPlusPipeline,
            )
            from lora.dual_lora_fusion import patch_model_with_dual_lora

            pipeline = QwenImageEditPlusPipeline.from_pretrained(
                args.backbone_path, torch_dtype=torch.float32
            )
            backbone = pipeline.transformer.to(device)
            patched = patch_model_with_dual_lora(
                backbone,
                target_patterns=args.lora_patterns,
                rank=args.rank,
                pano_cond_dim=512,
                aesg_cond_dim=512,
                alpha=args.alpha,
                dropout=args.dropout,
            )
            use_stub = False
            print(f"[Stage 2] Patched {len(patched)} layers")
        except Exception as exc:
            print(f"[Stage 2] WARNING: Could not load backbone ({exc}), using stub.")

    # Load Stage 1 checkpoint
    if args.stage1_ckpt and Path(args.stage1_ckpt).exists():
        ckpt = torch.load(args.stage1_ckpt, map_location=device)
        distortion_enc.load_state_dict(ckpt["distortion_encoder"])
        if use_stub:
            backbone.load_state_dict(ckpt["backbone"])
        print(f"[Stage 2] Loaded Stage 1 checkpoint from {args.stage1_ckpt}")
    else:
        print("[Stage 2] No Stage 1 checkpoint found, training from scratch.")

    model = DualLoRAStage2Net(
        backbone=backbone,
        distortion_enc=distortion_enc,
        aesg_aggregator=aesg_aggregator,
        gating=gating,
        use_stub=use_stub,
    ).to(device)

    # ------------------------------------------------------------------
    # Parameter groups
    # ------------------------------------------------------------------
    # Real backbone (Qwen): frozen — only LoRA adapters train.
    # Stub backbone: keep training so Stage 2 can build on Stage 1 weights.
    if not use_stub:
        for p in backbone.parameters():
            p.requires_grad_(False)

    # Separate pano params (10× lower LR) from everything else
    pano_params, new_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "distortion_enc" in name or "lora_pano_adapter" in name:
            pano_params.append(param)
        else:
            new_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": new_params,   "lr": args.lr},
            {"params": pano_params,  "lr": args.lr / 10.0},
        ],
        weight_decay=1e-4,
    )

    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr * 0.01
    )

    recon_loss_fn = ReconstructionLoss(
        lambda_perc=args.lambda_perc,
        lambda_ssim=args.lambda_ssim,
        use_perceptual=args.use_perceptual,
    ).to(device)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loss_config = {
        "lambda_recon": args.lambda_recon,
        "lambda_rel":   args.lambda_rel,
        "lambda_aff":   args.lambda_aff,
        "lambda_ctx":   args.lambda_ctx,
        "lambda_seam":  args.lambda_seam,
        "lambda_pano":  args.lambda_pano,
    }

    best_val_loss = float("inf")
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses: dict[str, float] = {}

        for step, batch in enumerate(train_loader):
            I_p_deg = batch["I_p_deg"].to(device)
            I_p_GT  = batch["I_p_GT"].to(device)
            theta   = batch["theta"].to(device)
            mask    = batch["mask"].to(device)
            meta    = batch["meta"]

            # Extract AESG conditions from meta (may be None for some samples)
            aesg_condition = _gather_aesg(meta, device)

            pred = model(
                I_p_deg=I_p_deg,
                mask=mask,
                theta=theta,
                aesg_condition=aesg_condition,
                task_type="reconstruct",
            )

            losses = compute_stage2_loss(
                pred=pred,
                target=I_p_GT,
                mask=mask,
                recon_loss_fn=recon_loss_fn,
                config=loss_config,
            )

            optimizer.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(
                [p for group in optimizer.param_groups for p in group["params"]],
                max_norm=1.0,
            )
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
                    f"rel={losses['L_rel'].item():.4f}  "
                    f"aff={losses['L_aff'].item():.4f}  "
                    f"lr={lr_now:.2e}"
                )

        n = len(train_loader)
        avg = {k: v / n for k, v in epoch_losses.items()}

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                I_p_deg = batch["I_p_deg"].to(device)
                I_p_GT  = batch["I_p_GT"].to(device)
                theta   = batch["theta"].to(device)
                mask    = batch["mask"].to(device)
                meta    = batch["meta"]

                aesg_condition = _gather_aesg(meta, device)
                pred = model(
                    I_p_deg=I_p_deg, mask=mask, theta=theta,
                    aesg_condition=aesg_condition,
                )
                losses = compute_stage2_loss(
                    pred=pred, target=I_p_GT, mask=mask,
                    recon_loss_fn=recon_loss_fn, config=loss_config,
                )
                val_loss += losses["total"].item()

        val_loss /= max(len(val_loader), 1)
        avg["val_total"] = val_loss
        history.append({"epoch": epoch, **avg})

        print(f"[Epoch {epoch:03d}] train={avg['total']:.4f}  val={val_loss:.4f}")

        # Checkpoint
        ckpt = {
            "epoch":             epoch,
            "distortion_encoder": model.distortion_enc.state_dict(),
            "aesg_aggregator":    model.aesg_aggregator.state_dict(),
            "gating":             model.gating.state_dict(),
            "optimizer":          optimizer.state_dict(),
            "val_loss":           val_loss,
        }
        if use_stub:
            ckpt["backbone"] = model.backbone.state_dict()
            if hasattr(model, "aesg_cross_attn"):
                ckpt["aesg_cross_attn"]  = model.aesg_cross_attn.state_dict()
                ckpt["img_feat_proj"]    = model.img_feat_proj.state_dict()
                ckpt["img_feat_unproj"]  = model.img_feat_unproj.state_dict()
        for name, layer in patched.items():
            ckpt[f"lora_pano__{name}"]  = layer.lora_pano_adapter.state_dict()
            ckpt[f"lora_aesg__{name}"]  = layer.lora_aesg_adapter.state_dict()

        torch.save(ckpt, output_dir / f"checkpoint_epoch{epoch:03d}.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, output_dir / "best_checkpoint.pt")
            print(f"  [Stage 2] New best val_loss={best_val_loss:.4f}")

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"[Stage 2] Done. Best val_loss={best_val_loss:.4f}")
    print(f"[Stage 2] Checkpoints saved to {output_dir}")


# ---------------------------------------------------------------------------
# Helper: gather AESG conditions from batch meta
# ---------------------------------------------------------------------------

def _gather_aesg(meta: list[dict], device: torch.device) -> dict | None:
    """Aggregate AESG conditions from meta list into a batched dict."""
    keys = ["anchor_tokens", "object_tokens", "context_tokens", "relation_tokens"]
    first_cond = meta[0].get("aesg_condition") if meta else None
    if first_cond is None:
        return None
    batched: dict = {}
    for k in keys:
        tensors = []
        for m in meta:
            cond = m.get("aesg_condition", {})
            t = cond.get(k)
            if t is None:
                shape = first_cond.get(k, torch.zeros(1, 1, 3584)).shape
                t = torch.zeros(shape)
            tensors.append(t.squeeze(0) if t.dim() == 3 else t)
        batched[k] = torch.stack(tensors).to(device)
    return batched


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2: Joint Dual-LoRA training")
    parser.add_argument("--data_root", type=str,
                        default="/users/2522553y/liangyue_ws/vlm_panorama/data")
    parser.add_argument("--stage1_ckpt", type=str, default="./checkpoints/stage1/best_checkpoint.pt")
    parser.add_argument("--output_dir",  type=str, default="./checkpoints/stage2")
    parser.add_argument("--backbone_path", type=str, default="")
    parser.add_argument("--epochs",      type=int, default=30)
    parser.add_argument("--batch_size",  type=int, default=4)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--img_size",    type=int, default=512)
    parser.add_argument("--rank",        type=int, default=8)
    parser.add_argument("--alpha",       type=float, default=8.0)
    parser.add_argument("--dropout",     type=float, default=0.0)
    # Loss weights
    parser.add_argument("--lambda_recon", type=float, default=1.0)
    parser.add_argument("--lambda_rel",   type=float, default=0.25)
    parser.add_argument("--lambda_aff",   type=float, default=0.25)
    parser.add_argument("--lambda_ctx",   type=float, default=0.25)
    parser.add_argument("--lambda_seam",  type=float, default=0.1)
    parser.add_argument("--lambda_pano",  type=float, default=0.3)
    parser.add_argument("--lambda_perc",  type=float, default=0.1)
    parser.add_argument("--lambda_ssim",  type=float, default=0.1)
    parser.add_argument("--use_perceptual", action="store_true", default=True)
    parser.add_argument("--logit_threshold", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every",   type=int, default=20)
    parser.add_argument("--max_scenes",  type=int, default=None)
    parser.add_argument("--max_val_scenes", type=int, default=None)
    parser.add_argument(
        "--lora_patterns", nargs="+",
        default=[".*attn.*\\.q_proj$", ".*attn.*\\.k_proj$",
                 ".*attn.*\\.v_proj$", ".*attn.*\\.out_proj$"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    train_stage2(parse_args())
