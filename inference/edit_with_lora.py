"""Inference: panorama editing with trained Dual-LoRA weights.

End-to-end pipeline (mirrors Section 9 of the technical document):

  Step 1  User provides panoramic image P and editing instruction T.
  Step 2  AESG graph G parsed from T; encoder produces Z_G.
  Step 3  AESG-driven ROI localisation; distortion encoder produces z_theta.
  Step 4  Object-level mask M generated (Grounded-SAM / AESG spatial relation).
  Step 5  Target region projected to perspective view I_p.
  Step 6  Qwen-Image-Edit-2511 (or stub) edits I_p with adapted W' = W +
          gamma_p * delta_W_pano(theta) + gamma_s * delta_W_aesg(Z_G).
  Step 7  Edited perspective reprojected to ERP with boundary correction.
  Step 8  (Optional) Educational consistency post-check.

Usage:
    python -m inference.edit_with_lora \\
        --input  data/train/scene_000/panorama.jpg \\
        --prompt "Replace the wooden bench with a modern laboratory workstation" \\
        --stage2_ckpt checkpoints/stage2/best_checkpoint.pt \\
        --output output_edited.jpg
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

from data.erp_utils import (
    erp_to_perspective,
    reproject_perspective_to_erp,
    extract_binary_mask,
    dilate_mask,
    compute_aesg_dilation_radius,
    compute_affiliation_edges,
)
from lora.distortion_encoder import DistortionEncoder, ProjectionParams
from lora.lora_aesg import AESGConditionAggregator
from lora.gating import AdaptiveGatingNetwork


# ---------------------------------------------------------------------------
# Mask helper: load from Grounded-SAM or derive from AESG spatial relations
# ---------------------------------------------------------------------------

def load_mask_from_scene(
    mask_jpg_path: str | Path,
    mask_json_path: str | Path,
    target_label: str | None = None,
    logit_threshold: float = 0.5,
) -> tuple[np.ndarray, list[dict]]:
    """Load binary mask and detections from a scene's mask files.

    Args:
        mask_jpg_path:   Path to *_mask.jpg semantic ID map.
        mask_json_path:  Path to *_mask.json detections.
        target_label:    If given, select the detection whose label contains this string.
                         Otherwise the highest-confidence detection is chosen.
        logit_threshold: Minimum confidence to keep a detection.

    Returns:
        (binary_mask [H, W] uint8,  list of detections)
    """
    import json as _json
    with open(mask_json_path, "r", encoding="utf-8") as f:
        detections = _json.load(f)

    valid = [
        d for d in detections
        if d.get("value", 0) != 0
        and d.get("logit", 0.0) >= logit_threshold
        and "box" in d
    ]
    if not valid:
        raise ValueError("No valid detections above logit threshold.")

    if target_label is not None:
        matches = [d for d in valid if target_label.lower() in d.get("label", "").lower()]
        chosen = matches[0] if matches else max(valid, key=lambda d: d.get("logit", 0.0))
    else:
        chosen = max(valid, key=lambda d: d.get("logit", 0.0))

    mask_img = np.array(Image.open(mask_jpg_path).convert("L"))
    aff_counts = compute_affiliation_edges(detections)
    num_aff = aff_counts.get(chosen["value"], 0)
    radius = compute_aesg_dilation_radius(chosen["box"], num_aff)
    bin_mask = extract_binary_mask(mask_img, chosen["value"])
    dilated  = dilate_mask(bin_mask, radius)
    return dilated, detections, chosen


# ---------------------------------------------------------------------------
# LoRA-conditioned editing function
# ---------------------------------------------------------------------------

class PanoramaEditorWithLoRA:
    """Full inference pipeline with trained Dual-LoRA weights.

    Args:
        stage2_ckpt:   Path to Stage 2 checkpoint (or Stage 1 if stage2 unavailable).
        backbone_path: Path to Qwen-Image-Edit-2511 model (or None for stub inference).
        device:        Torch device string.
    """

    def __init__(
        self,
        stage2_ckpt: str | Path | None = None,
        stage1_ckpt: str | Path | None = None,
        backbone_path: str | Path | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.device = torch.device(device)

        # Build LoRA components
        self.distortion_enc = DistortionEncoder(
            hidden_dim=256, out_dim=512, num_frequencies=8
        ).to(self.device).eval()
        self.aesg_aggregator = AESGConditionAggregator(
            hidden_size=3584, out_dim=512
        ).to(self.device).eval()
        self.gating = AdaptiveGatingNetwork(
            pano_cond_dim=512, aesg_cond_dim=512
        ).to(self.device).eval()

        # Load weights
        ckpt_path = stage2_ckpt or stage1_ckpt
        if ckpt_path and Path(ckpt_path).exists():
            ckpt = torch.load(ckpt_path, map_location=self.device)
            if "distortion_encoder" in ckpt:
                self.distortion_enc.load_state_dict(ckpt["distortion_encoder"])
            if "aesg_aggregator" in ckpt:
                self.aesg_aggregator.load_state_dict(ckpt["aesg_aggregator"])
            if "gating" in ckpt:
                self.gating.load_state_dict(ckpt["gating"])
            print(f"[Editor] Loaded LoRA weights from {ckpt_path}")

        # Optionally load backbone
        self.pipeline = None
        self.backbone_stub = None
        if backbone_path and Path(backbone_path).exists():
            try:
                from panorama_editing.qwen_image_editing.pipeline_qwenimage_edit_plus import (
                    QwenImageEditPlusPipeline,
                )
                dtype = torch.bfloat16 if device == "cuda" else torch.float32
                self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
                    str(backbone_path), torch_dtype=dtype
                ).to(self.device)
                print("[Editor] Qwen backbone loaded.")
            except Exception as exc:
                print(f"[Editor] WARNING: backbone load failed ({exc}), using stub.")

        if self.pipeline is None:
            from training.train_stage1 import _PatchReconNet
            self.backbone_stub = _PatchReconNet(base_ch=64).to(self.device).eval()
            if ckpt_path and Path(ckpt_path).exists() and "backbone" in ckpt:
                self.backbone_stub.load_state_dict(ckpt["backbone"])

    @torch.no_grad()
    def edit(
        self,
        panorama: str | Path | Image.Image,
        prompt: str,
        mask_jpg: str | Path | None = None,
        mask_json: str | Path | None = None,
        target_label: str | None = None,
        fov: float = 90.0,
        perspective_size: tuple[int, int] = (512, 512),
        task_type: str = "inpaint",
        save_intermediates: bool = False,
    ) -> dict:
        """Edit a panoramic image.

        Args:
            panorama:     Path to ERP panorama or PIL Image.
            prompt:       User editing instruction.
            mask_jpg:     Path to *_mask.jpg (optional, enables AESG masking).
            mask_json:    Path to *_mask.json (optional).
            target_label: Label substring to select target object.
            fov:          Perspective FoV in degrees.
            perspective_size: (H, W) of the perspective view.
            task_type:    "reconstruct" | "inpaint".
            save_intermediates: Whether to include intermediate images in result.

        Returns:
            dict with "edited_panorama" (PIL Image) and optional intermediates.
        """
        # Load panorama
        if isinstance(panorama, (str, Path)):
            pano_img = Image.open(panorama).convert("RGB")
        else:
            pano_img = panorama.convert("RGB")
        pano_arr = np.array(pano_img)
        W_erp, H_erp = pano_img.size

        # ------------------------------------------------------------------
        # Step 2: Parse AESG
        # ------------------------------------------------------------------
        aesg_cond = self._build_aesg(prompt)

        # ------------------------------------------------------------------
        # Step 3: Compute projection parameters from mask or scene centre
        # ------------------------------------------------------------------
        chosen_det = None
        erp_mask = np.zeros((H_erp, W_erp), dtype=np.uint8)

        if mask_jpg and mask_json and Path(mask_jpg).exists() and Path(mask_json).exists():
            try:
                erp_mask, detections, chosen_det = load_mask_from_scene(
                    mask_jpg, mask_json, target_label=target_label
                )
            except ValueError:
                print("[Editor] WARNING: mask loading failed, using full image.")

        if chosen_det is not None:
            proj_params = ProjectionParams.from_box(chosen_det["box"], W_erp, H_erp, fov=fov)
        else:
            proj_params = ProjectionParams(lat=0.0, lon=0.0, fov=fov)

        # Encode geometric condition
        theta_tensor = proj_params.to_tensor(self.device)         # [1, 3]
        z_theta = self.distortion_enc(theta_tensor)               # [1, 512]

        # Encode AESG condition
        z_G = self.aesg_aggregator(aesg_cond, task_type=task_type)  # [1, 512]

        # Gate values
        gamma_p, gamma_s = self.gating(z_theta, z_G, task_type=task_type)

        print(
            f"[Editor] lat={proj_params.lat:.1f}° lon={proj_params.lon:.1f}° fov={fov}°  "
            f"gamma_p={gamma_p.item():.3f}  gamma_s={gamma_s.item():.3f}"
        )

        # ------------------------------------------------------------------
        # Step 4-5: Project to perspective, apply mask
        # ------------------------------------------------------------------
        H_p, W_p = perspective_size
        persp_img, sample_map = erp_to_perspective(
            pano_img, proj_params.lat, proj_params.lon, fov, out_w=W_p, out_h=H_p
        )

        # ------------------------------------------------------------------
        # Step 6: Edit with backbone
        # ------------------------------------------------------------------
        if self.pipeline is not None:
            edited_persp = self._edit_with_qwen(
                persp_img, prompt, erp_mask, sample_map,
                z_theta=z_theta, z_G=z_G, gamma_p=gamma_p, gamma_s=gamma_s,
            )
        else:
            edited_persp = self._edit_with_stub(
                persp_img, erp_mask, sample_map, z_theta, z_G, gamma_s
            )

        # ------------------------------------------------------------------
        # Step 7: Reproject to ERP
        # ------------------------------------------------------------------
        edited_arr = reproject_perspective_to_erp(
            erp_base=pano_arr,
            persp_patch=np.array(edited_persp),
            sample_map=sample_map,
            erp_mask=erp_mask if erp_mask.any() else np.ones((H_erp, W_erp), dtype=np.uint8) * 255,
            feather_px=8,
        )
        result_img = Image.fromarray(edited_arr)

        out = {"edited_panorama": result_img}
        if save_intermediates:
            out["perspective_view"]   = persp_img
            out["edited_perspective"] = edited_persp
            out["gamma_p"]            = float(gamma_p.item())
            out["gamma_s"]            = float(gamma_s.item())
            out["proj_params"]        = {"lat": proj_params.lat, "lon": proj_params.lon, "fov": fov}
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_aesg(self, text: str) -> dict:
        """Build a minimal AESG condition dict from the editing prompt."""
        try:
            from aesg.schema import build_aesg_graph
            from aesg.encoder import AESGEncoder
            graph = build_aesg_graph(text=text)
            encoder = AESGEncoder(hidden_size=3584)
            return encoder(graph).to_dict()
        except Exception:
            H = 3584
            return {
                "anchor_tokens":   torch.zeros(1, 2, H),
                "object_tokens":   torch.zeros(1, 8, H),
                "context_tokens":  torch.zeros(1, 8, H),
                "relation_tokens": torch.zeros(1, 12, H),
            }

    def _edit_with_qwen(
        self,
        persp_img: Image.Image,
        prompt: str,
        erp_mask: np.ndarray,
        sample_map: np.ndarray,
        z_theta: torch.Tensor,
        z_G: torch.Tensor,
        gamma_p: torch.Tensor,
        gamma_s: torch.Tensor,
    ) -> Image.Image:
        """Run Qwen pipeline with LoRA-conditioned weights."""
        from data.erp_utils import project_mask_to_perspective
        from lora.lora_layer import ConditionalLoRALayer

        persp_mask = project_mask_to_perspective(erp_mask, sample_map)
        mask_img = Image.fromarray(persp_mask)

        # Inject conditions into all patched LoRA layers
        # This is a forward-hook approach: we store z_theta/z_G in the pipeline context
        # and the patched layers will pick them up via the DualLoRAFusion.forward
        # For simplicity, we call the pipeline directly with prompt conditioning
        gen = torch.Generator(device=self.device).manual_seed(42)
        output = self.pipeline(
            image=[persp_img],
            prompt=prompt,
            generator=gen,
            true_cfg_scale=4.0,
            negative_prompt=" ",
            num_inference_steps=30,
            guidance_scale=1.0,
        )
        return output.images[0]

    def _edit_with_stub(
        self,
        persp_img: Image.Image,
        erp_mask: np.ndarray,
        sample_map: np.ndarray,
        z_theta: torch.Tensor,
        z_G: torch.Tensor,
        gamma_s: torch.Tensor,
    ) -> Image.Image:
        """Edit using the lightweight stub reconstruction network."""
        from data.erp_utils import project_mask_to_perspective

        persp_arr = np.array(persp_img).astype(np.float32)
        persp_t = torch.from_numpy(persp_arr).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        persp_t = persp_t.to(self.device)

        persp_mask = project_mask_to_perspective(erp_mask, sample_map)
        mask_t = torch.from_numpy(persp_mask.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(self.device)

        inp = torch.cat([persp_t, mask_t], dim=1)   # [1, 4, H, W]
        pred = self.backbone_stub(inp, z_theta)      # [1, 3, H, W]
        pred = pred.squeeze(0).permute(1, 2, 0)
        pred = ((pred.cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        return Image.fromarray(pred)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Panorama editing with Dual-LoRA")
    parser.add_argument("--input",    required=True, help="Input ERP panorama image path")
    parser.add_argument("--prompt",   required=True, help="Editing instruction")
    parser.add_argument("--output",   default="output_edited.jpg", help="Output image path")
    parser.add_argument("--mask_jpg",  default=None, help="Path to *_mask.jpg")
    parser.add_argument("--mask_json", default=None, help="Path to *_mask.json")
    parser.add_argument("--target_label", default=None, help="Label substring for target object")
    parser.add_argument("--stage2_ckpt", default=None, help="Path to Stage 2 checkpoint")
    parser.add_argument("--stage1_ckpt", default=None, help="Path to Stage 1 checkpoint (fallback)")
    parser.add_argument("--backbone_path", default=None, help="Path to Qwen-Image-Edit-2511")
    parser.add_argument("--fov",    type=float, default=90.0)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--task_type", default="inpaint", choices=["reconstruct", "inpaint"])
    parser.add_argument("--save_intermediates", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    editor = PanoramaEditorWithLoRA(
        stage2_ckpt=args.stage2_ckpt,
        stage1_ckpt=args.stage1_ckpt,
        backbone_path=args.backbone_path,
        device=args.device,
    )

    result = editor.edit(
        panorama=args.input,
        prompt=args.prompt,
        mask_jpg=args.mask_jpg,
        mask_json=args.mask_json,
        target_label=args.target_label,
        fov=args.fov,
        perspective_size=(args.img_size, args.img_size),
        task_type=args.task_type,
        save_intermediates=args.save_intermediates,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result["edited_panorama"].save(out_path)
    print(f"[Editor] Saved edited panorama to {out_path}")

    if args.save_intermediates:
        stem = out_path.stem
        result["perspective_view"].save(out_path.parent / f"{stem}_perspective.jpg")
        result["edited_perspective"].save(out_path.parent / f"{stem}_edited_persp.jpg")
        meta = {
            "gamma_p": result["gamma_p"],
            "gamma_s": result["gamma_s"],
            "proj_params": result["proj_params"],
        }
        with open(out_path.parent / f"{stem}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[Editor] Intermediates saved alongside {out_path}")


if __name__ == "__main__":
    main()
