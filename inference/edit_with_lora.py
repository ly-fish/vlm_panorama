"""Inference: panorama editing with Dual-LoRA on top of Qwen-Image-Edit-2511.

This script wraps the existing ``panorama_editing.executor`` pipeline and adds
Dual-LoRA geometric + semantic conditioning on top.  The Qwen backbone call is
identical to the already-working executor; the only difference is that every
DualLoRAFusion layer is primed with (z_theta, z_G, gamma_p, gamma_s) before
the pipeline runs so the LoRA deltas are applied to every cross-attention layer.

Minimal usage — only panorama and prompt are required:

    python -m inference.edit_with_lora \\
        --input  data/train/scene_000/panorama.jpg \\
        --prompt "Replace the wooden bench with a red modern chair" \\
        --stage2_ckpt checkpoints/stage2/best_checkpoint.pt \\
        --output  output_edited.jpg

If --stage2_ckpt is omitted the script runs the base Qwen pipeline without
any LoRA conditioning (useful as a baseline).

Model loading: the Qwen backbone is loaded exactly as executor.get_pipeline()
does — using the HuggingFace model ID "Qwen/Qwen-Image-Edit-2511" (or the
local path provided via --backbone_path).  The model is cached in HF cache
after the first download.
"""
from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image

from lora.distortion_encoder import DistortionEncoder, ProjectionParams
from lora.lora_aesg import AESGConditionAggregator
from lora.gating import AdaptiveGatingNetwork
from lora.dual_lora_fusion import DualLoRAFusion, patch_model_with_dual_lora


# ---------------------------------------------------------------------------
# Condition injection context manager
# ---------------------------------------------------------------------------

@contextmanager
def _primed_lora(
    fusion_layers: dict[str, DualLoRAFusion],
    z_theta: torch.Tensor,
    z_G: torch.Tensor | None,
    gamma_p: torch.Tensor,
    gamma_s: torch.Tensor | None,
):
    """Prime every DualLoRAFusion layer before the pipeline call, reset after."""
    for layer in fusion_layers.values():
        layer.prime(z_theta, z_G, gamma_p, gamma_s)
    try:
        yield
    finally:
        for layer in fusion_layers.values():
            layer.reset_prime()


# ---------------------------------------------------------------------------
# Main editor
# ---------------------------------------------------------------------------

class PanoramaEditorWithLoRA:
    """Dual-LoRA inference wrapper around the existing Qwen-Image-Edit-2511 pipeline.

    Args:
        stage2_ckpt:   Path to Stage 2 checkpoint (contains LoRA weights).
        stage1_ckpt:   Path to Stage 1 checkpoint (fallback).
        backbone_path: HuggingFace model ID or local path to Qwen-Image-Edit-2511.
                       Defaults to "Qwen/Qwen-Image-Edit-2511".
        device:        "cuda" | "cpu" | "auto".
        lora_rank:     LoRA rank used during training (default 8).
    """

    def __init__(
        self,
        stage2_ckpt: str | Path | None = None,
        stage1_ckpt: str | Path | None = None,
        backbone_path: str = "Qwen/Qwen-Image-Edit-2511",
        device: str = "auto",
        lora_rank: int = 8,
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # ------------------------------------------------------------------
        # Load Qwen pipeline exactly as executor.get_pipeline() does
        # ------------------------------------------------------------------
        from panorama_editing.qwen_image_editing.pipeline_qwenimage_edit_plus import (
            QwenImageEditPlusPipeline,
        )
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        print(f"[Editor] Loading Qwen pipeline from '{backbone_path}' ...")
        self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
            backbone_path, torch_dtype=dtype
        ).to(self.device)
        self.pipeline.set_progress_bar_config(disable=None)
        print("[Editor] Qwen pipeline ready.")

        # ------------------------------------------------------------------
        # Patch transformer with DualLoRA
        # ------------------------------------------------------------------
        self.fusion_layers: dict[str, DualLoRAFusion] = patch_model_with_dual_lora(
            self.pipeline.transformer,
            rank=lora_rank,
            pano_cond_dim=512,
            aesg_cond_dim=512,
        )
        print(f"[Editor] Patched {len(self.fusion_layers)} attention layers with DualLoRA.")

        # ------------------------------------------------------------------
        # Shared condition encoders
        # ------------------------------------------------------------------
        self.distortion_enc = DistortionEncoder(
            hidden_dim=256, out_dim=512, num_frequencies=8
        ).to(self.device).eval()
        self.aesg_aggregator = AESGConditionAggregator(
            hidden_size=3584, out_dim=512
        ).to(self.device).eval()
        self.gating = AdaptiveGatingNetwork(
            pano_cond_dim=512, aesg_cond_dim=512
        ).to(self.device).eval()

        # ------------------------------------------------------------------
        # Load LoRA weights from checkpoint
        # ------------------------------------------------------------------
        ckpt_path = stage2_ckpt or stage1_ckpt
        if ckpt_path and Path(ckpt_path).exists():
            ckpt = torch.load(ckpt_path, map_location=self.device)
            if "distortion_encoder" in ckpt:
                self.distortion_enc.load_state_dict(ckpt["distortion_encoder"])
            if "aesg_aggregator" in ckpt:
                self.aesg_aggregator.load_state_dict(ckpt["aesg_aggregator"])
            if "gating" in ckpt:
                self.gating.load_state_dict(ckpt["gating"])
            self._load_lora_adapters(ckpt)
            print(f"[Editor] Loaded LoRA weights from {ckpt_path}")
        else:
            print(
                "[Editor] No checkpoint provided — running with untrained LoRA weights. "
                "The output is equivalent to the base Qwen model."
            )

        self._has_lora = bool(ckpt_path and Path(ckpt_path).exists())

    def _load_lora_adapters(self, ckpt: dict) -> None:
        loaded = 0
        for layer_name, layer in self.fusion_layers.items():
            safe = layer_name.replace(".", "__")
            if f"lora_pano__{safe}" in ckpt:
                layer.lora_pano_adapter.load_state_dict(ckpt[f"lora_pano__{safe}"])
                loaded += 1
            if f"lora_aesg__{safe}" in ckpt:
                layer.lora_aesg_adapter.load_state_dict(ckpt[f"lora_aesg__{safe}"])
        if loaded:
            print(f"[Editor] Loaded LoRA adapter weights for {loaded} layers.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def edit(
        self,
        panorama: str | Path | Image.Image,
        prompt: str,
        task_type: str = "inpaint",
        save_intermediates: bool = False,
        detection_text: str | None = None,
    ) -> dict:
        """Edit a panoramic image.

        Args:
            panorama:          Path to ERP panorama image or PIL Image.
            prompt:            Natural-language editing instruction.
            task_type:         "reconstruct" | "inpaint".
            save_intermediates: Include intermediate data in result dict.
            detection_text:    Optional text query for the object detector
                               (e.g. ``"wooden bench"``).  When *None* the
                               target object name is inferred from the AESG
                               anchor node.

        Returns:
            dict with "edited_panorama" (PIL Image) and, when
            save_intermediates=True, "gamma_p", "gamma_s", "roi_result",
            "effective_prompt", "aesg_prompt".
        """
        from aesg.schema import build_aesg_graph, build_aesg_prompt
        from aesg.encoder import encode_aesg
        from panorama_editing.executor import load_config
        from panorama_editing.roi.roi_localization import localize_and_project_roi
        from panorama_editing.reproject import reproject_to_erp

        if isinstance(panorama, (str, Path)):
            pano_img = Image.open(panorama).convert("RGB")
        else:
            pano_img = panorama.convert("RGB")

        config = load_config()

        # ------------------------------------------------------------------
        # Build AESG graph and condition tokens (same as executor)
        # ------------------------------------------------------------------
        aesg_graph = build_aesg_graph(text=prompt)
        condition_tokens = encode_aesg(aesg_graph)   # dict of tensors, used by Qwen pipeline
        aesg_prompt = build_aesg_prompt(aesg_graph)
        effective_prompt = prompt
        if aesg_prompt:
            effective_prompt = f"{prompt}\n\nStructured constraints: {aesg_prompt}"

        # ------------------------------------------------------------------
        # Object detection (Grounded-SAM / DINO) → feeds precise ROI
        # Falls back gracefully when the library is not installed.
        # ------------------------------------------------------------------
        det_query = detection_text or _anchor_text_from_aesg(aesg_graph)
        detection = _detect_object(pano_img, det_query) if det_query else None

        # ------------------------------------------------------------------
        # ROI localisation — uses detection result when available
        # ------------------------------------------------------------------
        roi_result = localize_and_project_roi(
            pano_img, aesg_graph,
            detection=detection,
            config=config,
        )
        W_erp, H_erp = pano_img.size

        # Convert ROI angles to ProjectionParams for LoRA-Pano encoding
        # roi_result["theta"] = azimuth (lon), roi_result["phi"] = elevation (lat)
        proj_params = ProjectionParams(
            lat=float(roi_result["phi"]),
            lon=float(roi_result["theta"]),
            fov=float(roi_result["fov"]),
        )

        # ------------------------------------------------------------------
        # Encode LoRA conditions
        # ------------------------------------------------------------------
        theta_t = proj_params.to_tensor(self.device)
        z_theta = self.distortion_enc(theta_t)

        # Build AESG tensor condition for LoRA-AESG (same tokens, different shape)
        z_G = self.aesg_aggregator(
            self._aesg_tokens_to_lora_cond(condition_tokens),
            task_type=task_type,
        )
        gamma_p, gamma_s = self.gating(z_theta, z_G, task_type=task_type)

        print(
            f"[Editor] lat={proj_params.lat:.2f} lon={proj_params.lon:.2f} "
            f"fov={proj_params.fov:.1f}  "
            f"gamma_p={gamma_p.item():.3f}  gamma_s={gamma_s.item():.3f}"
        )

        # ------------------------------------------------------------------
        # Call Qwen pipeline with LoRA conditions primed in every fusion layer
        # ------------------------------------------------------------------
        gen = torch.Generator(device=self.device).manual_seed(0)

        local_img  = roi_result["local_image"]
        confidence = float(roi_result["projection_meta"].get("confidence", 0.0))
        strategy   = roi_result["projection_meta"].get("strategy", "heuristic_crop")
        threshold  = float(config.get("roi_confidence_threshold", 0.7))
        # Only do local patch editing when we have a *real* detection or an
        # explicit roi_hint.  The heuristic fallback ("heuristic_crop") relies
        # on keyword-inferred angles that are too imprecise to crop the correct
        # area — editing a wrongly-cropped patch (e.g. steps instead of a
        # walkway) and reprojecting it back produces worse results than editing
        # the full panorama directly.
        use_local  = (
            confidence >= threshold
            and strategy in ("detection_perspective", "hint_perspective")
        )

        # Prime LoRA conditioning only when doing local patch editing.
        # The LoRA adapters were trained exclusively on perspective patches, so
        # applying them to a full ERP edit introduces a spatial bias (z_theta
        # from the ROI leaks into the full-panorama generation) that degrades
        # object placement.  When falling back to full-panorama editing we
        # behave identically to the baseline executor.
        ctx = (
            _primed_lora(self.fusion_layers, z_theta, z_G, gamma_p, gamma_s)
            if (self._has_lora and use_local)
            else _noop_ctx()
        )

        with ctx:
            if use_local:
                # ----------------------------------------------------------
                # Match training setup: degrade the local patch in the object
                # mask region before passing to the pipeline.
                #
                # During training the model saw I_p_deg (masked-out / degraded
                # perspective patch) as input, conditioned on z_theta, and
                # learned to fill the object region with ERP-distortion-aware
                # appearance.  Passing a clean patch at inference breaks this
                # conditioning path — the LoRA's distortion injection never
                # activates because the input domain doesn't match training.
                #
                # By applying the same degradation here we restore the
                # train/inference alignment: the model sees a degraded patch,
                # uses z_theta (via LoRA-Pano) to know where in the ERP this
                # view is, and generates the object with correct spherical
                # distortion pre-baked before reprojection.
                # ----------------------------------------------------------
                if self._has_lora:
                    local_img = _degrade_local_patch(
                        local_img,
                        roi_result["mask"],
                        strategy=config.get("degrade_strategy", "gray"),
                    )

                edited_local = self.pipeline(
                    image=[local_img],
                    prompt=effective_prompt,
                    generator=gen,
                    true_cfg_scale=4.0,
                    negative_prompt=" ",
                    num_inference_steps=40,
                    guidance_scale=1.0,
                    num_images_per_prompt=1,
                    aesg_condition=condition_tokens,
                    aesg_config=config,
                ).images[0]
                final_img = reproject_to_erp(pano_img, edited_local, roi_result)
            else:
                # Low confidence → edit full panorama (same as executor fallback)
                final_img = self.pipeline(
                    image=[pano_img],
                    prompt=effective_prompt,
                    generator=gen,
                    true_cfg_scale=4.0,
                    negative_prompt=" ",
                    num_inference_steps=40,
                    guidance_scale=1.0,
                    num_images_per_prompt=1,
                    aesg_condition=condition_tokens,
                    aesg_config=config,
                ).images[0]
                edited_local = None

        out: dict = {"edited_panorama": final_img}
        if save_intermediates:
            out["gamma_p"] = float(gamma_p.item())
            out["gamma_s"] = float(gamma_s.item())
            out["roi_result"] = roi_result
            out["effective_prompt"] = effective_prompt
            out["aesg_prompt"] = aesg_prompt
            if edited_local is not None:
                out["edited_local"] = edited_local
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _aesg_tokens_to_lora_cond(self, condition_tokens: dict) -> dict:
        """Move encode_aesg() output tensors to the correct device.

        encode_aesg() already returns {"anchor_tokens", "object_tokens",
        "context_tokens", "relation_tokens"} — exactly what AESGConditionAggregator
        expects.  We just ensure each tensor is [1, N, H] and on self.device.
        """
        H = 3584
        keys = ["anchor_tokens", "object_tokens", "context_tokens", "relation_tokens"]
        result: dict = {}
        for k in keys:
            t = condition_tokens.get(k, torch.zeros(1, 2, H))
            if t.dim() == 2:
                t = t.unsqueeze(0)
            result[k] = t.to(self.device)
        return result


@contextmanager
def _noop_ctx():
    yield


# ---------------------------------------------------------------------------
# Degradation helper — matches panorama_dataset.py training setup
# ---------------------------------------------------------------------------

def _degrade_local_patch(
    local_img: "Image.Image",
    mask: "Image.Image",
    strategy: str = "soft_blend",
    degrade_alpha: float = 0.6,
) -> "Image.Image":
    """Apply the same degradation used during LoRA training to a local patch.

    During training, ``I_p_deg`` is built by replacing the object mask region
    with gray / noise / blur before feeding into the model.  At inference,
    applying the same degradation restores the train/inference alignment so
    that the LoRA's z_theta-conditioned distortion injection activates
    correctly.

    Args:
        local_img:     PIL Image of the perspective patch (clean).
        mask:          PIL Image (mode "L") marking the object region in patch
                       coordinates (255 = object, 0 = background).
        strategy:      ``"soft_blend"`` | ``"gray"`` | ``"noise"`` |
                       ``"blur"`` | ``"random"``.
                       Defaults to ``"soft_blend"`` which preserves high-
                       frequency texture (reflections, structure) better than
                       the hard ``"gray"`` replacement used during training.
        degrade_alpha: Blend weight for ``"soft_blend"`` strategy.
                       0.0 = original unchanged, 1.0 = full gray.
                       Defaults to 0.6 (keeps 40 % of original texture so
                       the model still sees reflections / structure cues while
                       the LoRA conditioning pathway activates correctly).

    Returns:
        Degraded PIL Image of the same size as ``local_img``.
    """
    import numpy as np
    import random as _random
    from PIL import Image as _PILImage

    img_arr  = np.array(local_img.convert("RGB")).astype(np.float32)
    mask_arr = np.array(mask.convert("L"))

    obj_region = mask_arr > 0

    if strategy == "random":
        strategy = _random.choice(["soft_blend", "noise", "blur"])

    if strategy == "soft_blend":
        # Blend original pixels toward gray in the object region.
        # Unlike hard "gray" this keeps partial texture / reflection
        # information (high-frequency detail), preventing the model from
        # having to reconstruct surface appearance entirely from scratch.
        alpha = float(degrade_alpha)
        gray = np.full_like(img_arr, 128.0)
        img_arr[obj_region] = (
            (1.0 - alpha) * img_arr[obj_region] + alpha * gray[obj_region]
        )
    elif strategy == "noise":
        noise = np.random.normal(128, 40, img_arr.shape).clip(0, 255)
        img_arr[obj_region] = noise[obj_region]
    elif strategy == "blur":
        import cv2  # type: ignore
        # Use a moderate sigma (12) rather than 40 so structural edges and
        # reflections are softened but not completely destroyed.
        sigma = 12.0
        k = int(sigma * 3) | 1
        src = img_arr.astype(np.uint8)
        blurred = cv2.GaussianBlur(src, (k, k), sigma).astype(np.float32)
        img_arr[obj_region] = blurred[obj_region]
    else:  # "gray" — original hard replacement (kept for training compatibility)
        img_arr[obj_region] = 128.0

    return _PILImage.fromarray(img_arr.clip(0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Object detection helpers
# ---------------------------------------------------------------------------

def _anchor_text_from_aesg(aesg_graph) -> str | None:
    """Extract the anchor object name from the AESG graph for use as a
    detection query.  Returns *None* when the graph has no anchor."""
    try:
        anchor = aesg_graph.anchor
        if anchor and anchor.name:
            return anchor.name.strip()
    except AttributeError:
        pass
    # Fallback: use the first core object name
    try:
        if aesg_graph.core_objects:
            return aesg_graph.core_objects[0].name.strip()
    except (AttributeError, IndexError):
        pass
    return None


def _detect_object(
    image: "Image.Image",
    text_query: str,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
) -> dict | None:
    """Run Grounded-DINO + SAM to localise *text_query* in *image*.

    Returns a detection dict ``{"box": [x1,y1,x2,y2], "mask": np.ndarray|None,
    "score": float}`` for the highest-confidence detection, or *None* when the
    library is unavailable or no object is found above threshold.

    The function is intentionally lenient — any import error or runtime error
    causes a graceful *None* return so the pipeline degrades to the heuristic
    fallback.
    """
    try:
        import numpy as np
        # Try Grounded-SAM 2 / groundingdino package
        from groundingdino.util.inference import load_model, predict, annotate  # type: ignore
        import groundingdino.datasets.transforms as T  # type: ignore
        import torchvision  # type: ignore
    except ImportError:
        try:
            # Fallback: try the older grounded_sam package layout
            from groundingdino.util.inference import load_model, predict  # type: ignore
        except ImportError:
            # Neither package available — skip detection silently
            return None

    try:
        import torch
        from torchvision.ops import box_convert  # type: ignore

        # Lazy-init: cache the model on first call to avoid repeated loads
        if not hasattr(_detect_object, "_gdino_model"):
            import os, glob
            # Search for a Grounding DINO config and checkpoint in common locations
            cfg_candidates = glob.glob(
                os.path.join(os.path.dirname(__file__), "..", "**", "groundingdino_swint_ogc.py"),
                recursive=True,
            )
            ckpt_candidates = glob.glob(
                os.path.join(os.path.dirname(__file__), "..", "**", "groundingdino_swint_ogc.pth"),
                recursive=True,
            )
            if not cfg_candidates or not ckpt_candidates:
                # Config / checkpoint not found — skip detection
                _detect_object._gdino_model = None  # type: ignore[attr-defined]
            else:
                _detect_object._gdino_model = load_model(  # type: ignore[attr-defined]
                    cfg_candidates[0], ckpt_candidates[0]
                )

        model = _detect_object._gdino_model  # type: ignore[attr-defined]
        if model is None:
            return None

        # Prepare image tensor
        import torchvision.transforms as TVT
        transform = TVT.Compose([
            TVT.Resize((800, 1333)),
            TVT.ToTensor(),
            TVT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        img_tensor = transform(image).unsqueeze(0)

        W_orig, H_orig = image.size

        boxes, logits, _ = predict(
            model=model,
            image=img_tensor,
            caption=text_query,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        if len(boxes) == 0:
            return None

        # boxes are in [cx, cy, w, h] normalised; convert to [x1,y1,x2,y2] pixels
        best_idx = int(logits.argmax())
        box_norm = boxes[best_idx]  # [4]
        box_xyxy = box_convert(box_norm.unsqueeze(0), in_fmt="cxcywh", out_fmt="xyxy")[0]
        x1 = float(box_xyxy[0]) * W_orig
        y1 = float(box_xyxy[1]) * H_orig
        x2 = float(box_xyxy[2]) * W_orig
        y2 = float(box_xyxy[3]) * H_orig

        return {
            "box":   [x1, y1, x2, y2],
            "mask":  None,               # SAM mask integration can be added later
            "score": float(logits[best_idx]),
        }

    except Exception as exc:
        # Any runtime error → degrade gracefully
        print(f"[Editor] Detection failed ({exc}); falling back to heuristic ROI.")
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dual-LoRA panorama editing.  Only --input and --prompt are required."
    )
    parser.add_argument("--input",   required=True, help="Input ERP panorama image")
    parser.add_argument("--prompt",  required=True, help="Editing instruction in natural language")
    parser.add_argument("--output",  default="output_edited.jpg")
    parser.add_argument(
        "--backbone_path",
        default="Qwen/Qwen-Image-Edit-2511",
        help="HuggingFace model ID or local path to Qwen-Image-Edit-2511",
    )
    parser.add_argument("--stage2_ckpt",   default=None, help="Stage 2 checkpoint (LoRA weights)")
    parser.add_argument("--stage1_ckpt",   default=None, help="Stage 1 checkpoint (fallback)")
    parser.add_argument("--lora_rank",     type=int, default=8)
    parser.add_argument("--task_type",     default="inpaint", choices=["reconstruct", "inpaint"])
    parser.add_argument("--save_intermediates", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    editor = PanoramaEditorWithLoRA(
        stage2_ckpt=args.stage2_ckpt,
        stage1_ckpt=args.stage1_ckpt,
        backbone_path=args.backbone_path,
        device=args.device,
        lora_rank=args.lora_rank,
    )

    result = editor.edit(
        panorama=args.input,
        prompt=args.prompt,
        task_type=args.task_type,
        save_intermediates=args.save_intermediates,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result["edited_panorama"].save(out_path)
    print(f"[Editor] Saved → {out_path}")

    if args.save_intermediates:
        stem = out_path.stem
        if "edited_local" in result:
            result["edited_local"].save(out_path.parent / f"{stem}_local_edit.jpg")
        meta = {
            "gamma_p":          result.get("gamma_p"),
            "gamma_s":          result.get("gamma_s"),
            "effective_prompt": result.get("effective_prompt"),
            "aesg_prompt":      result.get("aesg_prompt"),
        }
        with open(out_path.parent / f"{stem}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[Editor] Intermediates saved alongside {out_path}")


if __name__ == "__main__":
    main()
