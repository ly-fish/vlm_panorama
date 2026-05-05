"""Evaluation script for panorama editing outputs.

Metrics
-------
  CLIP Score        — text–image alignment (edited image vs. prompt)
  CLIP Dir Score    — directional CLIP similarity (edit consistency)
  PSNR / SSIM       — pixel-level / structural fidelity vs. original
  BG PSNR / SSIM    — same metrics but ONLY on the non-masked background region
                      (most important for panorama editing: measures whether the
                       system pollutes the background outside the edit region)
  LPIPS             — perceptual similarity vs. original  (if lpips installed)

Usage examples
--------------
  # Single image + prompt
  python evaluate.py \\
      --input   data/train/scene_000/panorama.jpg \\
      --edited  output/scene_000_lora.png \\
      --prompt  "Replace the wooden bench with a red modern chair"

  # With mask for background-preservation metrics
  python evaluate.py \\
      --input   data/train/scene_000/panorama.jpg \\
      --edited  output/scene_000_lora.png \\
      --prompt  "Replace the wooden bench with a red modern chair" \\
      --mask    data/train/scene_000/result_*_mask.jpg

  # Compare baseline vs. LoRA side-by-side (auto-discovers files)
  python evaluate.py \\
      --input      data/train/scene_000/panorama.jpg \\
      --output_dir output/comparison_001 \\
      --prompt     "Replace the wooden bench with a red modern chair"

  # Batch mode: JSON file listing {input, edited, prompt, mask(opt)} entries
  python evaluate.py --batch eval_list.json --output_csv results.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---------------------------------------------------------------------------
# Optional heavy deps — degrade gracefully
# ---------------------------------------------------------------------------
try:
    import lpips as _lpips_mod
    _LPIPS_AVAILABLE = True
except ImportError:
    _LPIPS_AVAILABLE = False
    warnings.warn("lpips not installed — LPIPS metric skipped. pip install lpips")

try:
    from skimage.metrics import structural_similarity as _ssim_fn
    from skimage.metrics import peak_signal_noise_ratio as _psnr_fn
    _SKIMAGE_AVAILABLE = True
except ImportError:
    _SKIMAGE_AVAILABLE = False
    warnings.warn("scikit-image not installed — using numpy fallback for PSNR/SSIM.")


# ---------------------------------------------------------------------------
# CLIP loader (singleton)
# ---------------------------------------------------------------------------
_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_CLIP_DEVICE = None


def _get_clip(device: str = "auto") -> tuple:
    global _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_DEVICE
    if _CLIP_MODEL is None:
        import clip
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        _CLIP_MODEL, _CLIP_PREPROCESS = clip.load("ViT-B/32", device=device)
        _CLIP_MODEL.eval()
        _CLIP_DEVICE = device
    return _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_DEVICE


# ---------------------------------------------------------------------------
# LPIPS loader (singleton)
# ---------------------------------------------------------------------------
_LPIPS_NET = None


def _get_lpips(device: str) -> Optional[object]:
    global _LPIPS_NET
    if not _LPIPS_AVAILABLE:
        return None
    if _LPIPS_NET is None:
        _LPIPS_NET = _lpips_mod.LPIPS(net="alex").to(device)
        _LPIPS_NET.eval()
    return _LPIPS_NET


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _load_rgb(path: str | Path) -> np.ndarray:
    """Load image as HxWx3 uint8 array."""
    img = Image.open(path).convert("RGB")
    return np.array(img)


def _to_tensor_01(arr: np.ndarray, device: str) -> torch.Tensor:
    """HxWx3 uint8 → 1x3xHxW float32 in [0,1]."""
    t = torch.from_numpy(arr).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return t.to(device)


def _to_tensor_lpips(arr: np.ndarray, device: str) -> torch.Tensor:
    """HxWx3 uint8 → 1x3xHxW float32 in [-1,1] (LPIPS convention)."""
    return _to_tensor_01(arr, device) * 2.0 - 1.0


def _resize_match(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resize b to match a's spatial dimensions if needed."""
    if a.shape == b.shape:
        return a, b
    h, w = a.shape[:2]
    b_pil = Image.fromarray(b).resize((w, h), Image.LANCZOS)
    return a, np.array(b_pil)


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def compute_clip_score(image: np.ndarray, prompt: str, device: str = "auto") -> float:
    """CLIP cosine similarity between an image and a text prompt (0–100)."""
    import clip
    model, preprocess, dev = _get_clip(device)
    pil = Image.fromarray(image)
    img_t = preprocess(pil).unsqueeze(0).to(dev)
    text_t = clip.tokenize([prompt], truncate=True).to(dev)
    with torch.no_grad():
        img_feat = model.encode_image(img_t)
        txt_feat = model.encode_text(text_t)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        score = (img_feat * txt_feat).sum().item()
    return float(score) * 100.0  # scale to 0-100 for readability


def compute_clip_dir_score(
    original: np.ndarray,
    edited: np.ndarray,
    src_prompt: str,
    tgt_prompt: str,
    device: str = "auto",
) -> float:
    """Directional CLIP score: cosine similarity between image delta and text delta.

    Measures whether the edit direction in CLIP space matches the intended
    text direction (src_prompt → tgt_prompt).  Higher is better.
    """
    import clip
    model, preprocess, dev = _get_clip(device)

    orig_t = preprocess(Image.fromarray(original)).unsqueeze(0).to(dev)
    edit_t = preprocess(Image.fromarray(edited)).unsqueeze(0).to(dev)
    src_t = clip.tokenize([src_prompt], truncate=True).to(dev)
    tgt_t = clip.tokenize([tgt_prompt], truncate=True).to(dev)

    with torch.no_grad():
        f_orig = model.encode_image(orig_t)
        f_edit = model.encode_image(edit_t)
        f_src  = model.encode_text(src_t)
        f_tgt  = model.encode_text(tgt_t)

        delta_img  = f_edit - f_orig
        delta_text = f_tgt  - f_src
        delta_img  = delta_img  / (delta_img.norm(dim=-1, keepdim=True)  + 1e-8)
        delta_text = delta_text / (delta_text.norm(dim=-1, keepdim=True) + 1e-8)
        score = (delta_img * delta_text).sum().item()

    return float(score) * 100.0


def compute_psnr(original: np.ndarray, edited: np.ndarray) -> float:
    """PSNR in dB. Higher = more similar to original (useful for preservation check)."""
    original, edited = _resize_match(original, edited)
    if _SKIMAGE_AVAILABLE:
        return float(_psnr_fn(original, edited, data_range=255))
    mse = np.mean((original.astype(np.float32) - edited.astype(np.float32)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(255.0**2 / mse))


def compute_ssim(original: np.ndarray, edited: np.ndarray) -> float:
    """SSIM in [-1, 1]. Higher = more structurally similar to original."""
    original, edited = _resize_match(original, edited)
    if _SKIMAGE_AVAILABLE:
        return float(_ssim_fn(original, edited, channel_axis=2, data_range=255))
    # Minimal numpy fallback (less accurate)
    mu1 = original.astype(np.float32).mean()
    mu2 = edited.astype(np.float32).mean()
    s1  = original.astype(np.float32).std()
    s2  = edited.astype(np.float32).std()
    s12 = np.mean(
        (original.astype(np.float32) - mu1) * (edited.astype(np.float32) - mu2)
    )
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return float(
        (2 * mu1 * mu2 + c1) * (2 * s12 + c2)
        / ((mu1**2 + mu2**2 + c1) * (s1**2 + s2**2 + c2))
    )


def compute_lpips(original: np.ndarray, edited: np.ndarray, device: str) -> Optional[float]:
    """LPIPS perceptual distance. Lower = more similar to original."""
    net = _get_lpips(device)
    if net is None:
        return None
    original, edited = _resize_match(original, edited)
    t_orig = _to_tensor_lpips(original, device)
    t_edit = _to_tensor_lpips(edited, device)
    with torch.no_grad():
        dist = net(t_orig, t_edit).item()
    return float(dist)


def compute_background_metrics(
    original: np.ndarray,
    edited: np.ndarray,
    erp_mask: np.ndarray,
    object_value: int = 1,
) -> dict:
    """Compute PSNR and SSIM only on the background (non-edited) region.

    This is the most important metric for panorama editing: it measures whether
    the model pollutes areas outside the edit target.  A perfect system should
    preserve the background pixel-for-pixel; LoRA's distortion-aware conditioning
    should show an advantage over the plain baseline here.

    Args:
        original:     [H, W, 3] uint8 original ERP panorama.
        edited:       [H, W, 3] uint8 edited ERP panorama.
        erp_mask:     [H, W] or [H, W, C] uint8 semantic ID map (from mask.jpg),
                      or a pre-binarised mask where >0 = edited region.
        object_value: Object ID to treat as the edited region (default 1 = first
                      detected object).  Pass -1 to treat any non-zero pixel as
                      edited region (for pre-binarised masks).

    Returns:
        dict with keys bg_psnr, bg_ssim, bg_pixel_ratio.
    """
    original, edited = _resize_match(original, edited)

    # Build binary background mask (True = background = NOT edited)
    if erp_mask.ndim == 3:
        erp_mask = erp_mask[..., 0]

    # Resize mask to match image if needed
    if erp_mask.shape[:2] != original.shape[:2]:
        mask_pil = Image.fromarray(erp_mask).resize(
            (original.shape[1], original.shape[0]), Image.NEAREST
        )
        erp_mask = np.array(mask_pil)

    if object_value == -1:
        fg_mask = erp_mask > 0
    else:
        fg_mask = erp_mask == object_value

    bg_mask = ~fg_mask  # background = NOT the edited object
    bg_ratio = float(bg_mask.mean())

    if bg_ratio < 0.05:
        # Mask covers >95% of image — metrics meaningless
        return {"bg_psnr": None, "bg_ssim": None, "bg_pixel_ratio": bg_ratio}

    # Extract background pixels as flat arrays
    orig_bg = original[bg_mask]   # [N, 3]
    edit_bg = edited[bg_mask]     # [N, 3]

    mse = np.mean((orig_bg.astype(np.float32) - edit_bg.astype(np.float32)) ** 2)
    bg_psnr = float("inf") if mse == 0 else float(10 * np.log10(255.0**2 / mse))

    # For SSIM we need 2-D patches; use the masked region of each channel
    if _SKIMAGE_AVAILABLE:
        # Compute per-channel SSIM on flattened 1-D signal (fast approximation)
        # For a proper 2-D SSIM, fall back to full-image SSIM with the mask weight
        orig_f = original.astype(np.float32)
        edit_f = edited.astype(np.float32)
        bg_ssim = float(_ssim_fn(
            orig_f, edit_f,
            channel_axis=2,
            data_range=255,
            full=True,
        )[1][bg_mask].mean())
    else:
        # Simple correlation-based fallback on background pixels only
        mu1, mu2 = orig_bg.astype(np.float32).mean(), edit_bg.astype(np.float32).mean()
        s1 = orig_bg.astype(np.float32).std()
        s2 = edit_bg.astype(np.float32).std()
        s12 = np.mean(
            (orig_bg.astype(np.float32) - mu1) * (edit_bg.astype(np.float32) - mu2)
        )
        c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        bg_ssim = float(
            (2 * mu1 * mu2 + c1) * (2 * s12 + c2)
            / ((mu1**2 + mu2**2 + c1) * (s1**2 + s2**2 + c2))
        )

    return {
        "bg_psnr":        bg_psnr,
        "bg_ssim":        bg_ssim,
        "bg_pixel_ratio": bg_ratio,
    }


# ---------------------------------------------------------------------------
# High-level evaluate function
# ---------------------------------------------------------------------------

def evaluate_pair(
    input_path: str | Path,
    edited_path: str | Path,
    prompt: str,
    src_prompt: str = "a panoramic scene",
    mask_path: str | Path | None = None,
    object_value: int = 1,
    device: str = "auto",
    label: str = "",
) -> dict:
    """Evaluate one (original, edited) pair.

    Args:
        input_path:   Original ERP panorama.
        edited_path:  Edited ERP panorama.
        prompt:       Editing instruction (target prompt).
        src_prompt:   Neutral / scene-description prompt for CLIPDir.
        mask_path:    Path to the semantic mask image (result_*_mask.jpg).
                      When provided, background-preservation metrics are computed.
        object_value: Object ID in the mask treated as the edited region.
        device:       "auto" | "cuda" | "cpu".
        label:        Display label for comparison tables.

    Returns:
        Flat dict of all metrics.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    original = _load_rgb(input_path)
    edited   = _load_rgb(edited_path)

    results: dict = {
        "label":        label or Path(edited_path).stem,
        "input":        str(input_path),
        "edited":       str(edited_path),
        "prompt":       prompt,
    }

    print(f"  [CLIP Score]   computing…", end="", flush=True)
    results["clip_score"] = compute_clip_score(edited, prompt, device)
    print(f"  {results['clip_score']:.2f}")

    print(f"  [CLIP Dir]     computing…", end="", flush=True)
    results["clip_dir_score"] = compute_clip_dir_score(
        original, edited, src_prompt, prompt, device
    )
    print(f"  {results['clip_dir_score']:.2f}")

    print(f"  [PSNR]         computing…", end="", flush=True)
    results["psnr"] = compute_psnr(original, edited)
    print(f"  {results['psnr']:.2f} dB")

    print(f"  [SSIM]         computing…", end="", flush=True)
    results["ssim"] = compute_ssim(original, edited)
    print(f"  {results['ssim']:.4f}")

    print(f"  [LPIPS]        computing…", end="", flush=True)
    lpips_val = compute_lpips(original, edited, device)
    results["lpips"] = lpips_val
    print(f"  {lpips_val:.4f}" if lpips_val is not None else "  N/A (lpips not installed)")

    # Background-preservation metrics (only when mask is provided)
    if mask_path is not None:
        print(f"  [BG PSNR/SSIM] computing…", end="", flush=True)
        erp_mask = _load_rgb(mask_path)[..., 0]  # take red channel as ID map
        bg = compute_background_metrics(original, edited, erp_mask, object_value)
        results.update(bg)
        if bg["bg_psnr"] is not None:
            print(f"  PSNR={bg['bg_psnr']:.2f} dB  SSIM={bg['bg_ssim']:.4f}"
                  f"  (bg={bg['bg_pixel_ratio']:.1%})")
        else:
            print("  mask covers >95% of image, skipped")
    else:
        results["bg_psnr"] = None
        results["bg_ssim"] = None
        results["bg_pixel_ratio"] = None

    return results


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

_METRIC_FMT = {
    "clip_score":     ("CLIP Score (↑)",      "{:.2f}"),
    "clip_dir_score": ("CLIP Dir Score (↑)",  "{:.2f}"),
    "psnr":           ("PSNR dB (↑)",         "{:.2f}"),
    "ssim":           ("SSIM (↑)",            "{:.4f}"),
    "lpips":          ("LPIPS (↓)",           "{:.4f}"),
    "bg_psnr":        ("BG PSNR dB (↑)",      "{:.2f}"),
    "bg_ssim":        ("BG SSIM (↑)",         "{:.4f}"),
}


def print_comparison(results_list: list[dict]) -> None:
    """Print a side-by-side comparison table."""
    if not results_list:
        return

    col_w = 22
    label_w = 20
    header = f"{'Metric':<{label_w}}" + "".join(
        f"{r['label']:>{col_w}}" for r in results_list
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for key, (name, fmt) in _METRIC_FMT.items():
        row = f"{name:<{label_w}}"
        for r in results_list:
            val = r.get(key)
            if val is None:
                row += f"{'N/A':>{col_w}}"
            else:
                row += f"{fmt.format(val):>{col_w}}"
        print(row)

    print("=" * len(header) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate panorama editing outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Single-pair mode
    p.add_argument("--input",   default=None, help="Original input ERP image.")
    p.add_argument("--edited",  default=None, help="Edited output image to evaluate.")
    p.add_argument("--prompt",  default=None, help="Editing instruction (text).")
    p.add_argument(
        "--src_prompt", default="a panoramic scene",
        help="Source / neutral prompt for CLIPDir score.",
    )
    p.add_argument(
        "--mask", default=None,
        help="Semantic mask image (result_*_mask.jpg) for background-preservation metrics.",
    )
    p.add_argument(
        "--object_value", type=int, default=1,
        help="Object ID in the mask treated as the edited region (default: 1).",
    )

    # Auto-compare mode: discover *_baseline.png and *_lora.png in a directory
    p.add_argument(
        "--output_dir", default=None,
        help=(
            "Directory produced by run_comparison.py. "
            "Auto-discovers <stem>_baseline.png and <stem>_lora.png."
        ),
    )

    # Batch mode
    p.add_argument(
        "--batch", default=None,
        help=(
            "JSON file with a list of dicts, each having "
            "{\"input\", \"edited\", \"prompt\", \"mask\"(opt)} keys."
        ),
    )

    # Output
    p.add_argument("--output_json", default=None, help="Save results as JSON.")
    p.add_argument("--output_csv",  default=None, help="Save results as CSV.")

    # Device
    p.add_argument(
        "--device", default="auto",
        choices=["auto", "cuda", "cpu"],
    )

    return p.parse_args()


def _collect_pairs(args: argparse.Namespace) -> list[dict]:
    """Build list of {input, edited, prompt, src_prompt, label} dicts."""
    pairs: list[dict] = []

    # --- Batch JSON ---
    if args.batch:
        with open(args.batch) as f:
            entries = json.load(f)
        for e in entries:
            pairs.append({
                "input":         e["input"],
                "edited":        e["edited"],
                "prompt":        e["prompt"],
                "src_prompt":    e.get("src_prompt", "a panoramic scene"),
                "mask":          e.get("mask"),
                "object_value":  e.get("object_value", 1),
                "label":         e.get("label", Path(e["edited"]).stem),
            })
        return pairs

    # --- Auto-compare: discover files in output_dir ---
    if args.output_dir:
        out_dir = Path(args.output_dir)
        if not args.input:
            raise ValueError("--input is required with --output_dir")
        if not args.prompt:
            raise ValueError("--prompt is required with --output_dir")
        stem = Path(args.input).stem
        for suffix, label in [("_baseline", "Baseline"), ("_lora", "LoRA")]:
            candidate = out_dir / f"{stem}{suffix}.png"
            if candidate.exists():
                pairs.append({
                    "input":        args.input,
                    "edited":       str(candidate),
                    "prompt":       args.prompt,
                    "src_prompt":   args.src_prompt,
                    "mask":         getattr(args, "mask", None),
                    "object_value": getattr(args, "object_value", 1),
                    "label":        label,
                })
        if not pairs:
            raise FileNotFoundError(
                f"No *_baseline.png or *_lora.png found in {out_dir}"
            )
        return pairs

    # --- Single pair ---
    if args.input and args.edited and args.prompt:
        pairs.append({
            "input":        args.input,
            "edited":       args.edited,
            "prompt":       args.prompt,
            "src_prompt":   args.src_prompt,
            "mask":         getattr(args, "mask", None),
            "object_value": getattr(args, "object_value", 1),
            "label":        Path(args.edited).stem,
        })
        return pairs

    raise ValueError(
        "Provide one of: (--input + --edited + --prompt), "
        "(--input + --output_dir + --prompt), or --batch."
    )


def _save_csv(results_list: list[dict], path: str) -> None:
    keys = list(results_list[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results_list)
    print(f"[Eval] CSV saved → {path}")


def main() -> None:
    args = parse_args()
    pairs = _collect_pairs(args)

    all_results: list[dict] = []
    for pair in pairs:
        print(f"\n[Eval] === {pair['label']} ===")
        print(f"       input:  {pair['input']}")
        print(f"       edited: {pair['edited']}")
        print(f"       prompt: {pair['prompt']}")
        res = evaluate_pair(
            input_path=pair["input"],
            edited_path=pair["edited"],
            prompt=pair["prompt"],
            src_prompt=pair["src_prompt"],
            mask_path=pair.get("mask"),
            object_value=pair.get("object_value", 1),
            device=args.device,
            label=pair["label"],
        )
        all_results.append(res)

    print_comparison(all_results)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"[Eval] JSON saved → {args.output_json}")

    if args.output_csv:
        _save_csv(all_results, args.output_csv)


if __name__ == "__main__":
    main()
