"""Reproject an edited perspective patch back into the ERP panorama.

Two code paths:

* **sample_map path** (preferred): when ``roi_result["sample_map"]`` is
  available the edited patch is scattered back to ERP pixel positions using
  the exact inverse of the perspective projection.  This handles panorama
  seam wrapping (lon ≈ ±180°) automatically because the scatter uses modular
  ERP coordinates.

* **bbox fallback**: when no sample_map is available (heuristic-crop branch)
  a rectangular paste is performed.  Seam wrapping is handled by splitting
  the paste into two sub-rectangles when the bbox crosses the ERP boundary.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageFilter


def reproject_to_erp(
    image_erp: Image.Image,
    edited_local: Image.Image,
    roi_result: dict[str, Any],
    feather_radius: int = 8,
) -> Image.Image:
    """Blend an edited local patch back into the full ERP panorama.

    Args:
        image_erp:    Original ERP panorama (PIL Image).
        edited_local: Edited patch returned by the diffusion pipeline.
        roi_result:   Output of ``localize_and_project_roi()``.
        feather_radius: Gaussian feather radius for boundary blending (pixels).

    Returns:
        Full ERP panorama with the edit blended in (PIL Image, RGB).
    """
    sample_map = roi_result.get("sample_map")

    if sample_map is not None:
        return _reproject_perspective(image_erp, edited_local, roi_result, feather_radius)
    else:
        return _reproject_bbox(image_erp, edited_local, roi_result, feather_radius)


# ---------------------------------------------------------------------------
# Path 1: perspective projection — geometrically correct, seam-safe
# ---------------------------------------------------------------------------

def _reproject_perspective(
    image_erp: Image.Image,
    edited_local: Image.Image,
    roi_result: dict[str, Any],
    feather_radius: int,
) -> Image.Image:
    """Reproject the edited perspective patch back into the ERP panorama.

    Uses the analytical inverse-mapping path (``cv2.remap``) when
    ``roi_result["proj_params"]`` is available, which fills every ERP pixel in
    the object region without scatter holes.  Falls back to the forward-scatter
    implementation otherwise.
    """
    from data.erp_utils import reproject_perspective_to_erp as _reproject_np

    sample_map: np.ndarray = roi_result["sample_map"]   # [H_p, W_p, 2]
    H_p, W_p = sample_map.shape[:2]

    erp_arr  = np.array(image_erp.convert("RGB"))
    H_erp, W_erp = erp_arr.shape[:2]

    # Resize edited patch to match sample_map dimensions if needed
    patch = edited_local.convert("RGB")
    if patch.size != (W_p, H_p):
        patch = patch.resize((W_p, H_p), Image.Resampling.BICUBIC)
    patch_arr = np.array(patch)

    # Perspective-space object mask (needed by both paths)
    persp_mask = np.array(roi_result["mask"].convert("L"))
    if persp_mask.shape != (H_p, W_p):
        persp_mask = np.array(
            roi_result["mask"].resize((W_p, H_p), Image.Resampling.NEAREST).convert("L")
        )

    # Build ERP-space mask via forward scatter (used only by the fallback path
    # when proj_params is absent; kept here so the API stays backward compatible).
    u_idx = (sample_map[..., 0].astype(np.int32)) % W_erp          # seam-safe
    v_idx = np.clip(sample_map[..., 1].astype(np.int32), 0, H_erp - 1)

    erp_mask = np.zeros((H_erp, W_erp), dtype=np.uint8)
    valid_px = persp_mask > 0
    erp_mask[v_idx[valid_px], u_idx[valid_px]] = 255

    # Prefer the analytical inverse-mapping path when projection parameters are
    # available (detection_perspective and hint_perspective branches).
    proj_params = roi_result.get("proj_params")

    result_arr = _reproject_np(
        erp_arr, patch_arr, sample_map, erp_mask,
        feather_px=feather_radius,
        proj_params=proj_params,
        persp_mask=persp_mask,
    )
    return Image.fromarray(result_arr)


# ---------------------------------------------------------------------------
# Path 2: rectangular bbox paste — fallback for heuristic-crop branch
# ---------------------------------------------------------------------------

def _reproject_bbox(
    image_erp: Image.Image,
    edited_local: Image.Image,
    roi_result: dict[str, Any],
    feather_radius: int,
) -> Image.Image:
    """Paste edited patch at the ERP bbox; handles horizontal seam wrapping."""
    base  = image_erp.convert("RGB").copy()
    W_erp, H_erp = base.size

    patch = edited_local.convert("RGB")
    bbox  = tuple(roi_result["bbox"])              # (left, top, right, bottom)
    mask  = roi_result["mask"].convert("L")

    if feather_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_radius))

    left, top, right, bottom = bbox
    region_w = right - left
    region_h = bottom - top

    if patch.size != (region_w, region_h):
        patch = patch.resize((region_w, region_h), Image.Resampling.BICUBIC)
    if mask.size != (region_w, region_h):
        mask = mask.resize((region_w, region_h), Image.Resampling.BICUBIC)

    if right <= W_erp:
        # Normal case — no seam crossing
        region  = base.crop(bbox)
        blended = Image.composite(patch, region, mask)
        base.paste(blended, (left, top))
    else:
        # Seam crossing: split into right portion and wrapped left portion
        right_w = W_erp - left          # pixels before the right edge
        left_w  = right - W_erp         # pixels wrapping to left edge

        # Right segment [left : W_erp]
        right_patch  = patch.crop((0, 0, right_w, region_h))
        right_mask   = mask.crop((0, 0, right_w, region_h))
        right_region = base.crop((left, top, W_erp, bottom))
        base.paste(Image.composite(right_patch, right_region, right_mask), (left, top))

        # Wrapped left segment [0 : left_w]
        wrap_patch  = patch.crop((right_w, 0, region_w, region_h))
        wrap_mask   = mask.crop((right_w, 0, region_w, region_h))
        wrap_region = base.crop((0, top, left_w, bottom))
        base.paste(Image.composite(wrap_patch, wrap_region, wrap_mask), (0, top))

    return base
