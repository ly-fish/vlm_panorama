"""ROI localisation for panorama object-level editing.

Priority of localisation strategies (highest → lowest confidence):

1. **Detection result** (``detection`` arg): a real bounding box + optional mask
   from an object detector (e.g. Grounded-SAM / DINO).  The ERP bbox is
   converted to lat/lon angles and an adaptive FoV, then a geometrically
   correct perspective patch is extracted via ``erp_to_perspective()``.

2. **Manual roi_hint** (``roi_hint`` arg): caller-supplied spherical angles
   ``{theta, phi, fov}``.  A perspective patch is extracted at the given
   viewing direction.

3. **Heuristic fallback**: azimuth/elevation are inferred from keywords in
   the AESG graph (left/right/up/down …).  A direct rectangular ERP crop is
   returned.  This path has low confidence (0.2) and should only be used when
   no detection or hint is available.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from aesg.schema import AESGGraph
from data.erp_utils import erp_to_perspective, project_mask_to_perspective
from lora.distortion_encoder import ProjectionParams


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def localize_and_project_roi(
    image_erp: Image.Image,
    aesg_graph: AESGGraph,
    roi_hint: dict[str, Any] | None = None,
    detection: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Localise the object of interest and extract a perspective patch.

    Args:
        image_erp:  ERP panorama (PIL Image, any size).
        aesg_graph: Parsed AESG scene graph built from the edit prompt.
        roi_hint:   Manual viewing direction override::

                        {"theta": float,  # azimuth in radians
                         "phi":   float,  # elevation in radians
                         "fov":   float}  # horizontal FoV in degrees

        detection:  Object detection result (from Grounded-SAM / DINO)::

                        {"box":   [x1, y1, x2, y2],  # ERP pixel coords
                         "mask":  np.ndarray | None,  # [H_erp, W_erp] uint8
                         "score": float}              # detector confidence

        config:     Pipeline configuration dict (see executor.load_config).

    Returns:
        dict with keys:

        * ``local_image``     – PIL Image of the extracted patch.
        * ``mask``            – PIL Image (mode "L") marking the edit region
                                in *patch* space.
        * ``bbox``            – (left, top, right, bottom) in ERP pixels
                                (approximate; used by the fallback reprojector).
        * ``theta``           – azimuth  of patch centre in **radians**.
        * ``phi``             – elevation of patch centre in **radians**.
        * ``fov``             – horizontal FoV of the perspective patch.
        * ``sample_map``      – [H_p, W_p, 2] float32 (u, v) ERP pixel coords
                                for each perspective output pixel, or ``None``
                                for the heuristic-crop fallback.
        * ``proj_params``     – :class:`ProjectionParams` or ``None``.
        * ``projection_meta`` – dict with ``strategy`` and ``confidence``.
    """
    image = image_erp.convert("RGB")
    W_erp, H_erp = image.size
    config = config or {}
    out_size = int(config.get("local_edit_size", 512))

    # ------------------------------------------------------------------
    # Branch 1: Real detection result → proper perspective projection
    # ------------------------------------------------------------------
    if detection is not None:
        return _from_detection(image, W_erp, H_erp, detection, out_size)

    # ------------------------------------------------------------------
    # Branch 2: Manual roi_hint → perspective projection at given angles
    # ------------------------------------------------------------------
    if roi_hint is not None:
        return _from_roi_hint(image, W_erp, H_erp, roi_hint, config, out_size)

    # ------------------------------------------------------------------
    # Branch 3: Heuristic fallback — keyword-based angles, ERP crop
    # ------------------------------------------------------------------
    return _from_heuristic(image, W_erp, H_erp, aesg_graph, config)


# ---------------------------------------------------------------------------
# Branch implementations
# ---------------------------------------------------------------------------

def _from_detection(
    image: Image.Image,
    W_erp: int,
    H_erp: int,
    detection: dict[str, Any],
    out_size: int,
) -> dict[str, Any]:
    box = detection["box"]          # [x1, y1, x2, y2] in ERP pixels
    score = float(detection.get("score", 1.0))

    # Adaptive FoV from box dimensions (1.5× margin, clamped 30–150°)
    proj_params = ProjectionParams.from_box(box, W_erp, H_erp)

    local_image, sample_map = erp_to_perspective(
        image,
        lat_deg=proj_params.lat,
        lon_deg=proj_params.lon,
        fov_deg=proj_params.fov,
        out_w=out_size,
        out_h=out_size,
    )

    # Project detection mask; if none supplied, rasterise bbox in persp. space
    erp_mask_np = detection.get("mask")
    if erp_mask_np is not None:
        persp_mask_np = project_mask_to_perspective(erp_mask_np, sample_map)
        mask = Image.fromarray(persp_mask_np, mode="L")
    else:
        mask = _bbox_mask_in_perspective(box, W_erp, H_erp, sample_map, out_size)

    bbox_erp = (
        max(0, int(box[0])), max(0, int(box[1])),
        min(W_erp, int(box[2])), min(H_erp, int(box[3])),
    )

    return {
        "local_image": local_image,
        "mask": mask,
        "bbox": bbox_erp,
        "theta": math.radians(proj_params.lon),
        "phi": math.radians(proj_params.lat),
        "fov": proj_params.fov,
        "sample_map": sample_map,
        "proj_params": proj_params,
        "projection_meta": {
            "erp_size": (W_erp, H_erp),
            "strategy": "detection_perspective",
            # Clamp so that even a low-score detection triggers local editing
            "confidence": max(0.7, min(1.0, score)),
        },
    }


def _from_roi_hint(
    image: Image.Image,
    W_erp: int,
    H_erp: int,
    roi_hint: dict[str, Any],
    config: dict[str, Any],
    out_size: int,
) -> dict[str, Any]:
    theta = float(roi_hint.get("theta", 0.0))
    phi   = float(roi_hint.get("phi", 0.0))
    fov   = float(roi_hint.get("fov", float(config.get("default_fov", 75.0))))

    lon_deg = math.degrees(theta)
    lat_deg = math.degrees(phi)

    local_image, sample_map = erp_to_perspective(
        image,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        fov_deg=fov,
        out_w=out_size,
        out_h=out_size,
    )
    mask = _center_oval_mask(out_size, out_size)

    # Approximate ERP bbox for the fallback reprojector
    center_x = int(((theta + math.pi) / (2 * math.pi)) * W_erp) % W_erp
    center_y = int(((math.pi / 2 - phi) / math.pi) * H_erp)
    crop_w = int(W_erp * float(config.get("roi_width_ratio", 0.33)))
    crop_h = int(H_erp * float(config.get("roi_height_ratio", 0.45)))
    left = max(0, min(W_erp - crop_w, center_x - crop_w // 2))
    top  = max(0, min(H_erp - crop_h, center_y - crop_h // 2))

    return {
        "local_image": local_image,
        "mask": mask,
        "bbox": (left, top, left + crop_w, top + crop_h),
        "theta": theta,
        "phi": phi,
        "fov": fov,
        "sample_map": sample_map,
        "proj_params": ProjectionParams(lat=lat_deg, lon=lon_deg, fov=fov),
        "projection_meta": {
            "erp_size": (W_erp, H_erp),
            "strategy": "hint_perspective",
            "confidence": 1.0,
        },
    }


def _from_heuristic(
    image: Image.Image,
    W_erp: int,
    H_erp: int,
    aesg_graph: AESGGraph,
    config: dict[str, Any],
) -> dict[str, Any]:
    roi_width_ratio  = float(config.get("roi_width_ratio", 0.33))
    roi_height_ratio = float(config.get("roi_height_ratio", 0.45))
    crop_w = max(64, min(W_erp, int(W_erp * roi_width_ratio)))
    crop_h = max(64, min(H_erp, int(H_erp * roi_height_ratio)))

    theta, phi = _infer_angles_from_graph(aesg_graph)
    confidence = 0.75 if _has_explicit_localization(aesg_graph) else 0.2

    center_x = int(((theta + math.pi) / (2 * math.pi)) * W_erp) % W_erp
    center_y = int(((math.pi / 2 - phi) / math.pi) * H_erp)
    center_y = max(crop_h // 2, min(H_erp - crop_h // 2, center_y))
    left = max(0, min(W_erp - crop_w, center_x - crop_w // 2))
    top  = max(0, min(H_erp - crop_h, center_y - crop_h // 2))
    bbox = (left, top, left + crop_w, top + crop_h)

    local_image = image.crop(bbox)
    mask = _center_oval_mask(crop_w, crop_h)

    return {
        "local_image": local_image,
        "mask": mask,
        "bbox": bbox,
        "theta": theta,
        "phi": phi,
        "fov": float(config.get("default_fov", 75.0)),
        "sample_map": None,
        "proj_params": None,
        "projection_meta": {
            "center_x": center_x,
            "center_y": center_y,
            "erp_size": (W_erp, H_erp),
            "strategy": "heuristic_crop",
            "confidence": confidence,
        },
    }


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _center_oval_mask(w: int, h: int) -> Image.Image:
    """A rounded-rectangle mask centred in a (w × h) patch."""
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    mx = max(8, w // 8)
    my = max(8, h // 8)
    draw.rounded_rectangle(
        (mx, my, w - mx, h - my),
        radius=min(w, h) // 10,
        fill=255,
    )
    return mask


def _bbox_mask_in_perspective(
    box: list[float],
    W_erp: int,
    H_erp: int,
    sample_map: np.ndarray,
    out_size: int,
) -> Image.Image:
    """Build a perspective-space mask from an ERP bounding box.

    For each perspective pixel we check whether its corresponding ERP
    coordinate falls inside the detected bounding box.
    """
    x1, y1, x2, y2 = box
    u = sample_map[..., 0]   # [H_p, W_p]
    v = sample_map[..., 1]

    inside = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
    mask_np = inside.astype(np.uint8) * 255
    return Image.fromarray(mask_np, mode="L")


# ---------------------------------------------------------------------------
# Heuristic helpers (retained for fallback branch)
# ---------------------------------------------------------------------------

def _infer_angles_from_graph(graph: AESGGraph) -> tuple[float, float]:
    relation_text = " ".join(rel.relation_type for rel in graph.relations).lower()
    object_text   = " ".join(obj.name for obj in graph.core_objects).lower()
    text = f"{relation_text} {object_text}"
    theta = 0.0
    phi   = 0.0
    if "left" in text:
        theta = -math.pi / 2
    elif "right" in text:
        theta =  math.pi / 2
    elif "back" in text:
        theta =  math.pi
    if "up" in text or "top" in text:
        phi =  math.pi / 8
    elif "down" in text or "ground" in text or "floor" in text:
        phi = -math.pi / 8
    return theta, phi


def _has_explicit_localization(graph: AESGGraph) -> bool:
    text = " ".join(
        [rel.relation_type for rel in graph.relations]
        + [rel.direction      for rel in graph.relations]
        + [obj.action         for obj in graph.core_objects]
    ).lower()
    tokens = (
        "left", "right", "top", "bottom", "center",
        "front", "back", "ground", "floor", "ceiling",
        "near", "beside",
    )
    return any(token in text for token in tokens)
