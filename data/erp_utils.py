"""ERP (Equirectangular Projection) ↔ perspective conversion utilities.

Coordinate conventions
----------------------
ERP image of size (H, W):
  pixel (u, v) -> longitude = (u / W - 0.5) * 360  degrees
               -> latitude  = (0.5 - v / H) * 180  degrees

Perspective (gnomonic / rectilinear) view:
  centred at (lat_c, lon_c) with horizontal FoV.
  pixel (i, j) in an (H_p x W_p) perspective image is obtained by sampling
  the ERP at the ray direction computed from the angular offsets.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# ERP -> perspective projection
# ---------------------------------------------------------------------------

def erp_to_perspective(
    erp_img: Image.Image,
    lat_deg: float,
    lon_deg: float,
    fov_deg: float,
    out_w: int = 512,
    out_h: int = 512,
) -> tuple[Image.Image, np.ndarray]:
    """Sample a rectilinear perspective patch from an ERP panorama.

    Args:
        erp_img:  ERP panorama as a PIL Image.
        lat_deg:  Latitude of view centre in degrees  [-90, 90].
        lon_deg:  Longitude of view centre in degrees [-180, 180].
        fov_deg:  Horizontal field-of-view in degrees.
        out_w:    Width of the output perspective image in pixels.
        out_h:    Height of the output perspective image in pixels.

    Returns:
        (perspective_image, sample_map) where sample_map is a [H_p, W_p, 2]
        float32 array of (u, v) ERP pixel coordinates used for each output
        pixel (useful for mask projection).
    """
    erp_arr = np.array(erp_img.convert("RGB"), dtype=np.float32)
    H_erp, W_erp = erp_arr.shape[:2]

    lat_c = math.radians(lat_deg)
    lon_c = math.radians(lon_deg)
    fov   = math.radians(fov_deg)

    # Focal length from horizontal FoV
    f = (out_w / 2.0) / math.tan(fov / 2.0)

    # Pixel grid in perspective image (x right, y down, z forward)
    xs = (np.arange(out_w) - out_w / 2.0 + 0.5) / f   # [W_p]
    ys = (np.arange(out_h) - out_h / 2.0 + 0.5) / f   # [H_p]
    xv, yv = np.meshgrid(xs, ys)                        # [H_p, W_p]

    # 3-D unit vectors in camera space (z=1 plane)
    zv = np.ones_like(xv)
    norm = np.sqrt(xv**2 + yv**2 + zv**2)
    xv, yv, zv = xv / norm, yv / norm, zv / norm

    # Rotate from camera space to world space
    # Camera looks along +z; tilt by lat_c (pitch) then rotate by lon_c (yaw)
    # Pitch rotation (around x-axis, upward positive)
    cos_lat, sin_lat = math.cos(lat_c), math.sin(lat_c)
    xw =  xv
    yw =  yv * cos_lat - zv * sin_lat
    zw =  yv * sin_lat + zv * cos_lat

    # Yaw rotation (around y-axis)
    cos_lon, sin_lon = math.cos(lon_c), math.sin(lon_c)
    xr =  xw * cos_lon + zw * sin_lon
    yr =  yw
    zr = -xw * sin_lon + zw * cos_lon

    # Convert world unit-vector to spherical coordinates
    lat_map = np.arcsin(np.clip(yr, -1.0, 1.0))       # [-pi/2, pi/2]
    lon_map = np.arctan2(xr, zr)                        # [-pi, pi]

    # Map spherical -> ERP pixel coordinates
    u_map = (lon_map / (2 * math.pi) + 0.5) * W_erp   # [0, W_erp]
    v_map = (0.5 - lat_map / math.pi) * H_erp          # [0, H_erp]

    u_map = np.clip(u_map, 0, W_erp - 1).astype(np.float32)
    v_map = np.clip(v_map, 0, H_erp - 1).astype(np.float32)

    # Bilinear sampling
    out_arr = _bilinear_sample(erp_arr, u_map, v_map)
    out_img = Image.fromarray(out_arr.astype(np.uint8))

    sample_map = np.stack([u_map, v_map], axis=-1)     # [H_p, W_p, 2]
    return out_img, sample_map


def project_mask_to_perspective(
    erp_mask: np.ndarray,
    sample_map: np.ndarray,
) -> np.ndarray:
    """Project an ERP binary mask to perspective view using the sample map.

    Args:
        erp_mask:   [H_erp, W_erp] uint8 binary mask (0 or 255).
        sample_map: [H_p, W_p, 2] array of (u, v) ERP pixel coords.

    Returns:
        Perspective binary mask [H_p, W_p] uint8.
    """
    u_map = sample_map[..., 0].astype(np.int32)
    v_map = sample_map[..., 1].astype(np.int32)
    H_erp, W_erp = erp_mask.shape[:2]
    u_map = np.clip(u_map, 0, W_erp - 1)
    v_map = np.clip(v_map, 0, H_erp - 1)
    return erp_mask[v_map, u_map]


def reproject_perspective_to_erp(
    erp_base: np.ndarray,
    persp_patch: np.ndarray,
    sample_map: np.ndarray,
    erp_mask: np.ndarray,
    feather_px: int = 8,
) -> np.ndarray:
    """Blend an edited perspective patch back into an ERP image.

    Args:
        erp_base:    [H, W, 3] original ERP image.
        persp_patch: [H_p, W_p, 3] edited perspective patch.
        sample_map:  [H_p, W_p, 2] (u, v) ERP coordinates (from erp_to_perspective).
        erp_mask:    [H, W] binary mask (255 = edited region) in ERP space.
        feather_px:  Gaussian feather radius for blending.

    Returns:
        Blended ERP image [H, W, 3] uint8.
    """
    import cv2

    H_erp, W_erp = erp_base.shape[:2]
    result = erp_base.copy().astype(np.float32)

    # For each perspective pixel, scatter into ERP if inside the mask
    H_p, W_p = sample_map.shape[:2]
    u_map = sample_map[..., 0].astype(np.int32)
    v_map = sample_map[..., 1].astype(np.int32)

    patch_f = persp_patch.astype(np.float32)
    for py in range(H_p):
        for px in range(W_p):
            eu, ev = int(u_map[py, px]), int(v_map[py, px])
            if 0 <= eu < W_erp and 0 <= ev < H_erp and erp_mask[ev, eu] > 0:
                result[ev, eu] = patch_f[py, px]

    # Feather the boundary
    mask_f = erp_mask.astype(np.float32) / 255.0
    if feather_px > 0:
        mask_f = cv2.GaussianBlur(mask_f, (0, 0), feather_px)
    mask_f = mask_f[..., np.newaxis]
    blended = result * mask_f + erp_base.astype(np.float32) * (1.0 - mask_f)

    # Horizontal seam smoothing
    blended = _smooth_horizontal_seam(blended)
    return blended.clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bilinear_sample(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of *img* at fractional pixel coordinates (u, v)."""
    H, W = img.shape[:2]
    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    u1 = np.clip(u0 + 1, 0, W - 1)
    v1 = np.clip(v0 + 1, 0, H - 1)
    u0 = np.clip(u0, 0, W - 1)
    v0 = np.clip(v0, 0, H - 1)

    du = (u - u0.astype(np.float32))[..., np.newaxis]
    dv = (v - v0.astype(np.float32))[..., np.newaxis]

    top    = img[v0, u0] * (1 - du) + img[v0, u1] * du
    bottom = img[v1, u0] * (1 - du) + img[v1, u1] * du
    return top * (1 - dv) + bottom * dv


def _smooth_horizontal_seam(img: np.ndarray) -> np.ndarray:
    """Average the leftmost and rightmost columns to close the ERP seam."""
    result = img.copy()
    avg = (img[:, 0, :].astype(np.float32) + img[:, -1, :].astype(np.float32)) / 2.0
    result[:, 0, :] = avg
    result[:, -1, :] = avg
    return result


# ---------------------------------------------------------------------------
# Mask extraction utilities
# ---------------------------------------------------------------------------

def extract_binary_mask(mask_img: np.ndarray, object_value: int) -> np.ndarray:
    """Extract a binary mask for a specific object value from a Grounded-SAM mask image.

    Args:
        mask_img:     [H, W] or [H, W, C] uint8 semantic ID map (pixel value == object index).
        object_value: The integer ID of the target object (from mask.json "value" field).

    Returns:
        Binary mask [H, W] uint8 with 255 where object is present, 0 elsewhere.
    """
    if mask_img.ndim == 3:
        # The mask may be stored as an RGB image where R == G == B == object_value
        mask_img = mask_img[..., 0]
    return (mask_img == object_value).astype(np.uint8) * 255


def dilate_mask(
    binary_mask: np.ndarray,
    dilation_radius: int,
) -> np.ndarray:
    """Apply morphological dilation to a binary mask.

    Args:
        binary_mask:     [H, W] uint8 binary mask.
        dilation_radius: Radius in pixels.

    Returns:
        Dilated mask [H, W] uint8.
    """
    import cv2

    if dilation_radius <= 0:
        return binary_mask
    kernel_size = 2 * dilation_radius + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    return cv2.dilate(binary_mask, kernel)


def compute_aesg_dilation_radius(
    box: list[float],
    num_affiliation_edges: int,
    d_base_ratio: float = 0.15,
    beta: float = 0.1,
) -> int:
    """Compute AESG-driven semantic dilation radius.

    d = d_base * (1 + beta * |E_aff(v_c)|)

    where d_base = 15% of the object bounding box diagonal.

    Args:
        box:                   [x1, y1, x2, y2] bounding box in pixels.
        num_affiliation_edges: Number of educational affiliation edges for this object.
        d_base_ratio:          Fraction of bounding box diagonal for d_base.
        beta:                  Scaling factor for affiliation degree.

    Returns:
        Dilation radius in pixels (integer).
    """
    w = box[2] - box[0]
    h = box[3] - box[1]
    diagonal = math.sqrt(w**2 + h**2)
    d_base = d_base_ratio * diagonal
    d = d_base * (1.0 + beta * num_affiliation_edges)
    return max(1, int(round(d)))


def compute_affiliation_edges(
    detections: list[dict],
    iou_threshold: float = 0.0,
    gap_threshold: float = 50.0,
) -> dict[int, int]:
    """Count educational affiliation edge candidates for each detection.

    Two objects are considered affiliated if their boxes overlap (IoU > 0)
    or their gap is smaller than gap_threshold pixels.

    Args:
        detections:     List of detection dicts from mask.json (with "value" and "box").
        iou_threshold:  Minimum IoU for affiliation.
        gap_threshold:  Maximum box gap in pixels for affiliation.

    Returns:
        Dict mapping object value -> number of affiliated objects.
    """
    objs = [d for d in detections if d.get("value", 0) != 0]
    counts: dict[int, int] = {o["value"]: 0 for o in objs}

    for i, a in enumerate(objs):
        for j, b in enumerate(objs):
            if i >= j:
                continue
            if _boxes_affiliated(a["box"], b["box"], iou_threshold, gap_threshold):
                counts[a["value"]] = counts.get(a["value"], 0) + 1
                counts[b["value"]] = counts.get(b["value"], 0) + 1
    return counts


def _boxes_affiliated(
    box_a: list[float],
    box_b: list[float],
    iou_threshold: float,
    gap_threshold: float,
) -> bool:
    iou = _box_iou(box_a, box_b)
    if iou > iou_threshold:
        return True
    gap = _box_gap(box_a, box_b)
    return gap < gap_threshold


def _box_iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def _box_gap(a: list[float], b: list[float]) -> float:
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return math.sqrt(dx**2 + dy**2)
