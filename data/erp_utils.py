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

    # Use modular wrapping for longitude (handles objects crossing the ±180° seam)
    u_map = (u_map % W_erp).astype(np.float32)
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
    """Project an ERP binary mask to perspective view using bilinear interpolation.

    Bilinear interpolation avoids the ragged boundaries that nearest-neighbour
    produces at high latitudes where ERP pixels are strongly stretched.  The
    result is thresholded at 127 to keep the output binary.

    Args:
        erp_mask:   [H_erp, W_erp] uint8 binary mask (0 or 255).
        sample_map: [H_p, W_p, 2] array of (u, v) ERP pixel coords.

    Returns:
        Perspective binary mask [H_p, W_p] uint8.
    """
    H_erp, W_erp = erp_mask.shape[:2]
    mask_f = erp_mask.astype(np.float32)

    u = sample_map[..., 0]   # fractional ERP x
    v = sample_map[..., 1]   # fractional ERP y

    # Bilinear neighbours (with modular wrapping on u for the seam)
    u0 = np.floor(u).astype(np.int32) % W_erp
    u1 = (u0 + 1) % W_erp
    v0 = np.clip(np.floor(v).astype(np.int32), 0, H_erp - 1)
    v1 = np.clip(v0 + 1, 0, H_erp - 1)

    du = (u - np.floor(u))[..., np.newaxis]   # [H_p, W_p, 1]
    dv = (v - np.floor(v))[..., np.newaxis]

    top    = mask_f[v0, u0, np.newaxis] * (1 - du) + mask_f[v0, u1, np.newaxis] * du
    bottom = mask_f[v1, u0, np.newaxis] * (1 - du) + mask_f[v1, u1, np.newaxis] * du
    interp = (top * (1 - dv) + bottom * dv)[..., 0]   # [H_p, W_p]

    return (interp >= 127).astype(np.uint8) * 255


def _compute_perspective_grid(
    lat_deg: float,
    lon_deg: float,
    fov_deg: float,
    persp_w: int,
    persp_h: int,
    erp_h: int,
    erp_w: int,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Compute per-ERP-pixel perspective-image coordinates (inverse of erp_to_perspective).

    For each ERP pixel (u, v) this function returns the perspective-patch pixel
    coordinates (map_x, map_y) that the ERP pixel corresponds to.  This is the
    analytical inverse of the forward projection used in ``erp_to_perspective``
    and enables hole-free ``cv2.remap``-based reprojection.

    The forward projection is::

        world = R_y(lon_c) @ R_x(lat_c) @ camera_ray

    The inverse is therefore::

        camera_ray = R_x(lat_c)^T @ R_y(lon_c)^T @ world

    Args:
        lat_deg, lon_deg: Viewing-direction centre in degrees.
        fov_deg:          Horizontal field-of-view in degrees.
        persp_w, persp_h: Perspective patch size in pixels.
        erp_h, erp_w:     ERP image size in pixels.

    Returns:
        map_x: [H_erp, W_erp] float32 — perspective x-coord for each ERP pixel.
        map_y: [H_erp, W_erp] float32 — perspective y-coord for each ERP pixel.
        valid: [H_erp, W_erp] bool    — True where the ERP pixel is inside the
               camera frustum and within the perspective image bounds.
    """
    lat_c = math.radians(lat_deg)
    lon_c = math.radians(lon_deg)
    fov_r = math.radians(fov_deg)
    f = (persp_w / 2.0) / math.tan(fov_r / 2.0)

    # Full ERP pixel grid
    u = np.arange(erp_w, dtype=np.float64)
    v = np.arange(erp_h, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)   # [H_erp, W_erp]

    # ERP pixel → spherical angles (same formula as erp_to_perspective)
    lon_map = (uu / erp_w - 0.5) * (2.0 * math.pi)   # [-π, π]
    lat_map = (0.5 - vv / erp_h) * math.pi             # [-π/2, π/2]

    # Spherical → world unit vectors
    cos_lat_m = np.cos(lat_map)
    xr = cos_lat_m * np.sin(lon_map)
    yr = np.sin(lat_map)
    zr = cos_lat_m * np.cos(lon_map)

    # Inverse rotation: camera = R_x(lat_c)^T @ R_y(lon_c)^T @ world
    #
    # R_y(lon_c) = [[cos_lon,0,sin_lon],[0,1,0],[-sin_lon,0,cos_lon]]
    # R_y(lon_c)^T= [[cos_lon,0,-sin_lon],[0,1,0],[sin_lon,0,cos_lon]]
    cos_lon, sin_lon = math.cos(lon_c), math.sin(lon_c)
    x_p =  xr * cos_lon - zr * sin_lon
    y_p =  yr
    z_p =  xr * sin_lon + zr * cos_lon

    # R_x(lat_c) = [[1,0,0],[0,cos_lat,-sin_lat],[0,sin_lat,cos_lat]]
    # R_x(lat_c)^T= [[1,0,0],[0,cos_lat,sin_lat],[0,-sin_lat,cos_lat]]
    cos_lat, sin_lat = math.cos(lat_c), math.sin(lat_c)
    xv =  x_p
    yv =  y_p * cos_lat + z_p * sin_lat
    zv = -y_p * sin_lat + z_p * cos_lat

    # Only pixels in front of the camera (zv > 0) are visible
    valid = zv > 1e-6

    # Project to perspective pixel coordinates
    # erp_to_perspective: xs = (j - W_p/2 + 0.5) / f  →  j = xs*f + W_p/2 - 0.5
    safe_zv = np.where(valid, zv, 1.0)   # avoid division by zero
    map_x = (xv / safe_zv * f + persp_w / 2.0 - 0.5).astype(np.float32)
    map_y = (yv / safe_zv * f + persp_h / 2.0 - 0.5).astype(np.float32)

    # Discard pixels projecting outside the perspective image boundaries
    in_bounds = (
        (map_x >= 0) & (map_x < persp_w - 1) &
        (map_y >= 0) & (map_y < persp_h - 1)
    )
    valid = valid & in_bounds

    # cv2.remap fills pixels with borderValue when map coords are out-of-bounds;
    # set invalid entries to -1 to trigger that fill path.
    map_x = np.where(valid, map_x, np.float32(-1.0))
    map_y = np.where(valid, map_y, np.float32(-1.0))

    return map_x, map_y, valid


def reproject_perspective_to_erp(
    erp_base: np.ndarray,
    persp_patch: np.ndarray,
    sample_map: np.ndarray,
    erp_mask: np.ndarray,
    feather_px: int = 8,
    proj_params: "Any | None" = None,
    persp_mask: "np.ndarray | None" = None,
) -> np.ndarray:
    """Blend an edited perspective patch back into an ERP image.

    Two code paths are available:

    * **Inverse-mapping path** (preferred, activated when *proj_params* is
      supplied): computes the analytical inverse of ``erp_to_perspective`` for
      every ERP pixel and samples the edited patch with ``cv2.remap``.
      Geometrically exact — no scatter holes at high latitudes.

    * **Forward-scatter fallback** (used when *proj_params* is ``None``): the
      original implementation that scatters perspective pixels back to ERP
      positions.  Kept for backward compatibility.

    Args:
        erp_base:    [H, W, 3] original ERP image.
        persp_patch: [H_p, W_p, 3] edited perspective patch.
        sample_map:  [H_p, W_p, 2] (u, v) ERP coordinates (from erp_to_perspective).
        erp_mask:    [H, W] binary mask (255 = edited region) in ERP space.
                     Used only by the forward-scatter fallback.
        feather_px:  Gaussian feather radius for boundary blending.
        proj_params: Optional object with ``.lat``, ``.lon``, ``.fov`` attributes
                     (a :class:`~lora.distortion_encoder.ProjectionParams` instance).
                     Activates the hole-free inverse-mapping path when provided.
        persp_mask:  [H_p, W_p] uint8 binary mask of the object region in
                     perspective space.  Used by the inverse path to build a
                     hole-free ERP blend mask.  Falls back to the full camera
                     frustum when ``None``.

    Returns:
        Blended ERP image [H, W, 3] uint8.
    """
    import cv2

    H_erp, W_erp = erp_base.shape[:2]
    H_p, W_p = persp_patch.shape[:2]

    if proj_params is not None:
        # ------------------------------------------------------------------
        # Inverse-mapping path: analytically project ERP pixels into the
        # perspective patch and sample with cv2.remap.  No scatter holes.
        # ------------------------------------------------------------------
        map_x, map_y, visible = _compute_perspective_grid(
            proj_params.lat, proj_params.lon, proj_params.fov,
            W_p, H_p, H_erp, W_erp,
        )

        # Remap the edited patch content into ERP space
        reprojected = cv2.remap(
            persp_patch.astype(np.float32),
            map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )   # [H_erp, W_erp, 3]

        # Build a hole-free ERP blend mask by remapping the perspective mask
        if persp_mask is not None:
            mask_remapped = cv2.remap(
                persp_mask.astype(np.float32),
                map_x, map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )   # [H_erp, W_erp]
            blend_mask = ((mask_remapped > 127) & visible).astype(np.float32)
        else:
            blend_mask = visible.astype(np.float32)

        # Feather the boundary and composite
        if feather_px > 0:
            blend_mask = cv2.GaussianBlur(blend_mask, (0, 0), feather_px)
        blend_mask_3d = blend_mask[..., np.newaxis]
        result = erp_base.copy().astype(np.float32)
        blended = reprojected * blend_mask_3d + result * (1.0 - blend_mask_3d)
        blended = _smooth_horizontal_seam(blended)
        return blended.clip(0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Forward-scatter fallback (original implementation)
    # ------------------------------------------------------------------
    result = erp_base.copy().astype(np.float32)

    u_idx = np.clip(sample_map[..., 0].astype(np.int32), 0, W_erp - 1)  # [H_p,W_p]
    v_idx = np.clip(sample_map[..., 1].astype(np.int32), 0, H_erp - 1)

    patch_f = persp_patch.astype(np.float32)

    # Only scatter pixels covered by erp_mask (avoids overwriting background)
    covered = erp_mask[v_idx, u_idx] > 0                    # [H_p, W_p] bool
    flat_v  = v_idx[covered]
    flat_u  = u_idx[covered]
    result[flat_v, flat_u] = patch_f.reshape(-1, 3)[covered.ravel()]

    # Feather the boundary using the ERP-space mask
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
    dilation_radius: int | tuple[int, int],
) -> np.ndarray:
    """Apply morphological dilation to a binary mask.

    Args:
        binary_mask:      [H, W] uint8 binary mask.
        dilation_radius:  Radius in pixels.  Pass a (h_radius, v_radius) tuple
                          for anisotropic dilation (e.g. latitude-compensated).

    Returns:
        Dilated mask [H, W] uint8.
    """
    import cv2

    if isinstance(dilation_radius, (tuple, list)):
        h_radius, v_radius = int(dilation_radius[0]), int(dilation_radius[1])
    else:
        h_radius = v_radius = int(dilation_radius)

    if h_radius <= 0 and v_radius <= 0:
        return binary_mask

    h_radius = max(h_radius, 1)
    v_radius = max(v_radius, 1)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * h_radius + 1, 2 * v_radius + 1)
    )
    return cv2.dilate(binary_mask, kernel)


def compute_aesg_dilation_radius(
    box: list[float],
    num_affiliation_edges: int,
    lat_deg: float = 0.0,
    d_base_ratio: float = 0.15,
    beta: float = 0.1,
) -> tuple[int, int]:
    """Compute AESG-driven semantic dilation radii with latitude compensation.

    The horizontal radius is scaled by 1/cos(lat) to account for ERP
    horizontal stretching near the poles: a physically circular region
    occupies progressively more ERP pixels horizontally as latitude increases.

    d = d_base * (1 + beta * |E_aff(v_c)|)
    h_radius = d * (1 / cos(lat))   # compensate ERP horizontal stretch
    v_radius = d                     # no vertical distortion in ERP

    Args:
        box:                   [x1, y1, x2, y2] bounding box in ERP pixels.
        num_affiliation_edges: Number of educational affiliation edges for this object.
        lat_deg:               Latitude of the object centre in degrees [-90, 90].
        d_base_ratio:          Fraction of bounding box diagonal for d_base.
        beta:                  Scaling factor for affiliation degree.

    Returns:
        (h_radius, v_radius) dilation radii in pixels.
    """
    w = box[2] - box[0]
    h = box[3] - box[1]
    diagonal = math.sqrt(w**2 + h**2)
    d_base = d_base_ratio * diagonal
    d = d_base * (1.0 + beta * num_affiliation_edges)

    # Latitude compensation: ERP horizontal stretch = 1 / cos(lat)
    lat_rad = math.radians(lat_deg)
    h_stretch = 1.0 / max(math.cos(lat_rad), 0.1)   # cap at 10× to avoid extreme polar values

    h_radius = max(1, int(round(d * h_stretch)))
    v_radius = max(1, int(round(d)))
    return h_radius, v_radius


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
