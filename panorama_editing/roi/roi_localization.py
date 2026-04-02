from __future__ import annotations

import math
from typing import Any

from PIL import Image, ImageDraw

from aesg.schema import AESGGraph


def localize_and_project_roi(
    image_erp: Image.Image,
    aesg_graph: AESGGraph,
    roi_hint: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    image = image_erp.convert("RGB")
    width, height = image.size
    config = config or {}
    roi_width_ratio = float(config.get("roi_width_ratio", 0.33))
    roi_height_ratio = float(config.get("roi_height_ratio", 0.45))
    crop_w = max(64, min(width, int(width * roi_width_ratio)))
    crop_h = max(64, min(height, int(height * roi_height_ratio)))

    theta = 0.0
    phi = 0.0
    fov = float(config.get("default_fov", 75.0))
    if roi_hint:
        theta = float(roi_hint.get("theta", theta))
        phi = float(roi_hint.get("phi", phi))
        fov = float(roi_hint.get("fov", fov))
    else:
        theta, phi = _infer_angles_from_graph(aesg_graph)
    confidence = 0.2
    if roi_hint:
        confidence = 1.0
    elif _has_explicit_localization(aesg_graph):
        confidence = 0.75

    center_x = int(((theta + math.pi) / (2 * math.pi)) * width) % width
    center_y = int(((math.pi / 2 - phi) / math.pi) * height)
    center_y = max(crop_h // 2, min(height - crop_h // 2, center_y))
    left = max(0, min(width - crop_w, center_x - crop_w // 2))
    top = max(0, min(height - crop_h, center_y - crop_h // 2))
    bbox = (left, top, left + crop_w, top + crop_h)

    local_image = image.crop(bbox)
    mask = Image.new("L", (crop_w, crop_h), 0)
    draw = ImageDraw.Draw(mask)
    margin_x = max(8, crop_w // 8)
    margin_y = max(8, crop_h // 8)
    draw.rounded_rectangle(
        (margin_x, margin_y, crop_w - margin_x, crop_h - margin_y),
        radius=min(crop_w, crop_h) // 10,
        fill=255,
    )

    return {
        "local_image": local_image,
        "mask": mask,
        "bbox": bbox,
        "theta": theta,
        "phi": phi,
        "fov": fov,
        "projection_meta": {
            "center_x": center_x,
            "center_y": center_y,
            "erp_size": (width, height),
            "strategy": "heuristic_crop",
            "confidence": confidence,
        },
    }


def _infer_angles_from_graph(graph: AESGGraph) -> tuple[float, float]:
    relation_text = " ".join(rel.relation_type for rel in graph.relations).lower()
    object_text = " ".join(obj.name for obj in graph.core_objects).lower()
    text = f"{relation_text} {object_text}"
    theta = 0.0
    phi = 0.0
    if "left" in text:
        theta = -math.pi / 2
    elif "right" in text:
        theta = math.pi / 2
    elif "back" in text:
        theta = math.pi
    if "up" in text or "top" in text:
        phi = math.pi / 8
    elif "down" in text or "ground" in text or "floor" in text:
        phi = -math.pi / 8
    return theta, phi


def _has_explicit_localization(graph: AESGGraph) -> bool:
    text = " ".join(
        [rel.relation_type for rel in graph.relations]
        + [rel.direction for rel in graph.relations]
        + [obj.action for obj in graph.core_objects]
    ).lower()
    tokens = (
        "left",
        "right",
        "top",
        "bottom",
        "center",
        "front",
        "back",
        "ground",
        "floor",
        "ceiling",
        "near",
        "beside",
    )
    return any(token in text for token in tokens)
