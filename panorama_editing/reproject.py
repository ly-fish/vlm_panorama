from __future__ import annotations

from typing import Any

from PIL import Image, ImageFilter


def reproject_to_erp(
    image_erp: Image.Image,
    edited_local: Image.Image,
    roi_result: dict[str, Any],
    feather_radius: int = 8,
) -> Image.Image:
    base = image_erp.convert("RGB").copy()
    patch = edited_local.convert("RGB")
    bbox = tuple(roi_result["bbox"])
    mask = roi_result["mask"].convert("L")
    if feather_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_radius))
    region = base.crop(bbox)
    if patch.size != region.size:
        patch = patch.resize(region.size, Image.Resampling.BICUBIC)
    if mask.size != region.size:
        mask = mask.resize(region.size, Image.Resampling.BICUBIC)
    blended = Image.composite(patch, region, mask)
    base.paste(blended, bbox)
    return _smooth_horizontal_seam(base)


def _smooth_horizontal_seam(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width < 4:
        return image
    pixels = image.load()
    for y in range(height):
        left = pixels[0, y]
        right = pixels[width - 1, y]
        avg = tuple(int((left[idx] + right[idx]) / 2) for idx in range(3))
        pixels[0, y] = avg
        pixels[width - 1, y] = avg
    return image
