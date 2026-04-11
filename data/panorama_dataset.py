"""Self-supervised panoramic reconstruction dataset.

Each training sample is a tuple:
    (I_p_deg, I_p_GT, theta_tensor, M_perspective, meta)

where
  I_p_deg      -- degraded perspective view (tensor [3, H_p, W_p])
  I_p_GT       -- original perspective view (tensor [3, H_p, W_p])
  theta_tensor -- [3] normalised projection params (lat_n, lon_n, fov_n)
  M_perspective-- perspective binary mask (tensor [1, H_p, W_p], float32)
  meta         -- dict with scene info, raw detections, etc.

The dataset also exposes ``aesg_condition`` (Stage 2) via the meta dict
when ``build_aesg=True``.

Directory layout expected:
  data_root/
    scene_000/
      panorama.jpg
      result_*_mask.jpg
      result_*_mask.json
      scene_*_instruction.txt    (optional, used in Stage 2)
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.erp_utils import (
    erp_to_perspective,
    extract_binary_mask,
    dilate_mask,
    project_mask_to_perspective,
    compute_aesg_dilation_radius,
    compute_affiliation_edges,
)
from lora.distortion_encoder import ProjectionParams


# ---------------------------------------------------------------------------
# Degradation strategies
# ---------------------------------------------------------------------------

def _degrade_gray(img_arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace masked region with neutral grey (128, 128, 128)."""
    out = img_arr.copy()
    out[mask > 0] = [128, 128, 128]
    return out


def _degrade_noise(img_arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace masked region with Gaussian noise."""
    out = img_arr.copy()
    noise = np.random.normal(128, 40, img_arr.shape).clip(0, 255).astype(np.uint8)
    out[mask > 0] = noise[mask > 0]
    return out


def _degrade_blur(img_arr: np.ndarray, mask: np.ndarray, sigma: float = 40.0) -> np.ndarray:
    """Apply heavy Gaussian blur to the masked region only."""
    import cv2

    out = img_arr.copy()
    k = int(sigma * 3) | 1  # ensure odd kernel
    blurred = cv2.GaussianBlur(img_arr, (k, k), sigma)
    out[mask > 0] = blurred[mask > 0]
    return out


_DEGRADE_FNS = [_degrade_gray, _degrade_noise, _degrade_blur]


def degrade_image(
    img_arr: np.ndarray,
    mask: np.ndarray,
    strategy: str = "random",
) -> np.ndarray:
    """Apply a degradation strategy to the masked region.

    Args:
        img_arr:  [H, W, 3] uint8 image.
        mask:     [H, W] uint8 binary mask (255 = masked region).
        strategy: "gray" | "noise" | "blur" | "random".
    """
    if strategy == "gray":
        return _degrade_gray(img_arr, mask)
    if strategy == "noise":
        return _degrade_noise(img_arr, mask)
    if strategy == "blur":
        return _degrade_blur(img_arr, mask)
    return random.choice(_DEGRADE_FNS)(img_arr, mask)


# ---------------------------------------------------------------------------
# Image -> tensor helpers
# ---------------------------------------------------------------------------

def _img_to_tensor(img: np.ndarray) -> torch.Tensor:
    """[H, W, 3] uint8 -> [3, H, W] float32 in [-1, 1]."""
    t = torch.from_numpy(img).float().permute(2, 0, 1) / 127.5 - 1.0
    return t


def _mask_to_tensor(mask: np.ndarray) -> torch.Tensor:
    """[H, W] uint8 -> [1, H, W] float32 in [0, 1]."""
    return torch.from_numpy(mask).float().unsqueeze(0) / 255.0


# ---------------------------------------------------------------------------
# Scene loader
# ---------------------------------------------------------------------------

def _load_scene(scene_dir: Path) -> dict[str, Any] | None:
    """Load all files for a single scene directory."""
    panorama_path = scene_dir / "panorama.jpg"
    if not panorama_path.exists():
        return None

    # Find mask files (one pair of .jpg + .json per scene)
    mask_jsons = sorted(scene_dir.glob("*_mask.json"))
    mask_jpgs  = sorted(scene_dir.glob("*_mask.jpg"))
    if not mask_jsons or not mask_jpgs:
        return None

    mask_json_path = mask_jsons[0]
    mask_jpg_path  = mask_jpgs[0]

    with open(mask_json_path, "r", encoding="utf-8") as f:
        detections = json.load(f)

    # Optional instruction text
    instruction_paths = sorted(scene_dir.glob("*_instruction.txt"))
    instruction = ""
    if instruction_paths:
        instruction = instruction_paths[0].read_text(encoding="utf-8").strip()

    return {
        "scene_dir": str(scene_dir),
        "panorama_path": str(panorama_path),
        "mask_jpg_path": str(mask_jpg_path),
        "detections": detections,
        "instruction": instruction,
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PanoramaDataset(Dataset):
    """Self-supervised panoramic reconstruction dataset.

    Args:
        data_root:           Root directory containing scene_XXX/ subdirectories.
        split:               "train" | "val" | "test"  (sub-directory name).
        perspective_size:    (H_p, W_p) of extracted perspective patches.
        fov_range:           (fov_min, fov_max) in degrees.
        fov_steps:           Number of FoV values to sample per scene-object pair.
        lat_perturb:         Maximum latitude perturbation in degrees.
        lon_perturb:         Maximum longitude perturbation in degrees.
        num_views_per_obj:   How many perspective views per target object (3-5 per doc).
        logit_threshold:     Minimum detection confidence to keep an object.
        degrade_strategy:    "random" | "gray" | "noise" | "blur".
        dilation_beta:       AESG-driven dilation scale factor beta.
        mask_perturb_frac:   Fraction by which to randomly vary dilation radius (±).
        build_aesg:          Whether to build AESG conditions (Stage 2).
        max_scenes:          Cap on number of scenes (for debug / scaling ablation).
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        perspective_size: tuple[int, int] = (512, 512),
        fov_range: tuple[float, float] = (60.0, 120.0),
        fov_steps: int = 4,
        lat_perturb: float = 5.0,
        lon_perturb: float = 5.0,
        num_views_per_obj: int = 4,
        logit_threshold: float = 0.5,
        degrade_strategy: str = "random",
        dilation_beta: float = 0.1,
        mask_perturb_frac: float = 0.1,
        build_aesg: bool = False,
        max_scenes: int | None = None,
    ) -> None:
        super().__init__()
        self.perspective_size  = perspective_size
        self.fov_range         = fov_range
        self.fov_steps         = fov_steps
        self.lat_perturb       = lat_perturb
        self.lon_perturb       = lon_perturb
        self.num_views_per_obj = num_views_per_obj
        self.logit_threshold   = logit_threshold
        self.degrade_strategy  = degrade_strategy
        self.dilation_beta     = dilation_beta
        self.mask_perturb_frac = mask_perturb_frac
        self.build_aesg        = build_aesg

        root = Path(data_root) / split
        if not root.exists():
            raise FileNotFoundError(f"Dataset split not found: {root}")

        scenes = sorted(root.iterdir())
        if max_scenes is not None:
            scenes = scenes[:max_scenes]

        # Build flat list of (scene_meta, target_det, fov, lat_delta, lon_delta)
        self.samples: list[dict[str, Any]] = []
        valid_dirs = [s for s in scenes if s.is_dir()]
        for i, scene_dir in enumerate(valid_dirs):
            scene = _load_scene(scene_dir)
            if scene is None:
                continue
            self._expand_scene(scene)
            if (i + 1) % 20 == 0 or (i + 1) == len(valid_dirs):
                print(f"  [Dataset/{split}] {i + 1}/{len(valid_dirs)} scenes loaded, "
                      f"{len(self.samples)} samples so far", flush=True)

    def _expand_scene(self, scene: dict[str, Any]) -> None:
        """Enumerate all (object, projection) samples for a scene."""
        detections = scene["detections"]
        valid_objs = [
            d for d in detections
            if d.get("value", 0) != 0
            and d.get("logit", 0.0) >= self.logit_threshold
            and "box" in d
        ]
        if not valid_objs:
            return

        panorama = Image.open(scene["panorama_path"])
        W_erp, H_erp = panorama.size

        affiliation_counts = compute_affiliation_edges(detections)

        for det in valid_objs:
            box = det["box"]
            num_aff = affiliation_counts.get(det["value"], 0)

            # Compute projection centre from box
            base_params = ProjectionParams.from_box(box, W_erp, H_erp, fov=90.0)

            # FoV sweep
            fov_values = np.linspace(
                self.fov_range[0], self.fov_range[1], self.fov_steps
            ).tolist()

            view_count = 0
            for fov in fov_values:
                if view_count >= self.num_views_per_obj:
                    break
                lat_d = random.uniform(-self.lat_perturb, self.lat_perturb)
                lon_d = random.uniform(-self.lon_perturb, self.lon_perturb)
                self.samples.append({
                    "scene": scene,
                    "det": det,
                    "num_affiliation_edges": num_aff,
                    "W_erp": W_erp,
                    "H_erp": H_erp,
                    "lat": base_params.lat + lat_d,
                    "lon": base_params.lon + lon_d,
                    "fov": fov,
                })
                view_count += 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        scene  = sample["scene"]

        # Load images
        pano_pil   = Image.open(scene["panorama_path"]).convert("RGB")
        panorama   = np.array(pano_pil)
        # Resize mask to match panorama dimensions (they may differ across scenes)
        mask_pil   = Image.open(scene["mask_jpg_path"]).convert("L")
        if mask_pil.size != pano_pil.size:
            mask_pil = mask_pil.resize(pano_pil.size, Image.Resampling.NEAREST)
        mask_img   = np.array(mask_pil)

        det    = sample["det"]
        W_erp  = sample["W_erp"]
        H_erp  = sample["H_erp"]
        lat    = float(np.clip(sample["lat"], -85.0, 85.0))
        lon    = sample["lon"]
        fov    = sample["fov"]

        # ------------------------------------------------------------------
        # Step 1: Extract binary mask and apply AESG-driven dilation
        # ------------------------------------------------------------------
        num_aff = sample["num_affiliation_edges"]
        h_rad, v_rad = compute_aesg_dilation_radius(
            det["box"], num_aff,
            lat_deg=lat,
            d_base_ratio=0.15,
            beta=self.dilation_beta,
        )
        # Mask perturbation for augmentation (applied independently per axis)
        perturb_h = random.uniform(1.0 - self.mask_perturb_frac, 1.0 + self.mask_perturb_frac)
        perturb_v = random.uniform(1.0 - self.mask_perturb_frac, 1.0 + self.mask_perturb_frac)
        h_rad = max(1, int(round(h_rad * perturb_h)))
        v_rad = max(1, int(round(v_rad * perturb_v)))

        bin_mask_erp = extract_binary_mask(mask_img, det["value"])
        dilated_mask = dilate_mask(bin_mask_erp, (h_rad, v_rad))  # [H, W] uint8

        # ------------------------------------------------------------------
        # Step 2: Degrade the ERP image in the mask region
        # ------------------------------------------------------------------
        degraded_erp = degrade_image(panorama, dilated_mask, strategy=self.degrade_strategy)

        # ------------------------------------------------------------------
        # Step 3: Project both GT and degraded ERP to perspective view
        # ------------------------------------------------------------------
        H_p, W_p = self.perspective_size

        gt_persp, sample_map = erp_to_perspective(
            Image.fromarray(panorama), lat, lon, fov, out_w=W_p, out_h=H_p
        )
        deg_persp, _ = erp_to_perspective(
            Image.fromarray(degraded_erp), lat, lon, fov, out_w=W_p, out_h=H_p
        )

        # ------------------------------------------------------------------
        # Step 4: Project the binary mask to perspective space
        # ------------------------------------------------------------------
        mask_persp = project_mask_to_perspective(dilated_mask, sample_map)  # [H_p, W_p]

        # ------------------------------------------------------------------
        # Step 5: Convert to tensors
        # ------------------------------------------------------------------
        I_p_GT  = _img_to_tensor(np.array(gt_persp))    # [3, H_p, W_p]
        I_p_deg = _img_to_tensor(np.array(deg_persp))   # [3, H_p, W_p]
        M_persp = _mask_to_tensor(mask_persp)            # [1, H_p, W_p]

        # Projection params tensor [3]
        theta_tensor = torch.tensor(
            [lat / 90.0, lon / 180.0, (fov - 90.0) / 90.0],
            dtype=torch.float32,
        )

        # ------------------------------------------------------------------
        # Meta / AESG (Stage 2)
        # ------------------------------------------------------------------
        meta: dict[str, Any] = {
            "scene_dir": scene["scene_dir"],
            "det_value": det["value"],
            "det_label": det.get("label", ""),
            "instruction": scene.get("instruction", ""),
            "lat": lat,
            "lon": lon,
            "fov": fov,
            "erp_W": W_erp,
            "erp_H": H_erp,
            "box": det["box"],
        }

        if self.build_aesg:
            meta["aesg_condition"] = self._build_aesg_condition(scene, det)

        return {
            "I_p_deg":    I_p_deg,                           # [3, H_p, W_p]
            "I_p_GT":     I_p_GT,                            # [3, H_p, W_p]
            "theta":      theta_tensor,                      # [3]
            "mask":       M_persp,                           # [1, H_p, W_p]
            "sample_map": torch.from_numpy(sample_map),      # [H_p, W_p, 2]  (u, v) in ERP pixels
            "meta":       meta,
        }

    # ------------------------------------------------------------------
    # AESG condition builder (lightweight version for training)
    # ------------------------------------------------------------------
    def _build_aesg_condition(
        self, scene: dict[str, Any], target_det: dict[str, Any]
    ) -> dict[str, Any]:
        """Build a minimal AESG condition dict compatible with AESGEncoder output format."""
        try:
            from aesg.schema import build_aesg_graph
            from aesg.encoder import AESGEncoder

            text = scene.get("instruction", target_det.get("label", "panorama scene"))
            graph = build_aesg_graph(text=text)
            encoder = AESGEncoder(hidden_size=3584)
            cond = encoder(graph)
            return cond.to_dict()
        except Exception:
            # Fallback: zero tensors compatible with AESGConditionAggregator
            H = 3584
            return {
                "anchor_tokens":   torch.zeros(1, 2, H),
                "object_tokens":   torch.zeros(1, 8, H),
                "context_tokens":  torch.zeros(1, 8, H),
                "relation_tokens": torch.zeros(1, 12, H),
            }


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Custom collate that stacks tensors and aggregates meta lists."""
    keys_tensor = ["I_p_deg", "I_p_GT", "theta", "mask", "sample_map"]
    out: dict[str, Any] = {k: torch.stack([b[k] for b in batch]) for k in keys_tensor}
    out["meta"] = [b["meta"] for b in batch]
    return out
