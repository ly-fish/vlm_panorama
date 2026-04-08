# Distortion-aware Dual-LoRA Fusion for Panorama-consistent Scene Editing

A parameter-efficient adaptation strategy for editing 360° panoramic images (ERP format) using two complementary LoRA branches injected into the cross-attention layers of **Qwen-Image-Edit-2511**:

- **LoRA-Pano** — encodes panoramic geometric priors (latitude-dependent distortion, FoV, projection centre) so edited content is geometrically consistent with ERP space.
- **LoRA-AESG** — encodes structural educational semantics from the Anchor-centred Educational Scene Graph (AESG) so the model respects hierarchical scene constraints.

An **adaptive gating network** dynamically fuses both branches based on task type and spatial position. The entire backbone remains frozen; only the LoRA parameters (~0.2–0.6 % of the model) are trained via a **self-supervised reconstruction** paradigm.

---

## Repository structure

```
vlm_panorama/
├── lora/                          # Core Dual-LoRA modules
│   ├── distortion_encoder.py      # Sinusoidal MLP: θ=(lat,lon,FoV) → z_θ ∈ ℝ⁵¹²
│   ├── film.py                    # FiLM layer: (1+γ(z))·h + β(z)
│   ├── lora_layer.py              # ConditionalLoRALayer — FiLM-modulated A matrix
│   ├── lora_pano.py               # LoRA-Pano branch: θ → z_θ → ΔW_pano
│   ├── lora_aesg.py               # LoRA-AESG branch + AESGConditionAggregator
│   ├── gating.py                  # AdaptiveGatingNetwork: [γ_p, γ_s] = σ(MLP([z_θ; z_G; e_task]))
│   └── dual_lora_fusion.py        # DualLoRAFusion layer, DualLoRAModel, save/load helpers
│
├── data/                          # Dataset pipeline
│   ├── erp_utils.py               # ERP ↔ perspective projection, mask utilities, AESG dilation
│   └── panorama_dataset.py        # Self-supervised dataset: mask → degrade → perspective
│
├── training/
│   ├── losses.py                  # MSE + VGG perceptual + SSIM; L_pano, L_rel, L_aff, L_ctx, L_seam
│   ├── train_stage1.py            # Stage 1: LoRA-Pano + distortion encoder
│   ├── train_stage2.py            # Stage 2: joint LoRA-AESG + gating (10× lower LR for pano)
│   └── losses_aesg.py             # Legacy AESG loss helper
│
├── inference/
│   └── edit_with_lora.py          # PanoramaEditorWithLoRA — 8-step end-to-end pipeline + CLI
│
├── aesg/                          # AESG graph schema and encoder
├── modules/                       # HCFM cross-attention fusion module
├── panorama_editing/              # Qwen pipeline wrapper and ERP reprojection
├── scene_graph/                   # Scene graph parser (Qwen LLM)
├── configs/
│   └── aesg_edit.yaml             # Branch flags, loss weights, ROI config
└── data/
    ├── train/  (244 scenes)
    └── test/   (31 scenes)
```

---

## Method overview

### Adapted weight formula

```
W' = W + γ_p · ΔW_pano(θ) + γ_s · ΔW_aesg(Z_G)
```

| Component | Condition input | Learning target |
|-----------|----------------|-----------------|
| LoRA-Pano | Projection θ | ERP distortion patterns, latitude geometry |
| LoRA-AESG | AESG tokens Z_G | Structural semantic control |
| Adaptive gating | θ + Z_G + task | Dynamic branch balancing |
| Backbone W | (frozen) | Base editing capability |

### Self-supervised training paradigm

No paired before/after editing data is required. For each scene:

1. Grounded-SAM detects objects → `*_mask.jpg` + `*_mask.json`.
2. A target object region is masked out (grey fill / Gaussian noise / heavy blur).
3. The model learns to reconstruct the original from the degraded input.
4. Because GT is a real panoramic image, the model inherently learns panorama-consistent generation.

### Two-stage training

| Stage | Trains | Loss |
|-------|--------|------|
| 1 | LoRA-Pano + distortion encoder | L_recon + λ_pano · L_pano |
| 2 | LoRA-AESG + gating (LoRA-Pano at 10× lower LR) | λ₁L_recon + λ₂L_rel + λ₃L_aff + λ₄L_ctx + λ₅L_seam + λ₆L_pano |

---

## Data format

Each scene directory contains:

```
scene_000/
├── panorama.jpg              # 2048 × 1024 ERP panorama (ground truth)
├── result_*_mask.jpg         # Semantic ID map (pixel value = object index)
├── result_*_mask.json        # Detections: [{value, label, logit, box [x1,y1,x2,y2]}, ...]
└── scene_*_instruction.txt   # Natural language editing instruction
```

---

## Installation

```bash
conda activate panorama
# Dependencies assumed present in the panorama conda environment:
# torch, torchvision, Pillow, numpy, opencv-python, pyyaml
```

---

## Training

### Stage 1 — Panoramic geometric adaptation (LoRA-Pano only)

Trains the distortion encoder and LoRA-Pano adapters via self-supervised reconstruction.

```bash
conda run -n panorama python -m training.train_stage1 \
    --data_root /users/2522553y/liangyue_ws/vlm_panorama/data \
    --output_dir ./checkpoints/stage1 \
    --epochs 30 \
    --batch_size 4
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root` | — | Root of the data directory (must contain `train/` and `test/`) |
| `--output_dir` | `./checkpoints/stage1` | Where checkpoints and `history.json` are saved |
| `--backbone_path` | *(empty)* | Path to Qwen-Image-Edit-2511; omit to use the lightweight stub |
| `--epochs` | 30 | Number of training epochs |
| `--batch_size` | 4 | Batch size |
| `--lr` | 1e-4 | Base learning rate (cosine decay) |
| `--rank` | 8 | LoRA rank r |
| `--lambda_pano` | 0.3 | Weight for reprojection consistency loss |
| `--img_size` | 512 | Perspective patch size (H = W) |
| `--max_scenes` | *(all)* | Cap scenes for ablation / debug |

### Stage 2 — Joint Dual-LoRA fusion

Loads Stage 1 weights, then jointly trains LoRA-AESG and the gating network.

```bash
conda run -n panorama python -u -m training.train_stage2 \
    --data_root /users/2522553y/liangyue_ws/vlm_panorama/data \
    --stage1_ckpt ./checkpoints/stage1/best_checkpoint.pt \
    --output_dir ./checkpoints/stage2 \
    --epochs 30
```

Additional Stage 2 arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--stage1_ckpt` | — | Path to Stage 1 best checkpoint |
| `--lambda_recon` | 1.0 | L_recon weight |
| `--lambda_rel` | 0.25 | Spatial relation loss weight |
| `--lambda_aff` | 0.25 | Affiliation completeness loss weight |
| `--lambda_ctx` | 0.25 | Context style consistency weight |
| `--lambda_seam` | 0.1 | ERP boundary seam loss weight |
| `--lambda_pano` | 0.3 | Reprojection consistency weight |

---

## Inference

Only the panorama image and an editing prompt are required. The pipeline
automatically extracts the target object noun from the prompt and runs
Grounded-SAM to locate and mask the region. If Grounded-SAM checkpoints are
not available it falls back to a centre-strip crop.

```bash
conda run -n panorama python -m inference.edit_with_lora \
    --input  data/train/scene_000/panorama.jpg \
    --prompt "Replace the bench with a modern lab workstation" \
    --stage2_ckpt ./checkpoints/stage2/best_checkpoint.pt \
    --output output_edited.jpg \
    --save_intermediates
```

With `--save_intermediates` the script additionally saves:
- `output_edited_perspective.jpg` — original perspective crop
- `output_edited_edited_persp.jpg` — edited perspective patch before reprojection
- `output_edited_meta.json` — gate values (γ_p, γ_s), projection parameters, detected subject and box

All inference arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | — | Input ERP panorama path |
| `--prompt` | — | Natural language editing instruction |
| `--output` | `output_edited.jpg` | Output path |
| `--stage2_ckpt` | *(optional)* | Trained Stage 2 checkpoint |
| `--stage1_ckpt` | *(optional)* | Stage 1 checkpoint (fallback if Stage 2 unavailable) |
| `--backbone_path` | *(optional)* | Path to Qwen-Image-Edit-2511 model |
| `--fov` | 90.0 | Perspective field-of-view in degrees |
| `--img_size` | 512 | Perspective patch size |
| `--task_type` | `inpaint` | `reconstruct` or `inpaint` |
| `--save_intermediates` | off | Save perspective crops and meta JSON |
| `--device` | auto | `cuda` or `cpu` |

### ROI detection strategy

1. The subject noun is extracted from the prompt with pattern matching
   (e.g. "Replace the **bench** with …" → `bench`).
2. Grounded-SAM (GroundingDINO + SAM) detects and segments that noun
   in the panorama if `checkpoints/groundingdino_swint_ogc.pth` and
   `checkpoints/sam_vit_h_4b8939.pth` are present.
3. If Grounded-SAM is unavailable, a centre-strip region is used as fallback.
4. AESG-driven mask dilation is applied to the detected region before editing.

### Python API

```python
from inference.edit_with_lora import PanoramaEditorWithLoRA

editor = PanoramaEditorWithLoRA(
    stage2_ckpt="./checkpoints/stage2/best_checkpoint.pt",
    backbone_path="Qwen/Qwen-Image-Edit-2511",   # optional
    device="cuda",
)

result = editor.edit(
    panorama="data/train/scene_000/panorama.jpg",
    prompt="Replace the bench with a modern lab workstation",
    fov=90.0,
    task_type="inpaint",
    save_intermediates=True,
)
result["edited_panorama"].save("output_edited.jpg")
```

---

## Parameter budget

| Component | Parameters (est.) | % of backbone |
|-----------|------------------|---------------|
| Backbone W (frozen) | ~7 B | 100 % |
| LoRA-Pano (r = 8) | ~4–10 M | ~0.06–0.15 % |
| LoRA-AESG (r = 8) | ~4–10 M | ~0.06–0.15 % |
| Distortion encoder | ~0.5–1 M | < 0.01 % |
| Gating network | ~0.1–0.5 M | < 0.01 % |
| **Total trainable** | **~9–22 M** | **~0.13–0.32 %** |

Single A100 40 GB is sufficient for training.

---

## Evaluation metrics

| Dimension | Metric | Description |
|-----------|--------|-------------|
| Visual quality | FID, LPIPS, SSIM, CLIP Score | Generation quality and instruction following |
| Panoramic consistency | SDS, RE, Lat-FID | Boundary continuity, reprojection error, latitude robustness |
| Educational semantics | OPA, RSR, ACS | Object placement, relation satisfaction, affiliation completeness |
| Mask quality | Mask IoU, Leakage Rate | Mask precision and edit containment |
| Human | Educator Rating (1–5) | Pedagogical appropriateness |

---

## Ablation configurations

14 ablation conditions are defined in the technical document. The key ones:

| Configuration | Validates |
|---------------|-----------|
| Full model | Upper bound |
| w/o LoRA-Pano | Geometric prior necessity |
| w/o LoRA-AESG | Semantic adaptation value |
| w/o Gating (γ_p = γ_s = 0.5) | Adaptive fusion benefit |
| Single LoRA (joint conditioning) | Dual-branch separation value |
| No LoRA | Overall LoRA contribution |
| Fixed-region mask | Object-level mask value |
| Data scaling (50/100/150/200 scenes) | Data sufficiency |

---

## Citation

> Distortion-aware Dual-LoRA Fusion for Panorama-consistent Scene Editing.
> Target journal: *Techniques and Applications of Multimodal Data Fusion*.
