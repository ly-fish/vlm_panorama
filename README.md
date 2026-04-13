# Distortion-aware Dual-LoRA Fusion for Panorama-consistent Scene Editing

---

## Project Overview

### Background and Motivation

360° panoramic images in Equirectangular Projection (ERP) format are increasingly used in virtual classrooms, immersive training scenarios, and educational VR environments. Editing these panoramas with natural-language instructions poses two fundamental challenges that standard image-editing models (trained on perspective photographs) fail to address:

1. **Geometric distortion**: ERP maps a sphere onto a flat canvas, introducing severe latitude-dependent stretching near the poles. Any content synthesised without awareness of this projection will appear warped or inconsistent when viewed as a spherical scene.
2. **Pedagogical semantic constraints**: Educational scenes contain hierarchical object relationships (teacher, whiteboard, instruments, students, …) that must be preserved or respected when an edit request alters part of the scene. Ignoring these relations breaks the pedagogical intent.

This project introduces a **parameter-efficient dual-branch LoRA adaptation** of the frozen **Qwen-Image-Edit-2511** vision-language model that simultaneously handles both challenges — without any paired before/after editing supervision.

---

### System Architecture

```
User instruction + ERP panorama
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Scene Understanding                        │
│                                                                  │
│  ┌───────────────────┐        ┌──────────────────────────────┐  │
│  │  SceneGraph        │        │  ROI Localization             │  │
│  │  (Qwen-max LLM)   │        │  (Grounded-SAM / fallback)   │  │
│  │  scene_graph/      │        │  panorama_editing/roi/        │  │
│  └────────┬──────────┘        └──────────────┬───────────────┘  │
│           │ AESG JSON                         │ bounding box     │
└───────────┼───────────────────────────────────┼─────────────────┘
            ▼                                   ▼
┌────────────────────────┐       ┌──────────────────────────────┐
│  AESG Encoder          │       │  ERP Projection Parameters   │
│  aesg/encoder.py       │       │  (lat, lon, FoV, centre)     │
│  → z_G ∈ ℝ⁵¹²         │       │  lora/distortion_encoder.py  │
└──────────┬─────────────┘       │  → z_θ ∈ ℝ⁵¹²               │
           │                     └──────────────┬───────────────┘
           └──────────────┬──────────────────────┘
                          ▼
              ┌──────────────────────┐
              │  Adaptive Gating     │
              │  lora/gating.py      │
              │  [γ_p, γ_s] = σ(MLP │
              │   ([z_θ; z_G; e_t]))│
              └──────────┬───────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────┐
│              Dual-LoRA Modulated Backbone                       │
│                                                                  │
│   Frozen Qwen-Image-Edit-2511 cross-attention layers           │
│                                                                  │
│   W' = W + γ_p · ΔW_pano(z_θ) + γ_s · ΔW_aesg(z_G)          │
│                                                                  │
│   ┌──────────────────┐   ┌──────────────────┐                  │
│   │  LoRA-Pano       │   │  LoRA-AESG       │                  │
│   │  FiLM(z_θ) → A  │   │  FiLM(z_G) → A  │                  │
│   │  lora/lora_pano  │   │  lora/lora_aesg  │                  │
│   └──────────────────┘   └──────────────────┘                  │
│                                                                  │
│   HCFM: AESG branch tokens prepended to text cross-attn        │
│   modules/hcfm.py                                               │
└───────────────────────────────┬────────────────────────────────┘
                                │ edited perspective patch
                                ▼
                   ┌────────────────────────┐
                   │  ERP Reprojection      │
                   │  panorama_editing/     │
                   │  reproject.py          │
                   └────────────────────────┘
                                │
                                ▼
                    Output ERP panorama (edited)
```

---

### Module Breakdown

| Module | Location | Responsibility |
|--------|----------|----------------|
| **Dual-LoRA core** | `lora/` | All LoRA parameter modules: distortion encoder, FiLM conditioning, conditional LoRA layer, LoRA-Pano, LoRA-AESG, adaptive gating, fusion orchestrator |
| **AESG schema & encoder** | `aesg/` | Typed dataclass schema for the Anchor-centred Educational Scene Graph; GNN-style token encoder that maps graph nodes/edges → z_G |
| **Scene graph parser** | `scene_graph/` | LangChain + Qwen-max LLM wrapper that converts a free-text editing instruction into a structured AESG JSON, incorporating Physical Affiliation Analysis (PAA) and SE360 spatial reasoning |
| **ERP data utilities** | `data/` | Sphere ↔ perspective projection helpers (ERP ↔ gnomonic), mask dilation, self-supervised degradation pipeline, PyTorch Dataset |
| **Panorama editing pipeline** | `panorama_editing/` | Qwen-Image-Edit-2511 pipeline wrapper (`qwen_image_editing/`), ERP reprojection back to full panorama, ROI localisation via Grounded-SAM |
| **HCFM fusion module** | `modules/hcfm.py` | Hierarchical Condition Fusion: prepends AESG branch tokens (anchor / object / context / relation) to the text hidden states feeding cross-attention |
| **Training** | `training/` | Two-stage trainers with full loss suite (MSE + VGG perceptual + SSIM + panorama reprojection + AESG semantic losses) |
| **Inference** | `inference/` | `PanoramaEditorWithLoRA` — 8-step end-to-end pipeline with CLI; saves intermediate crops and gate-value meta JSON |
| **Panorama generation** | `panorama_generation/` | Separate Flux-based diffusion pipeline (CLIP + T5 + VAE + FluxTransformer2D) for unconditional 360° panorama synthesis |
| **Configs** | `configs/` | YAML configuration for branch flags, loss weights, ROI thresholds, and FoV defaults |
| **Tests** | `tests/` | End-to-end smoke tests for the AESG pipeline |

---

### Technology Stack

| Layer | Tools / Models |
|-------|---------------|
| **Editing backbone** | Qwen-Image-Edit-2511 (7 B VLM, fully frozen during LoRA training) |
| **Scene understanding LLM** | Qwen-max via LangChain + OpenAI-compatible API |
| **Panorama generation** | Flux diffusion pipeline (FluxTransformer2D, CLIP, T5, VAE) via Diffusers |
| **Object detection / ROI** | Grounded-SAM (GroundingDINO + SAM) |
| **LoRA conditioning** | FiLM (Feature-wise Linear Modulation) |
| **Perceptual loss** | VGG-16 (relu1\_2, relu2\_2, relu3\_3) |
| **Image quality metrics** | SSIM (differentiable), LPIPS, FID, CLIP Score |
| **Deep learning framework** | PyTorch 2.x |
| **Configuration** | YAML (`configs/aesg_edit.yaml`) |
| **Environment** | Conda `panorama`; single NVIDIA A100 40 GB sufficient |

---

### Complete End-to-End Workflow

```
Step 1  User provides:
        • panorama.jpg  (2048×1024 ERP image)
        • Editing prompt, e.g. "Replace the bench with a modern lab workstation"

Step 2  Scene Graph Parser  (scene_graph/scene_graph.py)
        • Qwen-max LLM parses the prompt via LangChain
        • Outputs structured AESG JSON with core/context objects, spatial
          relations, PAA affiliations, and safety constraints

Step 3  AESG Encoding  (aesg/schema.py + aesg/encoder.py)
        • JSON → typed AESGGraph dataclass (validated)
        • Graph encoder maps nodes + edges → z_G ∈ ℝ⁵¹²

Step 4  ROI Localisation  (panorama_editing/roi/roi_localization.py)
        • Subject noun extracted from prompt by regex
        • Grounded-SAM detects and segments the noun in the ERP image
        • Detected box projected to perspective crop parameters (lat, lon, FoV)
        • Fallback: centre-strip crop if SAM checkpoints unavailable

Step 5  Distortion Encoding  (lora/distortion_encoder.py)
        • Projection parameters (lat, lon, FoV, cx, cy) → sinusoidal MLP
        • Outputs z_θ ∈ ℝ⁵¹²

Step 6  Adaptive Gating  (lora/gating.py)
        • MLP([z_θ; z_G; e_task]) → (γ_p, γ_s) ∈ [0,1]²
        • γ_p weights the geometric LoRA-Pano branch
        • γ_s weights the semantic LoRA-AESG branch

Step 7  Dual-LoRA Priming  (lora/dual_lora_fusion.py)
        • All DualLoRAFusion layers cached with z_θ, z_G, γ_p, γ_s via prime()
        • Each cross-attention linear layer computes:
          W'x = Wx + γ_p·ΔW_pano(z_θ)x + γ_s·ΔW_aesg(z_G)x

Step 8  HCFM Token Injection  (modules/hcfm.py)
        • AESG branch tokens (anchor / object / context / relation) are
          concatenated to text hidden states before cross-attention

Step 9  Backbone Editing  (panorama_editing/executor.py)
        • QwenImageEditPlusPipeline runs on the perspective crop
        • Backbone weights stay frozen; only LoRA deltas modify outputs

Step 10 ERP Reprojection  (panorama_editing/reproject.py)
        • Edited perspective patch re-blended into the original ERP panorama
        • Seam blending applied at left/right boundary (λ_seam loss during training)

Step 11 Output saved:
        • output_edited.jpg            — full ERP panorama (edited)
        • output_edited_meta.json      — γ_p, γ_s, projection params, detected box
        • output_edited_perspective.jpg — original crop (with --save_intermediates)
        • output_edited_edited_persp.jpg — edited crop before reprojection
```

---

### Training Paradigm Summary

This project uses **self-supervised reconstruction** — no paired editing data is required:

1. Grounded-SAM detects objects in each training panorama → `*_mask.jpg`.
2. A region is masked / degraded (grey fill, Gaussian noise, or heavy blur).
3. The model learns to reconstruct the original from the degraded input.
4. The reconstruction target is a real ERP panorama, so the model inherently learns geometry-consistent generation.

Two training stages are applied sequentially:

- **Stage 1**: trains only LoRA-Pano + distortion encoder (geometric adaptation).
- **Stage 2**: jointly fine-tunes LoRA-AESG + gating network, while LoRA-Pano continues at 1/10 of the base learning rate.

---

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
conda activate panorama 
python -m training.train_stage1 \
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
    --epochs 30 \
    --batch_size 4
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
python -m inference.edit_with_lora \
    --input  /users/2522553y/liangyue_ws/panorama_test/desk/result.png \
    --prompt "请将桌子材质替换为铝合金材质" \
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
