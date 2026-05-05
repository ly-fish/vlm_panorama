"""Batch evaluation runner over the test split.

For each scene in data/test/:
  1. Reads panorama.jpg, result_*_mask.jpg, scene_*_instruction.txt
  2. Constructs the editing prompt (from prompts JSON or auto-template)
  3. Runs both baseline and Dual-LoRA pipelines (unless outputs already exist)
  4. Computes all evaluation metrics via evaluate.py
  5. Saves per-scene JSON + aggregated CSV

Typical workflow
----------------
Step A — prepare editing prompts (one-time):
    python eval_runner.py --data_root data/test --dump_prompts_template eval_prompts.json
    # Edit eval_prompts.json: fill in the "prompt" field for each scene
    # The instruction.txt content is included as "scene_description" to guide you.

Step B — run evaluation (can resume; skips scenes whose outputs already exist):
    python eval_runner.py \\
        --data_root    data/test \\
        --prompts_json eval_prompts.json \\
        --output_dir   output/eval_test \\
        --stage2_ckpt  checkpoints/stage2/best_checkpoint.pt \\
        --output_csv   results/test_metrics.csv

Flags
-----
  --metrics_only   Skip pipeline inference; only compute metrics on existing outputs.
  --baseline_only  Evaluate baseline branch only (faster).
  --lora_only      Evaluate LoRA branch only.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))


# ---------------------------------------------------------------------------
# Scene discovery
# ---------------------------------------------------------------------------

def _find_mask_jpg(scene_dir: Path) -> Optional[Path]:
    candidates = sorted(scene_dir.glob("*_mask.jpg"))
    return candidates[0] if candidates else None


def _find_mask_json(scene_dir: Path) -> Optional[Path]:
    candidates = sorted(scene_dir.glob("*_mask.json"))
    return candidates[0] if candidates else None


def _find_instruction(scene_dir: Path) -> str:
    candidates = sorted(scene_dir.glob("*_instruction.txt"))
    if candidates:
        return candidates[0].read_text(encoding="utf-8").strip()
    return ""


def _primary_object(mask_json: Path) -> tuple[str, int]:
    """Return (label, value) of the highest-logit non-background object."""
    with open(mask_json, encoding="utf-8") as f:
        detections = json.load(f)
    objs = [d for d in detections if d.get("value", 0) != 0 and "logit" in d]
    if not objs:
        objs = [d for d in detections if d.get("value", 0) != 0]
    if not objs:
        return ("unknown", 1)
    best = max(objs, key=lambda d: d.get("logit", 0.0))
    return best.get("label", "object"), int(best.get("value", 1))


def discover_scenes(data_root: Path) -> list[dict]:
    scenes = []
    for scene_dir in sorted(data_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        panorama = scene_dir / "panorama.jpg"
        if not panorama.exists():
            continue
        mask_jpg  = _find_mask_jpg(scene_dir)
        mask_json = _find_mask_json(scene_dir)
        if mask_jpg is None or mask_json is None:
            continue
        label, value = _primary_object(mask_json)
        instruction   = _find_instruction(scene_dir)
        scenes.append({
            "scene_id":          scene_dir.name,
            "panorama":          str(panorama),
            "mask_jpg":          str(mask_jpg),
            "mask_json":         str(mask_json),
            "object_label":      label,
            "object_value":      value,
            "scene_description": instruction,
            "prompt":            "",   # to be filled by user
        })
    return scenes


# ---------------------------------------------------------------------------
# Prompt template helper
# ---------------------------------------------------------------------------

def _auto_prompt(label: str) -> str:
    """Minimal fallback prompt when no user-supplied prompt is available."""
    return f"Replace the {label} with a different one"


# ---------------------------------------------------------------------------
# Pipeline runner (calls run_comparison.py as subprocess)
# ---------------------------------------------------------------------------

def _run_pipeline(
    panorama: str,
    prompt: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    cmd = [
        sys.executable, "run_comparison.py",
        "--input",      panorama,
        "--prompt",     prompt,
        "--output_dir", str(output_dir),
        "--device",     args.device,
    ]
    if args.baseline_only:
        cmd.append("--baseline_only")
    elif args.lora_only:
        cmd.append("--lora_only")
    if args.stage2_ckpt:
        cmd += ["--stage2_ckpt", args.stage2_ckpt]
    if args.stage1_ckpt:
        cmd += ["--stage1_ckpt", args.stage1_ckpt]
    if args.backbone_path:
        cmd += ["--backbone_path", args.backbone_path]

    print(f"  [Runner] {' '.join(cmd[:6])} …")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"  [Runner] WARNING: pipeline exited with code {result.returncode}")


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def _compute_metrics_for_scene(
    scene: dict,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[dict]:
    from evaluate import evaluate_pair

    stem      = Path(scene["panorama"]).stem
    prompt    = scene["prompt"] or _auto_prompt(scene["object_label"])
    src_prompt = scene["scene_description"] or "a panoramic scene"
    mask_path  = scene["mask_jpg"]
    obj_value  = scene["object_value"]
    device     = args.device

    results = []
    branches = []
    if not args.lora_only:
        branches.append(("Baseline", output_dir / f"{stem}_baseline.png"))
    if not args.baseline_only:
        branches.append(("LoRA",     output_dir / f"{stem}_lora.png"))

    for label, edited_path in branches:
        if not edited_path.exists():
            print(f"  [Metrics] {label}: output not found at {edited_path}, skipping.")
            continue
        print(f"\n[Metrics] {scene['scene_id']} — {label}")
        res = evaluate_pair(
            input_path=scene["panorama"],
            edited_path=str(edited_path),
            prompt=prompt,
            src_prompt=src_prompt,
            mask_path=mask_path,
            object_value=obj_value,
            device=device,
            label=label,
        )
        res["scene_id"] = scene["scene_id"]
        res["object_label"] = scene["object_label"]
        results.append(res)

    return results


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

_NUMERIC_METRICS = [
    "clip_score", "clip_dir_score", "psnr", "ssim", "lpips",
    "bg_psnr", "bg_ssim",
]


def _aggregate(all_results: list[dict]) -> dict[str, dict]:
    """Compute per-label mean ± std for each numeric metric."""
    import numpy as np

    by_label: dict[str, list[dict]] = {}
    for r in all_results:
        by_label.setdefault(r["label"], []).append(r)

    summary: dict[str, dict] = {}
    for label, records in by_label.items():
        agg: dict = {"label": label, "n": len(records)}
        for m in _NUMERIC_METRICS:
            vals = [r[m] for r in records if r.get(m) is not None]
            if vals:
                agg[f"{m}_mean"] = float(np.mean(vals))
                agg[f"{m}_std"]  = float(np.std(vals))
            else:
                agg[f"{m}_mean"] = None
                agg[f"{m}_std"]  = None
        summary[label] = agg
    return summary


def _print_summary(summary: dict[str, dict]) -> None:
    labels = list(summary.keys())
    if not labels:
        return

    col_w    = 20
    label_w  = 22
    metric_w = 26

    header = f"{'Metric':<{metric_w}}" + "".join(f"{l:>{col_w}}" for l in labels)
    print("\n" + "=" * len(header))
    print("AGGREGATE RESULTS (mean ± std)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    metric_names = {
        "clip_score":     "CLIP Score (↑)",
        "clip_dir_score": "CLIP Dir Score (↑)",
        "psnr":           "PSNR dB (↑)",
        "ssim":           "SSIM (↑)",
        "lpips":          "LPIPS (↓)",
        "bg_psnr":        "BG PSNR dB (↑)",
        "bg_ssim":        "BG SSIM (↑)",
    }
    for m, name in metric_names.items():
        row = f"{name:<{metric_w}}"
        for label in labels:
            agg = summary[label]
            mean = agg.get(f"{m}_mean")
            std  = agg.get(f"{m}_std")
            if mean is None:
                row += f"{'N/A':>{col_w}}"
            else:
                cell = f"{mean:.3f}±{std:.3f}"
                row += f"{cell:>{col_w}}"
        print(row)

    print("=" * len(header) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch evaluation runner over the panorama editing test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data_root", default="data/test",
        help="Root directory of the test split (contains scene_XX/ subdirectories).",
    )
    p.add_argument(
        "--prompts_json", default=None,
        help=(
            "JSON file mapping scene_id → editing prompt. "
            "Generate a template with --dump_prompts_template."
        ),
    )
    p.add_argument(
        "--dump_prompts_template", default=None,
        help=(
            "If provided, write a prompts template JSON to this path and exit. "
            "Fill in the 'prompt' field for each scene before running evaluation."
        ),
    )
    p.add_argument(
        "--output_dir", default="output/eval_test",
        help="Root directory to save pipeline outputs and metrics.",
    )
    p.add_argument(
        "--output_csv", default=None,
        help="Save aggregated results to this CSV file.",
    )
    p.add_argument(
        "--metrics_only", action="store_true",
        help="Skip pipeline inference; only compute metrics on existing outputs.",
    )
    p.add_argument("--baseline_only", action="store_true")
    p.add_argument("--lora_only",     action="store_true")
    p.add_argument("--stage2_ckpt",   default=None)
    p.add_argument("--stage1_ckpt",   default=None)
    p.add_argument(
        "--backbone_path", default="Qwen/Qwen-Image-Edit-2511",
    )
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)

    scenes = discover_scenes(data_root)
    print(f"[Runner] Found {len(scenes)} scenes in {data_root}")

    # --- Dump template and exit ---
    if args.dump_prompts_template:
        template = [
            {
                "scene_id":          s["scene_id"],
                "object_label":      s["object_label"],
                "scene_description": s["scene_description"],
                "prompt":            "",   # <-- fill this in
            }
            for s in scenes
        ]
        with open(args.dump_prompts_template, "w", encoding="utf-8") as f:
            json.dump(template, f, indent=2, ensure_ascii=False)
        print(f"[Runner] Prompts template written to {args.dump_prompts_template}")
        print("[Runner] Fill in the 'prompt' field for each scene, then re-run without --dump_prompts_template.")
        return

    # --- Load user-supplied prompts ---
    prompts_map: dict[str, str] = {}
    if args.prompts_json:
        with open(args.prompts_json, encoding="utf-8") as f:
            entries = json.load(f)
        for e in entries:
            if e.get("prompt"):
                prompts_map[e["scene_id"]] = e["prompt"]
        print(f"[Runner] Loaded {len(prompts_map)} prompts from {args.prompts_json}")

    for s in scenes:
        if s["scene_id"] in prompts_map:
            s["prompt"] = prompts_map[s["scene_id"]]
        # else: falls back to _auto_prompt() inside _compute_metrics_for_scene

    # --- Per-scene loop ---
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    per_scene_json = output_root / "per_scene_results.json"

    # Resume from previous run if file exists
    if per_scene_json.exists():
        with open(per_scene_json, encoding="utf-8") as f:
            all_results = json.load(f)
        done_scenes = {r["scene_id"] for r in all_results}
        print(f"[Runner] Resuming: {len(done_scenes)} scenes already done.")
    else:
        done_scenes: set[str] = set()

    for scene in scenes:
        sid = scene["scene_id"]
        if sid in done_scenes:
            print(f"[Runner] {sid}: already evaluated, skipping.")
            continue

        scene_out = output_root / sid
        scene_out.mkdir(parents=True, exist_ok=True)

        effective_prompt = scene["prompt"] or _auto_prompt(scene["object_label"])
        print(f"\n[Runner] ===== {sid} =====")
        print(f"         object : {scene['object_label']}")
        print(f"         prompt : {effective_prompt}")

        # Run pipeline
        if not args.metrics_only:
            _run_pipeline(scene["panorama"], effective_prompt, scene_out, args)

        # Compute metrics
        scene_results = _compute_metrics_for_scene(scene, scene_out, args)
        all_results.extend(scene_results)

        # Checkpoint after every scene
        with open(per_scene_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    # --- Aggregate ---
    summary = _aggregate(all_results)
    _print_summary(summary)

    # --- Save CSV ---
    if args.output_csv:
        import csv
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        # Flat per-entry CSV
        if all_results:
            keys = list(all_results[0].keys())
            with open(args.output_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(all_results)
            print(f"[Runner] Per-scene CSV saved → {args.output_csv}")

        # Summary CSV
        summary_csv = Path(str(Path(args.output_csv).with_suffix("")) + "_summary.csv")
        summary_rows = list(summary.values())
        if summary_rows:
            with open(summary_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
                w.writeheader()
                w.writerows(summary_rows)
            print(f"[Runner] Summary CSV saved → {summary_csv}")

    print(f"\n[Runner] Done. Per-scene results → {per_scene_json}")


if __name__ == "__main__":
    main()
