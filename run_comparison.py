"""Unified comparison runner: Baseline (executor) vs Dual-LoRA editing.

Runs the two pipelines on the *same* input image and prompt, and saves
both outputs to the *same* directory so they are easy to compare side-by-side.

Output filenames
----------------
  <output_dir>/<input_stem>_baseline.png   — executor (AESG only, no LoRA)
  <output_dir>/<input_stem>_lora.png       — Dual-LoRA editor
  <output_dir>/<input_stem>_lora_meta.json — (optional, --save_intermediates)

Usage examples
--------------
  # Minimal — no LoRA checkpoint (LoRA branch runs with untrained weights)
  python run_comparison.py \\
      --input  data/train/scene_000/panorama.jpg \\
      --prompt "增加一把红色现代椅子替换木质长椅"

  # With a trained Stage-2 checkpoint
  python run_comparison.py \\
      --input  /path/to/panorama.png \\
      --prompt "Replace the wooden bench with a red modern chair" \\
      --stage2_ckpt checkpoints/stage2/best_checkpoint.pt \\
      --output_dir  output/comparison_001

  # Skip the LoRA branch (baseline only, useful for quick sanity-checks)
  python run_comparison.py \\
      --input  /path/to/panorama.png \\
      --prompt "..." \\
      --baseline_only
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

# Make sure the package root is on sys.path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    # Default: same directory as the input image
    return Path(args.input).resolve().parent


def _free_executor_pipeline() -> None:
    """Release the executor's global Qwen pipeline to free GPU memory."""
    try:
        import panorama_editing.executor as _exec
        import torch
        if _exec._PIPELINE is not None:
            _exec._PIPELINE = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            print("[Runner] Baseline pipeline released from GPU memory.")
    except Exception as exc:
        print(f"[Runner] Warning: could not free executor pipeline: {exc}")


# ---------------------------------------------------------------------------
# Run baseline (executor)
# ---------------------------------------------------------------------------

def run_baseline(
    input_path: str,
    prompt: str,
    output_path: Path,
    use_aesg: bool,
    save_intermediates: bool,
) -> None:
    from panorama_editing.executor import edit_panorama

    print("\n" + "=" * 60)
    print("[Runner] Running BASELINE (executor)")
    print(f"         use_aesg={use_aesg}")
    print("=" * 60)
    t0 = time.time()

    result = edit_panorama(
        image_erp=input_path,
        text=prompt,
        use_aesg=use_aesg,
        return_intermediates=save_intermediates,
    )

    if save_intermediates and isinstance(result, dict):
        img = result["final_image"]
    else:
        img = result

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    elapsed = time.time() - t0
    print(f"[Runner] Baseline saved → {output_path}  ({elapsed:.1f}s)")


# ---------------------------------------------------------------------------
# Run Dual-LoRA editor
# ---------------------------------------------------------------------------

def run_lora(
    input_path: str,
    prompt: str,
    output_path: Path,
    args: argparse.Namespace,
) -> None:
    from inference.edit_with_lora import PanoramaEditorWithLoRA

    print("\n" + "=" * 60)
    print("[Runner] Running DUAL-LORA editor")
    print(f"         stage2_ckpt={args.stage2_ckpt or '(none — untrained weights)'}")
    print(f"         stage1_ckpt={args.stage1_ckpt or '(none)'}")
    print("=" * 60)
    t0 = time.time()

    editor = PanoramaEditorWithLoRA(
        stage2_ckpt=args.stage2_ckpt,
        stage1_ckpt=args.stage1_ckpt,
        backbone_path=args.backbone_path,
        device=args.device,
        lora_rank=args.lora_rank,
    )

    result = editor.edit(
        panorama=input_path,
        prompt=prompt,
        task_type=args.task_type,
        save_intermediates=args.save_intermediates,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result["edited_panorama"].save(output_path)
    elapsed = time.time() - t0
    print(f"[Runner] LoRA output saved → {output_path}  ({elapsed:.1f}s)")

    if args.save_intermediates:
        stem = output_path.stem
        meta = {
            "gamma_p":          result.get("gamma_p"),
            "gamma_s":          result.get("gamma_s"),
            "effective_prompt": result.get("effective_prompt"),
            "aesg_prompt":      result.get("aesg_prompt"),
        }
        if "edited_local" in result:
            local_path = output_path.parent / f"{stem}_local.png"
            result["edited_local"].save(local_path)
            print(f"[Runner] Local patch saved → {local_path}")
        meta_path = output_path.parent / f"{stem}_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"[Runner] Metadata saved → {meta_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run both the Baseline (executor) and Dual-LoRA editor on the same "
            "input, saving outputs side-by-side for easy comparison."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Required ---
    parser.add_argument(
        "--input", required=True,
        help="Path to the input ERP panorama image.",
    )
    parser.add_argument(
        "--prompt", required=True,
        help="Editing instruction in natural language.",
    )

    # --- Output ---
    parser.add_argument(
        "--output_dir", default=None,
        help=(
            "Directory to save both output images. "
            "Defaults to the same directory as --input."
        ),
    )

    # --- Shared model ---
    parser.add_argument(
        "--backbone_path",
        default="Qwen/Qwen-Image-Edit-2511",
        help="HuggingFace model ID or local path to the Qwen backbone.",
    )
    parser.add_argument(
        "--device", default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run on.",
    )

    # --- Baseline options ---
    parser.add_argument(
        "--no_aesg", action="store_true",
        help="Disable AESG conditioning in the baseline executor.",
    )
    parser.add_argument(
        "--baseline_only", action="store_true",
        help="Run only the baseline; skip the Dual-LoRA branch.",
    )

    # --- LoRA options ---
    parser.add_argument(
        "--lora_only", action="store_true",
        help="Run only the Dual-LoRA branch; skip the baseline.",
    )
    parser.add_argument(
        "--stage2_ckpt", default=None,
        help="Path to Stage-2 checkpoint with LoRA weights.",
    )
    parser.add_argument(
        "--stage1_ckpt", default=None,
        help="Path to Stage-1 checkpoint (fallback).",
    )
    parser.add_argument(
        "--lora_rank", type=int, default=8,
        help="LoRA rank used during training.",
    )
    parser.add_argument(
        "--task_type", default="inpaint",
        choices=["inpaint", "reconstruct"],
        help="Editing task type passed to the LoRA editor.",
    )

    # --- Misc ---
    parser.add_argument(
        "--save_intermediates", action="store_true",
        help="Save intermediate outputs (local patch, metadata JSON).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = str(Path(args.input).resolve())
    prompt = args.prompt
    output_dir = _resolve_output_dir(args)
    stem = Path(args.input).stem

    print("\n[Runner] ========================================")
    print(f"[Runner] Input  : {input_path}")
    print(f"[Runner] Prompt : {prompt}")
    print(f"[Runner] Output : {output_dir}/")
    print("[Runner] ========================================\n")

    run_base = not args.lora_only
    run_lora_branch = not args.baseline_only

    # --- Baseline ---
    if run_base:
        baseline_out = output_dir / f"{stem}_baseline.png"
        run_baseline(
            input_path=input_path,
            prompt=prompt,
            output_path=baseline_out,
            use_aesg=not args.no_aesg,
            save_intermediates=args.save_intermediates,
        )

    # Free GPU memory before loading the LoRA model
    if run_base and run_lora_branch:
        _free_executor_pipeline()

    # --- Dual-LoRA ---
    if run_lora_branch:
        lora_out = output_dir / f"{stem}_lora.png"
        run_lora(
            input_path=input_path,
            prompt=prompt,
            output_path=lora_out,
            args=args,
        )

    # --- Summary ---
    print("\n[Runner] ========== DONE ==========")
    if run_base:
        print(f"  Baseline : {output_dir / f'{stem}_baseline.png'}")
    if run_lora_branch:
        print(f"  LoRA     : {output_dir / f'{stem}_lora.png'}")
    print("[Runner] ===================================\n")


if __name__ == "__main__":
    main()
