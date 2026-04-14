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
import multiprocessing as mp
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
    device: str | None = None,
) -> None:
    from panorama_editing.executor import edit_panorama

    print("\n" + "=" * 60)
    print("[Runner] Running BASELINE (executor)")
    print(f"         use_aesg={use_aesg}  device={device or 'auto'}")
    print("=" * 60)
    t0 = time.time()

    result = edit_panorama(
        image_erp=input_path,
        text=prompt,
        use_aesg=use_aesg,
        return_intermediates=save_intermediates,
        device=device,
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
    device: str | None = None,
) -> None:
    from inference.edit_with_lora import PanoramaEditorWithLoRA

    effective_device = device or args.device
    print("\n" + "=" * 60)
    print("[Runner] Running DUAL-LORA editor")
    print(f"         stage2_ckpt={args.stage2_ckpt or '(none — untrained weights)'}")
    print(f"         stage1_ckpt={args.stage1_ckpt or '(none)'}")
    print(f"         device={effective_device}")
    print("=" * 60)
    t0 = time.time()

    editor = PanoramaEditorWithLoRA(
        stage2_ckpt=args.stage2_ckpt,
        stage1_ckpt=args.stage1_ckpt,
        backbone_path=args.backbone_path,
        device=effective_device,
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

    # --- Multi-GPU ---
    parser.add_argument(
        "--device_baseline", default=None,
        help=(
            "Device for the baseline branch (e.g. 'cuda:0'). "
            "When omitted, auto-assigned to cuda:0 in dual-GPU mode."
        ),
    )
    parser.add_argument(
        "--device_lora", default=None,
        help=(
            "Device for the Dual-LoRA branch (e.g. 'cuda:1'). "
            "When omitted, auto-assigned to cuda:1 in dual-GPU mode."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Subprocess worker targets (must be top-level to be picklable)
# ---------------------------------------------------------------------------

def _worker_baseline(
    input_path: str,
    prompt: str,
    output_path_str: str,
    use_aesg: bool,
    save_intermediates: bool,
    device: str,
    pkg_root: str,
    error_queue: "mp.Queue",
) -> None:
    """Worker target: runs the baseline branch in a subprocess."""
    import sys as _sys
    if pkg_root not in _sys.path:
        _sys.path.insert(0, pkg_root)
    try:
        run_baseline(
            input_path=input_path,
            prompt=prompt,
            output_path=Path(output_path_str),
            use_aesg=use_aesg,
            save_intermediates=save_intermediates,
            device=device,
        )
    except Exception as exc:
        error_queue.put(f"[baseline] {exc}")
        raise


def _worker_lora(
    input_path: str,
    prompt: str,
    output_path_str: str,
    args_dict: dict,
    device: str,
    pkg_root: str,
    error_queue: "mp.Queue",
) -> None:
    """Worker target: runs the Dual-LoRA branch in a subprocess."""
    import sys as _sys
    if pkg_root not in _sys.path:
        _sys.path.insert(0, pkg_root)
    try:
        # Reconstruct a lightweight namespace so run_lora() still works
        fake_args = argparse.Namespace(**args_dict)
        run_lora(
            input_path=input_path,
            prompt=prompt,
            output_path=Path(output_path_str),
            args=fake_args,
            device=device,
        )
    except Exception as exc:
        error_queue.put(f"[lora] {exc}")
        raise


def _detect_gpu_count() -> int:
    try:
        import torch
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0


def main() -> None:
    args = parse_args()

    input_path = str(Path(args.input).resolve())
    prompt = args.prompt
    output_dir = _resolve_output_dir(args)
    stem = Path(args.input).stem
    pkg_root = str(Path(__file__).resolve().parent)

    run_base = not args.lora_only
    run_lora_branch = not args.baseline_only

    # ------------------------------------------------------------------
    # Decide execution mode: parallel (2+ GPUs) vs sequential (≤1 GPU)
    # ------------------------------------------------------------------
    gpu_count = _detect_gpu_count()
    use_parallel = (
        run_base
        and run_lora_branch
        and gpu_count >= 2
    )

    # Device assignment
    if use_parallel:
        dev_baseline = args.device_baseline or "cuda:0"
        dev_lora     = args.device_lora     or "cuda:1"
    else:
        dev_baseline = args.device_baseline or args.device
        dev_lora     = args.device_lora     or args.device

    print("\n[Runner] ========================================")
    print(f"[Runner] Input  : {input_path}")
    print(f"[Runner] Prompt : {prompt}")
    print(f"[Runner] Output : {output_dir}/")
    print(f"[Runner] GPUs detected : {gpu_count}")
    if use_parallel:
        print(f"[Runner] Mode   : PARALLEL  (baseline→{dev_baseline}, LoRA→{dev_lora})")
    else:
        print(f"[Runner] Mode   : SEQUENTIAL  (device={dev_baseline})")
    print("[Runner] ========================================\n")

    baseline_out = output_dir / f"{stem}_baseline.png"
    lora_out     = output_dir / f"{stem}_lora.png"

    if use_parallel:
        # ---------------------------------------------------------------
        # Parallel execution: each branch gets its own GPU in a subprocess.
        # 'spawn' avoids CUDA fork issues (CUDA contexts must not be
        # inherited; each child process initialises its own).
        # ---------------------------------------------------------------
        ctx = mp.get_context("spawn")
        error_queue: mp.Queue = ctx.Queue()

        # Serialise args for the LoRA worker (Namespace is not reliably
        # picklable across spawn, so we convert to a plain dict first)
        args_dict = {
            "stage2_ckpt":       args.stage2_ckpt,
            "stage1_ckpt":       args.stage1_ckpt,
            "backbone_path":     args.backbone_path,
            "device":            dev_lora,
            "lora_rank":         args.lora_rank,
            "task_type":         args.task_type,
            "save_intermediates": args.save_intermediates,
        }

        p_baseline = ctx.Process(
            target=_worker_baseline,
            name="baseline",
            args=(
                input_path, prompt, str(baseline_out),
                not args.no_aesg, args.save_intermediates,
                dev_baseline, pkg_root, error_queue,
            ),
            daemon=False,
        )
        p_lora = ctx.Process(
            target=_worker_lora,
            name="lora",
            args=(
                input_path, prompt, str(lora_out),
                args_dict, dev_lora, pkg_root, error_queue,
            ),
            daemon=False,
        )

        t0 = time.time()
        p_baseline.start()
        p_lora.start()
        p_baseline.join()
        p_lora.join()
        elapsed = time.time() - t0

        # Collect errors reported by workers
        errors = []
        while not error_queue.empty():
            errors.append(error_queue.get_nowait())

        if p_baseline.exitcode != 0 or p_lora.exitcode != 0:
            print("\n[Runner] WARNING: one or both worker processes failed.")
            for e in errors:
                print(f"  Error: {e}")
        else:
            print(f"\n[Runner] Parallel execution finished in {elapsed:.1f}s")

    else:
        # ---------------------------------------------------------------
        # Sequential execution (original behaviour)
        # ---------------------------------------------------------------
        if run_base:
            run_baseline(
                input_path=input_path,
                prompt=prompt,
                output_path=baseline_out,
                use_aesg=not args.no_aesg,
                save_intermediates=args.save_intermediates,
                device=dev_baseline,
            )

        # Free GPU memory before loading the LoRA model
        if run_base and run_lora_branch:
            _free_executor_pipeline()

        if run_lora_branch:
            run_lora(
                input_path=input_path,
                prompt=prompt,
                output_path=lora_out,
                args=args,
                device=dev_lora,
            )

    # --- Promote LoRA output as primary result when available ---
    import shutil
    primary_result = output_dir / f"{stem}_result.png"
    if run_lora_branch and lora_out.exists():
        shutil.copy2(lora_out, primary_result)
        print(f"[Runner] Primary result (LoRA) → {primary_result}")
    elif run_base and baseline_out.exists():
        shutil.copy2(baseline_out, primary_result)
        print(f"[Runner] Primary result (baseline) → {primary_result}")

    # --- Summary ---
    print("\n[Runner] ========== DONE ==========")
    if run_base:
        print(f"  Baseline : {baseline_out}")
    if run_lora_branch:
        print(f"  LoRA     : {lora_out}")
    if primary_result.exists():
        print(f"  Primary  : {primary_result}")
    print("[Runner] ===================================\n")


if __name__ == "__main__":
    main()
