"""Visualisation of panorama editing evaluation results.

Reads results/test_metrics.csv and produces a set of charts saved to
results/figures/.

Charts generated
----------------
  01_metric_comparison_bar.png   — grouped bar chart for all metrics
  02_metric_boxplot.png          — box-plot distribution per metric
  03_per_scene_heatmap.png       — per-scene metric heatmap (Baseline vs LoRA delta)
  04_scatter_clip_vs_lpips.png   — CLIP Score vs LPIPS scatter
  05_radar_chart.png             — radar / spider chart summary
  06_improvement_delta.png       — LoRA − Baseline delta per scene (sorted)

Usage
-----
  python eval/visualize.py                          # default paths
  python eval/visualize.py --csv results/test_metrics.csv --out results/figures
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
    "savefig.bbox":      "tight",
    "savefig.dpi":       180,
})

COLORS = {"Baseline": "#5B8DB8", "LoRA": "#E07B54"}

METRICS_META = {
    # key: (display name, direction, good_color_direction)
    "clip_score":     ("CLIP Score",     "↑", +1),
    "clip_dir_score": ("CLIP Dir Score", "↑", +1),
    "psnr":           ("PSNR (dB)",      "↑", +1),
    "ssim":           ("SSIM",           "↑", +1),
    "lpips":          ("LPIPS",          "↓", -1),
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Keep only rows that have valid numeric metrics
    numeric_cols = list(METRICS_META.keys())
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=numeric_cols)
    return df


def split_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = df[df["label"] == "Baseline"].set_index("scene_id").sort_index()
    lora = df[df["label"] == "LoRA"].set_index("scene_id").sort_index()
    return base, lora


# ---------------------------------------------------------------------------
# Chart 1 — Grouped bar chart (mean ± std per metric)
# ---------------------------------------------------------------------------

def plot_metric_bar(df: pd.DataFrame, out_dir: Path) -> None:
    metrics = list(METRICS_META.keys())
    n = len(metrics)
    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, label in enumerate(["Baseline", "LoRA"]):
        sub  = df[df["label"] == label]
        means = [sub[m].mean() for m in metrics]
        stds  = [sub[m].std()  for m in metrics]
        bars = ax.bar(
            x + (i - 0.5) * width, means, width,
            yerr=stds, capsize=4,
            color=COLORS[label], label=label, alpha=0.88,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{METRICS_META[m][0]}\n({METRICS_META[m][1]})" for m in metrics]
    )
    ax.set_ylabel("Score")
    ax.set_title("Evaluation Metrics — Baseline vs Dual-LoRA (mean ± std, N=31)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "01_metric_comparison_bar.png")
    plt.close(fig)
    print(f"  Saved 01_metric_comparison_bar.png")


# ---------------------------------------------------------------------------
# Chart 2 — Box plots
# ---------------------------------------------------------------------------

def plot_boxplot(df: pd.DataFrame, out_dir: Path) -> None:
    metrics = list(METRICS_META.keys())
    n = len(metrics)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))

    for ax, m in zip(axes, metrics):
        data_base = df[df["label"] == "Baseline"][m].dropna().values
        data_lora = df[df["label"] == "LoRA"][m].dropna().values

        bp = ax.boxplot(
            [data_base, data_lora],
            tick_labels=["Baseline", "LoRA"],
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
        )
        for patch, color in zip(bp["boxes"], COLORS.values()):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)

        name, direction, _ = METRICS_META[m]
        ax.set_title(f"{name} ({direction})")
        ax.set_ylabel(name)

    fig.suptitle("Score Distributions — Baseline vs Dual-LoRA", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "02_metric_boxplot.png")
    plt.close(fig)
    print(f"  Saved 02_metric_boxplot.png")


# ---------------------------------------------------------------------------
# Chart 3 — Per-scene delta heatmap (LoRA − Baseline)
# ---------------------------------------------------------------------------

def plot_delta_heatmap(base: pd.DataFrame, lora: pd.DataFrame, out_dir: Path) -> None:
    metrics = list(METRICS_META.keys())
    common  = base.index.intersection(lora.index)
    base_v  = base.loc[common, metrics]
    lora_v  = lora.loc[common, metrics]

    # Flip sign for LPIPS so that "positive delta" always means LoRA is better
    delta = lora_v - base_v
    delta["lpips"] = -delta["lpips"]

    col_labels = [
        f"{METRICS_META[m][0]}\n(LoRA−Base{'↑' if METRICS_META[m][2] > 0 else ', flipped↑'})"
        for m in metrics
    ]

    fig, ax = plt.subplots(figsize=(len(metrics) * 1.8, len(common) * 0.38 + 1.5))
    vmax = delta.abs().max().max()
    im = ax.imshow(delta.values, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(len(common)))
    ax.set_yticklabels(common, fontsize=8)
    ax.set_title("Per-scene LoRA − Baseline Delta\n(green = LoRA better, red = Baseline better)")

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_dir / "03_per_scene_heatmap.png")
    plt.close(fig)
    print(f"  Saved 03_per_scene_heatmap.png")


# ---------------------------------------------------------------------------
# Chart 4 — CLIP Score vs LPIPS scatter
# ---------------------------------------------------------------------------

def plot_scatter(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    for label in ["Baseline", "LoRA"]:
        sub = df[df["label"] == label]
        ax.scatter(
            sub["lpips"], sub["clip_score"],
            color=COLORS[label], label=label,
            alpha=0.75, s=60, edgecolors="white", linewidths=0.5,
        )

    ax.set_xlabel("LPIPS (↓ better)")
    ax.set_ylabel("CLIP Score (↑ better)")
    ax.set_title("CLIP Score vs LPIPS — per scene")
    ax.legend()

    # Add quadrant annotation
    ax.axvline(df["lpips"].median(), color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(df["clip_score"].median(), color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_dir / "04_scatter_clip_vs_lpips.png")
    plt.close(fig)
    print(f"  Saved 04_scatter_clip_vs_lpips.png")


# ---------------------------------------------------------------------------
# Chart 5 — Radar chart
# ---------------------------------------------------------------------------

def plot_radar(df: pd.DataFrame, out_dir: Path) -> None:
    metrics = list(METRICS_META.keys())
    n = len(metrics)

    # Normalise each metric to [0, 1] across both methods combined,
    # flip LPIPS so that higher = better on the radar
    normed: dict[str, dict[str, float]] = {"Baseline": {}, "LoRA": {}}
    for m in metrics:
        vals = df[m].dropna()
        lo, hi = vals.min(), vals.max()
        span = hi - lo if hi != lo else 1.0
        _, _, direction = METRICS_META[m]
        for label in ["Baseline", "LoRA"]:
            mean = df[df["label"] == label][m].mean()
            norm = (mean - lo) / span
            if direction < 0:
                norm = 1.0 - norm  # flip so bigger = better on radar
            normed[label][m] = norm

    angles = [n / float(n) * 2 * math.pi * i for i in range(n)]
    angles += angles[:1]

    labels = [f"{METRICS_META[m][0]}\n({METRICS_META[m][1]})" for m in metrics]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for label in ["Baseline", "LoRA"]:
        vals = [normed[label][m] for m in metrics] + [normed[label][metrics[0]]]
        ax.plot(angles, vals, color=COLORS[label], linewidth=2, label=label)
        ax.fill(angles, vals, color=COLORS[label], alpha=0.18)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels([])
    ax.set_title("Radar Chart — Normalised Metric Summary", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))

    fig.tight_layout()
    fig.savefig(out_dir / "05_radar_chart.png")
    plt.close(fig)
    print(f"  Saved 05_radar_chart.png")


# ---------------------------------------------------------------------------
# Chart 6 — Per-scene LoRA improvement delta (sorted by LPIPS improvement)
# ---------------------------------------------------------------------------

def plot_improvement_delta(base: pd.DataFrame, lora: pd.DataFrame, out_dir: Path) -> None:
    common = base.index.intersection(lora.index)

    # LPIPS: lower is better, so improvement = base − lora (positive = LoRA better)
    lpips_delta = (base.loc[common, "lpips"] - lora.loc[common, "lpips"])
    lpips_delta_sorted = lpips_delta.sort_values(ascending=False)

    psnr_delta = (lora.loc[lpips_delta_sorted.index, "psnr"] -
                  base.loc[lpips_delta_sorted.index, "psnr"])
    clip_delta = (lora.loc[lpips_delta_sorted.index, "clip_score"] -
                  base.loc[lpips_delta_sorted.index, "clip_score"])

    scenes = lpips_delta_sorted.index.tolist()
    x = np.arange(len(scenes))

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    for ax, delta, title, color in [
        (axes[0], lpips_delta_sorted.values, "LPIPS improvement (Base − LoRA, ↑ better)", "#E07B54"),
        (axes[1], psnr_delta.values,        "PSNR improvement (LoRA − Base dB, ↑ better)",   "#5B8DB8"),
        (axes[2], clip_delta.values,        "CLIP Score improvement (LoRA − Base, ↑ better)", "#7DBD8A"),
    ]:
        bar_colors = [color if v >= 0 else "#CCCCCC" for v in delta]
        ax.bar(x, delta, color=bar_colors, edgecolor="white", linewidth=0.4)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhline(np.mean(delta), color="red", linewidth=1.2,
                   linestyle="--", label=f"mean={np.mean(delta):.3f}")
        ax.set_ylabel("Δ Score")
        ax.set_title(title)
        ax.legend(fontsize=9)

    axes[2].set_xticks(x)
    axes[2].set_xticklabels(scenes, rotation=45, ha="right", fontsize=8)

    fig.suptitle("Per-scene Improvement: LoRA vs Baseline", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "06_improvement_delta.png")
    plt.close(fig)
    print(f"  Saved 06_improvement_delta.png")


# ---------------------------------------------------------------------------
# Summary table printed to stdout
# ---------------------------------------------------------------------------

def print_summary_table(df: pd.DataFrame) -> None:
    metrics = list(METRICS_META.keys())
    print("\n" + "=" * 70)
    print(f"{'Metric':<22} {'Baseline':>16} {'LoRA':>16} {'Delta':>12}")
    print("-" * 70)
    for m in metrics:
        name, direction, sign = METRICS_META[m]
        b_mean = df[df["label"] == "Baseline"][m].mean()
        b_std  = df[df["label"] == "Baseline"][m].std()
        l_mean = df[df["label"] == "LoRA"][m].mean()
        l_std  = df[df["label"] == "LoRA"][m].std()
        delta  = (l_mean - b_mean) * sign  # positive = LoRA better
        marker = "✓" if delta > 0 else "✗"
        print(
            f"{name+' ('+direction+')':22}"
            f" {b_mean:7.4f}±{b_std:.4f}"
            f" {l_mean:7.4f}±{l_std:.4f}"
            f"  {delta:+.4f} {marker}"
        )
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--csv", default="eval/results/test_metrics.csv",
        help="Path to test_metrics.csv produced by eval_runner.py",
    )
    p.add_argument(
        "--out", default="eval/results/figures",
        help="Output directory for figure PNGs",
    )
    return p.parse_args()


def main() -> None:
    args  = parse_args()
    csv_path = Path(args.csv)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Viz] Loading {csv_path}")
    df = load_data(csv_path)
    base, lora = split_labels(df)

    print(f"[Viz] {len(base)} baseline scenes, {len(lora)} LoRA scenes")
    print_summary_table(df)

    print("[Viz] Generating figures …")
    plot_metric_bar(df, out_dir)
    plot_boxplot(df, out_dir)
    plot_delta_heatmap(base, lora, out_dir)
    plot_scatter(df, out_dir)
    plot_radar(df, out_dir)
    plot_improvement_delta(base, lora, out_dir)

    print(f"\n[Viz] All figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
