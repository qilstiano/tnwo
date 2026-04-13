#!/usr/bin/env python3
"""
Publication-quality training curves for the TNWO experiment matrix.

For each reward family we plot:
    • individual runs as thin low-alpha traces (kept for transparency)
    • a smoothed group mean as a solid thick line
    • a ±1 s.d. band around that mean

Three panels are produced in one row:
    1. Rolling win rate (20)
    2. Rolling episodic return (20)
    3. Rolling final score (20)

Usage:
    python experiments/plot_training_curves_pub.py outputs/exp_main
    python experiments/plot_training_curves_pub.py outputs/exp_main --smooth 40
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


sys_path_me = Path(__file__).resolve().parent
sys.path.insert(0, str(sys_path_me))
from _pub_style import (  # noqa: E402
    FAMILY_META,
    FAMILY_ORDER as ORDER,
    infer_family,
    set_pub_style as _shared_set_pub_style,
)


# ── data loading ──────────────────────────────────────────────────────────────
def load_curve(path: Path) -> Dict[str, List[float]]:
    data = defaultdict(list)
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                try:
                    data[k].append(float(v))
                except (TypeError, ValueError):
                    pass
    return dict(data)


def load_all(exp_root: Path):
    runs = []
    for d in sorted(exp_root.iterdir()):
        if not d.is_dir():
            continue
        csv_path = d / "training_curve.csv"
        if not csv_path.exists():
            continue
        curve = load_curve(csv_path)
        if "episode" not in curve:
            continue
        runs.append({
            "name": d.name,
            "family": infer_family(d.name),
            "curve": curve,
        })
    return runs


# ── smoothing helper ──────────────────────────────────────────────────────────
def moving_avg(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(x) < w:
        return x
    # symmetric moving average using convolution; pad via edge replication
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(w) / w
    return np.convolve(xp, kernel, mode="valid")[: len(x)]


# ── resample runs to common grid ──────────────────────────────────────────────
def aggregate_family(runs: List[dict], key: str, n_points: int = 400):
    """
    Returns (common_x, runs_y_resampled[list of arrays]) where every run has
    been linearly interpolated onto the same episode grid that spans the
    intersection of all runs in the family (so the band is defined everywhere).
    """
    valid = [r for r in runs if key in r["curve"] and len(r["curve"][key]) > 5]
    if not valid:
        return None, []

    x_max = min(max(r["curve"]["episode"]) for r in valid)
    x_min = max(min(r["curve"]["episode"]) for r in valid)
    if x_max <= x_min:
        return None, []

    grid = np.linspace(x_min, x_max, n_points)
    resampled = []
    for r in valid:
        xs = np.asarray(r["curve"]["episode"])
        ys = np.asarray(r["curve"][key])
        # drop any non-monotonic duplicates
        order = np.argsort(xs)
        xs = xs[order]; ys = ys[order]
        resampled.append(np.interp(grid, xs, ys))
    return grid, resampled


# ── styling ───────────────────────────────────────────────────────────────────
def set_pub_style():
    _shared_set_pub_style()
    # training curves need a subtle grid
    mpl.rcParams.update({
        "axes.grid":      True,
        "grid.alpha":     0.28,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
    })


# ── plotting ──────────────────────────────────────────────────────────────────
PANELS = [
    ("rolling_winrate_20", "Rolling win rate",    "Win rate"),
    ("rolling_reward_20",  "Rolling return",      "Episodic return"),
    ("rolling_score_20",   "Rolling final score", "Final score"),
]


def plot_curves(runs, out_path_png: Path, out_path_pdf: Path, smooth: int):
    set_pub_style()

    # Group runs by family, preserving the canonical ORDER
    by_family: Dict[str, List[dict]] = {f: [] for f in ORDER}
    for r in runs:
        by_family[r["family"]].append(r)
    present_families = [f for f in ORDER if by_family[f]]

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 5.0))

    for ax, (key, title, ylabel) in zip(axes, PANELS):
        for fam in present_families:
            fam_runs = by_family[fam]
            color = FAMILY_META[fam]["color"]

            grid, resampled = aggregate_family(fam_runs, key, n_points=400)
            if grid is None:
                continue

            # individual runs — thin, faint, for transparency
            for y in resampled:
                y_smooth = moving_avg(y, smooth)
                ax.plot(grid, y_smooth,
                        color=color, alpha=0.18, linewidth=0.9,
                        zorder=2)

            # group statistics
            stack = np.stack([moving_avg(y, smooth) for y in resampled])
            mean = stack.mean(axis=0)
            std = stack.std(axis=0) if stack.shape[0] > 1 else np.zeros_like(mean)

            if stack.shape[0] > 1:
                ax.fill_between(grid, mean - std, mean + std,
                                color=color, alpha=0.18,
                                linewidth=0, zorder=3)
            ax.plot(grid, mean, color=color, linewidth=2.2,
                    label=FAMILY_META[fam]["label"], zorder=4,
                    solid_capstyle="round")

        ax.set_title(title, pad=8, fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.margins(x=0.01)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    # y-limits: win rate is a probability
    axes[0].set_ylim(-0.03, 1.05)
    axes[0].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

    # legend — centered below all three panels
    handles, labels = axes[0].get_legend_handles_labels()
    # add a note handle explaining the band
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    extra = [
        Line2D([], [], color="#555555", linewidth=2.2, label="group mean"),
        Patch(facecolor="#555555", alpha=0.22, label=r"$\pm 1$ s.d. band"),
        Line2D([], [], color="#555555", alpha=0.35, linewidth=0.9,
               label="individual run"),
    ]
    leg1 = fig.legend(handles=handles, labels=labels,
                      title="Reward family",
                      loc="lower center",
                      bbox_to_anchor=(0.30, 0.005),
                      ncol=len(handles), frameon=True, framealpha=0.95,
                      edgecolor="#cccccc")
    leg1.get_title().set_fontweight("bold")
    leg2 = fig.legend(handles=extra,
                      loc="lower center",
                      bbox_to_anchor=(0.79, 0.005),
                      ncol=3, frameon=True, framealpha=0.95,
                      edgecolor="#cccccc")

    fig.suptitle("Training curves grouped by reward family",
                 fontsize=14, fontweight="bold", y=0.99)

    fig.subplots_adjust(left=0.055, right=0.99, top=0.87, bottom=0.22, wspace=0.24)
    # Use bbox_inches=None here so our subplots_adjust spacing is preserved.
    fig.savefig(out_path_png, bbox_inches=None)
    fig.savefig(out_path_pdf, bbox_inches=None)
    plt.close(fig)
    print(f"Saved → {out_path_png}")
    print(f"Saved → {out_path_pdf}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_root")
    parser.add_argument("--smooth", type=int, default=40,
                        help="Extra moving-average window on top of the rolling-20 metric")
    args = parser.parse_args()

    exp_root = Path(args.exp_root)
    if not exp_root.exists():
        print(f"ERROR: {exp_root} not found"); return

    runs = load_all(exp_root)
    print(f"Loaded {len(runs)} runs from {exp_root}")

    fig_dir = exp_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_png = fig_dir / "training_curves_pub.png"
    out_pdf = fig_dir / "training_curves_pub.pdf"
    plot_curves(runs, out_png, out_pdf, smooth=args.smooth)


if __name__ == "__main__":
    main()
