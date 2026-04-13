#!/usr/bin/env python3
"""
Publication-quality strategy-fingerprint figures.

Consumes the `fingerprints_full.json` produced by `plot_fingerprint_full.py`
(one action-category distribution per (policy, suite) cell) and emits:

    outputs/<exp>/figures/fingerprint_grid_pub.{png,pdf}
        2 × 2 grid with one per-suite panel. Single paper-ready figure
        replacing the old cluttered 4-bars-per-policy comparison.

    outputs/<exp>/figures/fingerprint_<suite>_pub.{png,pdf}
        One clean per-suite figure for each of the four evaluation suites.

Usage:
    python experiments/plot_fingerprint_pub.py outputs/exp_main
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pub_style import (  # noqa: E402
    FAMILY_META, FAMILY_ORDER,
    CATEGORY_META, CATEGORY_ORDER,
    SUITE_META, SUITE_ORDER,
    OPP_LABEL,
    infer_family, infer_opponent, sort_key,
    set_pub_style,
)


def load_fingerprints(exp_root: Path):
    path = exp_root / "fingerprints_full.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run plot_fingerprint_full.py first")
    return json.loads(path.read_text())


def sorted_run_names(fingerprints):
    names = list(fingerprints.keys())
    names.sort(key=sort_key)
    return names


def family_separators(names):
    sep = set()
    prev = None
    for i, n in enumerate(names):
        fam = infer_family(n)
        if prev is not None and fam != prev:
            sep.add(i)
        prev = fam
    return sep


def build_matrix(fingerprints, names, suite_key):
    mat = np.zeros((len(names), len(CATEGORY_ORDER)))
    for i, n in enumerate(names):
        fp = fingerprints.get(n, {}).get(suite_key, {})
        for j, c in enumerate(CATEGORY_ORDER):
            mat[i, j] = fp.get(c, 0.0)
    return mat


def row_labels_for(names):
    """Short per-row label: training-opponent short name, with family tint."""
    labels = []
    for n in names:
        opp = infer_opponent(n)
        labels.append(OPP_LABEL.get(opp, opp))
    return labels


# ── core stacked-bar panel drawing ────────────────────────────────────────────
def draw_fingerprint_panel(ax, matrix, names, sep_rows,
                           show_ylabels=True,
                           suite_key=None,
                           annotate_threshold=0.06,
                           label_fontsize=9.5,
                           bar_fontsize=7.5):
    n = matrix.shape[0]
    y = np.arange(n)

    # Stacked bars
    left = np.zeros(n)
    for j, cat in enumerate(CATEGORY_ORDER):
        vals = matrix[:, j]
        ax.barh(y, vals, left=left,
                color=CATEGORY_META[cat]["color"],
                edgecolor="white", linewidth=0.45, height=0.78)
        for i, (v, l) in enumerate(zip(vals, left)):
            if v >= annotate_threshold:
                ax.text(l + v / 2, i, f"{v:.2f}",
                        ha="center", va="center",
                        fontsize=bar_fontsize, color="white",
                        fontweight="bold")
        left += vals

    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", ".25", ".5", ".75", "1"])
    ax.set_xlabel("Action share", fontsize=10.5)
    ax.set_yticks(y)
    if show_ylabels:
        labels = row_labels_for(names)
        ax.set_yticklabels(labels, fontsize=label_fontsize)
        # Tint tick labels by reward family
        for i, (lab, tick) in enumerate(zip(names, ax.get_yticklabels())):
            tick.set_color(FAMILY_META[infer_family(lab)]["color"])
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#555555")
        ax.spines[spine].set_linewidth(0.8)

    # Subtle family separators
    for s in sep_rows:
        ax.axhline(s - 0.5, color="#bbbbbb", linewidth=0.8, linestyle=(0, (4, 3)))

    if suite_key is not None:
        col = SUITE_META[suite_key]["color"]
        ax.set_title(SUITE_META[suite_key]["label"],
                     fontsize=12, fontweight="bold", color=col, pad=6)


# ── legends ───────────────────────────────────────────────────────────────────
def _category_patches():
    return [mpatches.Patch(facecolor=CATEGORY_META[c]["color"],
                           label=f"{CATEGORY_META[c]['abbr']} — {CATEGORY_META[c]['label']}")
            for c in CATEGORY_ORDER]


def _family_patches(names_present):
    fams_used = {infer_family(n) for n in names_present}
    return [mpatches.Patch(facecolor=FAMILY_META[f]["color"],
                           label=FAMILY_META[f]["label"])
            for f in FAMILY_ORDER if f in fams_used]


# ── single-suite figure ───────────────────────────────────────────────────────
def plot_single_suite(fingerprints, names, sep_rows, suite_key,
                      fig_dir: Path):
    set_pub_style()
    mat = build_matrix(fingerprints, names, suite_key)

    fig_h = max(6.2, 0.32 * len(names) + 3.0)
    fig, ax = plt.subplots(figsize=(10.0, fig_h))

    # Suppress the panel subtitle — suptitle already carries the suite name.
    draw_fingerprint_panel(ax, mat, names, sep_rows,
                           show_ylabels=True, suite_key=None)

    # Shared legends at the bottom in two well-separated columns
    cat_leg = fig.legend(handles=_category_patches(),
                         title="Action category",
                         loc="lower left",
                         bbox_to_anchor=(0.08, 0.005),
                         ncol=2, frameon=True, framealpha=0.95,
                         edgecolor="#cccccc", fontsize=9.5,
                         title_fontsize=10)
    cat_leg.get_title().set_fontweight("bold")

    fam_leg = fig.legend(handles=_family_patches(names),
                         title="Reward family (label color)",
                         loc="lower right",
                         bbox_to_anchor=(0.98, 0.005),
                         ncol=3, frameon=True, framealpha=0.95,
                         edgecolor="#cccccc", fontsize=9.5,
                         title_fontsize=10)
    fam_leg.get_title().set_fontweight("bold")

    suite_color = SUITE_META[suite_key]["color"]
    fig.suptitle(
        f"Strategy fingerprint — {SUITE_META[suite_key]['label']}",
        fontsize=13.5, fontweight="bold", y=0.985, color=suite_color,
    )
    fig.subplots_adjust(left=0.22, right=0.98, top=0.93, bottom=0.19)

    out_png = fig_dir / f"fingerprint_{suite_key}_pub.png"
    out_pdf = fig_dir / f"fingerprint_{suite_key}_pub.pdf"
    fig.savefig(out_png, bbox_inches=None)
    fig.savefig(out_pdf, bbox_inches=None)
    plt.close(fig)
    print(f"Saved → {out_png}")


# ── 2×2 grid figure (replaces the old "full comparison") ──────────────────────
def plot_grid(fingerprints, names, sep_rows, fig_dir: Path):
    set_pub_style()

    n = len(names)
    fig_h = max(10.5, 0.42 * n + 4.0)
    fig, axes = plt.subplots(2, 2, figsize=(14.0, fig_h))

    for (r, c), suite_key in zip(
        [(0, 0), (0, 1), (1, 0), (1, 1)], SUITE_ORDER
    ):
        ax = axes[r, c]
        mat = build_matrix(fingerprints, names, suite_key)
        is_left = (c == 0)
        draw_fingerprint_panel(
            ax, mat, names, sep_rows,
            show_ylabels=is_left,
            suite_key=suite_key,
            label_fontsize=9.5,
            bar_fontsize=7.2,
        )

    # Shared legends at the top, between title and panels
    cat_leg = fig.legend(handles=_category_patches(),
                         title="Action category",
                         loc="upper center",
                         bbox_to_anchor=(0.28, 0.955),
                         ncol=2, frameon=True, framealpha=0.95,
                         edgecolor="#cccccc", fontsize=9.5,
                         title_fontsize=10)
    cat_leg.get_title().set_fontweight("bold")

    fam_leg = fig.legend(handles=_family_patches(names),
                         title="Reward family (label color)",
                         loc="upper center",
                         bbox_to_anchor=(0.70, 0.955),
                         ncol=3, frameon=True, framealpha=0.95,
                         edgecolor="#cccccc", fontsize=9.5,
                         title_fontsize=10)
    fam_leg.get_title().set_fontweight("bold")

    fig.suptitle("Strategy fingerprint — all policies × all evaluation suites",
                 fontsize=14, fontweight="bold", y=0.995)

    fig.subplots_adjust(left=0.13, right=0.985, top=0.84, bottom=0.06,
                        wspace=0.30, hspace=0.24)

    out_png = fig_dir / "fingerprint_grid_pub.png"
    out_pdf = fig_dir / "fingerprint_grid_pub.pdf"
    fig.savefig(out_png, bbox_inches=None)
    fig.savefig(out_pdf, bbox_inches=None)
    plt.close(fig)
    print(f"Saved → {out_png}")
    print(f"Saved → {out_pdf}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_root")
    parser.add_argument("--only-grid", action="store_true",
                        help="Only produce the 2x2 grid figure, skip per-suite files")
    args = parser.parse_args()

    exp_root = Path(args.exp_root)
    fingerprints = load_fingerprints(exp_root)
    names = sorted_run_names(fingerprints)
    sep_rows = family_separators(names)

    fig_dir = exp_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loaded fingerprints for {len(names)} runs from {exp_root}")

    plot_grid(fingerprints, names, sep_rows, fig_dir)
    if not args.only_grid:
        for sk in SUITE_ORDER:
            plot_single_suite(fingerprints, names, sep_rows, sk, fig_dir)


if __name__ == "__main__":
    main()
