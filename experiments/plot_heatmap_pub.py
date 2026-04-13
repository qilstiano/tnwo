#!/usr/bin/env python3
"""
Publication-quality eval win-rate heatmap.

Produces:
    outputs/<exp>/figures/eval_heatmap_pub.png
    outputs/<exp>/figures/eval_heatmap_pub.pdf

Usage:
    python experiments/plot_heatmap_pub.py outputs/exp_main
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
    SUITE_META, SUITE_ORDER,
    OPP_LABEL,
    infer_family, infer_opponent, sort_key,
    set_pub_style,
)


def load_records(exp_root: Path):
    path = exp_root / "eval_matrix.json"
    if not path.exists():
        raise FileNotFoundError(path)
    recs = json.loads(path.read_text())
    recs.sort(key=lambda r: sort_key(r["run_name"]))
    return recs


def plot(records, out_png: Path, out_pdf: Path):
    set_pub_style()

    n_rows = len(records)
    n_cols = len(SUITE_ORDER)

    # Win-rate matrix and row-wise mean
    W = np.zeros((n_rows, n_cols))
    for i, rec in enumerate(records):
        for j, suite in enumerate(SUITE_ORDER):
            W[i, j] = rec["eval"].get(suite, {}).get("win_rate", 0.0)
    avg = W.mean(axis=1)

    # Figure layout: one axes for the heatmap + a narrow axes for the average
    # column + a colorbar. Sizes are in data fractions of the figure.
    fig_h = max(5.6, 0.34 * n_rows + 2.8)
    fig_w = 11.5
    fig = plt.figure(figsize=(fig_w, fig_h))

    left_band  = 0.018   # colored family strip
    label_w    = 0.205   # row label column
    hmap_left  = 0.070 + left_band + label_w
    hmap_w     = 0.44
    avg_gap    = 0.012
    avg_w      = 0.050
    cbar_gap   = 0.022
    cbar_w     = 0.016
    top        = 0.80
    bottom     = 0.09
    hmap_h     = top - bottom

    ax_band  = fig.add_axes([0.065, bottom, left_band, hmap_h])
    ax_label = fig.add_axes([0.065 + left_band, bottom, label_w, hmap_h])
    ax_hmap  = fig.add_axes([hmap_left, bottom, hmap_w, hmap_h])
    ax_avg   = fig.add_axes([hmap_left + hmap_w + avg_gap, bottom, avg_w, hmap_h])
    ax_cbar  = fig.add_axes([hmap_left + hmap_w + avg_gap + avg_w + cbar_gap,
                              bottom, cbar_w, hmap_h])

    # ── identify family spans ────────────────────────────────────────────────
    families = [infer_family(r["run_name"]) for r in records]
    fam_spans = []
    start = 0
    for i in range(1, n_rows + 1):
        if i == n_rows or families[i] != families[start]:
            fam_spans.append((families[start], start, i))
            start = i
    sep_rows = {s[1] for s in fam_spans[1:]}

    # ── family colored band (left) ───────────────────────────────────────────
    ax_band.set_xlim(0, 1); ax_band.set_ylim(0, n_rows)
    ax_band.axis("off")
    for fam, rs, re in fam_spans:
        color = FAMILY_META[fam]["color"]
        y0 = n_rows - re + 0.08
        y1 = n_rows - rs - 0.08
        ax_band.add_patch(mpatches.FancyBboxPatch(
            (0.12, y0), 0.76, y1 - y0,
            boxstyle="round,pad=0.0,rounding_size=0.08",
            linewidth=0, facecolor=color, alpha=0.92))

    # ── row labels ───────────────────────────────────────────────────────────
    ax_label.set_xlim(0, 1); ax_label.set_ylim(0, n_rows)
    ax_label.axis("off")
    for sep in sep_rows:
        ax_label.axhline(n_rows - sep, color="#cfcfcf",
                         linewidth=0.7, xmin=0, xmax=1.02)
    for i, rec in enumerate(records):
        name = rec["run_name"]
        opp = infer_opponent(name)
        label = OPP_LABEL.get(opp, opp)
        yc = n_rows - i - 0.5
        ax_label.text(0.98, yc, label, ha="right", va="center",
                      fontsize=10, color="#111111")

    # ── heatmap ──────────────────────────────────────────────────────────────
    cmap = plt.get_cmap("RdYlGn")
    im = ax_hmap.imshow(W, cmap=cmap, vmin=0, vmax=1,
                        aspect="auto", origin="upper")

    ax_hmap.set_xticks(range(n_cols))
    ax_hmap.set_xticklabels(
        [r"$\textsc{" + SUITE_META[s]["label"] + "}$" if False
         else SUITE_META[s]["label"] for s in SUITE_ORDER],
        fontsize=10.5)
    ax_hmap.tick_params(axis="x", which="both", bottom=False, top=True,
                        labelbottom=False, labeltop=True, pad=3)
    ax_hmap.set_yticks([])
    for spine in ax_hmap.spines.values():
        spine.set_linewidth(0.8); spine.set_edgecolor("#555555")

    for i in range(n_rows):
        for j in range(n_cols):
            v = W[i, j]
            tc = "white" if v < 0.22 or v > 0.82 else "black"
            ax_hmap.text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=9, color=tc, fontweight="bold")
    for sep in sep_rows:
        ax_hmap.axhline(sep - 0.5, color="white", linewidth=1.8)

    # Axis-top label: "Evaluation suite" — placed in figure coords above the
    # column headers so it never overlaps the legend.
    fig.text((hmap_left + hmap_w / 2), top + 0.065,
             "Evaluation suite", ha="center", va="bottom",
             fontsize=11, fontstyle="italic", color="#444444")

    # ── row-mean column ──────────────────────────────────────────────────────
    ax_avg.imshow(avg.reshape(-1, 1), cmap=cmap, vmin=0, vmax=1,
                  aspect="auto", origin="upper")
    ax_avg.set_xticks([0])
    ax_avg.set_xticklabels([r"$\bar{w}$"], fontsize=11)
    ax_avg.tick_params(axis="x", bottom=False, top=True,
                       labelbottom=False, labeltop=True, pad=3)
    ax_avg.set_yticks([])
    for spine in ax_avg.spines.values():
        spine.set_linewidth(0.8); spine.set_edgecolor("#555555")
    for i, v in enumerate(avg):
        tc = "white" if v < 0.22 or v > 0.82 else "black"
        ax_avg.text(0, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=9, color=tc, fontweight="bold")
    for sep in sep_rows:
        ax_avg.axhline(sep - 0.5, color="white", linewidth=1.8)

    # ── colorbar ─────────────────────────────────────────────────────────────
    cb = fig.colorbar(im, cax=ax_cbar)
    cb.set_label("Win rate", fontsize=10)
    cb.ax.tick_params(labelsize=9)
    cb.outline.set_linewidth(0.7)
    cb.outline.set_edgecolor("#555555")

    # ── left-side column header: "Training opponent" ─────────────────────────
    fig.text(0.065 + left_band + label_w - 0.005, top + 0.065,
             "Training opponent", ha="right", va="bottom",
             fontsize=11, fontstyle="italic", color="#444444")

    # ── family legend at top ─────────────────────────────────────────────────
    seen = [f for f in FAMILY_ORDER if f in families]
    handles = [mpatches.Patch(facecolor=FAMILY_META[f]["color"], alpha=0.92,
                              label=FAMILY_META[f]["label"])
               for f in seen]
    leg = fig.legend(handles=handles, loc="upper center",
                     bbox_to_anchor=(0.5, 0.97),
                     ncol=len(seen), fontsize=10,
                     title="Reward family", title_fontsize=10.5,
                     frameon=True, framealpha=0.95, edgecolor="#cccccc")
    leg.get_title().set_fontweight("bold")

    # ── save ─────────────────────────────────────────────────────────────────
    fig.savefig(out_png, bbox_inches=None)
    fig.savefig(out_pdf, bbox_inches=None)
    plt.close(fig)
    print(f"Saved → {out_png}")
    print(f"Saved → {out_pdf}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_root")
    args = parser.parse_args()

    exp_root = Path(args.exp_root)
    if not exp_root.exists():
        print(f"ERROR: {exp_root} not found"); return

    records = load_records(exp_root)
    print(f"Loaded {len(records)} eval records from {exp_root}")

    fig_dir = exp_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot(records, fig_dir / "eval_heatmap_pub.png",
                  fig_dir / "eval_heatmap_pub.pdf")


if __name__ == "__main__":
    main()
