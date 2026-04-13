#!/usr/bin/env python3
"""
Annotated eval heatmap — clearer labels, family grouping, legend.
Usage:
    python experiments/plot_heatmap_annotated.py outputs/exp_main
    python experiments/plot_heatmap_annotated.py outputs/exp_ext
"""

import json
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ── colour palette ────────────────────────────────────────────────────────────
FAMILY_META = {
    "baseline":      {"color": "#3b82f6", "marker": "■", "label": "Baseline"},
    "annex_heavy":   {"color": "#ef4444", "marker": "▲", "label": "Annex-Heavy"},
    "peace_heavy":   {"color": "#10b981", "marker": "●", "label": "Peace-Heavy"},
    "dense_only":    {"color": "#f59e0b", "marker": "◆", "label": "Dense-Only"},
    "terminal_only": {"color": "#8b5cf6", "marker": "★", "label": "Terminal-Only"},
    "other":         {"color": "#94a3b8", "marker": "?", "label": "Other"},
}

# Opponent pool → short readable label
OPP_LABEL = {
    "balanced":    "Balanced x4\n(all-rounder)",
    "aggressor":   "Aggressor x4\n(warmonger)",
    "diverse":     "Diverse mix\n(Agg+Turtle+Sci+Dip)",
    "diplomat":    "Diplomat x4\n(alliance-focused)",
    "diverse_200t":"Diverse mix\n(200 turns / long game)",
    "seed1":       "Diverse mix\n(seed 1 — variance check)",
    "seed2":       "Diverse mix\n(seed 2 — variance check)",
}

# Eval column → header text (two lines)
COL_HEADER = {
    "vs_balanced":  "Test:\nBalanced ×4",
    "vs_aggressor": "Test:\nAggressor ×4",
    "vs_diverse":   "Test:\nDiverse mix",
    "vs_diplomat":  "Test:\nDiplomat ×4",
}
SUITES = ["vs_balanced", "vs_aggressor", "vs_diverse", "vs_diplomat"]


def infer_family(name: str) -> str:
    if name.startswith("base_"):
        return "baseline"
    if name.startswith("annex_"):
        return "annex_heavy"
    if name.startswith("peace_"):
        return "peace_heavy"
    if name.startswith("dense_only"):
        return "dense_only"
    if name.startswith("terminal_only"):
        return "terminal_only"
    return "other"


def infer_opp_key(name: str) -> str:
    if "200t" in name:
        return "diverse_200t"
    if "seed1" in name:
        return "seed1"
    if "seed2" in name:
        return "seed2"
    for k in ("balanced", "aggressor", "diverse", "diplomat"):
        if name.endswith(k):
            return k
    return name


def row_label(name: str) -> str:
    opp = OPP_LABEL.get(infer_opp_key(name), name)
    return opp


# ── fixed group order (so same-family rows stay together) ─────────────────────
GROUP_ORDER = ["baseline", "annex_heavy", "peace_heavy", "dense_only", "terminal_only", "other"]


def sort_key(name):
    fam = infer_family(name)
    fi = GROUP_ORDER.index(fam)
    # within family keep a stable sub-order
    for i, opp in enumerate(["balanced", "aggressor", "diverse", "diplomat",
                              "seed1", "seed2", "diverse_200t"]):
        if opp in name:
            return (fi, i)
    return (fi, 99)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    exp_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/exp_main")
    json_path = exp_root / "eval_matrix.json"
    if not json_path.exists():
        print(f"ERROR: {json_path} not found"); return

    records = json.loads(json_path.read_text())
    records.sort(key=lambda r: sort_key(r["run_name"]))

    n_rows = len(records)
    n_cols = len(SUITES)

    # Build matrix + average column
    matrix = np.zeros((n_rows, n_cols))
    for i, rec in enumerate(records):
        for j, s in enumerate(SUITES):
            matrix[i, j] = rec["eval"].get(s, {}).get("win_rate", 0.0)
    avg_col = matrix.mean(axis=1)

    # ── layout ────────────────────────────────────────────────────────────────
    # Columns: [family-band | row-labels | heatmap cols… | avg | colorbar space]
    fig_w = 16
    fig_h = max(7, 0.45 * n_rows + 4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_visible(False)

    # Manual axes positions (fractions of fig)
    left_band  = 0.01   # width of family colour band
    left_label = 0.02   # gap
    label_w    = 0.19   # row-label column width
    hmap_left  = left_band + left_label + label_w
    hmap_w     = 0.46
    avg_gap    = 0.01
    avg_w      = 0.05
    cbar_gap   = 0.01
    cbar_w     = 0.015
    top        = 0.86   # just below the legend
    bottom     = 0.14
    hmap_h     = top - bottom

    ax_band  = fig.add_axes([0.005, bottom, left_band, hmap_h])
    ax_label = fig.add_axes([left_band + left_label, bottom, label_w, hmap_h])
    ax_hmap  = fig.add_axes([hmap_left, bottom, hmap_w, hmap_h])
    ax_avg   = fig.add_axes([hmap_left + hmap_w + avg_gap, bottom, avg_w, hmap_h])
    ax_cbar  = fig.add_axes([hmap_left + hmap_w + avg_gap + avg_w + cbar_gap,
                              bottom, cbar_w, hmap_h])

    # ── family colour band ────────────────────────────────────────────────────
    ax_band.set_xlim(0, 1); ax_band.set_ylim(0, n_rows)
    ax_band.axis("off")
    prev_fam = None
    fam_spans = []   # (family, row_start, row_end)
    start = 0
    for i, rec in enumerate(records):
        fam = infer_family(rec["run_name"])
        if fam != prev_fam:
            if prev_fam is not None:
                fam_spans.append((prev_fam, start, i))
            prev_fam = fam; start = i
    fam_spans.append((prev_fam, start, n_rows))

    for fam, rs, re in fam_spans:
        color = FAMILY_META[fam]["color"]
        # bar from bottom (row 0 is top visually, so invert)
        y0 = n_rows - re
        ax_band.add_patch(mpatches.FancyBboxPatch(
            (0.1, y0 + 0.05), 0.8, (re - rs) - 0.1,
            boxstyle="round,pad=0.0", linewidth=0,
            facecolor=color, alpha=0.85))

    # ── row labels ────────────────────────────────────────────────────────────
    ax_label.set_xlim(0, 1); ax_label.set_ylim(0, n_rows)
    ax_label.axis("off")

    # separator lines between families
    sep_rows = set()
    prev = None
    for i, rec in enumerate(records):
        fam = infer_family(rec["run_name"])
        if prev and fam != prev:
            sep_rows.add(i)
        prev = fam

    for sep in sep_rows:
        y = n_rows - sep
        ax_label.axhline(y, color="#cccccc", linewidth=0.8, xmin=0, xmax=1.02)

    for i, rec in enumerate(records):
        name = rec["run_name"]
        fam  = infer_family(name)
        color = FAMILY_META[fam]["color"]
        # row centre in data coords (rows run top→bottom)
        yc = n_rows - i - 0.5
        label = row_label(name)
        ax_label.text(0.97, yc, label, ha="right", va="center",
                      fontsize=8.5, color="#111111",
                      linespacing=1.3)
        # small coloured dot on left
        ax_label.plot(0.02, yc, "o", color=color, markersize=6)

    # ── heatmap ───────────────────────────────────────────────────────────────
    im = ax_hmap.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1,
                        aspect="auto", origin="upper")
    ax_hmap.set_xticks(range(n_cols))
    ax_hmap.set_xticklabels([COL_HEADER[s] for s in SUITES],
                             fontsize=9, linespacing=1.35)
    ax_hmap.tick_params(axis="x", which="both", bottom=False,
                        top=True, labelbottom=False, labeltop=True)
    ax_hmap.set_yticks([])

    # cell text
    for i in range(n_rows):
        for j in range(n_cols):
            v = matrix[i, j]
            txt_color = "black" if 0.2 < v < 0.8 else ("white" if v <= 0.2 else "black")
            ax_hmap.text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=8, color=txt_color, fontweight="bold")

    # separator lines between families (on heatmap)
    for sep in sep_rows:
        ax_hmap.axhline(sep - 0.5, color="white", linewidth=1.5)

    # ── average column ────────────────────────────────────────────────────────
    avg_mat = avg_col.reshape(-1, 1)
    ax_avg.imshow(avg_mat, cmap="RdYlGn", vmin=0, vmax=1,
                  aspect="auto", origin="upper")
    ax_avg.set_xticks([0])
    ax_avg.set_xticklabels(["Avg\nWin%"], fontsize=9)
    ax_avg.tick_params(axis="x", bottom=False, top=True,
                       labelbottom=False, labeltop=True)
    ax_avg.set_yticks([])
    for i, v in enumerate(avg_col):
        tc = "black" if 0.2 < v < 0.8 else ("white" if v <= 0.2 else "black")
        ax_avg.text(0, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color=tc, fontweight="bold")
    for sep in sep_rows:
        ax_avg.axhline(sep - 0.5, color="white", linewidth=1.5)

    # ── colorbar ──────────────────────────────────────────────────────────────
    cb = fig.colorbar(im, cax=ax_cbar)
    cb.set_label("Win Rate", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    # ── legend: reward families ───────────────────────────────────────────────
    # which families actually appear?
    seen_fams = []
    for fam in GROUP_ORDER:
        if any(infer_family(r["run_name"]) == fam for r in records):
            seen_fams.append(fam)

    reward_patches = [
        mpatches.Patch(facecolor=FAMILY_META[f]["color"],
                       label=f"{FAMILY_META[f]['marker']} {FAMILY_META[f]['label']}")
        for f in seen_fams
    ]

    # Opponent type explanation (bottom)
    opp_lines = [
        "Training opponents (row labels) — rule-based AI strategies:",
        "  Balanced x4   all four opponents use balanced strategy (mixed economy/military/diplomacy)",
        "  Aggressor x4  all four use aggressor: declares war early (turn 20+), no diplomacy",
        "  Diverse mix   one each: Aggressor + Turtle (defensive) + Scientist + Diplomat",
        "  Diplomat x4   all four use diplomat: no war, maximises alliances and trade",
        "  200 turns     same as Diverse mix but max_turns=200 (long-game cross-check)",
        "  seed 1/2      same config as Diverse mix, different random seeds (variance check)",
        "",
        "Evaluation columns: same four opponent pools applied to EVERY trained policy.",
        "Cell value = win rate over 20 deterministic episodes  (0.00 = all losses, 1.00 = all wins).",
    ]

    fig.text(0.01, 0.12, "\n".join(opp_lines),
             fontsize=8, va="top", ha="left",
             family="monospace", color="#333333",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f8f8",
                       edgecolor="#cccccc", linewidth=0.8))

    # ── title (very top) ──────────────────────────────────────────────────────
    fig.suptitle(
        "Eval Win Rate:  Trained Policy  ×  Opponent Suite",
        fontsize=12, fontweight="bold", y=0.995, va="top")

    fig.text(0.5, 0.968,
             "Rows = trained policies (grouped by reward family, labelled by training opponent)  |  "
             "Columns = evaluation suites applied to every policy",
             fontsize=8.5, ha="center", va="top", color="#444444")

    # ── legend — tight below subtitle ─────────────────────────────────────────
    leg = fig.legend(handles=reward_patches,
                     title="Reward Family (row dot colour)",
                     loc="upper center",
                     bbox_to_anchor=(0.5, 0.948),
                     fontsize=8.5, title_fontsize=9,
                     framealpha=0.9, edgecolor="#cccccc",
                     ncol=len(seen_fams))

    out = exp_root / "figures" / "eval_heatmap_annotated.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
