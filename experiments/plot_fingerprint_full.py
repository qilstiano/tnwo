#!/usr/bin/env python3
"""
Full strategy-fingerprint sweep.

For every trained policy in an experiment folder, run deterministic rollouts
against ALL four evaluation opponent suites and record the action-category mix.

Outputs (written to <exp_root>/figures/):
    fingerprint_full.json                 — raw data
    fingerprint_full_<suite>.png          — one horizontal stacked-bar chart per suite
    fingerprint_full_comparison.png       — all suites side-by-side for every policy

Usage:
    python experiments/plot_fingerprint_full.py outputs/exp_main
    python experiments/plot_fingerprint_full.py outputs/exp_main --episodes 10
    python experiments/plot_fingerprint_full.py outputs/exp_main --skip-compute
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sb3_contrib import MaskablePPO                         # noqa: E402
from sb3_contrib.common.wrappers import ActionMasker        # noqa: E402

from rl.env import NationEnv                                # noqa: E402
from rl.action_space import build_action_catalog, command_from_action  # noqa: E402
from visualize import _classify_action, CATEGORY_ORDER, CATEGORY_COLORS  # noqa: E402

# ── suite definitions ─────────────────────────────────────────────────────────
SUITES: Dict[str, List[str]] = {
    "vs_balanced":  ["balanced", "balanced", "balanced", "balanced"],
    "vs_aggressor": ["aggressor", "aggressor", "aggressor", "aggressor"],
    "vs_diverse":   ["aggressor", "turtle", "scientist", "diplomat"],
    "vs_diplomat":  ["diplomat", "diplomat", "diplomat", "diplomat"],
}

SUITE_LABEL = {
    "vs_balanced":  "Balanced ×4",
    "vs_aggressor": "Aggressor ×4",
    "vs_diverse":   "Diverse mix",
    "vs_diplomat":  "Diplomat ×4",
}

SUITE_ICON = {
    "vs_balanced":  "■",
    "vs_aggressor": "▲",
    "vs_diverse":   "◆",
    "vs_diplomat":  "●",
}

SUITE_COLOR = {
    "vs_balanced":  "#3b82f6",
    "vs_aggressor": "#ef4444",
    "vs_diverse":   "#8b5cf6",
    "vs_diplomat":  "#10b981",
}

# ── reward-family metadata ────────────────────────────────────────────────────
FAMILY_META = {
    "baseline":      {"color": "#3b82f6", "label": "Baseline"},
    "annex_heavy":   {"color": "#ef4444", "label": "Annex-Heavy"},
    "peace_heavy":   {"color": "#10b981", "label": "Peace-Heavy"},
    "dense_only":    {"color": "#f59e0b", "label": "Dense-Only"},
    "terminal_only": {"color": "#8b5cf6", "label": "Terminal-Only"},
    "other":         {"color": "#94a3b8", "label": "Other"},
}
GROUP_ORDER = ["baseline", "annex_heavy", "peace_heavy", "dense_only", "terminal_only", "other"]


def infer_family(name: str) -> str:
    if name.startswith("base_"):     return "baseline"
    if name.startswith("annex_"):    return "annex_heavy"
    if name.startswith("peace_"):    return "peace_heavy"
    if name.startswith("dense_only"):return "dense_only"
    if name.startswith("terminal_only"): return "terminal_only"
    return "other"


def sort_key(name: str):
    fam = infer_family(name)
    fi = GROUP_ORDER.index(fam)
    for i, opp in enumerate(["balanced", "aggressor", "diverse", "diplomat",
                              "seed1", "seed2", "200t"]):
        if opp in name:
            return (fi, i)
    return (fi, 99)


# ── fingerprint computation ───────────────────────────────────────────────────

def compute_fingerprint(
    model_path: Path,
    opponent_strategies: List[str],
    episodes: int = 5,
    max_turns: int = 100,
    num_players: int = 5,
    learner_id: int = 0,
    base_seed: int = 99000,
    grace_period_turns: int = 25,
    starting_asymmetry: str = "current",
    starting_action_points: int = 3,
) -> Dict[str, float]:
    env = ActionMasker(
        NationEnv(
            num_players=num_players,
            learner_id=learner_id,
            max_turns=max_turns,
            seed=base_seed,
            opponent_mode="rulebased",
            rule_opponent_strategies=opponent_strategies,
            grace_period_turns=grace_period_turns,
            starting_asymmetry=starting_asymmetry,
            starting_action_points=starting_action_points,
        ),
        lambda e: e.action_masks(),
    )
    model = MaskablePPO.load(str(model_path), device="cpu")
    catalog = build_action_catalog(num_players)

    counts = {c: 0 for c in CATEGORY_ORDER}
    total = 0
    for ep in range(episodes):
        obs, _info = env.reset(seed=base_seed + ep * 31)
        done = trunc = False
        while not (done or trunc):
            masks = env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=masks)
            action_id = int(action)
            if not masks[action_id]:
                action_id = int(np.flatnonzero(masks)[0])
            label = command_from_action(catalog[action_id])
            counts[_classify_action(label)] += 1
            total += 1
            obs, _r, done, trunc, _info = env.step(action_id)

    return {c: (counts[c] / total if total else 0.0) for c in CATEGORY_ORDER}


# ── data loading ──────────────────────────────────────────────────────────────

def load_runs(exp_root: Path):
    runs = []
    for d in sorted(exp_root.iterdir()):
        if not d.is_dir():
            continue
        model = d / "maskable_ppo_nation.zip"
        cfg_path = d / "train_config.json"
        if not model.exists():
            continue
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        runs.append({"name": d.name, "dir": d, "config": cfg})
    return runs


# ── plot helpers ──────────────────────────────────────────────────────────────

CAT_ICONS = {
    "aggression": "⚔",
    "science":    "🔬",
    "economy":    "💰",
    "diplomacy":  "🤝",
}

# Use text-safe icons (matplotlib has no emoji font on most Linux):
CAT_ABBREV = {
    "aggression": "Agg",
    "science":    "Sci",
    "economy":    "Eco",
    "diplomacy":  "Dip",
}


def _stacked_bars(ax, names, data_matrix, title: str,
                  suite_key: str, sep_rows: set):
    """
    data_matrix: shape (n_runs, n_categories)
    """
    n = len(names)
    y = np.arange(n)
    left = np.zeros(n)

    for j, cat in enumerate(CATEGORY_ORDER):
        bars = ax.barh(y, data_matrix[:, j], left=left,
                       color=CATEGORY_COLORS[cat],
                       edgecolor="white", linewidth=0.4,
                       label=f"{CAT_ABBREV[cat]} ({cat})")
        # annotate if wide enough
        for i, (val, lft) in enumerate(zip(data_matrix[:, j], left)):
            if val > 0.06:
                ax.text(lft + val / 2, i, f"{val:.2f}",
                        ha="center", va="center", fontsize=6.5, color="white",
                        fontweight="bold")
        left += data_matrix[:, j]

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Action share", fontsize=8)
    ax.invert_yaxis()

    # family separator lines
    for sep in sep_rows:
        ax.axhline(sep - 0.5, color="#888888", linewidth=0.8, linestyle="--")

    # suite badge in title
    icon = SUITE_ICON[suite_key]
    color = SUITE_COLOR[suite_key]
    ax.set_title(f"{icon}  {title}", fontsize=10, fontweight="bold", color=color)


def plot_per_suite(results, sorted_names, sep_rows, fig_dir: Path):
    """One plot per eval suite."""
    n = len(sorted_names)
    fig_h = max(6, 0.32 * n + 3)

    for suite_key in SUITES:
        matrix = np.zeros((n, len(CATEGORY_ORDER)))
        for i, name in enumerate(sorted_names):
            fp = results.get(name, {}).get(suite_key, {})
            for j, cat in enumerate(CATEGORY_ORDER):
                matrix[i, j] = fp.get(cat, 0.0)

        fig, ax = plt.subplots(figsize=(10, fig_h))
        _stacked_bars(ax, sorted_names, matrix,
                      title=f"Opponent: {SUITE_LABEL[suite_key]}",
                      suite_key=suite_key, sep_rows=sep_rows)

        # legend: categories
        cat_patches = [mpatches.Patch(facecolor=CATEGORY_COLORS[c],
                                      label=f"{CAT_ABBREV[c]} = {c}")
                       for c in CATEGORY_ORDER]
        ax.legend(handles=cat_patches, loc="lower right", fontsize=8, ncol=2)

        fig.suptitle(
            f"Strategy fingerprint — {SUITE_LABEL[suite_key]}",
            fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out = fig_dir / f"fingerprint_{suite_key}.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out}")


def plot_comparison(results, sorted_names, sep_rows, fig_dir: Path):
    """All four suites side-by-side for every policy."""
    n = len(sorted_names)
    n_suites = len(SUITES)
    suite_keys = list(SUITES.keys())

    # Build full matrix: shape (n, n_suites, n_cats)
    full = np.zeros((n, n_suites, len(CATEGORY_ORDER)))
    for i, name in enumerate(sorted_names):
        for si, sk in enumerate(suite_keys):
            fp = results.get(name, {}).get(sk, {})
            for j, cat in enumerate(CATEGORY_ORDER):
                full[i, si, j] = fp.get(cat, 0.0)

    # Each policy gets n_suites rows; total rows = n * n_suites + gaps
    gap = 0.5          # extra space between policy groups
    bar_h = 0.18       # height of each bar in data units
    spacing = 0.22     # spacing between bars within a group

    fig_h = max(8, n * (n_suites * spacing + gap) * 0.45 + 3)
    fig, ax = plt.subplots(figsize=(12, fig_h))

    # Compute y positions
    y_positions = []
    y_labels = []       # (y, label, family_color)
    y = 0.0
    policy_band_spans = []  # (y_top, y_bottom, family_color) for colored left band

    for i, name in enumerate(sorted_names):
        fam = infer_family(name)
        fam_color = FAMILY_META[fam]["color"]
        ys = []
        for si in range(n_suites):
            ys.append(y + si * spacing)
        y_positions.append(ys)
        # label at centre of this group
        yc = ys[0] + (n_suites - 1) * spacing / 2
        y_labels.append((yc, name, fam_color))
        policy_band_spans.append((ys[-1] + spacing * 0.5, ys[0] - spacing * 0.5, fam_color))
        y += n_suites * spacing + gap

    total_height = y

    # Draw bars
    for i, name in enumerate(sorted_names):
        for si, sk in enumerate(suite_keys):
            yi = y_positions[i][si]
            left = 0.0
            for j, cat in enumerate(CATEGORY_ORDER):
                val = full[i, si, j]
                ax.barh(yi, val, left=left, height=bar_h,
                        color=CATEGORY_COLORS[cat], edgecolor="white", linewidth=0.3)
                if val > 0.08:
                    ax.text(left + val / 2, yi, f"{val:.2f}",
                            ha="center", va="center", fontsize=5.5, color="white",
                            fontweight="bold")
                left += val
            # suite label on right
            ax.text(1.01, yi, f"{SUITE_ICON[sk]} {SUITE_LABEL[sk]}",
                    va="center", fontsize=6.5, color=SUITE_COLOR[sk])

    # Policy name labels on left
    ax.set_yticks([yc for yc, _, _ in y_labels])
    ax.set_yticklabels([lbl for _, lbl, _ in y_labels], fontsize=7.5)
    # Colour the tick labels by family
    for tick, (yc, lbl, col) in zip(ax.get_yticklabels(), y_labels):
        tick.set_color(col)

    # Family separator lines
    for i in range(1, n):
        cur_fam = infer_family(sorted_names[i])
        prev_fam = infer_family(sorted_names[i - 1])
        if cur_fam != prev_fam:
            # midpoint between last bar of prev and first bar of cur
            y_sep = (y_positions[i][0] - spacing * 0.5 + y_positions[i - 1][-1] + spacing * 0.5) / 2
            ax.axhline(y_sep, color="#aaaaaa", linewidth=0.9, linestyle="--")

    ax.set_xlim(0, 1)
    ax.set_xlabel("Action share", fontsize=9)
    ax.invert_yaxis()

    # Legend: categories + suites
    cat_patches = [mpatches.Patch(facecolor=CATEGORY_COLORS[c],
                                  label=f"{CAT_ABBREV[c]} = {c}")
                   for c in CATEGORY_ORDER]
    suite_patches = [mpatches.Patch(facecolor=SUITE_COLOR[sk],
                                    label=f"{SUITE_ICON[sk]} {SUITE_LABEL[sk]}")
                     for sk in suite_keys]
    fam_patches = [mpatches.Patch(facecolor=FAMILY_META[f]["color"],
                                  label=FAMILY_META[f]["label"])
                   for f in GROUP_ORDER
                   if any(infer_family(n) == f for n in sorted_names)]

    leg1 = ax.legend(handles=cat_patches, title="Action category",
                     loc="upper right", bbox_to_anchor=(1.38, 1.0),
                     fontsize=8, title_fontsize=8.5, framealpha=0.9)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=suite_patches, title="Eval suite",
                     loc="upper right", bbox_to_anchor=(1.38, 0.72),
                     fontsize=8, title_fontsize=8.5, framealpha=0.9)
    ax.add_artist(leg2)
    ax.legend(handles=fam_patches, title="Reward family",
              loc="upper right", bbox_to_anchor=(1.38, 0.42),
              fontsize=8, title_fontsize=8.5, framealpha=0.9)

    fig.suptitle("Strategy fingerprint — all policies × all eval suites",
                 fontsize=12, fontweight="bold")
    fig.subplots_adjust(left=0.22, right=0.72)
    out = fig_dir / "fingerprint_full_comparison.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def print_stats_table(results, sorted_names):
    """Print a readable statistics table to stdout."""
    suite_keys = list(SUITES.keys())
    header = f"{'Policy':<35}" + "".join(
        f"  {SUITE_ICON[sk]} {SUITE_LABEL[sk]:<16}" for sk in suite_keys
    )
    print("\n" + "=" * len(header))
    print("STRATEGY FINGERPRINT — DETAILED STATISTICS")
    print("=" * len(header))
    print(f"\nFormat: Agg / Sci / Eco / Dip  (action share per category)\n")

    cur_fam = None
    for name in sorted_names:
        fam = infer_family(name)
        if fam != cur_fam:
            cur_fam = fam
            print(f"\n--- {FAMILY_META[fam]['label']} ---")
        row = f"  {name:<33}"
        for sk in suite_keys:
            fp = results.get(name, {}).get(sk, {})
            if fp:
                parts = "/".join(f"{fp.get(c, 0):.2f}" for c in CATEGORY_ORDER)
            else:
                parts = "  n/a  "
            row += f"  {parts:<20}"
        print(row)

    print(f"\nCategories: {' | '.join(f'{CAT_ABBREV[c]}={c}' for c in CATEGORY_ORDER)}")
    print(f"Suites:     {' | '.join(f'{SUITE_ICON[sk]} {SUITE_LABEL[sk]}' for sk in suite_keys)}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_root", help="Path to experiment folder (e.g. outputs/exp_main)")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Deterministic episodes per policy × suite (default: 5)")
    parser.add_argument("--skip-compute", action="store_true",
                        help="Skip rollouts; reload fingerprint_full.json and re-plot only")
    args = parser.parse_args()

    exp_root = Path(args.exp_root)
    if not exp_root.exists():
        print(f"ERROR: {exp_root} not found")
        return

    fig_dir = exp_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fp_json_path = exp_root / "fingerprints_full.json"

    runs = load_runs(exp_root)
    runs.sort(key=lambda r: sort_key(r["name"]))
    sorted_names = [r["name"] for r in runs]
    print(f"Found {len(runs)} trained policies in {exp_root}")

    # ── compute or load ───────────────────────────────────────────────────────
    if args.skip_compute and fp_json_path.exists():
        results = json.loads(fp_json_path.read_text())
        print(f"Loaded cached fingerprints from {fp_json_path}")
    else:
        results: Dict[str, Dict[str, Dict[str, float]]] = {}
        n_total = len(runs) * len(SUITES)
        done = 0
        for run in runs:
            name = run["name"]
            cfg = run["config"]
            max_turns = cfg.get("max_turns", 100)
            grace = cfg.get("grace_period_turns", 25)
            asym = cfg.get("starting_asymmetry", "current")
            ap = cfg.get("starting_action_points", 3)
            results[name] = {}

            for suite_key, strategies in SUITES.items():
                done += 1
                print(f"  [{done:3d}/{n_total}]  {name}  ×  {SUITE_LABEL[suite_key]}", flush=True)
                try:
                    fp = compute_fingerprint(
                        model_path=run["dir"] / "maskable_ppo_nation.zip",
                        opponent_strategies=strategies,
                        episodes=args.episodes,
                        max_turns=max_turns,
                        grace_period_turns=grace,
                        starting_asymmetry=asym,
                        starting_action_points=ap,
                    )
                    results[name][suite_key] = fp
                    top = max(fp.items(), key=lambda kv: kv[1])
                    print(f"           top={CAT_ABBREV[top[0]]} ({top[1]:.2f})", flush=True)
                except Exception as exc:
                    print(f"           ERROR: {exc}", flush=True)
                    results[name][suite_key] = {}

        fp_json_path.write_text(json.dumps(results, indent=2))
        print(f"\nFingerprints saved → {fp_json_path}")

    # ── separator rows (between families) ─────────────────────────────────────
    sep_rows: set = set()
    prev = None
    for i, name in enumerate(sorted_names):
        fam = infer_family(name)
        if prev and fam != prev:
            sep_rows.add(i)
        prev = fam

    # ── plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_per_suite(results, sorted_names, sep_rows, fig_dir)
    plot_comparison(results, sorted_names, sep_rows, fig_dir)

    # ── statistics table ──────────────────────────────────────────────────────
    print_stats_table(results, sorted_names)


if __name__ == "__main__":
    main()
