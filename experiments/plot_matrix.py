#!/usr/bin/env python3
"""
Aggregate visualization for the TNWO experiment matrix.

Inputs (produced by run_matrix.py + eval_matrix.py):
    outputs/<exp>/<run>/training_curve.csv
    outputs/<exp>/<run>/train_config.json
    outputs/<exp>/<run>/maskable_ppo_nation.zip
    outputs/<exp>/eval_matrix.json     (from eval_matrix.py)

Outputs (written to outputs/<exp>/figures/):
    training_curves_grouped.png   — rolling win rate / reward / score per group
    eval_heatmap.png              — policy × eval-suite win-rate heatmap
    strategy_fingerprint.png      — action-category mix per policy (grouped bar)

Usage:
    python experiments/plot_matrix.py outputs/exp_main
    python experiments/plot_matrix.py outputs/exp_main --fingerprint-episodes 5
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sb3_contrib import MaskablePPO  # noqa: E402
from sb3_contrib.common.wrappers import ActionMasker  # noqa: E402

from rl.env import NationEnv  # noqa: E402
from rl.action_space import build_action_catalog, command_from_action  # noqa: E402
from visualize import _classify_action, CATEGORY_ORDER, CATEGORY_COLORS  # noqa: E402


# Same suites as eval_matrix so the heatmap and the matrix line up.
EVAL_SUITES = ["vs_balanced", "vs_aggressor", "vs_diverse", "vs_diplomat"]
FINGERPRINT_OPPONENTS = ["aggressor", "turtle", "scientist", "diplomat"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> Dict[str, List[float]]:
    data = defaultdict(list)
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                try:
                    data[k].append(float(v))
                except (TypeError, ValueError):
                    data[k].append(v)
    return dict(data)


def load_runs(exp_root: Path) -> List[dict]:
    runs = []
    for d in sorted(exp_root.iterdir()):
        if not d.is_dir():
            continue
        curve = d / "training_curve.csv"
        cfg = d / "train_config.json"
        model = d / "maskable_ppo_nation.zip"
        if not curve.exists() or not model.exists():
            continue
        runs.append({
            "name": d.name,
            "dir": d,
            "curve": load_csv(curve),
            "config": json.loads(cfg.read_text()) if cfg.exists() else {},
        })
    return runs


def load_eval_matrix(exp_root: Path) -> List[dict]:
    path = exp_root / "eval_matrix.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Fingerprint: step the trained policy through some episodes and tally
# action-category frequencies. Uses the diverse opponent set so every policy
# is profiled in the same environment.
# ---------------------------------------------------------------------------

def compute_fingerprint(model_path: Path, episodes: int = 5,
                        max_turns: int = 100, num_players: int = 5,
                        learner_id: int = 0, base_seed: int = 99000,
                        grace_period_turns: int = 25,
                        starting_asymmetry: str = "current",
                        starting_action_points: int = 3) -> Dict[str, float]:
    env = ActionMasker(
        NationEnv(
            num_players=num_players,
            learner_id=learner_id,
            max_turns=max_turns,
            seed=base_seed,
            opponent_mode="rulebased",
            rule_opponent_strategies=FINGERPRINT_OPPONENTS,
            grace_period_turns=grace_period_turns,
            starting_asymmetry=starting_asymmetry,
            starting_action_points=starting_action_points,
        ),
        lambda e: e.action_masks(),
    )
    # Force CPU loading for the fingerprint pass. The policy network is
    # tiny (256-hidden MLP) so CPU inference is milliseconds per step,
    # and keeping it off the GPU avoids any contention with concurrent
    # training jobs or other processes that may have claimed device
    # memory. Previously this defaulted to the model's original device
    # (cuda), which OOM'd the entire fingerprint sweep when the card
    # was busy.
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


# ---------------------------------------------------------------------------
# Plot 1: grouped training curves
# ---------------------------------------------------------------------------

def _group_of(run_name: str) -> str:
    """Group runs into reward-family buckets for the legend coloring."""
    if run_name.startswith("base_"):
        return "baseline"
    if run_name.startswith("annex_"):
        return "annex_heavy"
    if run_name.startswith("peace_"):
        return "peace_heavy"
    if run_name.startswith("dense_only"):
        return "dense_only"
    if run_name.startswith("terminal_only"):
        return "terminal_only"
    return "other"


GROUP_COLORS = {
    "baseline":      "#3b82f6",
    "annex_heavy":   "#ef4444",
    "peace_heavy":   "#10b981",
    "dense_only":    "#f59e0b",
    "terminal_only": "#8b5cf6",
    "other":         "#94a3b8",
}


def plot_training_curves(runs, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [
        ("rolling_winrate_20", "Rolling Win Rate (20)"),
        ("rolling_reward_20",  "Rolling Reward (20)"),
        ("rolling_score_20",   "Rolling Score (20)"),
    ]
    for ax, (key, title) in zip(axes, metrics):
        for run in runs:
            curve = run["curve"]
            if key not in curve or "episode" not in curve:
                continue
            color = GROUP_COLORS[_group_of(run["name"])]
            ax.plot(curve["episode"], curve[key], color=color, alpha=0.7,
                    linewidth=1.0, label=run["name"])
        ax.set_xlabel("Episode")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    # Single grouped legend in the rightmost subplot
    handles = [plt.Line2D([], [], color=c, linewidth=2, label=g)
               for g, c in GROUP_COLORS.items()
               if any(_group_of(r["name"]) == g for r in runs)]
    axes[-1].legend(handles=handles, fontsize=8, loc="lower right")
    fig.suptitle("Training curves grouped by reward family", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: eval-suite × policy heatmap
# ---------------------------------------------------------------------------

def plot_eval_heatmap(eval_records, out_path: Path):
    if not eval_records:
        print("  (no eval_matrix.json — skipping heatmap)")
        return

    names = [r["run_name"] for r in eval_records]
    matrix = np.zeros((len(names), len(EVAL_SUITES)))
    for i, rec in enumerate(eval_records):
        for j, suite in enumerate(EVAL_SUITES):
            matrix[i, j] = rec["eval"].get(suite, {}).get("win_rate", 0.0)

    fig_h = max(4, 0.35 * len(names) + 2)
    fig, ax = plt.subplots(figsize=(8, fig_h))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(EVAL_SUITES)))
    ax.set_xticklabels(EVAL_SUITES, rotation=20, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(EVAL_SUITES)):
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black")
    fig.colorbar(im, ax=ax, label="Win Rate")
    ax.set_title("Eval win rate by policy × opponent suite")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# Plot 3: strategy fingerprint (action-category mix per policy)
# ---------------------------------------------------------------------------

def plot_strategy_fingerprint(fingerprints: Dict[str, Dict[str, float]],
                              out_path: Path):
    if not fingerprints:
        print("  (no fingerprints — skipping)")
        return

    names = list(fingerprints.keys())
    n = len(names)
    cats = CATEGORY_ORDER
    matrix = np.zeros((n, len(cats)))
    for i, name in enumerate(names):
        for j, cat in enumerate(cats):
            matrix[i, j] = fingerprints[name].get(cat, 0.0)

    fig_h = max(4, 0.35 * n + 2)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    y = np.arange(n)
    left = np.zeros(n)
    for j, cat in enumerate(cats):
        ax.barh(y, matrix[:, j], left=left, color=CATEGORY_COLORS[cat],
                edgecolor="white", label=cat)
        left += matrix[:, j]

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Action share (deterministic eval vs diverse opponents)")
    ax.set_title("Strategy fingerprint per policy")
    ax.legend(loc="lower right", fontsize=8, ncol=4)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_root")
    parser.add_argument("--fingerprint-episodes", type=int, default=5,
                        help="Episodes per policy when computing the action mix")
    parser.add_argument("--skip-fingerprint", action="store_true")
    args = parser.parse_args()

    exp_root = Path(args.exp_root)
    if not exp_root.exists():
        print(f"ERROR: {exp_root} not found")
        return

    fig_dir = exp_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(exp_root)
    print(f"Loaded {len(runs)} training runs from {exp_root}")

    plot_training_curves(runs, fig_dir / "training_curves_grouped.png")

    eval_records = load_eval_matrix(exp_root)
    plot_eval_heatmap(eval_records, fig_dir / "eval_heatmap.png")

    if args.skip_fingerprint:
        print("  (--skip-fingerprint set — no strategy mix plot)")
        return

    print(f"Computing strategy fingerprints ({args.fingerprint_episodes} eps each)...")
    fingerprints = {}
    for run in runs:
        cfg = run["config"]
        max_turns = cfg.get("max_turns", 100)
        try:
            fp = compute_fingerprint(
                model_path=run["dir"] / "maskable_ppo_nation.zip",
                episodes=args.fingerprint_episodes,
                max_turns=max_turns,
                grace_period_turns=cfg.get("grace_period_turns", 25),
                starting_asymmetry=cfg.get("starting_asymmetry", "current"),
                starting_action_points=cfg.get("starting_action_points", 3),
            )
            fingerprints[run["name"]] = fp
            top = max(fp.items(), key=lambda kv: kv[1])
            print(f"  {run['name']:32s}  top={top[0]} ({top[1]:.2f})")
        except Exception as exc:
            print(f"  {run['name']}: ERROR {exc}")

    fp_json = exp_root / "fingerprints.json"
    fp_json.write_text(json.dumps(fingerprints, indent=2))
    plot_strategy_fingerprint(fingerprints, fig_dir / "strategy_fingerprint.png")
    print(f"  fingerprints written to {fp_json}")


if __name__ == "__main__":
    main()
