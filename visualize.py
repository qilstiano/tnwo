#!/usr/bin/env python3
"""
TNWO Training & Game Visualization

Usage:
  python visualize.py training --training-dir outputs/ppo_vs_llm_20260406_100555
  python visualize.py game --game-export outputs/game_export_20260402_064946.jsonl
  python visualize.py compare --compare-dirs outputs/ppo_vs_llm_20260406_082433 outputs/ppo_vs_llm_20260406_100555
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import numpy as np

# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────
NATION_COLORS = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6"]

CATEGORY_MAP = {
    "aggression": {
        "DECLARE_WAR", "MILITARY_STRIKE", "SKIRMISH", "SABOTAGE",
        "HARVEST MANPOWER", "INVEST MILITARY",
    },
    "science": {
        "RESEARCH", "PURSUE_CIVIC", "HARVEST SCIENCE", "HARVEST CIVICS",
        "PROPOSE_RESEARCH", "ACCEPT_RESEARCH", "INVEST SCIENCE", "INVEST CIVICS",
    },
    "economy": {
        "HARVEST GOLD", "HARVEST PRODUCTION", "INVEST INDUSTRY",
        "INVEST MANPOWER", "PROPOSE_TRADE", "ACCEPT_TRADE",
    },
    "diplomacy": {
        "PROPOSE_ALLIANCE", "ACCEPT_ALLIANCE",
        "PROPOSE_TRADE", "ACCEPT_TRADE",
        "PROPOSE_RESEARCH", "ACCEPT_RESEARCH",
    },
}

CATEGORY_COLORS = {
    "aggression": "#ef4444",
    "science": "#3b82f6",
    "economy": "#f59e0b",
    "diplomacy": "#10b981",
}

CATEGORY_ORDER = ["aggression", "science", "economy", "diplomacy"]

DIPLO_COLORS = {
    "NEUTRAL": "#e5e7eb",
    "ALLIED": "#10b981",
    "WAR": "#ef4444",
    "AT_WAR": "#ef4444",
    "ALLIANCE_PENDING": "#93c5fd",
}

GRACE_PERIOD = 25


# ─────────────────────────────────────────────────────────────────
# Data Loaders
# ─────────────────────────────────────────────────────────────────

def load_training_curve(csv_path):
    data = defaultdict(list)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                try:
                    data[k].append(float(v))
                except ValueError:
                    data[k].append(v)
    return dict(data)


def load_train_config(dir_path):
    cfg_path = os.path.join(dir_path, "train_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            return json.load(f)
    return {}


def load_game_export(jsonl_path):
    turns = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                turns.append(json.loads(line))
    return turns


def load_game_summary(jsonl_path):
    summary_path = jsonl_path.rsplit(".", 1)[0] + "_summary.json"
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            return json.load(f)
    return {}


def load_eval_results(json_path):
    with open(json_path) as f:
        return json.load(f)


def _classify_action(action_str):
    """Classify an action string into a category."""
    action_upper = action_str.strip().upper()
    # Extract the action verb (first 1-2 words)
    parts = action_upper.split()
    if not parts:
        return "economy"

    # Try matching full action (e.g. "HARVEST GOLD")
    verb = parts[0]
    if len(parts) >= 2:
        full_cmd = f"{parts[0]} {parts[1]}"
    else:
        full_cmd = verb

    for cat, actions in CATEGORY_MAP.items():
        if full_cmd in actions or verb in actions:
            return cat
    return "economy"  # default


def extract_nation_timeseries(turns):
    """Extract per-nation time series from game export JSONL data."""
    # Filter to pre-resolution snapshots only
    pre_turns = [t for t in turns if t.get("phase", "pre") == "pre"]

    if not pre_turns:
        return None

    # Discover nation IDs from first turn
    first = pre_turns[0]
    nation_ids = sorted(int(k) for k in first["agents"].keys())
    num_nations = len(nation_ids)

    # Initialize storage
    ts = {
        "turns": [],
        "nation_ids": nation_ids,
        "names": {},
        "personalities": {},
        "scores": {nid: [] for nid in nation_ids},
        "gold": {nid: [] for nid in nation_ids},
        "manpower": {nid: [] for nid in nation_ids},
        "production": {nid: [] for nid in nation_ids},
        "science": {nid: [] for nid in nation_ids},
        "civics": {nid: [] for nid in nation_ids},
        "infra": {nid: [] for nid in nation_ids},
        "war_exhaustion": {nid: [] for nid in nation_ids},
        "tech_count": {nid: [] for nid in nation_ids},
        "civic_count": {nid: [] for nid in nation_ids},
        "trade_pacts": {nid: [] for nid in nation_ids},
        "research_pacts": {nid: [] for nid in nation_ids},
        "actions": {nid: [] for nid in nation_ids},  # list of action lists
        "diplomacy": {nid: {} for nid in nation_ids},  # nid -> {other_nid: [status_per_turn]}
        "action_categories": {nid: [] for nid in nation_ids},  # list of category dicts
    }

    # Initialize diplomacy pairs
    for nid in nation_ids:
        for other in nation_ids:
            if other != nid:
                ts["diplomacy"][nid][other] = []

    for turn_data in pre_turns:
        turn_num = turn_data["turn"]
        ts["turns"].append(turn_num)

        for nid in nation_ids:
            agent_data = turn_data["agents"].get(str(nid), {})
            state = agent_data.get("state", {})
            my_nation = state.get("my_nation", {})
            stats = my_nation.get("stats", {})
            status = my_nation.get("status", {})
            tech = my_nation.get("tech", {})
            civic = my_nation.get("civic", {})
            diplo = my_nation.get("diplomacy", {})
            others = state.get("other_nations", [])
            queued = agent_data.get("queued_actions", [])

            # Names (from first turn)
            if turn_num == pre_turns[0]["turn"]:
                ts["names"][nid] = my_nation.get("name", f"Nation {nid}")
                ts["personalities"][nid] = my_nation.get("personality", "")

            # Resources
            ts["gold"][nid].append(stats.get("gold", 0))
            ts["manpower"][nid].append(stats.get("manpower", 0))
            ts["production"][nid].append(stats.get("production", 0))
            ts["science"][nid].append(stats.get("science", 0))
            ts["civics"][nid].append(stats.get("civics", 0))

            # Tech/civic
            n_techs = len(tech.get("unlocked", []))
            n_civics = len(civic.get("unlocked", []))
            ts["tech_count"][nid].append(n_techs)
            ts["civic_count"][nid].append(n_civics)

            # Score
            score = (stats.get("gold", 0) + stats.get("manpower", 0) +
                     stats.get("production", 0) + n_techs * 500 + n_civics * 500)
            ts["scores"][nid].append(score)

            # Status
            ts["infra"][nid].append(status.get("infrastructure_health", 100))
            ts["war_exhaustion"][nid].append(status.get("war_exhaustion", 0))

            # Diplomacy counts
            ts["trade_pacts"][nid].append(len(diplo.get("active_trade_agreements", [])))
            ts["research_pacts"][nid].append(len(diplo.get("active_research_pacts", [])))

            # Diplomacy status per other nation
            other_status = {}
            for o in others:
                other_status[o["id"]] = o.get("diplomatic_status", "NEUTRAL")
            for other_nid in nation_ids:
                if other_nid != nid:
                    ts["diplomacy"][nid][other_nid].append(
                        other_status.get(other_nid, "NEUTRAL")
                    )

            # Actions and categories
            ts["actions"][nid].append(queued)
            cat_counts = {c: 0 for c in CATEGORY_ORDER}
            for a in queued:
                cat = _classify_action(a)
                cat_counts[cat] += 1
            ts["action_categories"][nid].append(cat_counts)

    return ts


def _rolling_mean(values, window=5):
    """Compute rolling mean with given window size."""
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(sum(values[start:i+1]) / (i - start + 1))
    return result


# ─────────────────────────────────────────────────────────────────
# Plotting — Training Dashboard
# ─────────────────────────────────────────────────────────────────

def plot_rolling_reward(ax, data):
    episodes = data["episode"]
    ax.scatter(episodes, data["ep_reward"], alpha=0.08, s=2, color="#94a3b8", label="raw")
    ax.plot(episodes, data["rolling_reward_20"], color="#3b82f6", linewidth=1.5, label="rolling avg (20)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Episode Reward")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_rolling_winrate(ax, data):
    episodes = data["episode"]
    ax.plot(episodes, data["rolling_winrate_20"], color="#10b981", linewidth=1.5)
    ax.axhline(y=0.2, color="#ef4444", linestyle="--", alpha=0.5, label="random baseline (0.2)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Win Rate")
    ax.set_title("Rolling Win Rate (20-ep)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_rolling_score(ax, data):
    episodes = data["episode"]
    ax.scatter(episodes, data["final_score"], alpha=0.08, s=2, color="#94a3b8", label="raw")
    ax.plot(episodes, data["rolling_score_20"], color="#f59e0b", linewidth=1.5, label="rolling avg (20)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    ax.set_title("Final Score")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_episode_length_dist(ax, data):
    lengths = data["ep_length"]
    ax.hist(lengths, bins=30, color="#8b5cf6", alpha=0.7, edgecolor="white")
    mean_len = np.mean(lengths)
    ax.axvline(x=mean_len, color="#ef4444", linestyle="--", label=f"mean={mean_len:.0f}")
    ax.set_xlabel("Episode Length (turns)")
    ax.set_ylabel("Count")
    ax.set_title("Episode Length Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_training_dashboard(data, config, output_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Suptitle with config info
    lr = config.get("learning_rate", "?")
    opp = config.get("opponent_mode", "?")
    eps = len(data.get("episode", []))
    fig.suptitle(f"PPO Training Dashboard  |  lr={lr}  |  opponents={opp}  |  {eps} episodes",
                 fontsize=13, fontweight="bold")

    plot_rolling_reward(axes[0, 0], data)
    plot_rolling_winrate(axes[0, 1], data)
    plot_rolling_score(axes[1, 0], data)
    plot_episode_length_dist(axes[1, 1], data)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────
# Plotting — Game Replay
# ─────────────────────────────────────────────────────────────────

def _add_grace_band(ax, max_turn):
    """Add gray band for grace period (turns 1-25)."""
    ax.axvspan(0, GRACE_PERIOD, alpha=0.08, color="gray", zorder=0)


def _nation_label(ts, nid):
    name = ts["names"].get(nid, f"N{nid}")
    return f"{name} (#{nid})"


def plot_nation_scores(ax, ts, summary):
    turns = ts["turns"]
    for nid in ts["nation_ids"]:
        color = NATION_COLORS[nid % len(NATION_COLORS)]
        ax.plot(turns, ts["scores"][nid], color=color, linewidth=1.5,
                label=_nation_label(ts, nid))
        # Star on final point for winner
        winner = summary.get("winner")
        if winner is not None:
            winners = winner if isinstance(winner, list) else [winner]
            if nid in winners:
                ax.plot(turns[-1], ts["scores"][nid][-1], marker="*",
                        markersize=15, color=color, zorder=5)

    _add_grace_band(ax, turns[-1])
    ax.set_xlabel("Turn")
    ax.set_ylabel("Score")
    vtype = summary.get("victory_type", "")
    ax.set_title(f"Nation Scores Over Time  ({vtype} victory)" if vtype else "Nation Scores Over Time")
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    ax.grid(True, alpha=0.3)


def plot_resource_trajectories(axes, ts):
    resources = [("gold", "Gold"), ("manpower", "Manpower"), ("production", "Production")]
    for ax, (key, label) in zip(axes, resources):
        for nid in ts["nation_ids"]:
            color = NATION_COLORS[nid % len(NATION_COLORS)]
            ax.plot(ts["turns"], ts[key][nid], color=color, linewidth=1, alpha=0.8)
        _add_grace_band(ax, ts["turns"][-1])
        ax.set_xlabel("Turn")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)


def plot_strategy_drift(axes, ts):
    """Stacked area chart of action-mix shares per agent (rolling 5-turn window)."""
    nation_ids = ts["nation_ids"]
    turns = ts["turns"]
    n_agents = len(nation_ids)

    # Use up to 5 subplots (one per agent)
    for idx, nid in enumerate(nation_ids):
        if idx >= len(axes):
            break
        ax = axes[idx]

        # Build per-category share arrays
        cat_shares = {c: [] for c in CATEGORY_ORDER}
        for cat_dict in ts["action_categories"][nid]:
            total = sum(cat_dict.values())
            for c in CATEGORY_ORDER:
                cat_shares[c].append(cat_dict[c] / total if total > 0 else 0)

        # Rolling smooth
        window = 5
        smoothed = {c: _rolling_mean(cat_shares[c], window) for c in CATEGORY_ORDER}

        # Stacked area
        bottom = np.zeros(len(turns))
        for cat in CATEGORY_ORDER:
            vals = np.array(smoothed[cat])
            ax.fill_between(turns, bottom, bottom + vals,
                            color=CATEGORY_COLORS[cat], alpha=0.7, label=cat)
            bottom += vals

        _add_grace_band(ax, turns[-1])
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{_nation_label(ts, nid)}", fontsize=9)
        ax.set_xlabel("Turn", fontsize=8)
        if idx == 0:
            ax.legend(fontsize=6, ncol=2, loc="upper right")
        ax.grid(True, alpha=0.2)


def plot_state_pressure(axes, ts):
    """Plot war_exhaustion, infra_health, trade_pacts, research_pacts."""
    turns = ts["turns"]
    metrics = [
        ("infra", "Infrastructure Health", (0, 105)),
        ("war_exhaustion", "War Exhaustion", None),
        ("trade_pacts", "Trade Pacts", None),
        ("research_pacts", "Research Pacts", None),
    ]

    for ax, (key, label, ylim) in zip(axes, metrics):
        for nid in ts["nation_ids"]:
            color = NATION_COLORS[nid % len(NATION_COLORS)]
            ax.plot(turns, ts[key][nid], color=color, linewidth=1, alpha=0.8)
        _add_grace_band(ax, turns[-1])
        ax.set_xlabel("Turn", fontsize=8)
        ax.set_ylabel(label, fontsize=8)
        ax.set_title(label, fontsize=9)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)


def plot_tech_civic_progress(ax, ts):
    turns = ts["turns"]
    for nid in ts["nation_ids"]:
        color = NATION_COLORS[nid % len(NATION_COLORS)]
        ax.plot(turns, ts["tech_count"][nid], color=color, linewidth=1.5,
                linestyle="-", alpha=0.9)
        ax.plot(turns, ts["civic_count"][nid], color=color, linewidth=1,
                linestyle="--", alpha=0.7)

    _add_grace_band(ax, turns[-1])
    # Custom legend
    ax.plot([], [], color="gray", linestyle="-", label="Tech")
    ax.plot([], [], color="gray", linestyle="--", label="Civic")
    ax.legend(fontsize=7)
    ax.set_xlabel("Turn")
    ax.set_ylabel("Count")
    ax.set_title("Tech (solid) & Civic (dashed) Progress")
    ax.grid(True, alpha=0.3)


def plot_diplomacy_timeline(ax, ts):
    """Horizontal colored segments for each nation pair's diplomatic status."""
    nation_ids = ts["nation_ids"]
    turns = ts["turns"]
    pairs = []
    for i, nid_a in enumerate(nation_ids):
        for nid_b in nation_ids[i+1:]:
            pairs.append((nid_a, nid_b))

    y_labels = []
    for pair_idx, (a, b) in enumerate(pairs):
        y_labels.append(f"{a}-{b}")
        statuses = ts["diplomacy"][a].get(b, [])
        if not statuses:
            continue

        # Draw colored segments
        seg_start = 0
        current_status = statuses[0]
        for t_idx in range(1, len(statuses)):
            if statuses[t_idx] != current_status or t_idx == len(statuses) - 1:
                end = t_idx if statuses[t_idx] != current_status else t_idx + 1
                color = DIPLO_COLORS.get(current_status, DIPLO_COLORS["NEUTRAL"])
                ax.barh(pair_idx, end - seg_start, left=turns[seg_start] if seg_start < len(turns) else 0,
                        height=0.8, color=color, edgecolor="none")
                seg_start = t_idx
                current_status = statuses[t_idx]

    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.set_xlabel("Turn")
    ax.set_title("Diplomacy Timeline")

    # Legend
    legend_elements = [Patch(facecolor=DIPLO_COLORS[k], label=k)
                       for k in ["NEUTRAL", "ALLIED", "WAR", "ALLIANCE_PENDING"]]
    ax.legend(handles=legend_elements, fontsize=6, ncol=2, loc="upper right")


def plot_action_distribution(ax, ts):
    """Stacked bar chart of total action counts per nation, by category."""
    nation_ids = ts["nation_ids"]
    cat_totals = {nid: {c: 0 for c in CATEGORY_ORDER} for nid in nation_ids}

    for nid in nation_ids:
        for cat_dict in ts["action_categories"][nid]:
            for c in CATEGORY_ORDER:
                cat_totals[nid][c] += cat_dict[c]

    x = np.arange(len(nation_ids))
    width = 0.6
    bottom = np.zeros(len(nation_ids))

    for cat in CATEGORY_ORDER:
        vals = [cat_totals[nid][cat] for nid in nation_ids]
        ax.bar(x, vals, width, bottom=bottom, color=CATEGORY_COLORS[cat], label=cat)
        bottom += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels([ts["names"].get(nid, f"N{nid}") for nid in nation_ids],
                       fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("Total Actions")
    ax.set_title("Action Distribution by Category")
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.2, axis="y")


def plot_game_replay(turns_data, summary, output_path=None):
    ts = extract_nation_timeseries(turns_data)
    if ts is None:
        print("No pre-resolution data found in export.")
        return

    nation_ids = ts["nation_ids"]
    n_nations = len(nation_ids)

    fig = plt.figure(figsize=(18, 32))
    gs = gridspec.GridSpec(6, 5, figure=fig, hspace=0.4, wspace=0.35)

    # Row 0: Scores (span all cols)
    ax_scores = fig.add_subplot(gs[0, :])
    plot_nation_scores(ax_scores, ts, summary)

    # Row 1: Resource trajectories (3 panels)
    ax_gold = fig.add_subplot(gs[1, 0:2])
    ax_mp = fig.add_subplot(gs[1, 2:4])
    ax_prod = fig.add_subplot(gs[1, 4])
    plot_resource_trajectories([ax_gold, ax_mp, ax_prod], ts)

    # Row 2: Strategy drift — one stacked area per agent
    drift_axes = []
    for i in range(min(n_nations, 5)):
        drift_axes.append(fig.add_subplot(gs[2, i]))
    plot_strategy_drift(drift_axes, ts)

    # Row 3: State pressure (4 panels)
    pressure_axes = []
    for i in range(4):
        pressure_axes.append(fig.add_subplot(gs[3, i]))
    ax_tech = fig.add_subplot(gs[3, 4])
    plot_state_pressure(pressure_axes, ts)
    plot_tech_civic_progress(ax_tech, ts)

    # Row 4: Diplomacy timeline + action distribution
    ax_diplo = fig.add_subplot(gs[4, 0:3])
    ax_actions = fig.add_subplot(gs[4, 3:])
    plot_diplomacy_timeline(ax_diplo, ts)
    plot_action_distribution(ax_actions, ts)

    # Row 5: Per-nation score breakdown as text summary
    ax_summary = fig.add_subplot(gs[5, :])
    ax_summary.axis("off")
    summary_lines = []
    winner = summary.get("winner")
    vtype = summary.get("victory_type", "N/A")
    turns_played = summary.get("turns_played", len(ts["turns"]))
    summary_lines.append(f"Turns: {turns_played}  |  Victory: {vtype}  |  Winner: {winner}")
    for nid in nation_ids:
        name = ts["names"].get(nid, f"N{nid}")
        score = ts["scores"][nid][-1] if ts["scores"][nid] else 0
        techs = ts["tech_count"][nid][-1] if ts["tech_count"][nid] else 0
        civics = ts["civic_count"][nid][-1] if ts["civic_count"][nid] else 0
        summary_lines.append(f"  #{nid} {name:16s}  score={score:6d}  techs={techs}  civics={civics}")
    ax_summary.text(0.02, 0.95, "\n".join(summary_lines), transform=ax_summary.transAxes,
                    fontsize=9, verticalalignment="top", fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#e2e8f0"))

    fig.suptitle("Game Replay Dashboard", fontsize=15, fontweight="bold", y=0.995)

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────
# Plotting — Cross-Run Comparison
# ─────────────────────────────────────────────────────────────────

def plot_compare_curves(ax, runs_data, key, ylabel, title):
    colors = plt.cm.tab10(np.linspace(0, 1, len(runs_data)))
    for i, (label, data) in enumerate(runs_data):
        ax.plot(data["episode"], data[key], color=colors[i], linewidth=1.2, label=label, alpha=0.85)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def plot_compare_eval_metrics(ax, eval_results):
    if not eval_results:
        ax.text(0.5, 0.5, "No eval results", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Evaluation Metrics")
        return

    labels = [label for label, _ in eval_results]
    metrics = ["win_rate", "avg_final_score"]
    x = np.arange(len(labels))
    width = 0.35

    # Dual y-axis: win_rate on left, score on right
    win_rates = [d.get("win_rate", 0) for _, d in eval_results]
    scores = [d.get("avg_final_score", 0) for _, d in eval_results]

    ax.bar(x - width/2, win_rates, width, color="#10b981", alpha=0.8, label="Win Rate")
    ax.set_ylabel("Win Rate")
    ax.set_ylim(0, 1.1)

    ax2 = ax.twinx()
    ax2.bar(x + width/2, scores, width, color="#3b82f6", alpha=0.8, label="Avg Score")
    ax2.set_ylabel("Avg Score")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=15, ha="right")
    ax.set_title("Evaluation Metrics")

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)
    ax.grid(True, alpha=0.2, axis="y")


def plot_cross_run_comparison(runs_data, eval_results=None, output_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    plot_compare_curves(axes[0, 0], runs_data, "rolling_winrate_20", "Win Rate", "Rolling Win Rate")
    plot_compare_curves(axes[0, 1], runs_data, "rolling_reward_20", "Reward", "Rolling Reward")
    plot_compare_curves(axes[1, 0], runs_data, "rolling_score_20", "Score", "Rolling Score")
    plot_compare_eval_metrics(axes[1, 1], eval_results or [])

    fig.suptitle("Cross-Run Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TNWO Training & Game Visualization")
    sub = parser.add_subparsers(dest="mode", required=True)

    # Mode 1: training curves
    p_train = sub.add_parser("training", help="Visualize training curves from a single run")
    p_train.add_argument("--training-dir", required=True, help="Path to ppo_vs_llm_* directory")
    p_train.add_argument("--output-dir", default=None, help="Save PNGs here instead of showing")

    # Mode 2: game replay
    p_game = sub.add_parser("game", help="Visualize a single game replay from JSONL export")
    p_game.add_argument("--game-export", required=True, help="Path to game_export_*.jsonl file")
    p_game.add_argument("--output-dir", default=None)

    # Mode 3: cross-run comparison
    p_compare = sub.add_parser("compare", help="Compare multiple training runs")
    p_compare.add_argument("--compare-dirs", nargs="+", required=True)
    p_compare.add_argument("--eval-jsons", nargs="*", default=[])
    p_compare.add_argument("--output-dir", default=None)

    args = parser.parse_args()

    if args.mode == "training":
        csv_path = os.path.join(args.training_dir, "training_curve.csv")
        if not os.path.exists(csv_path):
            print(f"ERROR: {csv_path} not found")
            return
        data = load_training_curve(csv_path)
        config = load_train_config(args.training_dir)
        out = None
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            out = os.path.join(args.output_dir, "training_dashboard.png")
        plot_training_dashboard(data, config, out)

    elif args.mode == "game":
        if not os.path.exists(args.game_export):
            print(f"ERROR: {args.game_export} not found")
            return
        turns = load_game_export(args.game_export)
        summary = load_game_summary(args.game_export)
        out = None
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            base = Path(args.game_export).stem
            out = os.path.join(args.output_dir, f"game_replay_{base}.png")
        plot_game_replay(turns, summary, out)

    elif args.mode == "compare":
        runs_data = []
        for d in args.compare_dirs:
            csv_path = os.path.join(d, "training_curve.csv")
            if os.path.exists(csv_path):
                label = os.path.basename(d)
                runs_data.append((label, load_training_curve(csv_path)))
            else:
                print(f"WARNING: {csv_path} not found, skipping")
        if not runs_data:
            print("ERROR: no valid training directories found")
            return

        eval_results = []
        for ej in args.eval_jsons:
            if os.path.exists(ej):
                label = Path(ej).stem
                eval_results.append((label, load_eval_results(ej)))

        out = None
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            out = os.path.join(args.output_dir, "cross_run_comparison.png")
        plot_cross_run_comparison(runs_data, eval_results, out)


if __name__ == "__main__":
    main()
