#!/usr/bin/env python3
"""
Batch evaluation for the TNWO experiment matrix.

For every trained PPO policy in outputs/<experiment>/<run>/maskable_ppo_nation.zip,
evaluate it against a fixed battery of rule-based opponent compositions and
write a per-run JSON plus a single aggregated CSV/JSON.

Each policy is scored against the SAME set of test opponents so we can compare
generalization across runs (a policy trained vs aggressors might overfit and
collapse against diplomats — that's exactly what we want to see).

Usage:
    python experiments/eval_matrix.py outputs/exp_main
    python experiments/eval_matrix.py outputs/exp_main --episodes 30
    python experiments/eval_matrix.py outputs/exp_main --only base_vs_diverse
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sb3_contrib import MaskablePPO  # noqa: E402
from sb3_contrib.common.wrappers import ActionMasker  # noqa: E402

from rl.env import NationEnv  # noqa: E402


# Fixed evaluation suites — applied identically to every trained policy.
EVAL_SUITES: Dict[str, List[str]] = {
    "vs_balanced":  ["balanced", "balanced", "balanced", "balanced"],
    "vs_aggressor": ["aggressor", "aggressor", "aggressor", "aggressor"],
    "vs_diverse":   ["aggressor", "turtle", "scientist", "diplomat"],
    "vs_diplomat":  ["diplomat", "diplomat", "diplomat", "diplomat"],
}


def mask_fn(env):
    return env.action_masks()


def evaluate_policy(model_path: Path, opponent_strategies: List[str],
                    episodes: int, num_players: int, learner_id: int,
                    max_turns: int, base_seed: int,
                    grace_period_turns: int = 25,
                    starting_asymmetry: str = "current",
                    starting_action_points: int = 3) -> dict:
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
        mask_fn,
    )
    model = MaskablePPO.load(str(model_path))

    rewards, scores = [], []
    wins = peace_wins = losses = 0

    for ep in range(episodes):
        obs, _info = env.reset(seed=base_seed + ep * 17)
        done = trunc = False
        ep_reward = 0.0
        last_info = {}
        while not (done or trunc):
            masks = env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=masks)
            obs, r, done, trunc, last_info = env.step(int(action))
            ep_reward += r

        rewards.append(float(ep_reward))
        scores.append(float(last_info.get("score", 0.0)))
        winner = last_info.get("winner")
        if isinstance(winner, list):
            if learner_id in winner:
                wins += 1
                peace_wins += 1
            else:
                losses += 1
        elif winner == learner_id:
            wins += 1
        else:
            losses += 1

    return {
        "episodes": episodes,
        "win_rate": wins / episodes,
        "peace_win_rate": peace_wins / episodes,
        "loss_rate": losses / episodes,
        "avg_reward": float(np.mean(rewards)),
        "avg_final_score": float(np.mean(scores)),
        "std_final_score": float(np.std(scores)),
    }


def find_run_dirs(exp_root: Path) -> List[Path]:
    return sorted(
        d for d in exp_root.iterdir()
        if d.is_dir() and (d / "maskable_ppo_nation.zip").exists()
    )


def load_train_config(run_dir: Path) -> dict:
    cfg_path = run_dir / "train_config.json"
    if cfg_path.exists():
        return json.loads(cfg_path.read_text())
    return {}


def main():
    parser = argparse.ArgumentParser(description="Batch eval over experiment matrix")
    parser.add_argument("exp_root", help="Experiment directory, e.g. outputs/exp_main")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--num-players", type=int, default=5)
    parser.add_argument("--learner-id", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=10000)
    parser.add_argument("--only", default=None,
                        help="Comma-separated subset of run names to evaluate")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip runs that already have eval_results.json")
    args = parser.parse_args()

    exp_root = Path(args.exp_root)
    if not exp_root.exists():
        print(f"ERROR: {exp_root} does not exist")
        return

    run_dirs = find_run_dirs(exp_root)
    if args.only:
        wanted = {n.strip() for n in args.only.split(",") if n.strip()}
        run_dirs = [d for d in run_dirs if d.name in wanted]

    print(f"Found {len(run_dirs)} trained runs in {exp_root}")
    print(f"Eval suites: {list(EVAL_SUITES.keys())}")
    print(f"Episodes per suite: {args.episodes}")
    print()

    aggregated = []

    for idx, run_dir in enumerate(run_dirs, 1):
        eval_path = run_dir / "eval_results.json"
        if args.skip_existing and eval_path.exists():
            print(f"  [{idx}/{len(run_dirs)}] {run_dir.name}: (cached)")
            cached = json.loads(eval_path.read_text())
            aggregated.append(cached)
            continue

        print(f"  [{idx}/{len(run_dirs)}] {run_dir.name}")
        cfg = load_train_config(run_dir)
        # Reuse the run's own env config so each policy is evaluated in
        # the exact world it was trained in (same horizon, grace period,
        # action budget, starting asymmetry). Falls back to defaults for
        # older runs that predate the extended-axis fields.
        eval_max_turns = cfg.get("max_turns", args.max_turns)
        eval_grace = cfg.get("grace_period_turns", 25)
        eval_asymmetry = cfg.get("starting_asymmetry", "current")
        eval_action_points = cfg.get("starting_action_points", 3)

        per_suite = {}
        t_start = time.time()
        for suite_name, opps in EVAL_SUITES.items():
            res = evaluate_policy(
                model_path=run_dir / "maskable_ppo_nation.zip",
                opponent_strategies=opps,
                episodes=args.episodes,
                num_players=args.num_players,
                learner_id=args.learner_id,
                max_turns=eval_max_turns,
                base_seed=args.base_seed,
                grace_period_turns=eval_grace,
                starting_asymmetry=eval_asymmetry,
                starting_action_points=eval_action_points,
            )
            per_suite[suite_name] = res
            print(f"      {suite_name:14s}  win_rate={res['win_rate']:.2f}  "
                  f"score={res['avg_final_score']:.0f}")
        elapsed = time.time() - t_start
        print(f"      ({elapsed:.1f}s)")

        record = {
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            "train_config": {
                k: cfg.get(k) for k in (
                    "opponent_strategies", "dense_reward_scale",
                    "terminal_win_reward", "terminal_peace_reward",
                    "terminal_loss_penalty", "annex_bonus", "max_turns",
                    "total_timesteps", "seed",
                    "grace_period_turns", "starting_asymmetry",
                    "starting_action_points",
                )
            },
            "eval_episodes": args.episodes,
            "eval": per_suite,
        }
        eval_path.write_text(json.dumps(record, indent=2))
        aggregated.append(record)

    aggregate_path = exp_root / "eval_matrix.json"
    aggregate_path.write_text(json.dumps(aggregated, indent=2))
    print()
    print(f"Wrote aggregated results to {aggregate_path}")


if __name__ == "__main__":
    main()
