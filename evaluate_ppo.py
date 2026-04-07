#!/usr/bin/env python3
"""
Evaluate a trained PPO nation policy against rule-based opponents.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from rl.env import NationEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO nation policy.")
    parser.add_argument("--model-path", required=True, help="Path to saved MaskablePPO model")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--num-players", type=int, default=5)
    parser.add_argument("--learner-id", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output-json", default=None, help="Optional path to save aggregated metrics")
    return parser.parse_args()


def mask_fn(env):
    return env.action_masks()


def main():
    args = parse_args()

    env = ActionMasker(
        NationEnv(
            num_players=args.num_players,
            learner_id=args.learner_id,
            max_turns=args.max_turns,
            seed=args.seed,
        ),
        mask_fn,
    )
    model = MaskablePPO.load(args.model_path)

    episode_rewards = []
    final_scores = []
    win_count = 0
    peace_win_count = 0
    loss_count = 0

    for episode_idx in range(args.episodes):
        obs, info = env.reset(seed=args.seed + episode_idx)
        done = False
        truncated = False
        episode_reward = 0.0
        final_info = {}

        while not (done or truncated):
            masks = env.action_masks()
            action, _ = model.predict(obs, deterministic=args.deterministic, action_masks=masks)
            obs, reward, done, truncated, info = env.step(int(action))
            episode_reward += reward
            final_info = info

        winner = final_info.get("winner")
        learner_id = args.learner_id
        learner_score = final_info.get("score", 0.0)
        final_scores.append(float(learner_score))
        episode_rewards.append(float(episode_reward))

        if isinstance(winner, list):
            if learner_id in winner:
                win_count += 1
                peace_win_count += 1
            else:
                loss_count += 1
        elif winner == learner_id:
            win_count += 1
        else:
            loss_count += 1

    summary = {
        "episodes": args.episodes,
        "win_rate": win_count / args.episodes if args.episodes else 0.0,
        "peace_win_rate": peace_win_count / args.episodes if args.episodes else 0.0,
        "loss_rate": loss_count / args.episodes if args.episodes else 0.0,
        "avg_episode_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "avg_final_score": float(np.mean(final_scores)) if final_scores else 0.0,
        "std_final_score": float(np.std(final_scores)) if final_scores else 0.0,
    }

    print(json.dumps(summary, indent=2))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
