#!/usr/bin/env python3
"""
Train a model-free PPO nation policy with action masking.

Example:
  python train_ppo.py \
    --output-dir outputs/ppo_nation0 \
    --total-timesteps 200000
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from rl.env import NationEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Train a PPO policy for one TNWO nation.")
    parser.add_argument("--output-dir", required=True, help="Directory to save checkpoints and metadata")
    parser.add_argument("--total-timesteps", type=int, default=200000)
    parser.add_argument("--num-players", type=int, default=5)
    parser.add_argument("--learner-id", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--dense-reward-scale", type=float, default=0.01)
    parser.add_argument("--terminal-win-reward", type=float, default=100.0)
    parser.add_argument("--terminal-peace-reward", type=float, default=50.0)
    parser.add_argument("--terminal-loss-penalty", type=float, default=-100.0)
    parser.add_argument("--annex-bonus", type=float, default=20.0)
    parser.add_argument("--invalid-action-penalty", type=float, default=-1.0)
    parser.add_argument("--policy-hidden-size", type=int, default=256)
    parser.add_argument("--checkpoint-freq", type=int, default=20000,
                        help="Save a checkpoint every N timesteps")
    parser.add_argument("--tensorboard-log", default=None, help="Optional tensorboard log dir")

    # LLM opponent configuration
    parser.add_argument("--opponent-mode", choices=["rulebased", "llm"], default="rulebased",
                        help="Opponent type: rulebased (AIAgent) or llm (LLMAgent)")
    parser.add_argument("--opponent-provider", default="vllm",
                        help="LLM provider for all opponents (vllm or ollama)")
    parser.add_argument("--opponent-base-url", default="http://localhost:8001",
                        help="Base URL for the shared opponent LLM backend")
    parser.add_argument("--opponent-model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model name for the shared opponent LLM backend")
    parser.add_argument("--nation-backends-json", default=None,
                        help="Optional JSON file mapping nation_id -> {provider, base_url, model} "
                             "for per-nation backend overrides")
    return parser.parse_args()


def mask_fn(env):
    return env.action_masks()


class TrainingCurveLogger(BaseCallback):
    """Logs per-episode metrics to a CSV file and prints a rolling summary."""

    def __init__(self, log_path: Path, print_every: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self.log_path = log_path
        self.print_every = print_every
        self.csv_file = None
        self.csv_writer = None
        self.episode_count = 0
        self.episode_rewards: list = []
        self.episode_scores: list = []
        self.episode_wins: list = []
        self.episode_lengths: list = []
        self.start_time = 0.0
        self.fieldnames = [
            "episode", "timestep", "elapsed_sec",
            "ep_reward", "ep_length", "final_score",
            "won", "peace_win", "defeated",
            "rolling_reward_20", "rolling_score_20", "rolling_winrate_20",
        ]

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        self.csv_file = open(self.log_path, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
        self.csv_writer.writeheader()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" not in info:
                continue

            ep_info = info["episode"]
            self.episode_count += 1

            ep_reward = float(ep_info.get("r", 0.0))
            ep_length = int(ep_info.get("l", 0))
            final_score = float(info.get("score", 0.0))

            winner = info.get("winner")
            try:
                learner_id = self.training_env.envs[0].unwrapped.learner_id
            except (AttributeError, IndexError):
                learner_id = 0
            won = False
            peace_win = False
            defeated = bool(info.get("learner_defeated", False))
            if isinstance(winner, list):
                won = learner_id in winner
                peace_win = won
            elif winner is not None:
                won = winner == learner_id

            self.episode_rewards.append(ep_reward)
            self.episode_scores.append(final_score)
            self.episode_wins.append(1.0 if won else 0.0)
            self.episode_lengths.append(ep_length)

            window = 20
            rolling_reward = float(np.mean(self.episode_rewards[-window:]))
            rolling_score = float(np.mean(self.episode_scores[-window:]))
            rolling_winrate = float(np.mean(self.episode_wins[-window:]))

            row: Dict = {
                "episode": self.episode_count,
                "timestep": self.num_timesteps,
                "elapsed_sec": round(time.time() - self.start_time, 1),
                "ep_reward": round(ep_reward, 3),
                "ep_length": ep_length,
                "final_score": round(final_score, 1),
                "won": int(won),
                "peace_win": int(peace_win),
                "defeated": int(defeated),
                "rolling_reward_20": round(rolling_reward, 3),
                "rolling_score_20": round(rolling_score, 1),
                "rolling_winrate_20": round(rolling_winrate, 3),
            }
            if self.csv_writer is not None:
                self.csv_writer.writerow(row)
                self.csv_file.flush()

            if self.episode_count % self.print_every == 0:
                print(
                    f"  [ep {self.episode_count:5d} | ts {self.num_timesteps:8d}] "
                    f"reward={rolling_reward:+8.2f}  score={rolling_score:8.0f}  "
                    f"winrate={rolling_winrate:.1%}  len={np.mean(self.episode_lengths[-window:]):.0f}"
                )

        return True

    def _on_training_end(self) -> None:
        if self.csv_file is not None:
            self.csv_file.close()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build opponent backend config
    opponent_backend = None
    nation_backends = None
    if args.opponent_mode == "llm":
        opponent_backend = {
            "provider": args.opponent_provider,
            "base_url": args.opponent_base_url,
            "model": args.opponent_model,
        }
        if args.nation_backends_json:
            with open(args.nation_backends_json, "r") as f:
                raw = json.load(f)
            nation_backends = {int(k): v for k, v in raw.items()}

    base_env = NationEnv(
        num_players=args.num_players,
        learner_id=args.learner_id,
        max_turns=args.max_turns,
        dense_reward_scale=args.dense_reward_scale,
        terminal_win_reward=args.terminal_win_reward,
        terminal_peace_reward=args.terminal_peace_reward,
        terminal_loss_penalty=args.terminal_loss_penalty,
        annex_bonus=args.annex_bonus,
        invalid_action_penalty=args.invalid_action_penalty,
        seed=args.seed,
        opponent_mode=args.opponent_mode,
        opponent_backend=opponent_backend,
        nation_backends=nation_backends,
    )
    env = ActionMasker(base_env, mask_fn)

    policy_kwargs = {
        "net_arch": dict(
            pi=[args.policy_hidden_size, args.policy_hidden_size],
            vf=[args.policy_hidden_size, args.policy_hidden_size],
        )
    }

    tb_log = args.tensorboard_log
    if tb_log is None:
        tb_log = str(output_dir / "tb_logs")

    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        tensorboard_log=tb_log,
        policy_kwargs=policy_kwargs,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(output_dir / "checkpoints"),
        name_prefix="ppo_nation",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    curve_cb = TrainingCurveLogger(
        log_path=output_dir / "training_curve.csv",
        print_every=10,
    )

    opponent_desc = "rulebased"
    if args.opponent_mode == "llm":
        opponent_desc = f"llm ({args.opponent_model} @ {args.opponent_base_url})"
    print(f"Training PPO | timesteps={args.total_timesteps} | obs_dim={base_env.observation_space.shape[0]} | act_dim={base_env.action_space.n}")
    print(f"Opponents: {opponent_desc}")
    print(f"Checkpoints -> {output_dir / 'checkpoints'}")
    print(f"Training curve -> {output_dir / 'training_curve.csv'}")
    print(f"Tensorboard -> {tb_log}")

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_cb, curve_cb],
        progress_bar=True,
    )

    model_path = output_dir / "maskable_ppo_nation"
    model.save(model_path)

    metadata = {
        "total_timesteps": args.total_timesteps,
        "num_players": args.num_players,
        "learner_id": args.learner_id,
        "max_turns": args.max_turns,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "dense_reward_scale": args.dense_reward_scale,
        "terminal_win_reward": args.terminal_win_reward,
        "terminal_peace_reward": args.terminal_peace_reward,
        "terminal_loss_penalty": args.terminal_loss_penalty,
        "annex_bonus": args.annex_bonus,
        "invalid_action_penalty": args.invalid_action_penalty,
        "policy_hidden_size": args.policy_hidden_size,
        "checkpoint_freq": args.checkpoint_freq,
        "opponent_mode": args.opponent_mode,
        "opponent_provider": args.opponent_provider if args.opponent_mode == "llm" else None,
        "opponent_base_url": args.opponent_base_url if args.opponent_mode == "llm" else None,
        "opponent_model": args.opponent_model if args.opponent_mode == "llm" else None,
        "observation_dim": int(base_env.observation_space.shape[0]),
        "action_dim": int(base_env.action_space.n),
        "episodes_trained": curve_cb.episode_count,
    }
    (output_dir / "train_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved final model to {model_path}")
    print(f"Total episodes: {curve_cb.episode_count}")


if __name__ == "__main__":
    main()
