#!/usr/bin/env python3
"""
Headless experiment runner for multi-LLM nation competition.

Usage:
  # Generate a sample config, edit it with your actual vLLM endpoints:
  python run_experiment.py --gen-config > my_experiment.json

  # Run with config file:
  python run_experiment.py -c my_experiment.json

  # Override specific params via CLI:
  python run_experiment.py -c my_experiment.json --seed 123 --max-turns 50

  # Quick run with defaults (uses ai/config.py values):
  python run_experiment.py
"""

import argparse
import json
import random
import time
import sys
from typing import Any

# ---------------------------------------------------------------------------
# JSON encoder (replicated from server.py to avoid importing the HTTP server)
# ---------------------------------------------------------------------------
from core.constants import Resource, DiplomaticState, NationAction, Tech, Civic


class GameEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (Resource, DiplomaticState, NationAction, Tech, Civic)):
            return obj.value
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return super().default(obj)


# ---------------------------------------------------------------------------
# Example config template
# ---------------------------------------------------------------------------
EXAMPLE_CONFIG = {
    "num_players": 5,
    "max_turns": 100,
    "seed": 42,
    "temperature": 0.7,
    "output": "game_export.jsonl",
    "verbose": False,
    "backends": {
        "server_1": {
            "provider": "vllm",
            "base_url": "http://localhost:8001",
            "model": "Qwen/Qwen2.5-7B-Instruct",
        },
        "server_2": {
            "provider": "vllm",
            "base_url": "http://localhost:8002",
            "model": "meta-llama/Llama-3.1-8B-Instruct",
        },
        "server_3": {
            "provider": "vllm",
            "base_url": "http://localhost:8003",
            "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        },
    },
    "nation_backend_map": {
        "0": "server_1",
        "1": "server_1",
        "2": "server_2",
        "3": "server_2",
        "4": "server_3",
    },
    "strategies": {
        "0": "neutral",
        "1": "expansionist",
        "2": "scientific",
        "3": "mercantile",
        "4": "diplomatic",
    },
}


def build_parser():
    p = argparse.ArgumentParser(
        description="Run a headless multi-LLM nation competition experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Tip: run with --gen-config to create a template config JSON.",
    )
    p.add_argument("-c", "--config", metavar="FILE",
                   help="JSON config file (see --gen-config for format)")
    p.add_argument("-n", "--num-players", type=int, default=None,
                   help="number of nations (default: 5)")
    p.add_argument("-t", "--max-turns", type=int, default=None,
                   help="stop after this many turns (default: 100)")
    p.add_argument("-s", "--seed", type=int, default=None,
                   help="random seed for reproducibility")
    p.add_argument("-o", "--output", default=None,
                   help="export JSONL path (default: game_export.jsonl)")
    p.add_argument("--temperature", type=float, default=None,
                   help="LLM sampling temperature (default: 0.7)")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="print per-turn event logs")
    p.add_argument("--gen-config", action="store_true",
                   help="print example config JSON to stdout and exit")
    return p


def resolve_config(args):
    """Merge config-file defaults with CLI overrides."""
    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)

    def pick(cli_val, cfg_key, fallback):
        if cli_val is not None:
            return cli_val
        return cfg.get(cfg_key, fallback)

    return {
        "num_players": pick(args.num_players, "num_players", 5),
        "max_turns":   pick(args.max_turns,   "max_turns",   100),
        "seed":        pick(args.seed,         "seed",        None),
        "temperature": pick(args.temperature,  "temperature", 0.7),
        "output":      pick(args.output,       "output",      "game_export.jsonl"),
        "verbose":     args.verbose or cfg.get("verbose", False),
        "backends":            cfg.get("backends", None),
        "nation_backend_map":  cfg.get("nation_backend_map", None),
        "strategies":          cfg.get("strategies", None),
    }


def apply_config_to_module(cfg):
    """Patch ai.config module attributes BEFORE the game is imported."""
    import ai.config as ai_cfg

    ai_cfg.AI_MODE = "llm"
    ai_cfg.LLM_TEMPERATURE = cfg["temperature"]

    if cfg["backends"] is not None:
        ai_cfg.LLM_BACKENDS = cfg["backends"]
    if cfg["nation_backend_map"] is not None:
        ai_cfg.NATION_BACKEND_MAP = {
            int(k): v for k, v in cfg["nation_backend_map"].items()
        }
    if cfg["strategies"] is not None:
        ai_cfg.NATION_STRATEGIES = {
            int(k): v for k, v in cfg["strategies"].items()
        }


def print_header(cfg, game):
    """Print a summary of the experiment setup."""
    import ai.config as ai_cfg

    sep = "=" * 64
    print(sep)
    print("  MULTI-LLM NATION COMPETITION — EXPERIMENT RUNNER")
    print(sep)
    print(f"  Players    : {cfg['num_players']}")
    print(f"  Max turns  : {cfg['max_turns']}")
    print(f"  Temperature: {cfg['temperature']}")
    print(f"  Seed       : {cfg['seed'] if cfg['seed'] is not None else 'random'}")
    print(f"  Output     : {cfg['output']}")
    print()

    print("  NATION ASSIGNMENTS:")
    for i in range(cfg["num_players"]):
        agent = game.agents.get(i)
        nation = game.state.nations[i]
        strategy = ai_cfg.NATION_STRATEGIES.get(i, "neutral")
        if hasattr(agent, "backend_id") and agent.backend_id:
            print(f"    Nation {i} | {nation.name:16s} | "
                  f"{agent.model_name} [{agent.backend_id}] | "
                  f"strategy={strategy}")
        else:
            print(f"    Nation {i} | {nation.name:16s} | "
                  f"rule-based                    | "
                  f"strategy={strategy}")
    print(sep)
    print()


def run_game_loop(game, cfg):
    """Run turns until winner or max_turns, return results dict."""
    num_players = cfg["num_players"]
    max_turns = cfg["max_turns"]
    output_path = cfg["output"]
    verbose = cfg["verbose"]
    ai_players = list(range(num_players))

    open(output_path, "w").close()

    total_start = time.time()
    turn_timings = []

    while True:
        winner = game.check_winner()
        if winner is not None:
            break
        if game.state.turn > max_turns:
            break

        t0 = time.time()
        current_turn = game.state.turn
        intent_logs = []

        # --- AI decision phase ---
        for ai in ai_players:
            if ai in game.state.nations and not game.state.nations[ai].is_defeated:
                agent = game.agents[ai]
                cmds = agent.decide_actions(game.state, game.handler)
                for c in cmds:
                    game.handler.queue_action(ai, c)
                if hasattr(agent, "last_reasoning") and agent.last_reasoning:
                    nation_name = game.state.nations[ai].name
                    intent_logs.append(
                        f"[INTENT] {nation_name}: {agent.last_reasoning}"
                    )

        # --- Snapshot for export ---
        state_snapshot = {"turn": current_turn, "agents": {}}
        for nid, n in game.state.nations.items():
            agent = game.agents.get(nid)
            agent_meta = {
                "state": game.state.get_symbolic_state(nid),
                "queued_actions": list(n.queued_actions),
            }
            if hasattr(agent, "backend_id") and agent.backend_id:
                agent_meta["backend_id"] = agent.backend_id
                agent_meta["model"] = agent.model_name
            state_snapshot["agents"][nid] = agent_meta

        with open(output_path, "a") as f:
            f.write(json.dumps(state_snapshot, cls=GameEncoder) + "\n")

        # --- Resolve turn ---
        logs = game.handler.resolve_simultaneous_turn()
        logs = intent_logs + logs

        # --- Feed memory ---
        resolved_turn = game.state.turn - 1
        for ai in ai_players:
            agent = game.agents.get(ai)
            if hasattr(agent, "update_memory"):
                agent.update_memory(resolved_turn, logs)

        elapsed = time.time() - t0
        turn_timings.append(elapsed)
        active = sum(1 for n in game.state.nations.values() if not n.is_defeated)

        if verbose:
            print(f"  Turn {resolved_turn:3d}  |  {active} alive  |  {elapsed:.1f}s")
            for line in logs:
                print(f"    {line}")
        else:
            sys.stdout.write(
                f"\r  Turn {resolved_turn:3d} / {max_turns}  |  "
                f"{active} nations alive  |  {elapsed:.1f}s    "
            )
            sys.stdout.flush()

    if not verbose:
        print()

    total_elapsed = time.time() - total_start
    return {
        "total_elapsed": total_elapsed,
        "turn_timings": turn_timings,
        "turns_played": game.state.turn - 1,
    }


def print_results(game, cfg, run_info):
    """Print final scoreboard and save summary."""
    import ai.config as ai_cfg

    sep = "=" * 64
    print()
    print(sep)
    print("  FINAL RESULTS")
    print(sep)

    winner = game.check_winner()
    if winner is not None:
        if isinstance(winner, list):
            names = ", ".join(game.state.nations[w].name for w in winner)
            print(f"  PEACE VICTORY: {names}")
        else:
            name = game.state.nations[winner].name
            active = sum(1 for n in game.state.nations.values() if not n.is_defeated)
            vtype = "DOMINATION" if active <= 1 else "SCORE"
            print(f"  {vtype} VICTORY: {name} (Nation {winner})")
    else:
        print("  No winner (max turns reached without decisive outcome)")

    print()
    print("  SCOREBOARD:")
    scores = {}
    for n in sorted(game.state.nations.values(), key=lambda n: n.id):
        score = (n.gold + n.manpower + n.production
                 + len(n.unlocked_techs) * 500
                 + len(n.unlocked_civics) * 500)
        scores[n.id] = score
        agent = game.agents.get(n.id)
        model = (agent.model_name
                 if hasattr(agent, "model_name") and agent.model_name
                 else "rulebased")
        status = "DEFEATED" if n.is_defeated else "ALIVE"
        print(f"    #{n.id} {n.name:16s}  score={score:6d}  {status:8s}  model={model}")

    total = run_info["total_elapsed"]
    turns = run_info["turns_played"]
    avg = total / turns if turns else 0
    print()
    print(f"  Turns played : {turns}")
    print(f"  Total time   : {total:.1f}s")
    print(f"  Avg per turn : {avg:.2f}s")
    print(f"  Export file  : {cfg['output']}")

    # Save machine-readable summary alongside export
    summary_path = cfg["output"].rsplit(".", 1)[0] + "_summary.json"
    summary = {
        "seed": cfg["seed"],
        "num_players": cfg["num_players"],
        "max_turns": cfg["max_turns"],
        "temperature": cfg["temperature"],
        "turns_played": turns,
        "total_time_seconds": round(total, 2),
        "winner": winner,
        "victory_type": None,
        "scores": scores,
        "nation_models": {},
    }
    if winner is not None:
        if isinstance(winner, list):
            summary["victory_type"] = "PEACE"
        else:
            active = sum(1 for n in game.state.nations.values() if not n.is_defeated)
            summary["victory_type"] = "DOMINATION" if active <= 1 else "SCORE"

    for nid in game.state.nations:
        agent = game.agents.get(nid)
        if hasattr(agent, "backend_id") and agent.backend_id:
            summary["nation_models"][nid] = {
                "backend_id": agent.backend_id,
                "model": agent.model_name,
            }
        else:
            summary["nation_models"][nid] = {
                "backend_id": "rulebased",
                "model": None,
            }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary file : {summary_path}")
    print(sep)


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.gen_config:
        print(json.dumps(EXAMPLE_CONFIG, indent=2))
        return

    cfg = resolve_config(args)

    # Set seed BEFORE any game imports (GameState.__init__ uses random)
    if cfg["seed"] is not None:
        random.seed(cfg["seed"])

    # Patch ai.config BEFORE importing the game (module-level imports bind at load time)
    apply_config_to_module(cfg)

    from main import CivilizationGame

    game = CivilizationGame(num_players=cfg["num_players"])

    print_header(cfg, game)
    run_info = run_game_loop(game, cfg)
    print_results(game, cfg, run_info)


if __name__ == "__main__":
    main()
