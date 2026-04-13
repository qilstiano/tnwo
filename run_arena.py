#!/usr/bin/env python3
"""
Universal Arena: run any mix of PPO / LLM / Rule-based agents and auto-visualize.

Usage examples:

  # 5 rule-based agents, 100 turns
  python run_arena.py \
    --agents rule rule rule rule rule \
    --max-turns 100

  # PPO (nation 0) vs 4 rule-based, 800 turns
  python run_arena.py \
    --agents ppo:outputs/ppo_vs_llm_20260406_100555/maskable_ppo_nation.zip \
             rule rule rule rule \
    --max-turns 800

  # PPO vs 2 LLM + 2 rule-based
  python run_arena.py \
    --agents ppo:outputs/model.zip \
             llm:vllm:http://localhost:8001:qwen-7b \
             llm:vllm:http://localhost:8001:qwen-7b \
             rule rule \
    --max-turns 100

  # Customize rule-based strategies
  python run_arena.py \
    --agents ppo:outputs/model.zip rule rule rule rule \
    --strategies neutral expansionist scientific mercantile diplomatic \
    --max-turns 800

Agent spec formats:
  rule                                          — rule-based AIAgent
  ppo:<model_path>                              — PPO MaskablePPO model
  llm:<provider>:<base_url>:<model_name>        — LLM agent (vllm/ollama)
"""

import argparse
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

from core.constants import Resource, DiplomaticState, NationAction, Tech, Civic
from core.game_state import GameState
from engine.actions import ActionHandler
from rl.reward import compute_score


class GameEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (Resource, DiplomaticState, NationAction, Tech, Civic)):
            return obj.value
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return super().default(obj)


# ─────────────────────────────────────────────────────────────────
# Agent wrappers
# ─────────────────────────────────────────────────────────────────

class PPOAgentWrapper:
    """Wraps a MaskablePPO model to act as a game agent."""

    def __init__(self, player_id: int, model_path: str, num_players: int,
                 max_turns: int = 100):
        from sb3_contrib import MaskablePPO
        from rl.action_space import build_action_catalog, build_action_mask, command_from_action
        from rl.encoding import encode_observation

        self.player_id = player_id
        self.model = MaskablePPO.load(model_path)
        self.num_players = num_players
        self.max_turns = max_turns
        self.action_catalog = build_action_catalog(num_players)
        self._build_action_mask = build_action_mask
        self._command_from_action = command_from_action
        self._encode_observation = encode_observation
        self.agent_type = "ppo"
        self.model_path = model_path
        self.last_reasoning = ""
        self.current_action_slots = [-1, -1, -1]

    def decide_actions(self, state: GameState, handler: ActionHandler) -> List[str]:
        """Return action commands (caller is responsible for queuing)."""
        nation = state.nations[self.player_id]
        if nation.is_defeated:
            return []

        commands = []
        max_actions = nation.max_action_points

        self.current_action_slots = [-1, -1, -1]
        for slot in range(max_actions):
            obs = self._encode_observation(
                state, self.player_id, self.num_players,
                self.max_turns, self.current_action_slots,
                len(self.action_catalog),
            )
            mask = self._build_action_mask(state, self.player_id, self.action_catalog)

            if not mask.any():
                break

            action, _ = self.model.predict(obs, deterministic=True, action_masks=mask)
            action_id = int(action)

            if not mask[action_id]:
                action_id = int(np.flatnonzero(mask)[0])

            cmd = self._command_from_action(self.action_catalog[action_id])
            # Queue so next iteration's mask reflects the queued action
            queued = handler.queue_action(self.player_id, cmd)
            if queued:
                commands.append(cmd)
                if slot < len(self.current_action_slots):
                    self.current_action_slots[slot] = action_id
            else:
                fallback = "HARVEST GOLD"
                handler.queue_action(self.player_id, fallback)
                commands.append(fallback)

        self.last_reasoning = f"PPO policy selected: {commands}"
        return commands


def parse_agent_spec(spec: str, player_id: int, num_players: int, max_turns: int = 100):
    """Parse agent spec string and return agent instance + metadata."""
    parts = spec.split(":")

    if parts[0] == "rule":
        from ai.agent import AIAgent
        agent = AIAgent(player_id)
        agent.agent_type = "rulebased"
        agent.last_reasoning = ""
        return agent, {"type": "rulebased"}

    elif parts[0] == "ppo":
        if len(parts) < 2:
            raise ValueError(f"PPO agent spec needs model path: ppo:<path>, got: {spec}")
        model_path = ":".join(parts[1:])  # Handle Windows paths with colons
        agent = PPOAgentWrapper(player_id, model_path, num_players, max_turns)
        return agent, {"type": "ppo", "model_path": model_path}

    elif parts[0] == "llm":
        if len(parts) < 4:
            raise ValueError(
                f"LLM agent spec: llm:<provider>:<base_url>:<model>, got: {spec}"
            )
        provider = parts[1]
        base_url = ":".join(parts[2:-1])  # Handle URL with port
        model_name = parts[-1]

        from ai.agent import AIAgent
        from ai.llm_client import create_llm_client
        from ai.llm_agent import LLMAgent

        client = create_llm_client(provider, base_url, model_name)
        fallback = AIAgent(player_id)
        agent = LLMAgent(
            player_id, client, fallback,
            backend_id=f"{provider}@{base_url}",
            model_name=model_name,
        )
        agent.agent_type = "llm"
        return agent, {"type": "llm", "provider": provider,
                       "base_url": base_url, "model": model_name}

    else:
        raise ValueError(f"Unknown agent type: {parts[0]}. Use rule/ppo/llm.")


# ─────────────────────────────────────────────────────────────────
# Game loop
# ─────────────────────────────────────────────────────────────────

def run_arena(agents, state, handler, max_turns, output_path, verbose,
              strategies_map, agent_meta):
    """Run the game loop and export JSONL snapshots."""
    open(output_path, "w").close()
    total_start = time.time()

    while True:
        winner = state.check_winner()
        if winner is not None:
            break
        if state.turn > max_turns:
            break

        t0 = time.time()
        current_turn = state.turn
        intent_logs = []

        # --- Agent decision phase ---
        for nid, agent in agents.items():
            nation = state.nations[nid]
            if nation.is_defeated:
                continue

            cmds = agent.decide_actions(state, handler)
            # PPO wrapper queues internally (needs mask updates per slot).
            # Rule-based and LLM return commands without queuing.
            atype = getattr(agent, "agent_type", "")
            if atype != "ppo":
                for c in cmds:
                    handler.queue_action(nid, c)

            reasoning = getattr(agent, "last_reasoning", "")
            if reasoning:
                intent_logs.append(f"[INTENT] {nation.name}: {reasoning}")

        # --- Pre-resolution snapshot ---
        snapshot = {"turn": current_turn, "phase": "pre", "agents": {}}
        for nid, nation in state.nations.items():
            agent = agents.get(nid)
            meta = {
                "state": state.get_symbolic_state(nid),
                "queued_actions": list(nation.queued_actions),
                "reasoning": getattr(agent, "last_reasoning", ""),
                "agent_type": agent_meta.get(nid, {}).get("type", "unknown"),
            }
            if hasattr(agent, "model_path"):
                meta["model_path"] = agent.model_path
            if hasattr(agent, "model_name"):
                meta["model"] = agent.model_name
            snapshot["agents"][nid] = meta

        with open(output_path, "a") as f:
            f.write(json.dumps(snapshot, cls=GameEncoder) + "\n")

        # --- Resolve turn ---
        resolved_logs = handler.resolve_simultaneous_turn()
        logs = intent_logs + resolved_logs

        # --- Post-resolution snapshot ---
        post_snapshot = {"turn": current_turn, "phase": "post", "agents": {}}
        for nid in state.nations:
            post_snapshot["agents"][nid] = {
                "state": state.get_symbolic_state(nid),
            }
        with open(output_path, "a") as f:
            f.write(json.dumps(post_snapshot, cls=GameEncoder) + "\n")

        elapsed = time.time() - t0
        resolved_turn = state.turn - 1
        active = sum(1 for n in state.nations.values() if not n.is_defeated)

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
        "turns_played": state.turn - 1,
    }


def print_results(state, agents, agent_meta, strategies_map, run_info, output_path):
    """Print final scoreboard and save summary JSON."""
    sep = "=" * 64
    print()
    print(sep)
    print("  FINAL RESULTS")
    print(sep)

    winner = state.check_winner()
    turns_played = run_info["turns_played"]

    # Determine victory type
    victory_type = None
    if winner is not None:
        if isinstance(winner, list):
            names = ", ".join(state.nations[w].name for w in winner)
            print(f"  PEACE VICTORY: {names}")
            victory_type = "PEACE"
        else:
            name = state.nations[winner].name
            active = sum(1 for n in state.nations.values() if not n.is_defeated)
            victory_type = "DOMINATION" if active <= 1 else "SCORE"
            print(f"  {victory_type} VICTORY: {name} (Nation {winner})")
    else:
        # Max turns reached — highest score wins
        scores = {}
        for n in state.nations.values():
            scores[n.id] = compute_score(state, n.id)
        best_id = max(scores, key=scores.get)
        victory_type = "SCORE"
        print(f"  SCORE VICTORY: {state.nations[best_id].name} (Nation {best_id})")
        winner = best_id

    print()
    print("  SCOREBOARD:")
    scores = {}
    for n in sorted(state.nations.values(), key=lambda n: n.id):
        score = compute_score(state, n.id)
        scores[n.id] = score
        meta = agent_meta.get(n.id, {})
        atype = meta.get("type", "?")
        strategy = strategies_map.get(n.id, "neutral")
        status = "DEFEATED" if n.is_defeated else "ALIVE"
        extra = ""
        if atype == "ppo":
            extra = os.path.basename(meta.get("model_path", ""))
        elif atype == "llm":
            extra = meta.get("model", "")
        print(f"    #{n.id} {n.name:16s}  score={score:6d}  {status:8s}  "
              f"type={atype:10s}  strategy={strategy:12s}  {extra}")

    total = run_info["total_elapsed"]
    avg = total / turns_played if turns_played else 0
    print()
    print(f"  Turns played : {turns_played}")
    print(f"  Total time   : {total:.1f}s")
    print(f"  Avg per turn : {avg:.2f}s")
    print(f"  Export file  : {output_path}")

    # Save summary JSON
    summary_path = output_path.rsplit(".", 1)[0] + "_summary.json"
    summary = {
        "turns_played": turns_played,
        "total_time_seconds": round(total, 2),
        "winner": winner,
        "victory_type": victory_type,
        "scores": scores,
        "agents": {
            nid: {
                **agent_meta.get(nid, {}),
                "strategy": strategies_map.get(nid, "neutral"),
                "name": state.nations[nid].name,
            }
            for nid in state.nations
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary file : {summary_path}")
    print(sep)

    return summary


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description="Universal Arena: run any mix of PPO / LLM / Rule-based agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--agents", nargs="+", required=True,
                   help="Agent specs: rule, ppo:<path>, llm:<provider>:<url>:<model>")
    p.add_argument("--strategies", nargs="*", default=None,
                   help="Strategy per agent: neutral, expansionist, scientific, mercantile, diplomatic")
    p.add_argument("--max-turns", type=int, default=100)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("-o", "--output", default=None,
                   help="JSONL export path (default: auto-generated)")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--no-viz", action="store_true",
                   help="Skip auto-visualization at the end")
    p.add_argument("--output-dir", default="outputs",
                   help="Directory for outputs (default: outputs)")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    num_players = len(args.agents)

    # Strategies
    strategies_map = {}
    if args.strategies:
        if len(args.strategies) != num_players:
            print(f"ERROR: --strategies needs {num_players} values, got {len(args.strategies)}")
            return
        for i, s in enumerate(args.strategies):
            strategies_map[i] = s
    else:
        default_strats = ["neutral", "expansionist", "scientific", "mercantile", "diplomatic"]
        for i in range(num_players):
            strategies_map[i] = default_strats[i % len(default_strats)]

    # Apply strategies to config BEFORE importing game modules that read it
    import ai.config as ai_cfg
    ai_cfg.NATION_STRATEGIES = strategies_map

    # Seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Output path
    os.makedirs(args.output_dir, exist_ok=True)
    if args.output:
        output_path = args.output
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        agent_types = "_".join(s.split(":")[0] for s in args.agents)
        output_path = os.path.join(args.output_dir, f"arena_{agent_types}_{ts}.jsonl")

    # Initialize game
    state = GameState(num_players)
    handler = ActionHandler(state)

    # Build agents
    agents = {}
    agent_meta = {}
    print("=" * 64)
    print("  ARENA SETUP")
    print("=" * 64)
    print(f"  Players    : {num_players}")
    print(f"  Max turns  : {args.max_turns}")
    print(f"  Seed       : {args.seed if args.seed is not None else 'random'}")
    print(f"  Output     : {output_path}")
    print()

    for i, spec in enumerate(args.agents):
        agent, meta = parse_agent_spec(spec, i, num_players, args.max_turns)
        agents[i] = agent
        agent_meta[i] = meta
        strategy = strategies_map.get(i, "neutral")
        nation_name = state.nations[i].name
        print(f"  Nation {i} | {nation_name:16s} | type={meta['type']:10s} | "
              f"strategy={strategy:12s} | {spec}")

    print("=" * 64)
    print()

    # Run
    run_info = run_arena(
        agents, state, handler, args.max_turns,
        output_path, args.verbose, strategies_map, agent_meta,
    )

    summary = print_results(
        state, agents, agent_meta, strategies_map, run_info, output_path,
    )

    # Auto-visualize
    if not args.no_viz:
        fig_dir = os.path.join(args.output_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        print()
        print(f"  Generating visualization...")
        try:
            from visualize import load_game_export, load_game_summary, plot_game_replay
            turns = load_game_export(output_path)
            plot_game_replay(turns, summary, os.path.join(
                fig_dir,
                os.path.basename(output_path).rsplit(".", 1)[0] + ".png"
            ))
        except Exception as e:
            print(f"  Visualization failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
