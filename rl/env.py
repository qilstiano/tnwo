from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ai.agent import AIAgent
from ai.llm_agent import LLMAgent
from ai.llm_client import create_llm_client
from core.game_state import GameState
from engine.actions import ActionHandler
from rl.action_space import build_action_catalog, build_action_mask, command_from_action
from rl.encoding import encode_observation
from rl.reward import compute_reward, compute_score, determine_terminal_winner


class NationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        num_players: int = 5,
        learner_id: int = 0,
        max_turns: int = 100,
        dense_reward_scale: float = 0.01,
        terminal_win_reward: float = 100.0,
        terminal_peace_reward: float = 50.0,
        terminal_loss_penalty: float = -100.0,
        annex_bonus: float = 20.0,
        invalid_action_penalty: float = -1.0,
        seed: Optional[int] = None,
        opponent_mode: str = "rulebased",
        opponent_backend: Optional[Dict[str, str]] = None,
        nation_backends: Optional[Dict[int, Dict[str, str]]] = None,
        nation_strategies: Optional[Dict[int, str]] = None,
        rule_opponent_strategies: Optional[Any] = None,
        grace_period_turns: int = 25,
        starting_asymmetry: str = "current",
        starting_action_points: int = 3,
    ):
        super().__init__()
        if learner_id >= num_players:
            raise ValueError("learner_id must be smaller than num_players")
        if starting_action_points < 1:
            raise ValueError("starting_action_points must be >= 1")

        self.num_players = num_players
        self.learner_id = learner_id
        self.max_turns = max_turns
        self.dense_reward_scale = dense_reward_scale
        self.terminal_win_reward = terminal_win_reward
        self.terminal_peace_reward = terminal_peace_reward
        self.terminal_loss_penalty = terminal_loss_penalty
        self.annex_bonus = annex_bonus
        self.invalid_action_penalty = invalid_action_penalty
        self.base_seed = seed
        self.opponent_mode = opponent_mode
        self.opponent_backend = opponent_backend
        self.nation_backends = nation_backends
        self.nation_strategies = nation_strategies
        self.rule_opponent_strategies = self._normalize_rule_strategies(
            rule_opponent_strategies, num_players, learner_id
        )
        self.grace_period_turns = int(grace_period_turns)
        self.starting_asymmetry = starting_asymmetry
        self.starting_action_points = int(starting_action_points)

        self.action_catalog = build_action_catalog(num_players)
        self.action_space = spaces.Discrete(len(self.action_catalog))

        random.seed(seed)
        sample_state = GameState(
            num_players,
            grace_period_turns=self.grace_period_turns,
            starting_asymmetry=self.starting_asymmetry,
            starting_action_points=self.starting_action_points,
        )
        sample_obs = encode_observation(
            sample_state,
            learner_id=self.learner_id,
            num_players=self.num_players,
            max_turns=self.max_turns,
            queued_action_slots=[-1] * self.starting_action_points,
            catalog_size=len(self.action_catalog),
        )
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=sample_obs.shape,
            dtype=np.float32,
        )

        self.state: Optional[GameState] = None
        self.handler: Optional[ActionHandler] = None
        self.opponents: Dict[int, Any] = {}
        self.current_action_slots = [-1] * self.starting_action_points
        self.prev_score = 0
        self.last_logs: List[str] = []
        self.done = False

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        reset_seed = seed if seed is not None else self.base_seed
        if reset_seed is not None:
            random.seed(int(reset_seed))
            np.random.seed(int(reset_seed))

        self.state = GameState(
            self.num_players,
            grace_period_turns=self.grace_period_turns,
            starting_asymmetry=self.starting_asymmetry,
            starting_action_points=self.starting_action_points,
        )
        self.handler = ActionHandler(self.state)
        self.opponents = self._build_opponents()
        self.current_action_slots = [-1] * self.starting_action_points
        self.prev_score = compute_score(self.state, self.learner_id)
        self.last_logs = []
        self.done = False

        obs = self._get_obs()
        info = {
            "score": self.prev_score,
            "turn": self.state.turn,
            "action_mask": self.action_masks(),
        }
        return obs, info

    def step(self, action: int):
        if self.done or self.state is None or self.handler is None:
            raise RuntimeError("Environment must be reset before calling step().")

        reward = 0.0
        terminated = False
        truncated = False
        learner = self.state.nations[self.learner_id]
        mask = self.action_masks()

        invalid_action = not bool(mask[action])
        applied_action_id = action
        applied_command = ""

        if invalid_action:
            reward += self.invalid_action_penalty
            applied_action_id = int(np.flatnonzero(mask)[0])

        applied_command = command_from_action(self.action_catalog[applied_action_id])
        queued = self.handler.queue_action(self.learner_id, applied_command)
        if not queued:
            reward += self.invalid_action_penalty
            fallback_id = self._fallback_action_id()
            applied_action_id = fallback_id
            applied_command = command_from_action(self.action_catalog[fallback_id])
            self.handler.queue_action(self.learner_id, applied_command)

        slot_idx = len(learner.queued_actions) - 1
        if 0 <= slot_idx < len(self.current_action_slots):
            self.current_action_slots[slot_idx] = applied_action_id

        info: Dict[str, Any] = {
            "invalid_action": invalid_action,
            "applied_action_id": applied_action_id,
            "applied_command": applied_command,
            "resolved_logs": [],
            "score": self.prev_score,
        }

        if learner.action_points <= 0 or len(learner.queued_actions) >= learner.max_action_points:
            resolved_logs = self._resolve_turn()
            info["resolved_logs"] = resolved_logs

            next_score = compute_score(self.state, self.learner_id)
            max_turns_reached = self.state.turn > self.max_turns
            winner = determine_terminal_winner(self.state, max_turns_reached=max_turns_reached)

            learner_defeated = self.state.nations[self.learner_id].is_defeated
            terminated = winner is not None or learner_defeated
            truncated = max_turns_reached and winner is None
            done = terminated or truncated
            info["learner_defeated"] = learner_defeated

            shaped_reward, reward_info = compute_reward(
                prev_score=self.prev_score,
                next_score=next_score,
                learner_id=self.learner_id,
                learner_name=self.state.nations[self.learner_id].name,
                next_state=self.state,
                winner=winner,
                done=done,
                resolved_logs=resolved_logs,
                dense_reward_scale=self.dense_reward_scale,
                terminal_win_reward=self.terminal_win_reward,
                terminal_peace_reward=self.terminal_peace_reward,
                terminal_loss_penalty=self.terminal_loss_penalty,
                annex_bonus=self.annex_bonus,
            )
            reward += shaped_reward
            info.update(reward_info)
            info["score"] = next_score
            info["winner"] = winner
            info["turn"] = self.state.turn
            self.prev_score = next_score
            self.current_action_slots = [-1] * self.starting_action_points

        obs = self._get_obs()
        info["action_mask"] = self.action_masks()
        self.done = terminated or truncated
        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        if self.done or self.state is None:
            return np.zeros(self.action_space.n, dtype=bool)
        return build_action_mask(self.state, self.learner_id, self.action_catalog)

    def _resolve_turn(self) -> List[str]:
        assert self.state is not None
        assert self.handler is not None

        for opponent_id, opponent in self.opponents.items():
            if self.state.nations[opponent_id].is_defeated:
                continue
            commands = opponent.decide_actions(self.state, self.handler)
            for command in commands:
                self.handler.queue_action(opponent_id, command)

        self.last_logs = self.handler.resolve_simultaneous_turn()

        if self.opponent_mode == "llm":
            for opponent in self.opponents.values():
                if isinstance(opponent, LLMAgent):
                    opponent.update_memory(
                        self.state.turn, self.last_logs, self.state
                    )

        return list(self.last_logs)

    @staticmethod
    def _normalize_rule_strategies(value, num_players: int, learner_id: int) -> Dict[int, str]:
        """Accept either a list aligned with opponent order or a {nid: strategy} dict.

        Returns a dict {opponent_nid: strategy_name}. Missing entries default to
        'balanced'.
        """
        opponent_ids = [nid for nid in range(num_players) if nid != learner_id]
        result: Dict[int, str] = {nid: "balanced" for nid in opponent_ids}
        if value is None:
            return result
        if isinstance(value, dict):
            for k, v in value.items():
                k_int = int(k)
                if k_int in result:
                    result[k_int] = str(v)
            return result
        if isinstance(value, (list, tuple)):
            for nid, strat in zip(opponent_ids, value):
                result[nid] = str(strat)
            return result
        raise ValueError(
            f"rule_opponent_strategies must be list, dict, or None — got {type(value)}"
        )

    def _build_opponents(self) -> Dict[int, Any]:
        opponent_ids = [
            nid for nid in range(self.num_players) if nid != self.learner_id
        ]

        if self.opponent_mode != "llm":
            return {
                nid: AIAgent(nid, strategy=self.rule_opponent_strategies[nid])
                for nid in opponent_ids
            }

        from ai.config import NATION_STRATEGIES as DEFAULT_STRATEGIES

        strategies = self.nation_strategies or DEFAULT_STRATEGIES
        opponents: Dict[int, Any] = {}

        for nid in opponent_ids:
            fallback = AIAgent(nid)

            backend_cfg = None
            if self.nation_backends and nid in self.nation_backends:
                backend_cfg = self.nation_backends[nid]
            elif self.opponent_backend:
                backend_cfg = self.opponent_backend

            if backend_cfg is None:
                opponents[nid] = fallback
                continue

            client = create_llm_client(
                provider=backend_cfg["provider"],
                base_url=backend_cfg["base_url"],
                model=backend_cfg["model"],
            )
            opponents[nid] = LLMAgent(
                player_id=nid,
                client=client,
                fallback=fallback,
                backend_id=backend_cfg.get("backend_id", ""),
                model_name=backend_cfg["model"],
            )

        return opponents

    def _fallback_action_id(self) -> int:
        if self.state is None:
            return 0

        mask = build_action_mask(self.state, self.learner_id, self.action_catalog)
        for preferred in ("HARVEST GOLD", "HARVEST MANPOWER", "HARVEST PRODUCTION"):
            for spec in self.action_catalog:
                if spec.label == preferred and mask[spec.idx]:
                    return spec.idx

        valid = np.flatnonzero(mask)
        return int(valid[0]) if len(valid) else 0

    def _get_obs(self) -> np.ndarray:
        assert self.state is not None
        return encode_observation(
            self.state,
            learner_id=self.learner_id,
            num_players=self.num_players,
            max_turns=self.max_turns,
            queued_action_slots=self.current_action_slots,
            catalog_size=len(self.action_catalog),
        )
