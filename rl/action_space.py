from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from core.constants import Civic, DiplomaticState, Tech
from core.game_state import GameState


HARVEST_ACTIONS = ["GOLD", "MANPOWER", "PRODUCTION", "SCIENCE", "CIVICS"]
INVEST_ACTIONS = ["MANPOWER", "INDUSTRY", "SCIENCE", "CIVICS", "MILITARY"]
SINGLE_TARGET_ACTIONS = [
    "PROPOSE_ALLIANCE",
    "ACCEPT_ALLIANCE",
    "CANCEL_ALLIANCE",
    "DECLARE_WAR",
    "MILITARY_STRIKE",
    "SABOTAGE",
    "SKIRMISH",
    "PROPOSE_TRADE",
    "ACCEPT_TRADE",
    "PROPOSE_RESEARCH",
    "ACCEPT_RESEARCH",
]
JOINT_WAR_ACTIONS = ["PROPOSE_JOINT_WAR", "ACCEPT_JOINT_WAR"]


@dataclass(frozen=True)
class ActionSpec:
    idx: int
    command_type: str
    value: Optional[str] = None
    target: Optional[int] = None
    enemy: Optional[int] = None

    @property
    def is_foreign(self) -> bool:
        return self.command_type in SINGLE_TARGET_ACTIONS or self.command_type in JOINT_WAR_ACTIONS

    @property
    def label(self) -> str:
        if self.command_type in {"HARVEST", "INVEST", "RESEARCH", "PURSUE_CIVIC"}:
            return f"{self.command_type} {self.value}"
        if self.command_type in JOINT_WAR_ACTIONS:
            return f"{self.command_type} {self.target} {self.enemy}"
        return f"{self.command_type} {self.target}"


def build_action_catalog(num_players: int) -> List[ActionSpec]:
    catalog: List[ActionSpec] = []
    idx = 0

    for value in HARVEST_ACTIONS:
        catalog.append(ActionSpec(idx=idx, command_type="HARVEST", value=value))
        idx += 1

    for value in INVEST_ACTIONS:
        catalog.append(ActionSpec(idx=idx, command_type="INVEST", value=value))
        idx += 1

    for tech in Tech:
        catalog.append(ActionSpec(idx=idx, command_type="RESEARCH", value=tech.value))
        idx += 1

    for civic in Civic:
        catalog.append(ActionSpec(idx=idx, command_type="PURSUE_CIVIC", value=civic.value))
        idx += 1

    for target in range(num_players):
        for command_type in SINGLE_TARGET_ACTIONS:
            catalog.append(ActionSpec(idx=idx, command_type=command_type, target=target))
            idx += 1

    for target in range(num_players):
        for enemy in range(num_players):
            if target == enemy:
                continue
            for command_type in JOINT_WAR_ACTIONS:
                catalog.append(
                    ActionSpec(
                        idx=idx,
                        command_type=command_type,
                        target=target,
                        enemy=enemy,
                    )
                )
                idx += 1

    return catalog


def command_from_action(spec: ActionSpec) -> str:
    if spec.command_type in {"HARVEST", "INVEST", "RESEARCH", "PURSUE_CIVIC"}:
        return f"{spec.command_type} {spec.value}"
    if spec.command_type in JOINT_WAR_ACTIONS:
        return f"{spec.command_type} {spec.target} {spec.enemy}"
    return f"{spec.command_type} {spec.target}"


def build_action_mask(state: GameState, player_id: int, catalog: List[ActionSpec]) -> np.ndarray:
    mask = np.zeros(len(catalog), dtype=bool)
    for spec in catalog:
        mask[spec.idx] = is_action_valid(state, player_id, spec)
    return mask


def is_action_valid(state: GameState, player_id: int, spec: ActionSpec) -> bool:
    nation = state.nations[player_id]
    if nation.is_defeated or len(nation.queued_actions) >= nation.max_action_points:
        return False

    if spec.command_type == "HARVEST":
        return True

    if spec.command_type == "INVEST":
        return nation.gold >= 200

    if spec.command_type == "RESEARCH":
        return nation.current_tech is None and spec.value not in nation.unlocked_techs

    if spec.command_type == "PURSUE_CIVIC":
        return nation.current_civic is None and spec.value not in nation.unlocked_civics

    grace_turns = getattr(state, "grace_period_turns", 25)
    if spec.is_foreign and state.turn <= grace_turns:
        return False

    if spec.command_type in SINGLE_TARGET_ACTIONS:
        return _is_single_target_action_valid(state, player_id, spec)

    if spec.command_type in JOINT_WAR_ACTIONS:
        return _is_joint_war_action_valid(state, player_id, spec)

    return False


def _is_single_target_action_valid(state: GameState, player_id: int, spec: ActionSpec) -> bool:
    target = spec.target
    if target is None or target == player_id or target not in state.nations:
        return False

    target_nation = state.nations[target]
    if target_nation.is_defeated:
        return False

    nation = state.nations[player_id]
    relation = state.get_diplomatic_state(player_id, target)
    command = spec.command_type

    if command == "PROPOSE_ALLIANCE":
        return relation not in (DiplomaticState.ALLIED, DiplomaticState.WAR, DiplomaticState.ALLIANCE_PENDING)
    if command == "ACCEPT_ALLIANCE":
        return relation == DiplomaticState.ALLIANCE_PENDING
    if command == "CANCEL_ALLIANCE":
        return relation == DiplomaticState.ALLIED
    if command == "DECLARE_WAR":
        return relation != DiplomaticState.WAR
    if command == "MILITARY_STRIKE":
        return relation == DiplomaticState.WAR and nation.manpower >= 100 and nation.production >= 50
    if command == "SABOTAGE":
        return nation.gold >= 50
    if command == "SKIRMISH":
        return nation.manpower >= 20
    if command == "PROPOSE_TRADE":
        return relation != DiplomaticState.WAR and target not in nation.active_trade_agreements
    if command == "ACCEPT_TRADE":
        return target in nation.pending_trade_agreements
    if command == "PROPOSE_RESEARCH":
        return relation != DiplomaticState.WAR and target not in nation.active_research_pacts
    if command == "ACCEPT_RESEARCH":
        return target in nation.pending_research_pacts

    return False


def _is_joint_war_action_valid(state: GameState, player_id: int, spec: ActionSpec) -> bool:
    ally = spec.target
    enemy = spec.enemy
    if (
        ally is None
        or enemy is None
        or ally == player_id
        or enemy == player_id
        or ally == enemy
    ):
        return False

    if ally not in state.nations or enemy not in state.nations:
        return False

    ally_nation = state.nations[ally]
    enemy_nation = state.nations[enemy]
    if ally_nation.is_defeated or enemy_nation.is_defeated:
        return False

    nation = state.nations[player_id]

    if spec.command_type == "PROPOSE_JOINT_WAR":
        return state.get_diplomatic_state(player_id, ally) == DiplomaticState.ALLIED

    if spec.command_type == "ACCEPT_JOINT_WAR":
        return any(
            proposal["proposer"] == ally and proposal["enemy"] == enemy
            for proposal in nation.pending_joint_wars
        )

    return False
