from __future__ import annotations

from typing import List

import numpy as np

from core.constants import Civic, DiplomaticState, Tech
from core.game_state import GameState


PERSONALITY_ORDER = ["WARMONGER", "BALANCED", "TECHNOCRAT", "OPPORTUNIST"]
RESOURCE_CAPS = {
    "gold": 5000.0,
    "manpower": 5000.0,
    "production": 3000.0,
    "science": 3000.0,
    "civics": 3000.0,
    "military": 3000.0,
    "war_exhaustion": 20.0,
    "yield": 250.0,
    "absorbed_yield": 250.0,
    "grievance": 300.0,
}

DIPLOMACY_ENCODING = {
    DiplomaticState.NEUTRAL: 0.0,
    DiplomaticState.ALLIANCE_PENDING: 1.0 / 3.0,
    DiplomaticState.ALLIED: 2.0 / 3.0,
    DiplomaticState.WAR: 1.0,
}


def encode_observation(
    state: GameState,
    learner_id: int,
    num_players: int,
    max_turns: int,
    queued_action_slots: List[int],
    catalog_size: int,
) -> np.ndarray:
    features: List[float] = []

    learner = state.nations[learner_id]
    features.append(_clip_ratio(state.turn, max_turns))
    grace_turns = getattr(state, "grace_period_turns", 25)
    features.append(1.0 if state.turn <= grace_turns else 0.0)
    features.append(_clip_ratio(learner.action_points, learner.max_action_points))

    for nation_id in range(num_players):
        nation = state.nations[nation_id]
        features.extend(_encode_nation_block(nation, learner_id == nation_id, num_players))

    for left_id in range(num_players):
        for right_id in range(num_players):
            relation = state.get_diplomatic_state(left_id, right_id)
            features.append(DIPLOMACY_ENCODING[relation])

    for left_id in range(num_players):
        for right_id in range(num_players):
            grievance = state.nations[left_id].grievances.get(right_id, 0)
            features.append(_clip_ratio(grievance, RESOURCE_CAPS["grievance"]))

    denom = float(catalog_size + 1)
    for action_id in queued_action_slots:
        features.append(0.0 if action_id < 0 else (action_id + 1) / denom)

    return np.array(features, dtype=np.float32)


def _encode_nation_block(nation, is_learner: bool, num_players: int) -> List[float]:
    features: List[float] = []
    features.append(1.0 if is_learner else 0.0)
    features.append(1.0 if nation.is_defeated else 0.0)
    features.extend(_one_hot_personality(nation.personality))

    features.extend(
        [
            _clip_ratio(nation.gold, RESOURCE_CAPS["gold"]),
            _clip_ratio(nation.manpower, RESOURCE_CAPS["manpower"]),
            _clip_ratio(nation.production, RESOURCE_CAPS["production"]),
            _clip_ratio(nation.science, RESOURCE_CAPS["science"]),
            _clip_ratio(nation.civics, RESOURCE_CAPS["civics"]),
            _clip_ratio(nation.military, RESOURCE_CAPS["military"]),
            _clip_ratio(nation.infrastructure_health, 100.0),
            _clip_ratio(nation.war_exhaustion, RESOURCE_CAPS["war_exhaustion"]),
            _clip_ratio(nation.gold_yield + nation.absorbed_gold_yield, RESOURCE_CAPS["yield"]),
            _clip_ratio(nation.manpower_yield, RESOURCE_CAPS["yield"]),
            _clip_ratio(nation.production_yield + nation.absorbed_prod_yield, RESOURCE_CAPS["yield"]),
            _clip_ratio(nation.science_yield + nation.absorbed_sci_yield, RESOURCE_CAPS["yield"]),
            _clip_ratio(nation.civic_yield, RESOURCE_CAPS["yield"]),
            _clip_ratio(len(nation.unlocked_techs), len(Tech)),
            _clip_ratio(len(nation.unlocked_civics), len(Civic)),
            _encode_named_progress(nation.current_tech, nation.tech_progress, Tech),
            _encode_named_progress(nation.current_civic, nation.civic_progress, Civic),
            _clip_ratio(len(nation.active_trade_agreements), num_players),
            _clip_ratio(len(nation.active_research_pacts), num_players),
            _clip_ratio(len(nation.pending_trade_agreements), num_players),
            _clip_ratio(len(nation.pending_research_pacts), num_players),
            _clip_ratio(len(nation.pending_joint_wars), num_players),
        ]
    )
    return features


def _encode_named_progress(current_name: str, progress: int, enum_cls) -> float:
    if not current_name:
        return 0.0

    names = [item.value for item in enum_cls]
    try:
        idx = names.index(current_name)
    except ValueError:
        idx = 0

    position = (idx + 1) / max(len(names), 1)
    return min(1.0, position + _clip_ratio(progress, RESOURCE_CAPS["science"]) * 0.1)


def _one_hot_personality(personality: str) -> List[float]:
    return [1.0 if personality == name else 0.0 for name in PERSONALITY_ORDER]


def _clip_ratio(value: float, denom: float) -> float:
    if denom <= 0:
        return 0.0
    return float(max(0.0, min(1.0, value / denom)))
