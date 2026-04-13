import random
import json
from typing import List, Optional, Tuple
from core.game_state import GameState
from engine.actions import ActionHandler
from core.constants import Resource, Tech, Civic, DiplomaticState, TECH_COSTS, CIVIC_COSTS
from ai.symbolic import parse_agent_response, AgentTurnResponse, GameAction


# Techs sorted by cost (cheapest first for fastest 500-point gains)
TECH_ORDER = sorted(
    [(t.value, TECH_COSTS.get(t.value, 100)) for t in Tech],
    key=lambda x: x[1],
)

# Civics sorted by cost
CIVIC_ORDER = sorted(
    [(c.value, CIVIC_COSTS.get(c.value, 100)) for c in Civic],
    key=lambda x: x[1],
)


# ---------------------------------------------------------------------------
# Rule-based strategy profiles.
# Each profile is a dict of overrides applied on top of the balanced defaults
# in AIAgent.decide_actions. Only keys that differ from balanced need to be
# listed — missing keys fall back to the defaults in `_DEFAULT_PROFILE`.
# ---------------------------------------------------------------------------
_DEFAULT_PROFILE = {
    "skip_military": False,        # if True, never declare war or strike
    "skip_skirmish": False,        # if True, never skirmish raids
    "skip_diplomacy": False,       # if True, never propose alliance/trade/research
    "eager_alliances": False,      # if True, propose alliance to every neutral immediately
    "prefer_research_pacts": False,
    "prefer_trade": False,
    "aggressive_invest": False,    # if True, invest down to threshold regardless of turn
    "invest_threshold_early": 200, # grace-period invest threshold
    "invest_threshold_late": 400,  # post-grace invest threshold
    "war_turn_threshold": 40,      # don't declare war before this turn
    "war_mp_threshold": 300,
    "war_prod_threshold": 200,
    "war_needs_military_tech": True,
    "war_target_infra_cap": 80,    # only attack if target infra < this
    "skirmish_turn_threshold": 30,
    "skirmish_mp_floor": 100,
    "harvest_priority": ["GOLD", "SCIENCE", "CIVICS", "MANPOWER"],
}

STRATEGY_PROFILES = {
    "balanced": {},
    "aggressor": {
        "skip_diplomacy": True,
        "war_turn_threshold": 20,
        "war_mp_threshold": 150,
        "war_prod_threshold": 100,
        "war_needs_military_tech": False,
        "war_target_infra_cap": 100,
        "skirmish_turn_threshold": 12,
        "skirmish_mp_floor": 40,
        "harvest_priority": ["MANPOWER", "PRODUCTION", "GOLD", "SCIENCE"],
        "invest_threshold_early": 300,
        "invest_threshold_late": 500,
    },
    "turtle": {
        "skip_military": True,
        "skip_skirmish": True,
        "eager_alliances": True,
        "invest_threshold_early": 200,
        "invest_threshold_late": 250,
        "aggressive_invest": True,
        "harvest_priority": ["GOLD", "SCIENCE", "CIVICS", "PRODUCTION"],
    },
    "scientist": {
        "skip_military": True,
        "skip_skirmish": True,
        "prefer_research_pacts": True,
        "invest_threshold_early": 400,
        "invest_threshold_late": 600,
        "harvest_priority": ["SCIENCE", "CIVICS", "GOLD", "MANPOWER"],
    },
    "trader": {
        "skip_military": True,
        "skip_skirmish": True,
        "prefer_trade": True,
        "aggressive_invest": True,
        "invest_threshold_early": 200,
        "invest_threshold_late": 250,
        "harvest_priority": ["GOLD", "PRODUCTION", "SCIENCE", "MANPOWER"],
    },
    "diplomat": {
        "skip_military": True,
        "skip_skirmish": True,
        "eager_alliances": True,
        "prefer_trade": True,
        "prefer_research_pacts": True,
        "invest_threshold_early": 300,
        "invest_threshold_late": 400,
        "harvest_priority": ["GOLD", "CIVICS", "SCIENCE", "MANPOWER"],
    },
}


def get_profile(strategy: str) -> dict:
    overrides = STRATEGY_PROFILES.get(strategy, {})
    merged = dict(_DEFAULT_PROFILE)
    merged.update(overrides)
    return merged


class AIAgent:
    def __init__(self, player_id: int, strategy: str = "balanced"):
        self.player_id = player_id
        self.strategy = strategy if strategy in STRATEGY_PROFILES else "balanced"
        self.profile = get_profile(self.strategy)

    def decide_actions(self, state: GameState, handler: ActionHandler) -> List[str]:
        sym_state = state.get_symbolic_state(self.player_id)
        if "error" in sym_state:
            return []

        me = sym_state["my_nation"]
        if me["is_defeated"]:
            return []

        prof = self.profile
        aps = me["actions"]["current_points"]
        turn = sym_state["global_state"]["turn"]
        stats = me["stats"]
        diplo = me["diplomacy"]
        others = [n for n in sym_state["other_nations"] if not n["is_defeated"]]

        # Diplomacy categorization
        at_war       = [n["id"] for n in others if n["diplomatic_status"] == "WAR"]
        allies       = [n["id"] for n in others if n["diplomatic_status"] == "ALLIED"]
        neutrals     = [n["id"] for n in others if n["diplomatic_status"] == "NEUTRAL"]
        pending_ally = [n["id"] for n in others if n["diplomatic_status"] == "ALLIANCE_PENDING"]

        # Local copies of pending proposals
        pending_trades   = list(diplo.get("pending_trade_agreements", []))
        pending_research = list(diplo.get("pending_research_pacts", []))
        pending_jwars    = list(diplo.get("pending_joint_wars", []))

        # Grace period: no foreign actions. Read the length from state
        # so rule-based opponents automatically adapt when the engine is
        # configured with a different grace period.
        grace_turns = getattr(state, "grace_period_turns", 25)
        in_grace = turn <= grace_turns
        if in_grace:
            at_war = []
            allies = []
            neutrals = []
            pending_ally = []
            pending_trades = []
            pending_research = []
            pending_jwars = []

        active_trades   = set(diplo.get("active_trade_agreements", []))
        active_research = set(diplo.get("active_research_pacts", []))

        # Available tech/civic (cheapest first)
        unlocked_techs  = set(me["tech"]["unlocked"])
        unlocked_civics = set(me["civic"]["unlocked"])
        avail_techs  = [t for t, _ in TECH_ORDER  if t not in unlocked_techs]
        avail_civics = [c for c, _ in CIVIC_ORDER if c not in unlocked_civics]

        has_research = me["tech"]["current"] is not None
        has_civic    = me["civic"]["current"] is not None

        proposed_trade = set()
        proposed_research = set()

        actions = []

        # ──────────────────────────────────────────────────
        #  Helper to add actions (respects AP limit)
        # ──────────────────────────────────────────────────
        def emit(action_dict) -> bool:
            nonlocal aps
            if aps <= 0:
                return False
            actions.append(action_dict)
            aps -= 1
            return True

        # ──────────────────────────────────────────────────
        #  PHASE 0: Accept incoming proposals (highest ROI)
        #  Trade: +200 gold + 15%/turn; Research: +150 sci + 15%/turn; Alliance: +150 mp
        # ──────────────────────────────────────────────────
        for tid in pending_trades:
            if aps <= 0:
                break
            emit({"action": "ACCEPT_TRADE", "target": tid})

        for tid in pending_research:
            if aps <= 0:
                break
            emit({"action": "ACCEPT_RESEARCH", "target": tid})

        for tid in pending_ally:
            if aps <= 0:
                break
            emit({"action": "ACCEPT_ALLIANCE", "target": tid})
            allies.append(tid)
            if tid in neutrals:
                neutrals.remove(tid)

        # Accept joint wars only if the enemy is already weak
        for p in pending_jwars:
            if aps <= 0:
                break
            enemy_info = next((n for n in others if n["id"] == p["enemy"]), None)
            if enemy_info and enemy_info["visible_status"]["infrastructure_health"] < 60:
                emit({"action": "ACCEPT_JOINT_WAR", "target": p["proposer"], "enemy": p["enemy"]})

        # ──────────────────────────────────────────────────
        #  PHASE 1: Always keep tech & civic pipelines running
        #  (Each completion = 500 score; cheapest first)
        # ──────────────────────────────────────────────────
        if not has_research and avail_techs and aps > 0:
            emit({"action": "RESEARCH", "target": avail_techs[0]})
            has_research = True

        if not has_civic and avail_civics and aps > 0:
            emit({"action": "PURSUE_CIVIC", "target": avail_civics[0]})
            has_civic = True

        # ──────────────────────────────────────────────────
        #  PHASE 2: Invest for compound growth (early game priority)
        #  INVEST order: SCIENCE → MANPOWER → INDUSTRY → CIVICS
        #  Early INVESTs have 70-90 turns to compound.
        # ──────────────────────────────────────────────────
        if not in_grace:
            invest_threshold = prof["invest_threshold_late"]
        else:
            invest_threshold = prof["invest_threshold_early"]

        invest_priority = ["SCIENCE", "MANPOWER", "INDUSTRY", "CIVICS"]
        while aps > 0 and stats["gold"] >= invest_threshold:
            # Aggressive investors keep going; default profile only invests
            # late-game if there is real surplus.
            if not prof["aggressive_invest"] and turn > 60 and stats["gold"] < 600:
                break
            target = invest_priority[min(len(actions) % len(invest_priority), len(invest_priority) - 1)]
            emit({"action": "INVEST", "target": target})
            stats["gold"] -= 200  # Track locally

        # ──────────────────────────────────────────────────
        #  PHASE 3: Diplomacy — propose agreements for yield modifiers
        #  Trade: +15% gold/turn per agreement
        #  Research: +15% science/turn per pact
        #  Alliance: prerequisite for joint wars
        # ──────────────────────────────────────────────────
        if not in_grace and not prof["skip_diplomacy"]:
            # Propose alliances to neutrals.
            # Eager_alliances profiles always propose; default also proposes,
            # so this branch is currently equivalent — kept for clarity.
            if prof["eager_alliances"] or True:
                for tid in neutrals[:]:
                    if aps <= 0:
                        break
                    emit({"action": "PROPOSE_ALLIANCE", "target": tid})

            # Propose trade to anyone without active agreement.
            # Trade-preferring profiles get a second pass at allies later if AP remain.
            all_partners = allies + neutrals
            trade_loops = 2 if prof["prefer_trade"] else 1
            for _ in range(trade_loops):
                for tid in all_partners:
                    if aps <= 0:
                        break
                    if tid not in active_trades and tid not in proposed_trade:
                        emit({"action": "PROPOSE_TRADE", "target": tid})
                        proposed_trade.add(tid)

            # Propose research pacts
            research_loops = 2 if prof["prefer_research_pacts"] else 1
            for _ in range(research_loops):
                for tid in all_partners:
                    if aps <= 0:
                        break
                    if tid not in active_research and tid not in proposed_research:
                        emit({"action": "PROPOSE_RESEARCH", "target": tid})
                        proposed_research.add(tid)

        # ──────────────────────────────────────────────────
        #  PHASE 4: Military — selective war for yield absorption
        #  Only when strong and target is weak.
        #  War generates 50 global grievances → time it carefully.
        # ──────────────────────────────────────────────────
        if not in_grace and aps > 0 and not prof["skip_military"]:
            # If at war, strike if resources allow
            if at_war:
                enemy_infos = [(n["id"], n["visible_status"]["infrastructure_health"])
                               for n in others if n["id"] in at_war]
                enemy_infos.sort(key=lambda x: x[1])

                while aps > 0 and stats["manpower"] >= 100 and stats["production"] >= 50 and enemy_infos:
                    target_id, _ = enemy_infos[0]
                    emit({"action": "MILITARY_STRIKE", "target": target_id})
                    stats["manpower"] -= 100
                    stats["production"] -= 50

            # Consider declaring war
            if not at_war and aps > 0 and turn > prof["war_turn_threshold"]:
                if prof["war_needs_military_tech"]:
                    has_military_tech = (
                        "Steel" in unlocked_techs or "Gunpowder" in unlocked_techs
                    )
                else:
                    has_military_tech = True
                strong_enough = (
                    stats["manpower"] >= prof["war_mp_threshold"]
                    and stats["production"] >= prof["war_prod_threshold"]
                    and has_military_tech
                )

                if strong_enough:
                    targets = [(n["id"], n["visible_status"]["infrastructure_health"])
                               for n in others
                               if n["id"] not in allies and not n["is_defeated"]]
                    targets.sort(key=lambda x: x[1])

                    if targets and targets[0][1] < prof["war_target_infra_cap"]:
                        target_id = targets[0][0]
                        emit({"action": "DECLARE_WAR", "target": target_id})
                        at_war.append(target_id)
                        if allies and aps > 0:
                            emit({"action": "PROPOSE_JOINT_WAR",
                                  "target": random.choice(allies), "enemy": target_id})

        # Skirmish raiding (governed independently so military-skipping profiles
        # can still raid if they want — currently they don't).
        if (not in_grace and aps > 0 and not prof["skip_skirmish"]
                and turn > prof["skirmish_turn_threshold"]
                and stats["manpower"] >= prof["skirmish_mp_floor"]):
            raid_targets = [n["id"] for n in others
                            if n["id"] not in allies
                            and n["visible_status"]["infrastructure_health"] > 30]
            if raid_targets and aps > 0 and stats["manpower"] >= 20:
                emit({"action": "SKIRMISH", "target": random.choice(raid_targets)})
                stats["manpower"] -= 20

        # ──────────────────────────────────────────────────
        #  PHASE 5: Fill remaining AP with harvests, ordered by profile.
        # ──────────────────────────────────────────────────
        harvest_order = prof["harvest_priority"]
        while aps > 0:
            picked = harvest_order[len(actions) % len(harvest_order)]
            emit({"action": "HARVEST", "target": picked})
            if picked == "GOLD":
                stats["gold"] += me["yields"].get("gold_yield", 15) * 2

        # Package and parse through symbolic pipeline
        agent_response = {
            "reasoning": f"Turn {turn}: optimal priority-based strategy.",
            "actions": actions,
        }
        engine_commands, err = parse_agent_response(json.dumps(agent_response))
        if err:
            print(f"AI {self.player_id} generated invalid actions: {err}")
            return []
        return engine_commands
