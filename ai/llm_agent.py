import json
from typing import List, Optional, Tuple
import re

from core.game_state import GameState
from core.constants import Resource, DiplomaticState, Tech, Civic, TECH_COSTS, CIVIC_COSTS
from engine.actions import ActionHandler
from ai.agent import AIAgent
from ai.config import LLM_TEMPERATURE, MAX_RETRIES, STRATEGY_LIBRARY, NATION_STRATEGIES

VALID_RESOURCES = {"GOLD", "MANPOWER", "PRODUCTION", "SCIENCE", "CIVICS"}
VALID_TECHS = {t.value for t in Tech}
VALID_CIVICS = {c.value for c in Civic}

class LLMAgent:
    def __init__(self, player_id: int, client, fallback: AIAgent,
                 backend_id: str = "", model_name: str = ""):
        self.player_id = player_id
        self.client = client
        self.fallback = fallback
        self.backend_id = backend_id
        self.model_name = model_name
        self.last_reasoning = ""
        self.event_memory: list = []
        self.archived_event_summaries: dict = {}

    def decide_actions(self, state: GameState, handler: ActionHandler) -> List[str]:
        nation = state.nations[self.player_id]
        if nation.is_defeated:
            return []

        actions, reasoning = self._call_llm_with_retry(state)

        if actions is None:
            self.last_reasoning = "(Fallback: rule-based agent)"
            return self.fallback.decide_actions(state, handler)

        self.last_reasoning = reasoning
        return actions

    # Events that matter for long-term memory (wars, alliances, deaths, betrayals)
    SIGNIFICANT_KEYWORDS = [
        "WAR!", "declared war", "FALLEN", "Alliance", "annexed",
        "struck", "Trade Agreement", "Research Pact",
        "SABOTAGE", "SKIRMISH", "CANCEL", "JOINT WAR",
        "Peace", "Victory", "PARIAH", "BETRAYAL", "BORDER CONFLICT",
        "SHADOW WAR", "proposed an Alliance", "proposed a Trade Agreement",
        "proposed a Research Pact", "requested",
    ]

    VISIBLE_EVENT_KEYWORDS = [
        "proposed an Alliance",
        "proposed a Trade Agreement",
        "proposed a Research Pact",
        "requested",
        "formed an Alliance",
        "signed a Trade Agreement",
        "signed a Research Pact",
        "declared war",
        "WAR!",
        "JOINT WAR",
        "struck",
        "sabotaged",
        "skirmished",
        "raided",
        "BETRAYAL",
        "SHADOW WAR",
        "BORDER CONFLICT",
        "HAS FALLEN",
        "annexed",
    ]

    def update_memory(self, turn: int, events: List[str], state: Optional[GameState] = None):
        if state is None or self.player_id not in state.nations:
            return

        my_name = state.nations[self.player_id].name
        for event in events:
            if not self._is_visible_event(event, my_name):
                continue

            self.event_memory.append({
                "turn": turn,
                "text": event,
                "is_significant": any(kw in event for kw in self.SIGNIFICANT_KEYWORDS),
            })

        self._compact_memory(turn, my_name)

    def clear_memory(self):
        self.event_memory.clear()
        self.archived_event_summaries.clear()

    def _is_visible_event(self, event: str, my_name: str) -> bool:
        if event.startswith("[INTENT]"):
            return False
        if my_name not in event:
            return False
        return any(keyword in event for keyword in self.VISIBLE_EVENT_KEYWORDS)

    def _compact_memory(self, current_turn: int, my_name: str, recent_turns: int = 5):
        cutoff = current_turn - recent_turns
        kept_recent = []

        for entry in self.event_memory:
            if entry["turn"] >= cutoff:
                kept_recent.append(entry)
                continue

            if not entry["is_significant"]:
                continue

            summary = self._summarize_event(entry["text"], my_name)
            bucket = self.archived_event_summaries.setdefault(
                summary,
                {"count": 0, "last_turn": entry["turn"]},
            )
            bucket["count"] += 1
            bucket["last_turn"] = max(bucket["last_turn"], entry["turn"])

        self.event_memory = kept_recent

    def _summarize_event(self, event: str, my_name: str) -> str:
        patterns = [
            (r"WAR! (.+) declared war on (.+)!", "war"),
            (r"(.+) proposed an Alliance to (.+)\.", "alliance_proposal"),
            (r"(.+) proposed a Trade Agreement to (.+)\.", "trade_proposal"),
            (r"(.+) proposed a Research Pact to (.+)\.", "research_proposal"),
            (r"(.+) and (.+) formed an Alliance!.*", "alliance_formed"),
            (r"(.+) and (.+) signed a Trade Agreement.*", "trade_signed"),
            (r"(.+) and (.+) signed a Research Pact.*", "research_signed"),
            (r"JOINT WAR: (.+) answered (.+)'s call and declared war on (.+)!", "joint_war"),
            (r"(.+) struck (.+)!.*", "strike"),
            (r"\*\*\* (.+) HAS FALLEN! (.+) annexed.*", "fallen"),
            (r"BETRAYAL: (.+) broke their Alliance with (.+)!", "alliance_broken"),
            (r"BETRAYAL: (.+) sabotaged their ally (.+)!.*", "ally_sabotage"),
            (r"SHADOW WAR: (.+) sabotaged (.+)'s industry.*", "sabotage"),
            (r"BETRAYAL: (.+) raided their ally (.+)'s border!.*", "ally_skirmish"),
            (r"BORDER CONFLICT: (.+)'s forces skirmished with (.+)!.*", "skirmish"),
            (r"(.+) requested (.+) to join them in a war against (.+)!", "joint_war_request"),
        ]

        for pattern, kind in patterns:
            match = re.match(pattern, event)
            if not match:
                continue

            parties = match.groups()
            if kind == "war":
                attacker, target = parties
                if attacker == my_name:
                    return f"You declared war on {target}."
                if target == my_name:
                    return f"{attacker} declared war on you."
            elif kind in {"alliance_proposal", "trade_proposal", "research_proposal"}:
                proposer, target = parties
                label = {
                    "alliance_proposal": "Alliance",
                    "trade_proposal": "Trade Agreement",
                    "research_proposal": "Research Pact",
                }[kind]
                if proposer == my_name:
                    return f"You proposed a {label} to {target}."
                if target == my_name:
                    return f"{proposer} proposed a {label} to you."
            elif kind in {"alliance_formed", "trade_signed", "research_signed"}:
                left, right = parties
                counterpart = right if left == my_name else left
                label = {
                    "alliance_formed": "Alliance",
                    "trade_signed": "Trade Agreement",
                    "research_signed": "Research Pact",
                }[kind]
                return f"You and {counterpart} completed a {label}."
            elif kind == "joint_war":
                ally, proposer, enemy = parties
                if ally == my_name:
                    return f"You joined {proposer}'s war against {enemy}."
                if proposer == my_name:
                    return f"{ally} joined your war against {enemy}."
            elif kind == "strike":
                attacker, target = parties
                if attacker == my_name:
                    return f"You struck {target}."
                if target == my_name:
                    return f"{attacker} struck you."
            elif kind == "fallen":
                fallen, killer = parties
                if fallen == my_name:
                    return f"You were defeated by {killer}."
                if killer == my_name:
                    return f"You annexed {fallen}."
            elif kind == "alliance_broken":
                breaker, target = parties
                if breaker == my_name:
                    return f"You broke your Alliance with {target}."
                if target == my_name:
                    return f"{breaker} broke their Alliance with you."
            elif kind in {"ally_sabotage", "sabotage"}:
                attacker, target = parties
                if attacker == my_name:
                    return f"You sabotaged {target}."
                if target == my_name:
                    return f"{attacker} sabotaged you."
            elif kind in {"ally_skirmish", "skirmish"}:
                attacker, target = parties
                if attacker == my_name:
                    return f"You skirmished with {target}."
                if target == my_name:
                    return f"{attacker} skirmished with you."
            elif kind == "joint_war_request":
                proposer, target, enemy = parties
                if proposer == my_name:
                    return f"You asked {target} to join a war against {enemy}."
                if target == my_name:
                    return f"{proposer} asked you to join a war against {enemy}."

        return "A relevant diplomatic or military event involved you."

    def get_history_prompt(self, current_turn: int, recent_turns: int = 5) -> List[str]:
        # Build event history: use archived summaries for older events and keep
        # only recent visible events verbatim.
        cutoff = current_turn - recent_turns
        recent_events = [e for e in self.event_memory if e["turn"] >= cutoff]

        lines = []
        if self.archived_event_summaries:
            lines.append("OLDER STRATEGIC MEMORY:")
            summary_items = sorted(
                self.archived_event_summaries.items(),
                key=lambda item: item[1]["last_turn"],
                reverse=True,
            )
            for summary, meta in summary_items[:8]:
                if meta["count"] > 1:
                    lines.append(f"  - {summary} (x{meta['count']}, last turn {meta['last_turn']})")
                else:
                    lines.append(f"  - {summary} (turn {meta['last_turn']})")
        if recent_events:
            lines.append(f"RECENT VISIBLE EVENTS (last {recent_turns} turns):")
            for entry in recent_events[-8:]:
                lines.append(f"  - Turn {entry['turn']}: {entry['text']}")
        return lines

    # Prompt Engineering

    def _build_system_prompt(self, state: GameState) -> str:
        nation = state.nations[self.player_id]
        ap = nation.max_action_points

        strategy_key = NATION_STRATEGIES.get(self.player_id, "neutral")
        strategy_data = STRATEGY_LIBRARY.get(strategy_key)

        if strategy_data:
            strategy_block = (f"{strategy_data['directive']} "
                              f"Your preferred action types are: {strategy_data['priority_actions']}. "
                              f"Use at least 2 of these preferred types every turn when they are legal.")
        else:
            strategy_block = "Choose the best actions to maximize your score."

        all_techs = ", ".join(t.value for t in Tech)
        all_civics = ", ".join(c.value for c in Civic)

        other_ids = [n.id for n in state.nations.values() if n.id != self.player_id and not n.is_defeated]
        ex_id = other_ids[0] if other_ids else 1
        ex_tech = list(Tech)[0].value
        ex_civic = list(Civic)[0].value

        return f"""You are the ruler of nation {self.player_id}. {strategy_block}

=== ACTION SYNTAX (always "COMMAND argument", space-separated, never underscore-separated) ===
HARVEST GOLD | HARVEST MANPOWER | HARVEST PRODUCTION | HARVEST SCIENCE | HARVEST CIVICS
  → always available. Example: "HARVEST GOLD"
INVEST MANPOWER | INVEST INDUSTRY | INVEST SCIENCE | INVEST CIVICS | INVEST MILITARY
  → costs 200 gold. Valid categories: MANPOWER, INDUSTRY, SCIENCE, CIVICS, MILITARY. "INVEST GOLD" is INVALID.
RESEARCH {ex_tech}
  → replace "{ex_tech}" with any tech name from: {all_techs}
PURSUE_CIVIC {ex_civic}
  → replace "{ex_civic}" with any civic name from: {all_civics}
DECLARE_WAR {ex_id}
  → replace {ex_id} with the target nation's integer ID. Example: "DECLARE_WAR {ex_id}"
MILITARY_STRIKE {ex_id}
  → replace {ex_id} with target ID. Only valid when already at war with that nation.
PROPOSE_ALLIANCE {ex_id} | ACCEPT_ALLIANCE {ex_id}
PROPOSE_TRADE {ex_id} | ACCEPT_TRADE {ex_id}
PROPOSE_RESEARCH {ex_id} | ACCEPT_RESEARCH {ex_id}
SABOTAGE {ex_id} | SKIRMISH {ex_id}

=== RULES (strictly enforced — violations are silently replaced with HARVEST GOLD) ===
1. Pick EXACTLY {ap} actions. No more, no less.
2. Nation IDs are integers. You MUST include the ID: "DECLARE_WAR {ex_id}", NOT "DECLARE_WAR".
3. DECLARE_WAR takes effect NEXT turn. Never use MILITARY_STRIKE in the same turn as DECLARE_WAR.
4. MILITARY_STRIKE only when relation is already "At War".
5. ACCEPT_ALLIANCE / ACCEPT_TRADE / ACCEPT_RESEARCH: ONLY when a pending proposal is listed in THIS TURN NOTES. Using ACCEPT without a pending proposal is always rejected.
6. RESEARCH / PURSUE_CIVIC: cannot start a new one while one is already in progress.
7. INVEST needs 200 gold. SABOTAGE needs 50 gold. SKIRMISH needs 20 manpower.
8. Targets must be alive, non-defeated nations other than yourself.

=== SCORING ===
score = gold + manpower + production + (unlocked_techs × 500) + (unlocked_civics × 500)
A nation is eliminated when its infrastructure reaches 0.

Reply with ONLY valid JSON ({ap} actions), no other text:
{{"reasoning": "one sentence explaining your strategy", "actions": ["HARVEST MANPOWER", "DECLARE_WAR {ex_id}", "RESEARCH {ex_tech}"]}}"""

    def _build_turn_prompt(self, state: GameState) -> str:
        nation = state.nations[self.player_id]
        lines = []

        lines.append(f"=== TURN {state.turn} ===")
        lines.append("")

        # --- Your nation ---
        lines.append(f"YOUR NATION: {nation.id} ({nation.name})")
        lines.append(
            f"  Resources:  Gold={nation.gold}, Manpower={nation.manpower}, "
            f"Production={nation.production}, Science={nation.science}, Civics={nation.civics}"
        )
        lines.append(
            f"  Yields/turn: Gold={nation.gold_yield + nation.absorbed_gold_yield}, "
            f"Manpower={nation.manpower_yield}, "
            f"Production={nation.production_yield + nation.absorbed_prod_yield}, "
            f"Science={nation.science_yield + nation.absorbed_sci_yield}, "
            f"Civics={nation.civic_yield}"
        )
        lines.append(
            f"  Military={nation.military} | Infrastructure={nation.infrastructure_health}% "
            f"| War Exhaustion={nation.war_exhaustion}"
        )
        lines.append(f"  Techs unlocked:  {nation.unlocked_techs or 'None'}")
        lines.append(f"  Civics unlocked: {nation.unlocked_civics or 'None'}")
        if nation.current_tech:
            cost = TECH_COSTS.get(nation.current_tech, 100)
            lines.append(f"  Researching: {nation.current_tech} ({nation.tech_progress}/{cost})")
        if nation.current_civic:
            cost = CIVIC_COSTS.get(nation.current_civic, 100)
            lines.append(f"  Pursuing civic: {nation.current_civic} ({nation.civic_progress}/{cost})")

        # --- Opponents ---
        lines.append("")
        lines.append("OPPONENTS:")
        for other_id in sorted(state.nations):
            if other_id == self.player_id:
                continue
            other = state.nations[other_id]
            status = "DEFEATED" if other.is_defeated else "alive"
            lines.append(
                f"  Nation {other.id} ({other.name}) [{status}]: "
                f"Gold={other.gold}, Manpower={other.manpower}, Production={other.production}, "
                f"Military={other.military}, Infra={other.infrastructure_health}%, "
                f"Techs={other.unlocked_techs or 'None'}"
            )

        # --- Diplomacy ---
        lines.append("")
        lines.append("YOUR DIPLOMATIC RELATIONS:")
        alive_others = [n for n in state.nations.values()
                        if n.id != self.player_id and not n.is_defeated]
        for other in alive_others:
            rel = state.get_diplomatic_state(self.player_id, other.id)
            lines.append(f"  Nation {other.id} ({other.name}): {rel.value}")

        # --- Action availability summary (compact) ---
        lines.append("")
        lines.append("THIS TURN ACTION NOTES:")
        target_ids = ", ".join(
            f"{n.id} ({n.name})" for n in alive_others
        )
        lines.append(f"  VALID TARGET IDs: {target_ids}")
        lines.append(f"  (Use these IDs in commands: DECLARE_WAR 2, PROPOSE_TRADE 0, SKIRMISH 4, etc.)")

        # INVEST
        if nation.gold >= 200:
            lines.append(f"  INVEST: available (you have {nation.gold} gold)")
        else:
            lines.append(f"  INVEST: unavailable (need 200 gold, you have {nation.gold})")

        # RESEARCH
        if nation.current_tech:
            lines.append(f"  *** DO NOT use RESEARCH — already researching '{nation.current_tech}'. Any RESEARCH action will be REJECTED. ***")
        else:
            available_techs = [t.value for t in Tech if t.value not in nation.unlocked_techs]
            lines.append(f"  RESEARCH: available — choices: {', '.join(available_techs) or 'none left'}")

        # PURSUE_CIVIC
        if nation.current_civic:
            lines.append(f"  *** DO NOT use PURSUE_CIVIC — already pursuing '{nation.current_civic}'. Any PURSUE_CIVIC action will be REJECTED. ***")
        else:
            available_civics = [c.value for c in Civic if c.value not in nation.unlocked_civics]
            lines.append(f"  PURSUE_CIVIC: available — choices: {', '.join(available_civics) or 'none left'}")

        # War / military
        at_war_with = [n for n in alive_others
                       if state.get_diplomatic_state(self.player_id, n.id) == DiplomaticState.WAR]
        if at_war_with:
            for enemy in at_war_with:
                if nation.manpower >= 100 and nation.production >= 50:
                    lines.append(f"  MILITARY_STRIKE {enemy.id}: available (at war with {enemy.name})")
                else:
                    lines.append(
                        f"  MILITARY_STRIKE {enemy.id}: unavailable "
                        f"(need manpower>=100 prod>=50, you have {nation.manpower},{nation.production})"
                    )

        # Pending proposals (ACCEPT actions)
        pending_alliances = [
            oid for oid, rel in state.diplomacy[self.player_id].items()
            if rel == DiplomaticState.ALLIANCE_PENDING and oid != self.player_id
            and oid in state.nations and not state.nations[oid].is_defeated
        ]
        if pending_alliances:
            names = [f"Nation {oid} ({state.nations[oid].name})" for oid in pending_alliances]
            lines.append(f"  ACCEPT_ALLIANCE: available from {', '.join(names)}")
        if nation.pending_trade_agreements:
            names = [f"Nation {oid} ({state.nations[oid].name})"
                     for oid in nation.pending_trade_agreements if oid in state.nations]
            lines.append(f"  ACCEPT_TRADE: available from {', '.join(names)}")
        if nation.pending_research_pacts:
            names = [f"Nation {oid} ({state.nations[oid].name})"
                     for oid in nation.pending_research_pacts if oid in state.nations]
            lines.append(f"  ACCEPT_RESEARCH: available from {', '.join(names)}")

        if not pending_alliances:
            lines.append("  ACCEPT_ALLIANCE: NOT available (no pending proposals)")
        if not nation.pending_trade_agreements:
            lines.append("  ACCEPT_TRADE: NOT available (no pending proposals)")
        if not nation.pending_research_pacts:
            lines.append("  ACCEPT_RESEARCH: NOT available (no pending proposals)")

        # SABOTAGE / SKIRMISH resource check
        if nation.gold < 50:
            lines.append(f"  SABOTAGE: unavailable (need 50 gold, you have {nation.gold})")
        if nation.manpower < 20:
            lines.append(f"  SKIRMISH: unavailable (need 20 manpower, you have {nation.manpower})")

        return "\n".join(lines)

    # Output Validation

    def _get_rejection_reason(self, action: str, state: GameState) -> str:
        parts = action.strip().split()
        if not parts:
            return "empty action"
        cmd = parts[0].upper()
        nation = state.nations[self.player_id]

        if cmd == "HARVEST":
            if len(parts) < 2 or parts[1].upper() not in VALID_RESOURCES:
                return f"invalid resource; use one of: {', '.join(VALID_RESOURCES)}"
            return "valid"

        if cmd == "INVEST":
            valid_invest = {"MANPOWER", "INDUSTRY", "SCIENCE", "CIVICS", "MILITARY"}
            if len(parts) < 2 or parts[1].upper() not in valid_invest:
                return f"invalid type; use one of: {', '.join(valid_invest)}"
            if nation.gold < 200:
                return f"need 200 gold, you have {nation.gold}"
            return "valid"

        if cmd in ("DECLARE_WAR", "MILITARY_STRIKE", "PROPOSE_ALLIANCE", "ACCEPT_ALLIANCE",
                   "PROPOSE_TRADE", "ACCEPT_TRADE", "PROPOSE_RESEARCH", "ACCEPT_RESEARCH",
                   "SABOTAGE", "SKIRMISH"):
            if len(parts) < 2:
                return "missing target nation ID"
            try:
                tid = int(parts[1])
            except ValueError:
                return f"'{parts[1]}' is not an integer nation ID"
            if tid not in state.nations:
                return f"nation {tid} does not exist"
            if tid == self.player_id:
                return "cannot target yourself"
            if state.nations[tid].is_defeated:
                return f"nation {tid} is already defeated"
            if cmd == "MILITARY_STRIKE":
                dipl = state.get_diplomatic_state(self.player_id, tid)
                if dipl != DiplomaticState.WAR:
                    return (f"not at war with nation {tid} (relation: {dipl.value}); "
                            f"use DECLARE_WAR {tid} this turn, then MILITARY_STRIKE next turn")
                if nation.manpower < 100:
                    return f"need manpower>=100, you have {nation.manpower}"
                if nation.production < 50:
                    return f"need production>=50, you have {nation.production}"
            if cmd == "SABOTAGE" and nation.gold < 50:
                return f"need 50 gold, you have {nation.gold}"
            if cmd == "SKIRMISH" and nation.manpower < 20:
                return f"need 20 manpower, you have {nation.manpower}"
            if cmd == "ACCEPT_ALLIANCE":
                dipl = state.get_diplomatic_state(self.player_id, tid)
                if dipl != DiplomaticState.ALLIANCE_PENDING:
                    return f"no pending alliance from nation {tid}"
            if cmd == "ACCEPT_TRADE":
                if tid not in nation.pending_trade_agreements:
                    return f"no pending trade proposal from nation {tid}"
            if cmd == "ACCEPT_RESEARCH":
                if tid not in nation.pending_research_pacts:
                    return f"no pending research pact from nation {tid}"
            return "valid"

        if cmd == "RESEARCH":
            if nation.current_tech:
                return f"already researching '{nation.current_tech}'; wait until it finishes"
            tech_name = " ".join(parts[1:])
            if tech_name not in VALID_TECHS:
                return f"unknown tech '{tech_name}'"
            if tech_name in nation.unlocked_techs:
                return f"already unlocked '{tech_name}'"
            return "valid"

        if cmd == "PURSUE_CIVIC":
            if nation.current_civic:
                return f"already pursuing '{nation.current_civic}'; wait until it finishes"
            civic_name = " ".join(parts[1:])
            if civic_name not in VALID_CIVICS:
                return f"unknown civic '{civic_name}'"
            if civic_name in nation.unlocked_civics:
                return f"already unlocked '{civic_name}'"
            return "valid"

        return f"unknown command '{cmd}'"

    def _build_legal_actions(self, state: GameState) -> str:
        nation = state.nations[self.player_id]
        alive_others = [n for n in state.nations.values()
                        if n.id != self.player_id and not n.is_defeated]
        lines = ["YOUR LEGAL ACTIONS THIS TURN (ONLY use actions from this list):"]

        for res in sorted(VALID_RESOURCES):
            lines.append(f"  HARVEST {res}")

        if nation.gold >= 200:
            for inv in ["MANPOWER", "INDUSTRY", "SCIENCE", "CIVICS", "MILITARY"]:
                lines.append(f"  INVEST {inv}")
        else:
            lines.append(f"  (INVEST unavailable: need 200 gold, you have {nation.gold})")

        if nation.current_tech:
            lines.append(f"  (RESEARCH unavailable: already researching '{nation.current_tech}')")
        else:
            for t in Tech:
                if t.value not in nation.unlocked_techs:
                    lines.append(f"  RESEARCH {t.value}")

        if nation.current_civic:
            lines.append(f"  (PURSUE_CIVIC unavailable: already pursuing '{nation.current_civic}')")
        else:
            for c in Civic:
                if c.value not in nation.unlocked_civics:
                    lines.append(f"  PURSUE_CIVIC {c.value}")

        for other in alive_others:
            tid = other.id
            dipl = state.get_diplomatic_state(self.player_id, tid)

            if dipl == DiplomaticState.WAR:
                if nation.manpower >= 100 and nation.production >= 50:
                    lines.append(f"  MILITARY_STRIKE {tid}  # at war with {other.name}")
                else:
                    lines.append(
                        f"  (MILITARY_STRIKE {tid} unavailable: "
                        f"need manpower>=100,production>=50; "
                        f"you have {nation.manpower},{nation.production})"
                    )
            else:
                lines.append(f"  DECLARE_WAR {tid}  # current relation: {dipl.value}")

            if dipl not in (DiplomaticState.WAR, DiplomaticState.ALLIED,
                            DiplomaticState.ALLIANCE_PENDING):
                lines.append(f"  PROPOSE_ALLIANCE {tid}")
            if dipl == DiplomaticState.ALLIANCE_PENDING:
                lines.append(f"  ACCEPT_ALLIANCE {tid}  # {other.name} proposed an alliance to you")

            lines.append(f"  PROPOSE_TRADE {tid}")
            if tid in nation.pending_trade_agreements:
                lines.append(f"  ACCEPT_TRADE {tid}  # {other.name} proposed a trade")

            lines.append(f"  PROPOSE_RESEARCH {tid}")
            if tid in nation.pending_research_pacts:
                lines.append(f"  ACCEPT_RESEARCH {tid}  # {other.name} proposed a research pact")

            if nation.gold >= 50:
                lines.append(f"  SABOTAGE {tid}  # costs 50 gold")
            if nation.manpower >= 20:
                lines.append(f"  SKIRMISH {tid}  # costs 20 manpower")

        return "\n".join(lines)

    def _validate_and_extract(self, raw: str, state: GameState) -> Tuple[List[str], str, List[str]]:
        text = raw.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        # Extract JSON
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            text = text[start:end]

        # Parse LLM output mistakes (trailing comma)
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*]', ']', text)

        data = json.loads(text)

        if not isinstance(data.get("actions"), list):
            raise ValueError("'actions' must be a list")

        reasoning = data.get("reasoning", "No reasoning provided.")
        if not isinstance(reasoning, str):
            reasoning = "No reasoning provided."

        valid_actions = []
        rejections = []
        for action_str in data["actions"]:
            action_str = str(action_str).strip().upper()
            # Fix space-separated compound commands (LLM writes spaces instead of underscores)
            action_str = action_str.replace("DECLARE WAR", "DECLARE_WAR")
            action_str = action_str.replace("MILITARY STRIKE", "MILITARY_STRIKE")
            action_str = action_str.replace("PROPOSE ALLIANCE", "PROPOSE_ALLIANCE")
            action_str = action_str.replace("ACCEPT ALLIANCE", "ACCEPT_ALLIANCE")
            action_str = action_str.replace("PROPOSE TRADE", "PROPOSE_TRADE")
            action_str = action_str.replace("ACCEPT TRADE", "ACCEPT_TRADE")
            action_str = action_str.replace("PROPOSE RESEARCH", "PROPOSE_RESEARCH")
            action_str = action_str.replace("ACCEPT RESEARCH", "ACCEPT_RESEARCH")
            action_str = action_str.replace("PURSUE CIVIC", "PURSUE_CIVIC")
            # Fix HARVEST_X / INVEST_X (LLM uses underscore instead of space for argument)
            action_str = re.sub(r'^HARVEST_', 'HARVEST ', action_str)
            action_str = re.sub(r'^INVEST_', 'INVEST ', action_str)
            # Fix COMMAND_N where N is a digit (e.g. DECLARE_WAR_0 -> DECLARE_WAR 0)
            _CMDS_WITH_ID = (
                "DECLARE_WAR", "MILITARY_STRIKE", "PROPOSE_ALLIANCE", "ACCEPT_ALLIANCE",
                "PROPOSE_TRADE", "ACCEPT_TRADE", "PROPOSE_RESEARCH", "ACCEPT_RESEARCH",
                "SABOTAGE", "SKIRMISH",
            )
            for cmd in _CMDS_WITH_ID:
                action_str = re.sub(rf'^{cmd}_(\d+)$', rf'{cmd} \1', action_str)
            # Fix common INVEST confusions: INVEST GOLD -> HARVEST GOLD, INVEST PRODUCTION -> INVEST INDUSTRY
            if action_str == "INVEST GOLD":
                action_str = "HARVEST GOLD"
            elif action_str == "INVEST PRODUCTION":
                action_str = "INVEST INDUSTRY"
            # Restore original casing for tech/civic names via case-insensitive lookup
            parts = action_str.split(' ', 1)
            if len(parts) == 2 and parts[0] in ('RESEARCH', 'PURSUE_CIVIC'):
                raw_name = parts[1].replace('_', ' ')
                # Match against known tech/civic names (case-insensitive)
                canonical = None
                for name in (list(VALID_TECHS) + list(VALID_CIVICS)):
                    if name.upper() == raw_name.upper():
                        canonical = name
                        break
                if canonical:
                    # Auto-correct command if LLM used RESEARCH for a civic or PURSUE_CIVIC for a tech
                    if parts[0] == 'RESEARCH' and canonical in VALID_CIVICS:
                        action_str = 'PURSUE_CIVIC ' + canonical
                    elif parts[0] == 'PURSUE_CIVIC' and canonical in VALID_TECHS:
                        action_str = 'RESEARCH ' + canonical
                    else:
                        action_str = parts[0] + ' ' + canonical
                else:
                    action_str = parts[0] + ' ' + raw_name
            if self._is_valid_action(action_str, state):
                valid_actions.append(action_str)
            else:
                reason = self._get_rejection_reason(action_str, state)
                print(f"[LLM Agent {self.player_id}] REJECTED action: '{action_str}' ({reason})")
                rejections.append(f"'{action_str}': {reason}")

        nation = state.nations[self.player_id]
        padded = 0
        while len(valid_actions) < nation.max_action_points:
            valid_actions.append("HARVEST GOLD")
            padded += 1
        if padded:
            print(f"[LLM Agent {self.player_id}] Padded {padded} actions with HARVEST GOLD")

        return valid_actions[:nation.max_action_points], reasoning, rejections

    def _is_valid_action(self, action: str, state: GameState) -> bool:
        parts = action.strip().split()
        if not parts:
            return False

        cmd = parts[0].upper()
        nation = state.nations[self.player_id]

        if cmd == "HARVEST":
            return len(parts) >= 2 and parts[1].upper() in VALID_RESOURCES

        if cmd == "INVEST":
            if len(parts) < 2:
                return False
            valid_invest = {"MANPOWER", "INDUSTRY", "SCIENCE", "CIVICS", "MILITARY"}
            return parts[1].upper() in valid_invest and nation.gold >= 200

        if cmd in ("DECLARE_WAR", "MILITARY_STRIKE", "PROPOSE_ALLIANCE", "ACCEPT_ALLIANCE",
                    "PROPOSE_TRADE", "ACCEPT_TRADE", "PROPOSE_RESEARCH", "ACCEPT_RESEARCH",
                    "SABOTAGE", "SKIRMISH"):
            if len(parts) < 2:
                return False
            try:
                tid = int(parts[1])
            except ValueError:
                return False
            if tid not in state.nations or tid == self.player_id or state.nations[tid].is_defeated:
                return False

            if cmd == "MILITARY_STRIKE":
                if state.get_diplomatic_state(self.player_id, tid) != DiplomaticState.WAR:
                    return False
                if nation.manpower < 100 or nation.production < 50:
                    return False

            if cmd == "SABOTAGE":
                if nation.gold < 50:
                    return False

            if cmd == "SKIRMISH":
                if nation.manpower < 20:
                    return False

            if cmd == "ACCEPT_ALLIANCE":
                if state.get_diplomatic_state(self.player_id, tid) != DiplomaticState.ALLIANCE_PENDING:
                    return False

            if cmd == "ACCEPT_TRADE":
                if tid not in nation.pending_trade_agreements:
                    return False

            if cmd == "ACCEPT_RESEARCH":
                if tid not in nation.pending_research_pacts:
                    return False

            return True

        if cmd == "RESEARCH":
            if nation.current_tech:
                return False
            tech_name = " ".join(parts[1:])
            return tech_name in VALID_TECHS and tech_name not in nation.unlocked_techs

        if cmd == "PURSUE_CIVIC":
            if nation.current_civic:
                return False
            civic_name = " ".join(parts[1:])
            return civic_name in VALID_CIVICS and civic_name not in nation.unlocked_civics

        return False

    # LLM Call

    def _call_llm_with_retry(self, state: GameState) -> Tuple[Optional[List[str]], str]:
        system = self._build_system_prompt(state)
        user = self._build_turn_prompt(state)

        for attempt in range(MAX_RETRIES + 1):
            try:
                raw = self.client.chat(system, user, temperature=LLM_TEMPERATURE)
                actions, reasoning, rejections = self._validate_and_extract(raw, state)
                if rejections and attempt < MAX_RETRIES:
                    rejection_msg = "; ".join(rejections)
                    user += (
                        f"\n\nPREVIOUS ATTEMPT HAD INVALID ACTIONS — "
                        f"{rejection_msg}. "
                        f"Check THIS TURN ACTION NOTES above and pick valid replacements."
                    )
                    continue
                return actions, reasoning
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                print(f"[LLM Agent {self.player_id}] Validation error (attempt {attempt+1}): {e}")
                if attempt < MAX_RETRIES:
                    user += f"\n\nPREVIOUS ATTEMPT FAILED: {e}. Return ONLY valid JSON."
                continue
            except Exception as e:
                print(f"[LLM Agent {self.player_id}] API error: {e}")
                break

        return None, ""
