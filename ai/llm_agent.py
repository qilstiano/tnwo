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
            strategy_block = f"""{strategy_data['directive']} PRIORITY: {strategy_data['priority_actions']}. Use at least 2 priority actions."""
        else:
            strategy_block = "Choose the best actions to maximize your score."

        # Other nation IDs
        other_ids = [n.id for n in state.nations.values() if n.id != self.player_id and not n.is_defeated]
        ex = other_ids[0] if other_ids else 1

        return f"""You control nation {self.player_id} ("{nation.name}"). {strategy_block}

ACTIONS (pick exactly {ap}, use exact syntax shown):
HARVEST GOLD | HARVEST MANPOWER | HARVEST PRODUCTION | HARVEST SCIENCE | HARVEST CIVICS
RESEARCH <tech> | PURSUE_CIVIC <civic> (only if none in progress)
PROPOSE_ALLIANCE {ex} | ACCEPT_ALLIANCE {ex} | DECLARE_WAR {ex} | MILITARY_STRIKE {ex}
PROPOSE_TRADE {ex} | ACCEPT_TRADE {ex} | PROPOSE_RESEARCH {ex} | ACCEPT_RESEARCH {ex}
INVEST MILITARY | INVEST MANPOWER | INVEST INDUSTRY | INVEST SCIENCE | INVEST CIVICS (costs 200 gold)
SABOTAGE {ex} (costs 50 gold) | SKIRMISH {ex} (costs 20 manpower)

RULES: target_id is integer, not name. ACCEPT only if pending. MILITARY_STRIKE only at war. RESEARCH/PURSUE_CIVIC only if none in progress.
SCORE = gold + manpower + production + (techs x 500) + (civics x 500). Eliminated if infrastructure = 0.

Reply with ONLY this JSON, no other text:
{{"reasoning":"short sentence","actions":["HARVEST MANPOWER","DECLARE_WAR {ex}","MILITARY_STRIKE {ex}"]}}"""

    def _build_turn_prompt(self, state: GameState) -> str:
        nation = state.nations[self.player_id]
        lines = []

        lines.append(f"TURN {state.turn}")
        lines.append("Use only the current state snapshot below. Do not assume access to hidden past actions or intent logs.")
        lines.append("")
        lines.append(f"YOU ARE: Nation {nation.id} ({nation.name})")
        lines.append("")
        lines.append("GLOBAL STATE SNAPSHOT AFTER THE PREVIOUS TURN:")

        for other_id in sorted(state.nations):
            other = state.nations[other_id]
            lines.append("")
            lines.append(f"NATION {other.id}: {other.name}")
            lines.append(f"  STATUS: {'DEFEATED' if other.is_defeated else 'ALIVE'}")
            lines.append(
                f"  RESOURCES: Gold={other.gold}, Manpower={other.manpower}, "
                f"Production={other.production}, Science={other.science}, Civics={other.civics}"
            )
            lines.append(
                f"  MILITARY: {other.military} | INFRASTRUCTURE: {other.infrastructure_health}% | "
                f"WAR_EXHAUSTION: {other.war_exhaustion}"
            )
            lines.append(
                f"  YIELDS/TURN: Gold={other.gold_yield + other.absorbed_gold_yield}, "
                f"Manpower={other.manpower_yield}, "
                f"Production={other.production_yield + other.absorbed_prod_yield}, "
                f"Science={other.science_yield + other.absorbed_sci_yield}, "
                f"Civics={other.civic_yield}"
            )
            lines.append(f"  TECHS UNLOCKED: {other.unlocked_techs or 'None'}")
            lines.append(f"  CIVICS UNLOCKED: {other.unlocked_civics or 'None'}")
            if other.current_tech:
                cost = TECH_COSTS.get(other.current_tech, 100)
                lines.append(f"  RESEARCHING: {other.current_tech} (progress: {other.tech_progress}/{cost})")
            if other.current_civic:
                cost = CIVIC_COSTS.get(other.current_civic, 100)
                lines.append(f"  PURSUING CIVIC: {other.current_civic} (progress: {other.civic_progress}/{cost})")
            if other.pending_trade_agreements:
                lines.append(f"  PENDING TRADE PROPOSALS FROM: {other.pending_trade_agreements}")
            if other.pending_research_pacts:
                lines.append(f"  PENDING RESEARCH PROPOSALS FROM: {other.pending_research_pacts}")
            if other.pending_joint_wars:
                lines.append(f"  PENDING JOINT WAR PROPOSALS: {other.pending_joint_wars}")
            pending_alliances = [
                oid for oid, relation in state.diplomacy[other.id].items()
                if relation == DiplomaticState.ALLIANCE_PENDING and oid != other.id
            ]
            if pending_alliances:
                lines.append(f"  PENDING ALLIANCE PROPOSALS FROM: {pending_alliances}")

        lines.append("")
        lines.append("GLOBAL DIPLOMACY:")
        alive_ids = [nid for nid, other in state.nations.items() if not other.is_defeated]
        for i, left_id in enumerate(alive_ids):
            for right_id in alive_ids[i + 1:]:
                relation = state.get_diplomatic_state(left_id, right_id)
                left_name = state.nations[left_id].name
                right_name = state.nations[right_id].name
                lines.append(f"  {left_name} <-> {right_name}: {relation.value}")

        lines.append("")
        available_techs = [t.value for t in Tech if t.value not in nation.unlocked_techs]
        available_civics = [c.value for c in Civic if c.value not in nation.unlocked_civics]
        if not nation.current_tech and available_techs:
            lines.append(f"YOUR AVAILABLE TECHS: {', '.join(available_techs)}")
        if not nation.current_civic and available_civics:
            lines.append(f"YOUR AVAILABLE CIVICS: {', '.join(available_civics)}")

        return "\n".join(lines)

    # Output Validation

    def _validate_and_extract(self, raw: str, state: GameState) -> Tuple[List[str], str]:
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
        for action_str in data["actions"]:
            action_str = str(action_str).strip()
            # Fix common LLM formatting mistakes
            action_str = action_str.replace("DECLARE WAR", "DECLARE_WAR")
            action_str = action_str.replace("MILITARY STRIKE", "MILITARY_STRIKE")
            action_str = action_str.replace("PROPOSE ALLIANCE", "PROPOSE_ALLIANCE")
            action_str = action_str.replace("ACCEPT ALLIANCE", "ACCEPT_ALLIANCE")
            action_str = action_str.replace("PROPOSE TRADE", "PROPOSE_TRADE")
            action_str = action_str.replace("ACCEPT TRADE", "ACCEPT_TRADE")
            action_str = action_str.replace("PROPOSE RESEARCH", "PROPOSE_RESEARCH")
            action_str = action_str.replace("ACCEPT RESEARCH", "ACCEPT_RESEARCH")
            action_str = action_str.replace("PURSUE CIVIC", "PURSUE_CIVIC")
            action_str = action_str.replace("PROPOSE JOINT WAR", "PROPOSE_JOINT_WAR")
            action_str = action_str.replace("ACCEPT JOINT WAR", "ACCEPT_JOINT_WAR")
            if self._is_valid_action(action_str, state):
                valid_actions.append(action_str)
            else:
                print(f"[LLM Agent {self.player_id}] REJECTED action: '{action_str}'")

        nation = state.nations[self.player_id]
        padded = 0
        while len(valid_actions) < nation.max_action_points:
            valid_actions.append("HARVEST GOLD")
            padded += 1
        if padded:
            print(f"[LLM Agent {self.player_id}] Padded {padded} actions with HARVEST GOLD")

        return valid_actions[:nation.max_action_points], reasoning

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
                actions, reasoning = self._validate_and_extract(raw, state)
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
