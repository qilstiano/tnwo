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
        "INTENT", "struck", "Trade Agreement", "Research Pact",
        "SABOTAGE", "SKIRMISH", "CANCEL", "JOINT WAR",
        "Peace", "Victory", "PARIAH",
    ]

    def update_memory(self, turn: int, events: List[str]):
        significant = []
        routine = []
        for event in events:
            tagged = f"Turn {turn}: {event}"
            if any(kw in event for kw in self.SIGNIFICANT_KEYWORDS):
                significant.append(tagged)
            else:
                routine.append(tagged)
        for s in significant:
            self.event_memory.append(("significant", s))
        for r in routine:
            self.event_memory.append(("routine", r))

    def clear_memory(self):
        self.event_memory.clear()

    def get_history_prompt(self, current_turn: int, recent_turns: int = 5) -> List[str]:
        # Build event history: summarise past significant events + keep recent events in full
        cutoff = current_turn - recent_turns
        summary_events = []
        recent_events = []

        for event_type, text in self.event_memory:
            try:
                turn_num = int(text.split(":")[0].replace("Turn ", ""))
            except (ValueError, IndexError):
                turn_num = 0

            if turn_num >= cutoff:
                # Recent: include everything
                recent_events.append(text)
            elif event_type == "significant":
                # Old: only significant events
                summary_events.append(text)

        lines = []
        if summary_events:
            lines.append("KEY PAST EVENTS:")
            for s in summary_events:
                lines.append(f"  - {s}")
        if recent_events:
            lines.append(f"RECENT EVENTS (last {recent_turns} turns):")
            for r in recent_events:
                lines.append(f"  - {r}")
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
        lines.append(f"YOUR RESOURCES: Gold={nation.gold}, Manpower={nation.manpower}, "
                      f"Production={nation.production}, Science={nation.science}, Civics={nation.civics}")
        lines.append(f"MILITARY: {nation.military} | INFRASTRUCTURE: {nation.infrastructure_health}% | "
                      f"WAR_EXHAUSTION: {nation.war_exhaustion}")
        lines.append(f"YIELDS/TURN: Gold={nation.gold_yield}, Manpower={nation.manpower_yield}, "
                      f"Production={nation.production_yield}")
        lines.append(f"TECHS UNLOCKED: {nation.unlocked_techs or 'None'}")
        lines.append(f"CIVICS UNLOCKED: {nation.unlocked_civics or 'None'}")

        if nation.current_tech:
            cost = TECH_COSTS.get(nation.current_tech, 100)
            lines.append(f"RESEARCHING: {nation.current_tech} (progress: {nation.tech_progress}/{cost})")
        else:
            available = [t.value for t in Tech if t.value not in nation.unlocked_techs]
            if available:
                lines.append(f"AVAILABLE TECHS: {', '.join(available)}")

        if nation.current_civic:
            cost = CIVIC_COSTS.get(nation.current_civic, 100)
            lines.append(f"PURSUING CIVIC: {nation.current_civic} (progress: {nation.civic_progress}/{cost})")
        else:
            available = [c.value for c in Civic if c.value not in nation.unlocked_civics]
            if available:
                lines.append(f"AVAILABLE CIVICS: {', '.join(available)}")

        lines.append("")
        lines.append("OTHER NATIONS:")
        sym_state = state.get_symbolic_state(self.player_id)
        for other in sym_state["other_nations"]:
            if other["is_defeated"]:
                continue
            grievance = nation.grievances.get(other["id"], 0)
            vs = other["visible_status"]
            lines.append(f"  Nation {other['id']} ({other['name']}): {other['diplomatic_status']} | "
                          f"Infra={vs['infrastructure_health']}% TechTier={vs['estimated_tech_tier']} "
                          f"GlobalGrievances={vs['global_grievances']} | "
                          f"MyGrievance={grievance}")

        if nation.pending_trade_agreements:
            lines.append(f"PENDING TRADE PROPOSALS FROM: {nation.pending_trade_agreements}")
        if nation.pending_research_pacts:
            lines.append(f"PENDING RESEARCH PROPOSALS FROM: {nation.pending_research_pacts}")

        pending_alliances = [oid for oid, s in state.diplomacy[self.player_id].items()
                             if s == DiplomaticState.ALLIANCE_PENDING and oid != self.player_id]
        if pending_alliances:
            lines.append(f"PENDING ALLIANCE PROPOSALS FROM: {pending_alliances}")

        history_lines = self.get_history_prompt(state.turn)
        if history_lines:
            lines.append("")
            lines.extend(history_lines)

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
