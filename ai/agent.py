import random
import json
from typing import List
from core.game_state import GameState
from engine.actions import ActionHandler
from core.constants import Resource, Tech, Civic, DiplomaticState
from ai.symbolic import parse_agent_response, AgentTurnResponse, GameAction

class AIAgent:
    def __init__(self, player_id: int):
        self.player_id = player_id

    def decide_actions(self, state: GameState, handler: ActionHandler) -> List[str]:
        # 1. Receive state symbolically (enforcing the pipeline)
        sym_state = state.get_symbolic_state(self.player_id)
        
        if "error" in sym_state:
            return []
            
        me = sym_state["my_nation"]
        if me["is_defeated"]:
            return []
            
        actions = []
        aps = me["actions"]["current_points"]
        pers = me["personality"]
        
        # Archetype weight mapping
        weights = {
            "WARMONGER": {"military": 50, "economy": 15, "diplomacy": 10, "tech": 15, "expansion": 10},
            "BALANCED": {"military": 25, "economy": 25, "diplomacy": 20, "tech": 20, "expansion": 10},
            "TECHNOCRAT": {"military": 10, "economy": 25, "diplomacy": 15, "tech": 40, "expansion": 10},
            "OPPORTUNIST": {"military": 20, "economy": 25, "diplomacy": 35, "tech": 10, "expansion": 10}
        }.get(pers, {"military": 25, "economy": 25, "diplomacy": 20, "tech": 20, "expansion": 10})
        
        cats, probs = list(weights.keys()), list(weights.values())
        
        # Helper lists derived solely from symbolic state
        diplo = me["diplomacy"]
        others = [n for n in sym_state["other_nations"] if not n["is_defeated"]]
        at_war = [n["id"] for n in others if n["diplomatic_status"] == "WAR"]
        allies = [n["id"] for n in others if n["diplomatic_status"] == "ALLIED"]
        neutrals = [n["id"] for n in others if n["diplomatic_status"] == "NEUTRAL"]
        
        pending_allies = [n["id"] for n in others if n["diplomatic_status"] == "ALLIANCE_PENDING"]
        
        # Enforce 25-turn grace period (no foreign actions)
        if sym_state["global_state"]["turn"] <= 25:
            at_war = []
            allies = []
            neutrals = []
            pending_allies = []
            diplo["pending_trade_agreements"] = []
            diplo["pending_research_pacts"] = []
            
        # Pariah Engine: Avoid heavily hostile nations
        pariahs = [n["id"] for n in others if n["visible_status"].get("global_grievances", 0) >= 100]
        
        # Remove pariahs from standard diplomatic outreach
        allies = [a for a in allies if a not in pariahs]
        neutrals = [n_id for n_id in neutrals if n_id not in pariahs]
        pending_allies = [p for p in pending_allies if p not in pariahs]
        
        # Increase military action probability heavily against Pariahs if we are balanced or opportunist
        if pariahs and pers in ["BALANCED", "OPPORTUNIST"]:
            weights["military"] += 15
            probs = list(weights.values())
        
        available_techs = [t.value for t in Tech if t.value not in me["tech"]["unlocked"]]
        available_civics = [c.value for c in Civic if c.value not in me["civic"]["unlocked"]]
        
        war_thresholds = {"WARMONGER": 300, "OPPORTUNIST": 500, "BALANCED": 700, "TECHNOCRAT": 900}
        war_thresh = war_thresholds.get(pers, 600)

        proposed_trade = set()
        proposed_research = set()
        
        # Emergency Defense
        if at_war and me["stats"]["military"] < 400:
            weights["military"] += 50
            probs = list(weights.values())

        while aps > 0:
            # Force auto-responses for deals
            if diplo.get("pending_joint_wars") and random.random() < 0.7:
                p = diplo["pending_joint_wars"].pop(0)
                if p["enemy"] in pariahs or pers in ["WARMONGER", "OPPORTUNIST"]:
                    actions.append({"action": "ACCEPT_JOINT_WAR", "target": p["proposer"], "enemy": p["enemy"]})
                aps -= 1; continue
            if pending_allies and random.random() < 0.8:
                actions.append({"action": "ACCEPT_ALLIANCE", "target": pending_allies.pop(0)})
                aps -= 1; continue
            if diplo["pending_trade_agreements"] and random.random() < 0.8:
                actions.append({"action": "ACCEPT_TRADE", "target": diplo["pending_trade_agreements"].pop(0)})
                aps -= 1; continue
            if diplo["pending_research_pacts"] and random.random() < 0.8:
                actions.append({"action": "ACCEPT_RESEARCH", "target": diplo["pending_research_pacts"].pop(0)})
                aps -= 1; continue
                
            chosen_cat = random.choices(cats, weights=probs)[0]
            action_taken = False
            
            if chosen_cat == "military":
                if at_war and me["stats"]["manpower"] >= 100 and me["stats"]["production"] >= 50:
                    actions.append({"action": "MILITARY_STRIKE", "target": random.choice(at_war)})
                    action_taken = True
                elif not at_war and me["stats"]["military"] >= war_thresh:
                    targets = neutrals
                    if pers == "OPPORTUNIST":
                        targets = [n["id"] for n in others if n["diplomatic_status"] != "ALLIED" and n["visible_status"]["infrastructure_health"] < 80]
                        if not targets: targets = neutrals
                    if targets:
                        target = random.choice(targets)
                        actions.append({"action": "DECLARE_WAR", "target": target})
                        at_war.append(target)
                        if target in neutrals: neutrals.remove(target)
                        action_taken = True
                        if allies and aps > 1 and random.random() < 0.8:
                            actions.append({"action": "PROPOSE_JOINT_WAR", "target": random.choice(allies), "enemy": target})
                            aps -= 1
                
                # Opportunists and Warmongers try border skirmishes if no war targets
                if not action_taken and pers in ["OPPORTUNIST", "WARMONGER"] and (neutrals or allies) and me["stats"]["manpower"] >= 50:
                    actions.append({"action": "SKIRMISH", "target": random.choice(neutrals + allies)})
                    action_taken = True

            elif chosen_cat == "economy":
                # INVEST is a high-value option when gold is plentiful
                if me["stats"]["gold"] >= 200 and random.random() < 0.4:
                    invest_targets = {
                        "WARMONGER": ["MILITARY", "MANPOWER"],
                        "TECHNOCRAT": ["SCIENCE", "INDUSTRY"],
                        "BALANCED": ["MANPOWER", "INDUSTRY", "SCIENCE"],
                        "OPPORTUNIST": ["MILITARY", "SCIENCE", "CIVICS"]
                    }.get(pers, ["MANPOWER", "INDUSTRY"])
                    actions.append({"action": "INVEST", "target": random.choice(invest_targets)})
                    action_taken = True
                elif allies and random.random() > 0.5:
                    viable = [a for a in allies if a not in diplo["active_trade_agreements"] and a not in proposed_trade]
                    if viable:
                        t = random.choice(viable)
                        actions.append({"action": "PROPOSE_TRADE", "target": t})
                        proposed_trade.add(t)
                        action_taken = True
                if not action_taken:
                    actions.append({"action": "HARVEST", "target": random.choice(["GOLD", "PRODUCTION"])})
                    action_taken = True

            elif chosen_cat == "diplomacy":
                r = random.random()
                if pariahs and allies and r < 0.5 and me["stats"]["military"] >= 100:
                    actions.append({"action": "PROPOSE_JOINT_WAR", "target": random.choice(allies), "enemy": random.choice(pariahs)})
                    action_taken = True
                elif pers == "OPPORTUNIST" and r < 0.2 and allies:
                    t = random.choice(allies)
                    actions.append({"action": "CANCEL_ALLIANCE", "target": t})
                    allies.remove(t)
                    action_taken = True
                elif pers == "OPPORTUNIST" and r < 0.5 and (neutrals or allies) and me["stats"]["gold"] >= 50:
                    actions.append({"action": "SABOTAGE", "target": random.choice(neutrals + allies)})
                    action_taken = True
                elif r < 0.4 and neutrals:
                    actions.append({"action": "PROPOSE_ALLIANCE", "target": random.choice(neutrals)})
                    action_taken = True
                elif r < 0.8 and not me["civic"]["current"] and available_civics:
                    actions.append({"action": "PURSUE_CIVIC", "target": random.choice(available_civics)})
                    me["civic"]["current"] = "Pending"
                    action_taken = True
                else: 
                    actions.append({"action": "HARVEST", "target": "CIVICS"})
                    action_taken = True

            elif chosen_cat == "tech":
                if not me["tech"]["current"] and available_techs:
                    actions.append({"action": "RESEARCH", "target": random.choice(available_techs)})
                    me["tech"]["current"] = "Pending"
                    action_taken = True
                elif allies and random.random() > 0.5:
                    viable = [a for a in allies if a not in diplo["active_research_pacts"] and a not in proposed_research]
                    if viable:
                        t = random.choice(viable)
                        actions.append({"action": "PROPOSE_RESEARCH", "target": t})
                        proposed_research.add(t)
                        action_taken = True
                else:
                    if pers == "TECHNOCRAT" and (neutrals or allies) and random.random() < 0.4 and me["stats"]["gold"] >= 50:
                        actions.append({"action": "SABOTAGE", "target": random.choice(neutrals + allies)})
                        action_taken = True
                    else:
                        actions.append({"action": "HARVEST", "target": "SCIENCE"})
                        action_taken = True

            elif chosen_cat == "expansion":
                actions.append({"action": "HARVEST", "target": "MANPOWER"})
                action_taken = True
                
            if not action_taken:
                # Fallback purely harvest
                actions.append({"action": "HARVEST", "target": random.choice(["GOLD", "MANPOWER", "PRODUCTION", "SCIENCE"])})
                
            aps -= 1

        # 2. Package into foundation model format and dump to JSON string
        agent_response_dict = {
            "reasoning": f"Simulating archetypal weights ({pers}) to select: {cats[probs.index(max(probs))] if probs else 'fallback'}.",
            "actions": actions
        }
        
        response_json = json.dumps(agent_response_dict)
        
        # 3. Give state actions back via symbolic parser
        engine_commands, err = parse_agent_response(response_json)
        
        if err:
            print(f"AI {self.player_id} generated invalid actions: {err}")
            return []
            
        return engine_commands