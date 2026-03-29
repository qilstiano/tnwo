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
        
        # Helper lists derived solely from symbolic state
        diplo = me["diplomacy"]
        at_war = [n["id"] for n in sym_state["other_nations"] if n["diplomatic_status"] == "WAR" and not n["is_defeated"]]
        allies = [n["id"] for n in sym_state["other_nations"] if n["diplomatic_status"] == "ALLIED" and not n["is_defeated"]]
        
        pending_allies = []
        for n in sym_state["other_nations"]:
            if n["diplomatic_status"] == "ALLIANCE_PENDING":
                pending_allies.append(n["id"])
        
        proposed_trade = set()
        proposed_research = set()
        
        while aps > 0:
            if pending_allies:
                actions.append({"action": "ACCEPT_ALLIANCE", "target": pending_allies.pop(0)})
                aps -= 1
                continue
                
            if diplo["pending_trade_agreements"]:
                actions.append({"action": "ACCEPT_TRADE", "target": diplo["pending_trade_agreements"].pop(0)})
                aps -= 1
                continue
                
            if diplo["pending_research_pacts"]:
                actions.append({"action": "ACCEPT_RESEARCH", "target": diplo["pending_research_pacts"].pop(0)})
                aps -= 1
                continue
                
            if at_war and me["stats"]["manpower"] >= 100 and me["stats"]["production"] >= 50:
                actions.append({"action": "MILITARY_STRIKE", "target": random.choice(at_war)})
                aps -= 1
                continue
                
            if allies:
                if pers in ["MERCHANT", "BALANCED", "DIPLOMAT"] and aps > 0:
                    viable_trade = [a for a in allies if a not in diplo["active_trade_agreements"] and a not in proposed_trade]
                    if viable_trade:
                        target = random.choice(viable_trade)
                        actions.append({"action": "PROPOSE_TRADE", "target": target})
                        proposed_trade.add(target)
                        aps -= 1
                        continue
                        
                if pers in ["SCIENTIST", "BALANCED", "DIPLOMAT"] and aps > 0:
                    viable_sci = [a for a in allies if a not in diplo["active_research_pacts"] and a not in proposed_research]
                    if viable_sci:
                        target = random.choice(viable_sci)
                        actions.append({"action": "PROPOSE_RESEARCH", "target": target})
                        proposed_research.add(target)
                        aps -= 1
                        continue

            if not me["tech"]["current"] and pers in ["SCIENTIST", "BALANCED", "WARMONGER"]:
                available_techs = [t.value for t in Tech if t.value not in me["tech"]["unlocked"]]
                if available_techs:
                    actions.append({"action": "RESEARCH", "target": random.choice(available_techs)})
                    me["tech"]["current"] = "Pending" 
                    aps -= 1
                    continue
                    
            if not me["civic"]["current"] and pers in ["DIPLOMAT", "BALANCED", "MERCHANT"]:
                available_civics = [c.value for c in Civic if c.value not in me["civic"]["unlocked"]]
                if available_civics:
                    actions.append({"action": "PURSUE_CIVIC", "target": random.choice(available_civics)})
                    me["civic"]["current"] = "Pending"
                    aps -= 1
                    continue
                    
            war_threshold = {"WARMONGER": 400, "BALANCED": 700, "SCIENTIST": 1000, "MERCHANT": 1000, "DIPLOMAT": 1200}[pers]
            
            if not at_war and me["stats"]["military"] >= war_threshold:
                high_grievance = [n["id"] for n in sym_state["other_nations"] if str(n["id"]) in diplo["grievances"] and diplo["grievances"][str(n["id"])] >= 50 and not n["is_defeated"] and n["diplomatic_status"] != "ALLIED"]
                if high_grievance:
                    target = max(high_grievance, key=lambda t: diplo["grievances"][str(t)])
                    actions.append({"action": "DECLARE_WAR", "target": target})
                    aps -= 1
                    at_war.append(target)
                    continue
                else:
                    targets = [n["id"] for n in sym_state["other_nations"] if n["id"] != self.player_id and not n["is_defeated"] and n["diplomatic_status"] != "ALLIED"]
                    if targets:
                        actions.append({"action": "DECLARE_WAR", "target": random.choice(targets)})
                        aps -= 1
                        continue
                        
            if pers in ["DIPLOMAT", "MERCHANT", "BALANCED"] and random.random() < 0.3:
                neutrals = [n["id"] for n in sym_state["other_nations"] if n["id"] != self.player_id and not n["is_defeated"] and n["diplomatic_status"] == "NEUTRAL"]
                if neutrals:
                    actions.append({"action": "PROPOSE_ALLIANCE", "target": random.choice(neutrals)})
                    aps -= 1
                    continue

            pool = ["GOLD", "MANPOWER", "PRODUCTION", "SCIENCE", "CIVICS"]
            if pers == "WARMONGER": pool = ["MANPOWER", "PRODUCTION", "MANPOWER"]
            elif pers == "SCIENTIST": pool = ["SCIENCE", "SCIENCE", "PRODUCTION"]
            elif pers == "MERCHANT": pool = ["GOLD", "GOLD", "PRODUCTION"]
            elif pers == "DIPLOMAT": pool = ["CIVICS", "CIVICS", "GOLD"]

            actions.append({"action": "HARVEST", "target": random.choice(pool)})
            aps -= 1
            
        # 2. Package into foundation model format and dump to JSON string
        agent_response_dict = {
            "reasoning": f"Simulating hardcoded rules for {pers} personality based on symbolic state.",
            "actions": actions
        }
        
        response_json = json.dumps(agent_response_dict)
        
        # 3. Give state actions back via symbolic parser
        engine_commands, err = parse_agent_response(response_json)
        
        if err:
            print(f"AI {self.player_id} generated invalid actions: {err}")
            return []
            
        return engine_commands