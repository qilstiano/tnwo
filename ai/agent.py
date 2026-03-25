import random
from typing import List
from core.game_state import GameState
from engine.actions import ActionHandler
from core.constants import Resource, Tech, Civic, DiplomaticState

class AIAgent:
    def __init__(self, player_id: int):
        self.player_id = player_id

    def decide_actions(self, state: GameState, handler: ActionHandler) -> List[str]:
        nation = state.nations[self.player_id]
        if nation.is_defeated:
            return []
            
        actions = []
        aps = nation.max_action_points
        pers = nation.personality
        
        at_war = [n for n, s in state.diplomacy[self.player_id].items() if s == DiplomaticState.WAR and not state.nations[n].is_defeated]
        allies = [n for n, s in state.diplomacy[self.player_id].items() if s == DiplomaticState.ALLIED and not state.nations[n].is_defeated]
        pending_allies = [n for n, s in state.diplomacy[self.player_id].items() if s == DiplomaticState.ALLIANCE_PENDING]
        
        proposed_trade = set()
        proposed_research = set()
        
        while aps > 0:
            if pending_allies:
                actions.append(f"ACCEPT_ALLIANCE {pending_allies.pop(0)}")
                aps -= 1
                continue
                
            if nation.pending_trade_agreements:
                actions.append(f"ACCEPT_TRADE {nation.pending_trade_agreements.pop(0)}")
                aps -= 1
                continue
                
            if nation.pending_research_pacts:
                actions.append(f"ACCEPT_RESEARCH {nation.pending_research_pacts.pop(0)}")
                aps -= 1
                continue
                
            if at_war and nation.manpower >= 100 and nation.production >= 50:
                actions.append(f"MILITARY_STRIKE {random.choice(at_war)}")
                aps -= 1
                continue
                
            if allies:
                if pers in ["MERCHANT", "BALANCED", "DIPLOMAT"] and aps > 0:
                    viable_trade = [a for a in allies if a not in nation.active_trade_agreements and a not in proposed_trade]
                    if viable_trade:
                        target = random.choice(viable_trade)
                        actions.append(f"PROPOSE_TRADE {target}")
                        actions.append(f"ACCEPT_TRADE {target}")
                        proposed_trade.add(target)
                        aps -= 1
                        continue
                        
                if pers in ["SCIENTIST", "BALANCED", "DIPLOMAT"] and aps > 0:
                    viable_sci = [a for a in allies if a not in nation.active_research_pacts and a not in proposed_research]
                    if viable_sci:
                        target = random.choice(viable_sci)
                        actions.append(f"PROPOSE_RESEARCH {target}")
                        actions.append(f"ACCEPT_RESEARCH {target}")
                        proposed_research.add(target)
                        aps -= 1
                        continue

            if not nation.current_tech and pers in ["SCIENTIST", "BALANCED", "WARMONGER"]:
                available_techs = [t.value for t in Tech if t.value not in nation.unlocked_techs]
                if available_techs:
                    actions.append(f"RESEARCH {random.choice(available_techs)}")
                    nation.current_tech = "Pending" 
                    aps -= 1
                    continue
                    
            if not nation.current_civic and pers in ["DIPLOMAT", "BALANCED", "MERCHANT"]:
                available_civics = [c.value for c in Civic if c.value not in nation.unlocked_civics]
                if available_civics:
                    actions.append(f"PURSUE_CIVIC {random.choice(available_civics)}")
                    nation.current_civic = "Pending"
                    aps -= 1
                    continue
                    
            war_threshold = {"WARMONGER": 400, "BALANCED": 700, "SCIENTIST": 1000, "MERCHANT": 1000, "DIPLOMAT": 1200}[pers]
            
            if not at_war and nation.military >= war_threshold:
                high_grievance = [n for n, g in nation.grievances.items() if g >= 50 and not state.nations[n].is_defeated and state.get_diplomatic_state(self.player_id, n) != DiplomaticState.ALLIED]
                if high_grievance:
                    target = max(high_grievance, key=lambda t: nation.grievances[t])
                    actions.append(f"DECLARE_WAR {target}")
                    aps -= 1
                    at_war.append(target)
                    continue
                else:
                    targets = [n.id for n in state.nations.values() if n.id != self.player_id and not n.is_defeated and state.get_diplomatic_state(self.player_id, n.id) != DiplomaticState.ALLIED]
                    if targets:
                        actions.append(f"DECLARE_WAR {random.choice(targets)}")
                        aps -= 1
                        continue
                        
            if pers in ["DIPLOMAT", "MERCHANT", "BALANCED"] and random.random() < 0.3:
                neutrals = [n.id for n in state.nations.values() if n.id != self.player_id and not n.is_defeated and state.get_diplomatic_state(self.player_id, n.id) == DiplomaticState.NEUTRAL]
                if neutrals:
                    actions.append(f"PROPOSE_ALLIANCE {random.choice(neutrals)}")
                    aps -= 1
                    continue

            pool = ["GOLD", "MANPOWER", "PRODUCTION", "SCIENCE", "CIVICS"]
            if pers == "WARMONGER": pool = ["MANPOWER", "PRODUCTION", "MANPOWER"]
            elif pers == "SCIENTIST": pool = ["SCIENCE", "SCIENCE", "PRODUCTION"]
            elif pers == "MERCHANT": pool = ["GOLD", "GOLD", "PRODUCTION"]
            elif pers == "DIPLOMAT": pool = ["CIVICS", "CIVICS", "GOLD"]

            actions.append(f"HARVEST {random.choice(pool)}")
            aps -= 1
            
        if nation.current_tech == "Pending": nation.current_tech = None
        if nation.current_civic == "Pending": nation.current_civic = None
        
        return actions