import random
from typing import Dict, List, Optional
from typing import Dict, Optional, List
from .constants import Resource, DiplomaticState, TECH_COSTS, CIVIC_COSTS
from .models import Nation, Achievements


class GameState:
    def __init__(self, num_players: int = 4):
        self.nations: Dict[int, Nation] = {}
        self.turn: int = 1
        self.current_player: int = 0
        
        # Diplomatic matrix: dict of dicts diplomacy[id1][id2] = DiplomaticState
        self.diplomacy: Dict[int, Dict[int, DiplomaticState]] = {}
        
        # Generate some flavor names
        NATION_NAMES = ["Valnor", "Zephyria", "Ironforge", "Aeldria", "Crimson Order", "Sunpeak", "Mercia"]
        PERSONALITIES = ["WARMONGER", "SCIENTIST", "MERCHANT", "DIPLOMAT", "BALANCED"]
        
        for i in range(num_players):
            name = NATION_NAMES[i % len(NATION_NAMES)]
            color = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4"][i]
            pers = random.choice(PERSONALITIES)
            
            n = Nation(id=i, name=name, color=color, personality=pers)
            
            # Start with some base yields randomly shifted for asymmetry
            n.gold_yield = random.randint(10, 20)
            n.manpower_yield = random.randint(10, 20)
            n.production_yield = random.randint(2, 8)
            
            self.nations[i] = n
            self.diplomacy[i] = {}
            for j in range(num_players):
                self.diplomacy[i][j] = DiplomaticState.NEUTRAL
                    
    def process_turn_updates(self):
        """Called automatically at the end of simultaneous resolution"""
        for n_id, nation in self.nations.items():
            if nation.is_defeated:
                continue
            
            # Reset AP and Queue
            nation.action_points = nation.max_action_points
            nation.queued_actions.clear()
            
            # Assess War Exhaustion
            at_war_with = [oid for oid, st in self.diplomacy[nation.id].items() if st == DiplomaticState.WAR and not self.nations[oid].is_defeated]
            if at_war_with:
                nation.war_exhaustion += 1
            else:
                nation.war_exhaustion = max(0, nation.war_exhaustion - 2)

            # Apply passive yields
            # Calculate modifiers
            prod_mod = 1.0
            if "Steam Power" in nation.unlocked_techs: prod_mod += 0.25
            
            gold_mod = 1.0
            if nation.achievements.golden_age: gold_mod += 0.25
            gold_mod += (0.15 * len(nation.active_trade_agreements))
            
            sci_mod = 1.0
            sci_mod += (0.15 * len(nation.active_research_pacts))
            
            manpower_mod = 1.0
            if nation.achievements.national_identity: manpower_mod += 0.20
            if nation.achievements.levies: manpower_mod += 0.10
            
            # War exhaustion penalty to manpower
            manpower_mod -= (0.05 * nation.war_exhaustion)
            manpower_mod = max(0.1, manpower_mod) # heavily crippled but not backwards
            
            # Base generation + Annexation yields
            nation.gold += int((nation.gold_yield + nation.absorbed_gold_yield) * gold_mod)
            nation.manpower += int(nation.manpower_yield * manpower_mod)
            nation.production += int((nation.production_yield + nation.absorbed_prod_yield) * prod_mod)
            nation.science += nation.absorbed_sci_yield
            
            # Check tech/civic passives
            if getattr(nation.achievements, 'industrial_revolution', False):
                nation.science += nation.production // 10
            if getattr(nation.achievements, 'enlightenment', False):
                nation.science += nation.manpower // 10
                
            # Process Tech
            if nation.current_tech:
                nation.tech_progress += nation.science_yield + nation.science
                nation.science = 0 # consume lump science
                cost = self.get_tech_cost(nation.current_tech)
                if nation.tech_progress >= cost:
                    nation.unlocked_techs.append(nation.current_tech)
                    nation.current_tech = None
                    nation.tech_progress = 0
                    nation.achievements.techs_researched += 1
                    
            # Process Civic
            if nation.current_civic:
                nation.civic_progress += nation.civic_yield + nation.civics
                nation.civics = 0
                cost = self.get_civic_cost(nation.current_civic)
                if nation.civic_progress >= cost:
                    nation.unlocked_civics.append(nation.current_civic)
                    nation.current_civic = None
                    nation.civic_progress = 0
                    nation.achievements.civic_policies_adopted += 1
            
            # Check constant achievements
            if nation.gold > 1000:
                nation.achievements.consecutive_wealth_turns += 1
                if nation.achievements.consecutive_wealth_turns >= 10:
                    nation.achievements.mercantilism = True
            else:
                nation.achievements.consecutive_wealth_turns = 0
                
            if nation.gold >= 5000:
                nation.achievements.treasury_reserve = True
                
            if nation.military > 500:
                nation.achievements.consecutive_military_turns += 1
                if nation.achievements.consecutive_military_turns >= 5:
                    nation.achievements.standing_army = True
            else:
                nation.achievements.consecutive_military_turns = 0
              # Extra Achievements
            if nation.production_yield >= 500:
                nation.achievements.industrial_heartland = True
            if nation.gold > 2000:
                nation.achievements.surplus_economy = True
                gold_mod += 0.10
            if nation.manpower > 1500:
                nation.achievements.standing_army = True
            if len(nation.unlocked_techs) >= 3:
                nation.achievements.university_network = True
                sci_mod += 0.20
            if len(nation.unlocked_civics) >= 3:
                nation.achievements.bureaucracy = True
                nation.civic_yield = max(nation.civic_yield, int(nation.civic_yield * 1.2))
                
    def get_tech_cost(self, tech_name: str) -> int:
        cost = TECH_COSTS.get(tech_name, 100)
        # Check modifiers
        for n in self.nations.values():
            if tech_name == n.current_tech and n.achievements.scientific_method:
                return int(cost * 0.85)
        return cost

    def get_civic_cost(self, civic_name: str) -> int:
        return CIVIC_COSTS.get(civic_name, 100)
    
    def get_diplomatic_state(self, id1: int, id2: int) -> DiplomaticState:
        return self.diplomacy[id1].get(id2, DiplomaticState.NEUTRAL)
        
    def set_diplomatic_state(self, id1: int, id2: int, state: DiplomaticState):
        self.diplomacy[id1][id2] = state
        self.diplomacy[id2][id1] = state

    def check_winner(self) -> Optional[int]:
        active_nations = [n.id for n in self.nations.values() if not n.is_defeated]
        if len(active_nations) == 1:
            return active_nations[0]
        if self.turn >= 200:
            # Score based on military + gold + production + unlocked trees
            scores = {}
            for n in self.nations.values():
                if n.is_defeated: continue
                score = n.gold + n.manpower + n.production + (len(n.unlocked_techs)*500) + (len(n.unlocked_civics)*500)
                scores[n.id] = score
            if scores:
                return max(scores, key=scores.get)
        return None

    def get_symbolic_state(self, player_id: int) -> dict:
        """
        Generates a symbolic representation of the game state from the perspective
        of `player_id`. Hides specific exact stats of other nations unless visible.
        """
        my_nation = self.nations.get(player_id)
        if not my_nation:
             return {"error": "Invalid player ID"}

        # 1. Global State
        active_nations = [n.id for n in self.nations.values() if not n.is_defeated]
        global_data = {
            "turn": self.turn,
            "active_nations": active_nations
        }

        # 2. My State
        my_data = {
            "id": my_nation.id,
            "name": my_nation.name,
            "personality": my_nation.personality,
            "is_defeated": my_nation.is_defeated,
            "stats": {
                "gold": my_nation.gold,
                "manpower": my_nation.manpower,
                "production": my_nation.production,
                "science": my_nation.science,
                "civics": my_nation.civics,
                "military": my_nation.military
            },
            "yields": {
                "gold_yield": my_nation.gold_yield + my_nation.absorbed_gold_yield,
                "manpower_yield": my_nation.manpower_yield,
                "production_yield": my_nation.production_yield + my_nation.absorbed_prod_yield,
                "science_yield": my_nation.science_yield + my_nation.absorbed_sci_yield,
                "civic_yield": my_nation.civic_yield
            },
            "tech": {
                "unlocked": my_nation.unlocked_techs,
                "current": my_nation.current_tech,
                "progress": my_nation.tech_progress
            },
            "civic": {
                "unlocked": my_nation.unlocked_civics,
                "current": my_nation.current_civic,
                "progress": my_nation.civic_progress
            },
            "actions": {
                "max_points": my_nation.max_action_points,
                "current_points": my_nation.action_points
            },
            "status": {
                "infrastructure_health": my_nation.infrastructure_health,
                "war_exhaustion": my_nation.war_exhaustion
            },
            "diplomacy": {
                "active_trade_agreements": my_nation.active_trade_agreements,
                "active_research_pacts": my_nation.active_research_pacts,
                "pending_trade_agreements": my_nation.pending_trade_agreements,
                "pending_research_pacts": my_nation.pending_research_pacts,
                "grievances": my_nation.grievances
            }
        }

        # 3. Other Nations State
        others_data = []
        for n_id, n in self.nations.items():
            if n_id == player_id: continue
            
            diplomatic_status = self.get_diplomatic_state(player_id, n_id).name
            tier = len(n.unlocked_techs) + 1
            
            others_data.append({
                "id": n.id,
                "name": n.name,
                "is_defeated": n.is_defeated,
                "diplomatic_status": diplomatic_status,
                "visible_status": {
                    "infrastructure_health": n.infrastructure_health,
                    "estimated_tech_tier": tier
                }
            })

        return {
            "global_state": global_data,
            "my_nation": my_data,
            "other_nations": others_data
        }