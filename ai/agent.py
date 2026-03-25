# ai/agent.py
from typing import Optional, Tuple
import random
from core.game_state import GameState
from core.constants import UnitType, District, Tech
from core.models import Player, City, Unit
from engine.actions import ActionHandler

class AIAgent:
    def __init__(self, player_id: int, difficulty: str = "normal"):
        self.player_id = player_id
        self.difficulty = difficulty
    
    def decide_action(self, state: GameState, handler: ActionHandler) -> str:
        """Decide what action to take this turn"""
        if state.current_player != self.player_id:
            return "NEXT_TURN"
        
        player = state.players[self.player_id]
        
        # Priority 1: Expand if possible
        settler_action = self._find_settler_action(state, handler)
        if settler_action:
            return settler_action
        
        # Priority 2: Attack nearby enemies
        attack_action = self._find_attack_action(state, handler)
        if attack_action:
            return attack_action
        
        # Priority 3: Build in cities
        build_action = self._find_build_action(state, handler)
        if build_action:
            return build_action
        
        # Priority 4: Research tech
        if player.current_tech:
            tech_action = self._find_research_action(state, handler)
            if tech_action:
                return tech_action
        
        # Priority 5: Buy tiles
        buy_action = self._find_buy_tile_action(state, handler)
        if buy_action:
            return buy_action
        
        # Default: end turn
        return "NEXT_TURN"
    
    def _find_settler_action(self, state: GameState, handler: ActionHandler) -> Optional[str]:
        """Find a settler to move to a good settlement location"""
        for unit_id, unit in state.units.items():
            if unit.unit_type == UnitType.SETTLER and unit.player_id == self.player_id:
                # Find best nearby location to settle
                best_score = -1
                best_pos = None
                
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        x, y = unit.x + dx, unit.y + dy
                        tile = state.map.get_tile(x, y)
                        if tile and tile.is_passable and tile.district != District.CITY_CENTER:
                            # Check if location is valid (not too close to other cities)
                            valid_location = True
                            for city in state.cities.values():
                                if abs(city.x - x) + abs(city.y - y) < 3:
                                    valid_location = False
                                    break
                            
                            if not valid_location:
                                continue
                            
                            # Score location (mountains good for science, hills good for production)
                            score = 0
                            adjacent = state.map.get_adjacent_tiles(x, y)
                            for adj in adjacent:
                                if adj.terrain.value == "M":  # Mountain
                                    score += 3
                                elif adj.terrain.value == "H":  # Hill
                                    score += 2
                                elif adj.feature.value == "R":  # Resource
                                    score += 1
                            
                            # Bonus for being near existing cities (but not too close)
                            for city in state.cities.values():
                                dist = abs(city.x - x) + abs(city.y - y)
                                if dist == 3:
                                    score += 1
                                elif dist == 4:
                                    score += 0
                                else:
                                    score -= 1
                            
                            if score > best_score:
                                best_score = score
                                best_pos = (x, y)
                
                if best_pos and best_score > 0:
                    # If adjacent, found city; otherwise move towards it
                    if abs(unit.x - best_pos[0]) + abs(unit.y - best_pos[1]) == 1:
                        return f"FOUND_CITY {unit_id} {best_pos[0]} {best_pos[1]}"
                    else:
                        # Move towards best position
                        return f"MOVE {unit_id} {best_pos[0]} {best_pos[1]}"
        
        return None
    
    def _find_attack_action(self, state: GameState, handler: ActionHandler) -> Optional[str]:
        """Find an enemy unit to attack"""
        # Find closest enemy
        closest_dist = float('inf')
        closest_attack = None
        
        for attacker_id, attacker in state.units.items():
            if attacker.player_id != self.player_id:
                continue
            
            # Don't attack with settlers
            if attacker.unit_type == UnitType.SETTLER:
                continue
            
            for defender_id, defender in state.units.items():
                if defender.player_id == self.player_id:
                    continue
                
                dist = abs(attacker.x - defender.x) + abs(attacker.y - defender.y)
                if dist == 1 and dist < closest_dist:
                    closest_dist = dist
                    closest_attack = (attacker_id, defender_id)
        
        if closest_attack:
            return f"ATTACK {closest_attack[0]} {closest_attack[1]}"
        
        # If no adjacent enemies, consider moving towards them
        # Find nearest enemy to any of our units
        nearest_enemy = None
        nearest_dist = float('inf')
        nearest_attacker = None
        
        for attacker_id, attacker in state.units.items():
            if attacker.player_id != self.player_id:
                continue
            
            if attacker.unit_type == UnitType.SETTLER:
                continue
                
            for defender_id, defender in state.units.items():
                if defender.player_id == self.player_id:
                    continue
                
                dist = abs(attacker.x - defender.x) + abs(attacker.y - defender.y)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_enemy = defender
                    nearest_attacker = attacker
        
        if nearest_attacker and nearest_enemy and nearest_dist <= 5:
            # Move towards enemy
            dx = 0
            dy = 0
            if nearest_attacker.x < nearest_enemy.x:
                dx = 1
            elif nearest_attacker.x > nearest_enemy.x:
                dx = -1
            
            if nearest_attacker.y < nearest_enemy.y:
                dy = 1
            elif nearest_attacker.y > nearest_enemy.y:
                dy = -1
            
            new_x = nearest_attacker.x + dx
            new_y = nearest_attacker.y + dy
            
            tile = state.map.get_tile(new_x, new_y)
            if tile and tile.is_passable:
                return f"MOVE {nearest_attacker.id} {new_x} {new_y}"
        
        return None
    
    def _find_build_action(self, state: GameState, handler: ActionHandler) -> Optional[str]:
        """Decide what to build in cities"""
        player = state.players[self.player_id]
        
        # Count how many settlers we have
        settler_count = sum(1 for unit in state.units.values() 
                           if unit.player_id == self.player_id and unit.unit_type == UnitType.SETTLER)
        
        # Check if we need more settlers
        if len(player.cities) + settler_count < min(3, len(state.players) * 2):
            for city_id in player.cities:
                city = state.cities[city_id]
                if "Settler" not in city.build_queue and len(city.build_queue) < 2:
                    return f"BUILD {city_id} Settler"
        
        # Check if we need defense
        enemy_nearby = False
        for unit in state.units.values():
            if unit.player_id != self.player_id:
                for city_id in player.cities:
                    city = state.cities[city_id]
                    if abs(unit.x - city.x) + abs(unit.y - city.y) < 5:
                        enemy_nearby = True
                        break
        
        if enemy_nearby:
            for city_id in player.cities:
                city = state.cities[city_id]
                # Check if we have enough warriors
                warrior_count = sum(1 for unit in state.units.values() 
                                   if unit.player_id == self.player_id and unit.unit_type == UnitType.WARRIOR)
                if warrior_count < len(player.cities) and "Warrior" not in city.build_queue:
                    return f"BUILD {city_id} Warrior"
        
        # Build districts for science or production
        has_campus = False
        has_industrial = False
        
        for city_id in player.cities:
            city = state.cities[city_id]
            for x, y in city.worked_tiles:
                tile = state.map.get_tile(x, y)
                if tile and tile.district == District.SCIENCE:
                    has_campus = True
                elif tile and tile.district == District.INDUSTRIAL:
                    has_industrial = True
        
        # Build campus if we have writing tech
        if not has_campus and player.current_tech == Tech.WRITING:
            for city_id in player.cities:
                city = state.cities[city_id]
                # Check if there's a good location for campus (near mountains)
                best_score = 0
                best_pos = None
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        x, y = city.x + dx, city.y + dy
                        tile = state.map.get_tile(x, y)
                        if tile and tile.district == District.NONE and tile.is_passable:
                            score = state.get_adjacency_bonus(x, y, District.SCIENCE)
                            if score > best_score:
                                best_score = score
                                best_pos = (x, y)
                
                if best_pos:
                    return f"BUILD {city_id} Campus"
        
        # Build industrial zone if we have apprenticeship tech
        if not has_industrial and player.current_tech == Tech.APPRENTICESHIP:
            for city_id in player.cities:
                city = state.cities[city_id]
                best_score = 0
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        x, y = city.x + dx, city.y + dy
                        tile = state.map.get_tile(x, y)
                        if tile and tile.district == District.NONE and tile.is_passable:
                            score = state.get_adjacency_bonus(x, y, District.INDUSTRIAL)
                            if score > best_score:
                                best_score = score
                
                if best_score > 0:
                    return f"BUILD {city_id} IndustrialZone"
        
        return None
    
    def _find_research_action(self, state: GameState, handler: ActionHandler) -> Optional[str]:
        """Decide what technology to research"""
        player = state.players[self.player_id]
        
        # Always research the next available tech
        if player.current_tech:
            tech_order = list(Tech)
            current_index = tech_order.index(player.current_tech)
            
            if current_index < len(tech_order) - 1:
                next_tech = tech_order[current_index + 1]
                return f"RESEARCH {next_tech.value}"
        
        return None
    
    def _find_buy_tile_action(self, state: GameState, handler: ActionHandler) -> Optional[str]:
        """Buy a tile if we have excess gold"""
        player = state.players[self.player_id]
        
        # Don't buy if we don't have enough gold
        if player.gold < 100:
            return None
        
        for city_id in player.cities:
            city = state.cities[city_id]
            # Find an unowned adjacent tile with good yield
            best_score = -1
            best_tile = None
            
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    x, y = city.x + dx, city.y + dy
                    tile = state.map.get_tile(x, y)
                    if tile and tile.owned_by is None:
                        # Check adjacency to owned tiles
                        for owned_x, owned_y in city.worked_tiles:
                            if abs(owned_x - x) + abs(owned_y - y) == 1:
                                # Score tile based on value
                                score = tile.production_value + tile.food_value
                                
                                # Bonus for resources
                                if tile.feature.value == "R":
                                    score += 3
                                
                                if score > best_score:
                                    best_score = score
                                    best_tile = (x, y)
            
            if best_tile and best_score > 2:
                return f"BUY {city_id} {best_tile[0]} {best_tile[1]}"
        
        return None