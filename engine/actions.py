from typing import Optional, Tuple, List
from core.constants import UnitType, District, Tech
from core.models import Player, City, Unit
from core.game_state import GameState

class ActionHandler:
    def __init__(self, state: GameState):
        self.state = state
    
    def move_unit(self, unit_id: str, x: int, y: int) -> bool:
        """Move a unit to new coordinates"""
        unit = self.state.units.get(unit_id)
        if not unit or unit.player_id != self.state.current_player:
            return False
        
        target_tile = self.state.map.get_tile(x, y)
        if not target_tile or not target_tile.is_passable:
            return False
        
        # Check for other units
        for other_unit in self.state.units.values():
            if other_unit.x == x and other_unit.y == y:
                return False
        
        # Update unit position
        old_tile = self.state.map.get_tile(unit.x, unit.y)
        if old_tile:
            old_tile.unit = UnitType.NONE
        
        unit.x = x
        unit.y = y
        target_tile.unit = unit.unit_type
        
        return True
    
    def attack_unit(self, attacker_id: str, defender_id: str) -> bool:
        """Attack another unit"""
        attacker = self.state.units.get(attacker_id)
        defender = self.state.units.get(defender_id)
        
        if not attacker or not defender:
            return False
        
        if attacker.player_id != self.state.current_player:
            return False
        
        # Simple combat resolution
        terrain_bonus = 0
        defender_tile = self.state.map.get_tile(defender.x, defender.y)
        if defender_tile and defender_tile.terrain == "H":  # Hills
            terrain_bonus = 5
        
        attacker_strength = attacker.strength
        defender_strength = defender.strength + terrain_bonus
        
        if attacker_strength > defender_strength:
            # Defender defeated
            del self.state.units[defender_id]
            defender_tile.unit = UnitType.NONE
            return True
        else:
            # Attacker takes damage
            attacker.hp -= 50
            if attacker.hp <= 0:
                del self.state.units[attacker_id]
                attacker_tile = self.state.map.get_tile(attacker.x, attacker.y)
                if attacker_tile:
                    attacker_tile.unit = UnitType.NONE
            return False
    
    def found_city(self, settler_id: str, x: int, y: int) -> bool:
        """Found a new city with settler"""
        settler = self.state.units.get(settler_id)
        if not settler or settler.unit_type != UnitType.SETTLER:
            return False
        
        if settler.player_id != self.state.current_player:
            return False
        
        # Check if location is valid (no nearby cities)
        tile = self.state.map.get_tile(x, y)
        if not tile or tile.district == District.CITY_CENTER:
            return False
        
        # Check distance from other cities
        for city in self.state.cities.values():
            if abs(city.x - x) + abs(city.y - y) < 3:
                return False
        
        # Found city
        self.state._found_city(settler.player_id, x, y)
        
        # Remove settler
        del self.state.units[settler_id]
        tile.unit = UnitType.NONE
        
        return True
    
    def build_item(self, city_id: str, item: str) -> bool:
        """Add item to city build queue"""
        city = self.state.cities.get(city_id)
        if not city or city.player_id != self.state.current_player:
            return False
        
        valid_items = ["Settler", "Warrior", "Granary", "Campus", "IndustrialZone"]
        if item not in valid_items:
            return False
        
        city.build_queue.append(item)
        return True
    
    def research_tech(self, tech_name: str) -> bool:
        """Research a new technology"""
        player = self.state.players[self.state.current_player]
        
        try:
            tech = Tech(tech_name)
        except ValueError:
            return False
        
        # Check if tech is researchable (must be next in tree)
        tech_order = list(Tech)
        current_index = tech_order.index(player.current_tech) if player.current_tech else -1
        target_index = tech_order.index(tech)
        
        if target_index <= current_index:
            return False
        
        if target_index > current_index + 1:
            return False
        
        player.current_tech = tech
        player.tech_progress = 0
        return True
    
    def buy_tile(self, city_id: str, x: int, y: int) -> bool:
        """Purchase a tile for a city"""
        city = self.state.cities.get(city_id)
        player = self.state.players[self.state.current_player]
        
        if not city or city.player_id != self.state.current_player:
            return False
        
        tile = self.state.map.get_tile(x, y)
        if not tile or tile.owned_by is not None:
            return False
        
        # Check if tile is adjacent to city or owned tiles
        adjacent_to_owned = False
        for owned_x, owned_y in city.worked_tiles:
            if abs(owned_x - x) + abs(owned_y - y) == 1:
                adjacent_to_owned = True
                break
        
        if not adjacent_to_owned:
            return False
        
        # Purchase cost (simple formula)
        cost = 50
        if player.gold >= cost:
            player.gold -= cost
            tile.owned_by = city.player_id
            city.worked_tiles.append((x, y))
            return True
        
        return False
    
    def end_turn(self) -> bool:
        """End current turn and process end-of-turn updates"""
        self.state.current_player = (self.state.current_player + 1) % len(self.state.players)
        if self.state.current_player == 0:
            self.state.turn += 1
            self._process_turn_updates()
        return True
    
    def _process_turn_updates(self):
        """Process end-of-turn updates for all players"""
        for player in self.state.players.values():
            # Process city production
            for city_id in player.cities:
                city = self.state.cities[city_id]
                production = self.state.get_city_production(city)
                city.production_stock += production
                
                # Check if build queue has items
                while city.build_queue and city.production_stock >= self._get_build_cost(city.build_queue[0]):
                    item = city.build_queue.pop(0)
                    cost = self._get_build_cost(item)
                    city.production_stock -= cost
                    self._complete_build(player, city, item)
            
            # Process tech research
            if player.current_tech:
                player.tech_progress += player.science_flow
                # Simplified: tech costs 100 science
                if player.tech_progress >= 100:
                    self._complete_tech(player)
    
    def _get_build_cost(self, item: str) -> int:
        """Get production cost for an item"""
        costs = {
            "Warrior": 40,
            "Settler": 80,
            "Granary": 60,
            "Campus": 100,
            "IndustrialZone": 120
        }
        return costs.get(item, 50)
    
    def _complete_build(self, player: Player, city: City, item: str):
        """Handle completion of a build item"""
        if item == "Settler":
            # Create settler unit
            settler_id = f"S_{player.id}_{self.state.turn}"
            settler = Unit(
                id=settler_id,
                unit_type=UnitType.SETTLER,
                player_id=player.id,
                x=city.x + 1,
                y=city.y
            )
            self.state.units[settler_id] = settler
            tile = self.state.map.get_tile(settler.x, settler.y)
            if tile:
                tile.unit = UnitType.SETTLER
        
        elif item == "Warrior":
            warrior_id = f"W_{player.id}_{self.state.turn}"
            warrior = Unit(
                id=warrior_id,
                unit_type=UnitType.WARRIOR,
                player_id=player.id,
                x=city.x + 1,
                y=city.y
            )
            self.state.units[warrior_id] = warrior
            tile = self.state.map.get_tile(warrior.x, warrior.y)
            if tile:
                tile.unit = UnitType.WARRIOR
        
        elif item == "Campus":
            # Place campus district
            tile = self.state.map.get_tile(city.x + 1, city.y)
            if tile:
                tile.district = District.SCIENCE
                player.science_flow += 2 + self.state.get_adjacency_bonus(
                    city.x + 1, city.y, District.SCIENCE
                )
        
        elif item == "IndustrialZone":
            tile = self.state.map.get_tile(city.x + 1, city.y)
            if tile:
                tile.district = District.INDUSTRIAL
    
    def _complete_tech(self, player: Player):
        """Handle completion of a technology"""
        tech_order = list(Tech)
        current_index = tech_order.index(player.current_tech)
        
        if current_index < len(tech_order) - 1:
            player.current_tech = tech_order[current_index + 1]
            player.tech_progress = 0
            
            # Update science flow from new techs
            if player.current_tech == Tech.WRITING:
                player.science_flow += 2
            elif player.current_tech == Tech.APPRENTICESHIP:
                player.science_flow += 3