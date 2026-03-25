# core/game_state.py
from typing import Dict, List, Optional, Tuple
from .constants import Tech, Civic, UnitType, District, Terrain
from .models import City, Unit, Player, Tile
from .map import GameMap

class GameState:
    def __init__(self, width: int = 10, height: int = 10, num_players: int = 2):
        self.map = GameMap(width, height)
        self.players: Dict[int, Player] = {}
        self.cities: Dict[str, City] = {}
        self.units: Dict[str, Unit] = {}
        self.turn: int = 0
        self.current_player: int = 0
        
        # Initialize players
        for i in range(num_players):
            self.players[i] = Player(id=i)
        
        # Place starting cities
        self._place_starting_cities()
    
    def _place_starting_cities(self):
        """Place starting cities for each player"""
        start_positions = [
            (2, 2),
            (self.map.width - 3, self.map.height - 3)
        ]
        
        for i, (x, y) in enumerate(start_positions):
            if i in self.players:
                city = self._found_city(i, x, y)
                # Add starting warrior
                warrior = Unit(
                    id=f"W_{i}_0",
                    unit_type=UnitType.WARRIOR,
                    player_id=i,
                    x=x + 1,
                    y=y
                )
                self.units[warrior.id] = warrior
                tile = self.map.get_tile(warrior.x, warrior.y)
                if tile:
                    tile.unit = UnitType.WARRIOR
    
    def _found_city(self, player_id: int, x: int, y: int) -> City:
        """Found a new city at location"""
        city_id = f"C_{player_id}_{len(self.players[player_id].cities)}"
        city = City(
            id=city_id,
            player_id=player_id,
            x=x,
            y=y,
            population=1
        )
        
        # Update tile to have city center
        tile = self.map.get_tile(x, y)
        if tile:
            tile.district = District.CITY_CENTER
            tile.owned_by = player_id
        
        self.cities[city_id] = city
        self.players[player_id].cities.append(city_id)
        
        # Initial worked tiles (city center only)
        city.worked_tiles = [(x, y)]
        
        return city
    
    def get_adjacency_bonus(self, x: int, y: int, district_type: District) -> int:
        """Calculate adjacency bonus for a district"""
        adjacent = self.map.get_adjacent_tiles(x, y)
        bonus = 0
        
        if district_type == District.SCIENCE:
            # Campus: +1 for each adjacent mountain
            for adj in adjacent:
                if adj.terrain == Terrain.MOUNTAIN:
                    bonus += 1
        elif district_type == District.INDUSTRIAL:
            # Industrial Zone: +1 for each adjacent mine (hill with resource or forest)
            for adj in adjacent:
                if adj.terrain == Terrain.HILL:
                    bonus += 1
        
        return bonus
    
    def get_city_production(self, city: City) -> int:
        """Calculate total production for a city"""
        base = 2  # City center base production
        
        # Add production from worked tiles
        for x, y in city.worked_tiles:
            tile = self.map.get_tile(x, y)
            if tile:
                base += tile.production_value
        
        # Add industrial zone bonuses if present
        for x, y in city.worked_tiles:
            tile = self.map.get_tile(x, y)
            if tile and tile.district == District.INDUSTRIAL:
                base += self.get_adjacency_bonus(x, y, District.INDUSTRIAL)
        
        return base
    
    def render(self) -> str:
        """Render full game state"""
        output = f"\n=== TURN {self.turn} | PLAYER {self.current_player} ===\n"
        
        player = self.players[self.current_player]
        output += f"Gold: {player.gold} | Faith: {player.faith} | "
        output += f"Sci: {player.science_flow}/turn | Cul: {player.culture_flow}/turn\n"
        
        if player.current_tech:
            output += f"Tech: {player.current_tech.value} ({player.tech_progress}/100)\n"
        
        output += f"\n--- MAP ---\n"
        output += self.map.render(self.current_player)
        
        output += f"\n--- UNITS ---\n"
        for unit in self.units.values():
            if unit.player_id == self.current_player:
                output += f"{unit.unit_type.value} ({unit.x},{unit.y}) HP:{unit.hp} "
        output += "\n"
        
        output += f"\n--- CITIES ---\n"
        for city_id in player.cities:
            city = self.cities[city_id]
            prod = self.get_city_production(city)
            output += f"{city.id} [Pop {city.population}] [Prod {prod}] Queue: {city.build_queue}\n"
            output += f" Worked Tiles: {city.worked_tiles}\n"
        
        return output