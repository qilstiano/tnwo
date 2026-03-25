from typing import List, Optional, Tuple
import random
from .constants import Terrain, Feature, District, UnitType
from .models import Tile

class GameMap:
    def __init__(self, width: int = 10, height: int = 10):
        self.width = width
        self.height = height
        self.tiles: List[List[Tile]] = []
        self._generate_map()
    
    def _generate_map(self):
        """Generate a simple random map"""
        for y in range(self.height):
            row = []
            for x in range(self.width):
                # Simple terrain generation
                rand = random.random()
                if rand < 0.2:
                    terrain = Terrain.MOUNTAIN
                elif rand < 0.3:
                    terrain = Terrain.WATER
                elif rand < 0.5:
                    terrain = Terrain.HILL
                elif rand < 0.7:
                    terrain = Terrain.PLAINS
                else:
                    terrain = Terrain.GRASS
                
                # Features
                feature = Feature.NONE
                if terrain != Terrain.WATER and random.random() < 0.3:
                    feature = Feature.FOREST
                elif terrain != Terrain.WATER and random.random() < 0.2:
                    feature = Feature.RESOURCE
                
                tile = Tile(terrain=terrain, feature=feature, x=x, y=y)
                row.append(tile)
            self.tiles.append(row)
    
    def get_tile(self, x: int, y: int) -> Optional[Tile]:
        """Get tile at coordinates"""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.tiles[y][x]
        return None
    
    def set_tile(self, x: int, y: int, tile: Tile):
        """Set tile at coordinates"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.tiles[y][x] = tile
            tile.x = x
            tile.y = y
    
    def get_adjacent_tiles(self, x: int, y: int) -> List[Tile]:
        """Get all adjacent tiles (including diagonals)"""
        adjacent = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                tile = self.get_tile(x + dx, y + dy)
                if tile:
                    adjacent.append(tile)
        return adjacent
    
    def render(self, player_id: Optional[int] = None) -> str:
        """Render map to string representation"""
        output = "    " + " ".join(f"{i:2}" for i in range(self.width)) + "\n"
        for y in range(self.height):
            output += f"{y:2}  "
            for x in range(self.width):
                tile = self.tiles[y][x]
                output += tile.to_symbol() + " "
            output += "\n"
        return output