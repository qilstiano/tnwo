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
        """Generate map using cellular automata for continents"""
        # Step 1: Initialize random noise
        temp_grid = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                row.append(1 if random.random() < 0.55 else 0) # 1 is land, 0 is water
            temp_grid.append(row)
            
        # Step 2: Smooth with cellular automata rules
        for _ in range(4):
            new_grid = [[0 for _ in range(self.width)] for _ in range(self.height)]
            for y in range(self.height):
                for x in range(self.width):
                    # count land neighbors
                    land_count = 0
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.width and 0 <= ny < self.height:
                                land_count += temp_grid[ny][nx]
                            else:
                                land_count += 0 # Water at borders
                    
                    if land_count >= 5 or (land_count == 4 and temp_grid[y][x] == 1):
                        new_grid[y][x] = 1
                    else:
                        new_grid[y][x] = 0
            temp_grid = new_grid
            
        # Step 3: Assign specific terrains and features
        for y in range(self.height):
            row = []
            for x in range(self.width):
                if temp_grid[y][x] == 0:
                    terrain = Terrain.WATER
                else:
                    rand = random.random()
                    if rand < 0.15:
                        terrain = Terrain.MOUNTAIN
                    elif rand < 0.35:
                        terrain = Terrain.HILL
                    elif rand < 0.65:
                        terrain = Terrain.PLAINS
                    else:
                        terrain = Terrain.GRASS
                
                # Features
                feature = Feature.NONE
                if terrain != Terrain.WATER and random.random() < 0.25:
                    feature = Feature.FOREST
                elif terrain != Terrain.WATER and random.random() < 0.15:
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