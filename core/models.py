# core/models.py
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from .constants import Terrain, Feature, District, UnitType, Tech, Civic

@dataclass
class Tile:
    terrain: Terrain
    feature: Feature = Feature.NONE
    district: District = District.NONE
    unit: UnitType = UnitType.NONE
    owned_by: Optional[int] = None
    x: int = 0
    y: int = 0
    
    def to_symbol(self) -> str:
        return f"[{self.terrain}{self.feature}{self.district}{self.unit}]"
    
    @property
    def is_passable(self) -> bool:
        return self.terrain != Terrain.WATER and self.district != District.CITY_CENTER
    
    @property
    def production_value(self) -> int:
        value = 0
        if self.terrain == Terrain.PLAINS:
            value += 1
        elif self.terrain == Terrain.HILL:
            value += 2
        if self.feature == Feature.FOREST:
            value += 1
        return value
    
    @property
    def food_value(self) -> int:
        if self.terrain == Terrain.GRASS:
            return 2
        elif self.terrain == Terrain.PLAINS:
            return 1
        return 0

@dataclass
class City:
    id: str
    player_id: int
    x: int
    y: int
    population: int = 1
    production_flow: int = 0
    production_stock: int = 0
    build_queue: List[str] = field(default_factory=list)  # Fixed!
    worked_tiles: List[Tuple[int, int]] = field(default_factory=list)  # Fixed!

@dataclass
class Unit:
    id: str
    unit_type: UnitType
    player_id: int
    x: int
    y: int
    hp: int = 100
    strength: int = 10
    
    def __post_init__(self):
        if self.unit_type == UnitType.WARRIOR:
            self.strength = 15
        elif self.unit_type == UnitType.SETTLER:
            self.strength = 0
            self.hp = 50

@dataclass
class Player:
    id: int
    gold: int = 150
    faith: int = 50
    science_flow: int = 0
    culture_flow: int = 0
    current_tech: Optional[Tech] = None
    current_civic: Optional[Civic] = None
    tech_progress: int = 0
    civic_progress: int = 0
    cities: List[str] = field(default_factory=list)  # Fixed!