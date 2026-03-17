from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any

# this class describes the various land types available for each country
class LandType(Enum):
    URBAN = "urban"
    AGRICULTURAL = "agricultural"
    ENERGY = "energy"
    MINERAL = "mineral"
    STRATEGIC = "strategic"

# this class describes the various structures available for each country
#   - each structure has a cost, build time, and production rate
class StructureType(Enum):
    FARM = "farm"
    POWER_PLANT = "power_plant"
    MINE = "mine"
    SCHOOL = "school"
    MILITARY_BASE = "military_base"
    PORT = "port"
    FACTORY = "factory"
    RESEARCH_LAB = "research_lab"

# this class describes a single square of land
@dataclass
class LandSquare:
    square_id: str
    land_type: LandType
    structure: Optional[StructureType] = None
    improvement_level: int = 1
    under_construction_structure: Optional[StructureType] = None
    build_time_remaining: int = 0

# this class describes the state of a single country
@dataclass
class CountryState:
    name: str
    land_squares: List[LandSquare]
    food: float
    energy: float
    minerals: float
    wealth: float
    population: int
    tech_level: float
    military_power: float
    reputation: float = 1.0

# this class describes the actions that can be taken by each country
#   - each action has a type, target country, resource type, amount, 
#     square id, structure type, and parameters
@dataclass
class AgentAction:
    action_type: str  # e.g., "build", "trade", "war"
    target_country: Optional[str] = None
    resource_type: Optional[str] = None
    amount: float = 0.0
    square_id: Optional[str] = None
    structure_type: Optional[StructureType] = None
    parameters: Dict[str, Any] = field(default_factory=dict)

# this class describes the entire game world
#   - it has a current turn and a dictionary of countries
@dataclass
class GameWorld:
    current_turn: int
    countries: Dict[str, CountryState]

@dataclass
class GameConfig:
    structure_costs: Dict[StructureType, float]
    structure_build_times: Dict[StructureType, int]
    production_rates: Dict[StructureType, Dict[str, float]]
    food_per_capita: float = 1.0
    energy_per_capita: float = 0.5
    malthusian_decay: float = 0.1
