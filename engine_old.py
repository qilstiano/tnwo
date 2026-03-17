from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any
import math

"""
this is the old engine code, it is not used in the current version of the game
"""

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

class SimulationEngine:
    def __init__(self, initial_world: GameWorld, config: GameConfig):
        self.world = initial_world
        self.config = config
        self.pending_intents: Dict[str, List[AgentAction]] = {}

    def receive_agent_intents(self, intents: Dict[str, List[AgentAction]]):
        self.pending_intents = intents

    def resolve_turn(self) -> GameWorld:
        self._resolve_diplomacy()
        self._resolve_active_trade_deals()
        self._resolve_construction()
        
        for country_name, country_state in self.world.countries.items():
            self._resolve_production_and_consumption(country_state)
            
        self._resolve_war_and_sabotage()
        
        self.world.current_turn += 1
        self.pending_intents = {}
        return self.world

    def _resolve_diplomacy(self):
        # Placeholder for diplomacy resolution
        pass

    def _resolve_active_trade_deals(self):
        # Placeholder for trade resolution
        pass

    def _resolve_construction(self):
        """Process 'build' intents and progress active construction."""
        # 1. Progress active constructions for all countries
        for country in self.world.countries.values():
            for sq in country.land_squares:
                if sq.under_construction_structure:
                    sq.build_time_remaining -= 1
                    if sq.build_time_remaining <= 0:
                        sq.structure = sq.under_construction_structure
                        sq.under_construction_structure = None
                        sq.build_time_remaining = 0

        # 2. Process new build intents
        for country_name, actions in self.pending_intents.items():
            country = self.world.countries.get(country_name)
            if not country: continue
            
            for action in actions:
                if action.action_type == "build" and action.structure_type and action.square_id:
                    # Find the square
                    target_sq = next((sq for sq in country.land_squares if sq.square_id == action.square_id), None)
                    if target_sq and not target_sq.structure and not target_sq.under_construction_structure:
                        cost = self.config.structure_costs.get(action.structure_type, 0)
                        if country.minerals >= cost:
                            country.minerals -= cost
                            target_sq.under_construction_structure = action.structure_type
                            target_sq.build_time_remaining = self.config.structure_build_times.get(action.structure_type, 1)

    def _resolve_production_and_consumption(self, country: CountryState):
        """Calculate and apply resource changes."""
        # Production
        total_food_produced = 0.0
        total_energy_produced = 0.0
        total_minerals_produced = 0.0
        
        for sq in country.land_squares:
            if not sq.structure:
                continue
                
            rates = self.config.production_rates.get(sq.structure, {})
            
            # Apply land type bonuses (simple logic: +50% if matching)
            multiplier = 1.0 + (0.5 * sq.improvement_level)
            if sq.structure == StructureType.FARM and sq.land_type == LandType.AGRICULTURAL:
                multiplier *= 1.5
            elif sq.structure == StructureType.POWER_PLANT and sq.land_type == LandType.ENERGY:
                multiplier *= 1.5
            elif sq.structure == StructureType.MINE and sq.land_type == LandType.MINERAL:
                multiplier *= 1.5

            total_food_produced += rates.get("food", 0.0) * multiplier
            total_energy_produced += rates.get("energy", 0.0) * multiplier
            total_minerals_produced += rates.get("minerals", 0.0) * multiplier

        # Consumption
        food_consumed = country.population * self.config.food_per_capita
        energy_consumed = country.population * self.config.energy_per_capita
        
        # Apply net changes
        country.food += total_food_produced - food_consumed
        country.energy += total_energy_produced - energy_consumed
        country.minerals += total_minerals_produced
        
        # Starvation mechanics
        if country.food < 0:
            country.population -= int(country.population * self.config.malthusian_decay)
            country.food = 0
            
        # Wealth mechanics W_gen = (P * T * I) * min(1, E_avail / E_req)
        # We simplify Infra to number of structures
        infra = sum(1 for sq in country.land_squares if sq.structure)
        energy_required_for_wealth = infra * 10  # Arbitrary logic
        energy_available = max(0.0, country.energy)
        energy_ratio = min(1.0, energy_available / max(1.0, energy_required_for_wealth))
        
        wealth_generated = (country.population * max(1.0, country.tech_level) * max(1, infra)) * energy_ratio * 0.01
        country.wealth += wealth_generated
        
        # Tech growth mechanics (simple)
        # Assuming each school produces 0.1 tech points, each lab 0.5
        tech_growth = sum(0.1 for sq in country.land_squares if sq.structure == StructureType.SCHOOL)
        tech_growth += sum(0.5 for sq in country.land_squares if sq.structure == StructureType.RESEARCH_LAB)
        country.tech_level += (tech_growth * (1 - country.tech_level / 100)) # Approaching asymptotic limit 100
        
        # Military power update
        base_military_power = sum(10 for sq in country.land_squares if sq.structure == StructureType.MILITARY_BASE)
        base_military_power *= max(1.0, country.tech_level * 0.5)
        country.military_power = base_military_power * energy_ratio

    def _resolve_war_and_sabotage(self):
        # Gather war intents
        wars = []
        for country_name, actions in self.pending_intents.items():
            for action in actions:
                if action.action_type == "war" and action.target_country:
                    wars.append((country_name, action.target_country, action.amount))

        for attacker_name, defender_name, minerals_spent in wars:
            attacker = self.world.countries.get(attacker_name)
            defender = self.world.countries.get(defender_name)
            if not attacker or not defender:
                continue

            attacker.minerals = max(0.0, attacker.minerals - minerals_spent)

            # Calculate Power: Power = (P * T * I) + M_spent
            infra_a = sum(1 for sq in attacker.land_squares if sq.structure)
            infra_d = sum(1 for sq in defender.land_squares if sq.structure)
            
            power_a = (attacker.population * max(1.0, attacker.tech_level) * max(1, infra_a)) + minerals_spent
            power_d = (defender.population * max(1.0, defender.tech_level) * max(1, infra_d)) # Basic defense power

            # Calculate attrition using Lanchester's approach
            attrition_rate = 0.1
            p_loss_a = int(power_d * attrition_rate)
            p_loss_d = int(power_a * attrition_rate)
            
            # Apply population losses
            attacker.population = max(0, attacker.population - p_loss_a)
            defender.population = max(0, defender.population - p_loss_d)
            
            # Destroy some infrastructure based on intensity
            if attacker.population > 0:
                destroy_chance_a = p_loss_a / float(attacker.population + p_loss_a)
                for sq in attacker.land_squares:
                    if sq.structure and destroy_chance_a > 0.1:
                        sq.structure = None
            if defender.population > 0:
                destroy_chance_d = p_loss_d / float(defender.population + p_loss_d)
                for sq in defender.land_squares:
                    if sq.structure and destroy_chance_d > 0.1:
                        sq.structure = None

            # Determine winner and transfer resources
            if power_a > power_d:
                stolen_m = defender.minerals * 0.2
                stolen_e = defender.energy * 0.2
                defender.minerals -= stolen_m
                defender.energy -= stolen_e
                attacker.minerals += stolen_m
                attacker.energy += stolen_e
            elif power_d > power_a:
                stolen_m = attacker.minerals * 0.2
                stolen_e = attacker.energy * 0.2
                attacker.minerals -= stolen_m
                attacker.energy -= stolen_e
                defender.minerals += stolen_m
                defender.energy += stolen_e

    def export_state(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self.world)

def get_default_config() -> GameConfig:
    return GameConfig(
        structure_costs={
            StructureType.FARM: 10,
            StructureType.POWER_PLANT: 15,
            StructureType.MINE: 15,
            StructureType.SCHOOL: 20,
            StructureType.MILITARY_BASE: 50,
            StructureType.PORT: 20,
            StructureType.FACTORY: 40,
            StructureType.RESEARCH_LAB: 100
        },
        structure_build_times={
            StructureType.FARM: 1,
            StructureType.POWER_PLANT: 2,
            StructureType.MINE: 2,
            StructureType.SCHOOL: 3,
            StructureType.MILITARY_BASE: 4,
            StructureType.PORT: 3,
            StructureType.FACTORY: 3,
            StructureType.RESEARCH_LAB: 5
        },
        production_rates={
            StructureType.FARM: {"food": 50},
            StructureType.POWER_PLANT: {"energy": 50},
            StructureType.MINE: {"minerals": 30},
            StructureType.FACTORY: {}, # converts minerals to other goods (advanced logic later)
        }
    )
