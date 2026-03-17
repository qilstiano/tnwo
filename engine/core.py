from typing import Dict, List, Any
from engine.models import GameWorld, GameConfig, AgentAction

from engine.resolvers.construction import ConstructionResolverMixin
from engine.resolvers.production import ProductionResolverMixin
from engine.resolvers.war import WarResolverMixin
from engine.resolvers.diplomacy import DiplomacyResolverMixin
from engine.resolvers.trade import TradeResolverMixin

class SimulationEngine(
    DiplomacyResolverMixin,
    TradeResolverMixin,
    ConstructionResolverMixin,
    ProductionResolverMixin,
    WarResolverMixin
):
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

    def export_state(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self.world)
