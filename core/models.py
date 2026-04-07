import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from .constants import Resource, DiplomaticState, Tech, Civic

@dataclass
class Achievements:
    # Economy
    harvest_gold_count: int = 0
    total_gold_earned: int = 0
    consecutive_wealth_turns: int = 0
    fertile_lands: bool = False
    trade_routes: bool = False
    mercantilism: bool = False
    treasury_reserve: bool = False
    economic_hegemony: bool = False

    # Military
    draft_conscripts_count: int = 0
    military_strikes_won: int = 0
    consecutive_military_turns: int = 0
    draft_during_war_count: int = 0
    levies: bool = False
    standing_army: bool = False
    war_college: bool = False
    total_mobilization: bool = False
    marshal_legacy: bool = False
    master_of_seas: bool = False # New
    blitzkrieg: bool = False # New

    # Industry
    mobilize_industry_count: int = 0
    total_production_earned: int = 0
    workshops: bool = False
    factory_system: bool = False
    assembly_line: bool = False
    industrial_heartland: bool = False
    industrial_revolution: bool = False

    # Science
    fund_academies_count: int = 0
    techs_researched: int = 0
    literacy: bool = False
    university_system: bool = False
    scientific_method: bool = False
    enlightenment: bool = False
    technological_supremacy: bool = False
    university_network: bool = False # New (replaces university_system in the edit, but keeping both for now as per instruction)

    # Civics
    civics_generated: int = 0
    civic_policies_adopted: int = 0
    alliances_formed: int = 0
    common_law: bool = False
    representation: bool = False
    diplomatic_corps: bool = False
    national_identity: bool = False
    golden_age: bool = False
    bureaucracy: bool = False # New
    constitution: bool = False # New
    surplus_economy: bool = False # New


@dataclass
class Nation:
    id: int
    name: str
    color: str
    personality: str = "BALANCED" # WARMONGER, SCIENTIST, MERCHANT, DIPLOMAT, BALANCED
    
    # Core Resources
    gold: int = 500
    manpower: int = 200
    production: int = 100
    science: int = 0
    civics: int = 0
    military: int = 150
    
    # Per-turn Yields
    gold_yield: int = 15
    manpower_yield: int = 50
    production_yield: int = 10
    science_yield: int = 5
    civic_yield: int = 5
    
    # Development
    unlocked_techs: List[str] = field(default_factory=list)
    unlocked_civics: List[str] = field(default_factory=list)
    
    current_tech: Optional[str] = None
    tech_progress: int = 0
    
    current_civic: Optional[str] = None
    civic_progress: int = 0
    
    # Actions
    max_action_points: int = 3
    action_points: int = 3
    queued_actions: List[str] = field(default_factory=list)
    
    # Abstract State & Warfare
    infrastructure_health: int = 100
    is_defeated: bool = False
    war_exhaustion: int = 0
    grievances: Dict[int, int] = field(default_factory=dict) # target_id -> grievance points
    
    active_trade_agreements: List[int] = field(default_factory=list)
    active_research_pacts: List[int] = field(default_factory=list)
    
    pending_trade_agreements: List[int] = field(default_factory=list)
    pending_research_pacts: List[int] = field(default_factory=list)
    pending_joint_wars: List[Dict[str, int]] = field(default_factory=list) # e.g. [{"proposer": 1, "enemy": 2}]
    pending_peace_treaties: List[int] = field(default_factory=list)
    
    absorbed_gold_yield: int = 0
    absorbed_prod_yield: int = 0
    absorbed_sci_yield: int = 0
    
    # Progress
    achievements: Achievements = field(default_factory=Achievements)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "color": self.color,
            "gold": getattr(self, "gold", 0),
            "manpower": getattr(self, "manpower", 0),
            "production": getattr(self, "production", 0),
            "science": getattr(self, "science", 0),
            "civics": getattr(self, "civics", 0),
            "military": getattr(self, "military", 0),
            "gold_yield": self.gold_yield,
            "manpower_yield": self.manpower_yield,
            "production_yield": self.production_yield,
            "science_yield": self.science_yield,
            "civic_yield": self.civic_yield,
            "unlocked_techs": self.unlocked_techs,
            "unlocked_civics": self.unlocked_civics,
            "current_tech": self.current_tech,
            "tech_progress": self.tech_progress,
            "current_civic": self.current_civic,
            "civic_progress": self.civic_progress,
            "max_action_points": self.max_action_points,
            "action_points": self.action_points,
            "queued_actions": self.queued_actions,
            "infrastructure_health": self.infrastructure_health,
            "is_defeated": self.is_defeated,
            "war_exhaustion": self.war_exhaustion,
            "grievances": self.grievances,
            "active_trade_agreements": self.active_trade_agreements,
            "active_research_pacts": self.active_research_pacts,
            "pending_trade_agreements": self.pending_trade_agreements,
            "pending_research_pacts": self.pending_research_pacts,
            "pending_joint_wars": self.pending_joint_wars,
            "pending_peace_treaties": self.pending_peace_treaties,
            "absorbed_gold_yield": self.absorbed_gold_yield,
            "absorbed_prod_yield": self.absorbed_prod_yield,
            "absorbed_sci_yield": self.absorbed_sci_yield,
            "personality": self.personality,
            "achievements": self.achievements.__dict__
        }