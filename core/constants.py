from enum import Enum

class Resource(Enum):
    GOLD = "Gold"
    MANPOWER = "Manpower"
    PRODUCTION = "Production"
    SCIENCE = "Science"
    CIVICS = "Civics"
    MILITARY = "Military"

class DiplomaticState(Enum):
    NEUTRAL = "Neutral"
    ALLIANCE_PENDING = "Alliance Pending"
    ALLIED = "Allied"
    WAR = "At War"

class NationAction(Enum):
    HARVEST = "HARVEST"
    RESEARCH = "RESEARCH"
    PURSUE_CIVIC = "PURSUE_CIVIC"
    PROPOSE_ALLIANCE = "PROPOSE_ALLIANCE"
    ACCEPT_ALLIANCE = "ACCEPT_ALLIANCE"
    DECLARE_WAR = "DECLARE_WAR"
    MILITARY_STRIKE = "MILITARY_STRIKE"

# Expanded Tech tree
class Tech(Enum):
    BRONZE_WORKING = "Bronze Working"
    IRON_WORKING = "Iron Working"
    GUNPOWDER = "Gunpowder"
    STEEL = "Steel"
    ENGINEERING = "Engineering"
    CANNONS = "Cannons"
    MILITARY_TACTICS = "Military Tactics"
    INDUSTRIALIZATION = "Industrialization"
    RAILROADS = "Railroads"
    STEAM_POWER = "Steam Power"

# New Civics Tree
class Civic(Enum):
    CODE_OF_LAWS = "Code of Laws"
    REPRESENTATION = "Representation"
    DIPLOMATIC_CORPS = "Diplomatic Corps"
    NATIONAL_IDENTITY = "National Identity"
    GOLDEN_AGE = "Golden Age"

TECH_COSTS = {
    "Bronze Working": 50,
    "Iron Working": 100,
    "Gunpowder": 150,
    "Steel": 200,
    "Engineering": 200,
    "Cannons": 250,
    "Military Tactics": 250,
    "Industrialization": 300,
    "Railroads": 400,
    "Steam Power": 400
}

CIVIC_COSTS = {
    "Code of Laws": 100,
    "Representation": 150,
    "Diplomatic Corps": 200,
    "National Identity": 250,
    "Golden Age": 300
}