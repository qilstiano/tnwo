# core/constants.py - Only enums
from enum import Enum

class Terrain(Enum):
    PLAINS = "P"
    GRASS = "G"
    HILL = "H"
    MOUNTAIN = "M"
    WATER = "W"
    
    def __str__(self):
        return self.value

class Feature(Enum):
    FOREST = "F"
    RESOURCE = "R"
    NONE = "_"
    
    def __str__(self):
        return self.value

class District(Enum):
    CITY_CENTER = "C"
    INDUSTRIAL = "I"
    SCIENCE = "S"
    NONE = "_"
    
    def __str__(self):
        return self.value

class UnitType(Enum):
    SETTLER = "S"
    WARRIOR = "W"
    NONE = "_"
    
    def __str__(self):
        return self.value

class Tech(Enum):
    POTTERY = "Pottery"
    MINING = "Mining"
    BRONZE_WORKING = "Bronze Working"
    WRITING = "Writing"
    APPRENTICESHIP = "Apprenticeship"

class Civic(Enum):
    CRAFTSMANSHIP = "Craftsmanship"
    STATE_WORKFORCE = "State Workforce"
    POLITICAL_PHILOSOPHY = "Political Philosophy"