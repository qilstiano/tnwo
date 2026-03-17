from engine.models import StructureType, GameConfig

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
