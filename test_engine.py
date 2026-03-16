from engine import (
    SimulationEngine, GameWorld, CountryState, LandSquare, 
    LandType, StructureType, AgentAction, get_default_config
)

def test_engine():
    # Setup initial world
    country_a = CountryState(
        name="Alpha",
        land_squares=[
            LandSquare(square_id="A1", land_type=LandType.AGRICULTURAL),
            LandSquare(square_id="A2", land_type=LandType.MINERAL),
            LandSquare(square_id="A3", land_type=LandType.URBAN)
        ],
        food=100,
        energy=100,
        minerals=50,
        wealth=100,
        population=10,
        tech_level=1.0,
        military_power=10.0,
        reputation=1.0
    )

    country_b = CountryState(
        name="Beta",
        land_squares=[
            LandSquare(square_id="B1", land_type=LandType.ENERGY),
            LandSquare(square_id="B2", land_type=LandType.URBAN)
        ],
        food=50,
        energy=200,
        minerals=20,
        wealth=50,
        population=5,
        tech_level=1.0,
        military_power=5.0,
        reputation=1.0
    )

    world = GameWorld(current_turn=1, countries={"Alpha": country_a, "Beta": country_b})
    config = get_default_config()
    
    engine = SimulationEngine(initial_world=world, config=config)

    # 1. Provide intents
    intents = {
        "Alpha": [
            AgentAction(action_type="build", square_id="A1", structure_type=StructureType.FARM),
            AgentAction(action_type="build", square_id="A2", structure_type=StructureType.MINE)
        ],
        "Beta": [
            AgentAction(action_type="build", square_id="B1", structure_type=StructureType.POWER_PLANT)
        ]
    }
    
    print("--- Turn 1: Building Structures ---")
    engine.receive_agent_intents(intents)
    engine.resolve_turn()
    
    state = engine.export_state()
    for name, c_state in state["countries"].items():
        print(f"{name} Minerals: {c_state['minerals']}")
        
    print("\n--- Turn 2: Structures Finish Building ---")
    engine.resolve_turn() 
    
    for _ in range(3):
        engine.resolve_turn()
        
    print("\n--- Turn 5: After some production ---")
    state = engine.export_state()
    for name, c_state in state["countries"].items():
        print(f"{name} Food: {c_state['food']:.1f}, Energy: {c_state['energy']:.1f}, Minerals: {c_state['minerals']:.1f}")

    print("\n--- Turn 6: War ---")
    intents = {
        "Alpha": [
            AgentAction(action_type="war", target_country="Beta", amount=10.0) 
        ]
    }
    engine.receive_agent_intents(intents)
    engine.resolve_turn()
    
    state = engine.export_state()
    print("Alpha population:", state["countries"]["Alpha"]["population"])
    print("Beta population:", state["countries"]["Beta"]["population"])
    print("Beta minerals:", state["countries"]["Beta"]["minerals"])

if __name__ == "__main__":
    test_engine()
    print("Simulation completed successfully.")
