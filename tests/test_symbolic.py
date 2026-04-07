import json
import sys
import os

# Add parent path so we can import packages
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.game_state import GameState
from ai.symbolic import parse_agent_response

def test_symbolic_state():
    state = GameState(num_players=2)
    sym = state.get_symbolic_state(0)
    print("--- SYMBOLIC STATE (PLAYER 0) ---")
    print(json.dumps(sym, indent=2))
    assert sym["my_nation"]["id"] == 0
    assert len(sym["other_nations"]) == 1

def test_parse_valid_actions():
    llm_json = """
    {
        "reasoning": "I need more gold and should propose trade with player 1.",
        "actions": [
            {"action": "HARVEST", "target": "GOLD"},
            {"action": "PROPOSE_TRADE", "target": 1}
        ]
    }
    """
    actions, err = parse_agent_response(llm_json)
    assert err is None
    assert actions == ["HARVEST GOLD", "PROPOSE_TRADE 1"]
    print("--- PARSE VALID ACTIONS PASSED ---")

def test_parse_invalid_actions():
    llm_json_bad = """
    {
        "reasoning": "Harvesting water is good.",
        "actions": [
            {"action": "HARVEST", "target": "WATER"}
        ]
    }
    """
    actions, err = parse_agent_response(llm_json_bad)
    assert err is not None
    assert "Invalid harvest target" in err
    print("--- PARSE INVALID ACTIONS ERR ---")
    print(err)

if __name__ == "__main__":
    test_symbolic_state()
    test_parse_valid_actions()
    test_parse_invalid_actions()
    print("ALL TESTS PASSED")
