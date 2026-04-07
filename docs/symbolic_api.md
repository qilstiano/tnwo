# Symbolic Game State API

This document describes the structure of the symbolic communication payloads between Foundation Models (LLMs) and the game engine.

## 1. Receiving Game State
Every turn, the AI receives a JSON payload representing the world from its perspective. The engine handles "Fog of War" implicitly; data exposed in `other_nations` contains only publicly observable metrics.

### Structure of `get_symbolic_state(player_id)`
```json
{
  "global_state": {
    "turn": 15,
    "active_nations": [0, 1, 2]
  },
  "my_nation": {
    "id": 1,
    "name": "Ironforge",
    "personality": "BALANCED",
    "stats": {"gold": 1200, "manpower": 500, "production": 350, "science": 100, "civics": 50, "military": 200},
    "yields": {"gold_yield": 25, "manpower_yield": 50, "production_yield": 15, "science_yield": 5, "civic_yield": 0},
    "tech": {"unlocked": ["Bronze Working"], "current": "Iron Working", "progress": 50},
    "actions": {"max_points": 3, "current_points": 3},
    "status": {"infrastructure_health": 100, "war_exhaustion": 0},
    "diplomacy": {
        "active_trade_agreements": [0],
        "active_research_pacts": [],
        "pending_trade_agreements": [],
        "pending_research_pacts": [],
        "grievances": {}
    }
  },
  "other_nations": [
    {
      "id": 0,
      "name": "Valnor",
      "diplomatic_status": "ALLIED",
      "visible_status": {
        "infrastructure_health": 100,
        "estimated_tech_tier": 2
      }
    }
  ]
}
```

## 2. Emitting Turn Actions
The AI generates a strict JSON payload matching the `AgentTurnResponse` Pydantic class.

### Expected AI JSON Response
```json
{
    "reasoning": "I need to boost my production before launching an attack on Player 0.",
    "actions": [
        {"action": "HARVEST", "target": "PRODUCTION"},
        {"action": "RESEARCH", "target": "Engineering"},
        {"action": "DECLARE_WAR", "target": 0}
    ]
}
```

### Action Vocabulary Validations
- `HARVEST`: Target must be a resource string (`"GOLD"`, `"MANPOWER"`, `"PRODUCTION"`, `"SCIENCE"`, `"CIVICS"`).
- `DECLARE_WAR`, `PROPOSE_ALLIANCE`, `ACCEPT_ALLIANCE`, `CANCEL_ALLIANCE`, `PROPOSE_TRADE`, `ACCEPT_TRADE`, `PROPOSE_RESEARCH`, `ACCEPT_RESEARCH`, `MILITARY_STRIKE`, `SABOTAGE`, `SKIRMISH`: Target must be an integer, the `id` of another nation.
- `RESEARCH`, `PURSUE_CIVIC`: Target must be a string containing the name of the tech/civic.
- **`PROPOSE_JOINT_WAR`, `ACCEPT_JOINT_WAR`**: Target must be an integer representing the ally you are communicating with. AND you must supply the `"enemy"` field in the JSON with an integer representing the mutual target. Example: `{"action": "PROPOSE_JOINT_WAR", "target": 1, "enemy": 2}`.

> **Note on Limits:** Do not propose more actions in your `actions` array than your `actions.max_points`. Excess actions will fail validation or be ignored by the engine queue.
