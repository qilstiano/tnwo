import os

# Toggle: "llm" or "rulebased"
AI_MODE = "llm"

# Legacy single-backend settings (used as fallback when LLM_BACKENDS is empty)
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1:8b"
LLM_TEMPERATURE = 0.7
MAX_RETRIES = 2

# ---------------------------------------------------------------------------
# Multi-backend configuration for LLM competition
# Each pool is a distinct vLLM / Ollama endpoint serving a specific model.
# Nations sharing a pool_id share the same underlying server + model.
# ---------------------------------------------------------------------------
LLM_BACKENDS = {
    "server_1": {
        "provider": "vllm",
        "base_url": "http://localhost:8001",
        "model": "Qwen/Qwen2.5-7B-Instruct",
    },
    "server_2": {
        "provider": "vllm",
        "base_url": "http://localhost:8002",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
    },
    "server_3": {
        "provider": "vllm",
        "base_url": "http://localhost:8003",
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    },
}

# Map each nation_id -> pool_id in LLM_BACKENDS
NATION_BACKEND_MAP = {
    0: "server_1",
    1: "server_1",
    2: "server_2",
    3: "server_2",
    4: "server_3",
}

# Named strategy directives with priority actions
# Each strategy represents a distinct theory of how to win:
#   expansionist: win by eliminating rivals and annexing their yields
#   scientific:   win by score - techs and civics are worth 500 points each
#   mercantile:   win by compounding wealth — invest gold into permanent yield boosts, trade for ongoing bonuses
#   diplomatic:   win by alliances — form pacts for resource bonuses, aim for peace victory
STRATEGY_LIBRARY = {
    "expansionist": {
        "directive": "You are an EXPANSIONIST nation. Win by eliminating rival nations. Conquered nations give you 50% of their yields permanently. Build military strength, declare wars, and strike to reduce enemy infrastructure to 0.",
        "priority_actions": "DECLARE_WAR, MILITARY_STRIKE, HARVEST MANPOWER, HARVEST PRODUCTION, SKIRMISH, SABOTAGE",
    },
    "scientific": {
        "directive": "You are a SCIENTIFIC nation. Win by score — each unlocked tech is worth 500 points and each civic is worth 500 points. Research technologies, pursue civics, and form research pacts for +15% science/turn per pact.",
        "priority_actions": "RESEARCH, PURSUE_CIVIC, HARVEST SCIENCE, HARVEST CIVICS, PROPOSE_RESEARCH, ACCEPT_RESEARCH",
    },
    "mercantile": {
        "directive": "You are a MERCANTILE nation. Win by compounding wealth. Use INVEST (costs 200 gold) to permanently boost your yields. Form trade agreements for +15% gold/turn per agreement. Grow your economy faster than anyone else.",
        "priority_actions": "INVEST, PROPOSE_TRADE, ACCEPT_TRADE, HARVEST GOLD, HARVEST PRODUCTION, PROPOSE_ALLIANCE",
    },
    "diplomatic": {
        "directive": "You are a DIPLOMATIC nation. Win by achieving peace victory — if all surviving nations are allied, everyone shares the win. Form alliances with every nation, then establish trade and research pacts with allies for mutual benefit.",
        "priority_actions": "PROPOSE_ALLIANCE, ACCEPT_ALLIANCE, PROPOSE_TRADE, ACCEPT_TRADE, PROPOSE_RESEARCH, ACCEPT_RESEARCH",
    },
    "neutral": None,
}

# Assign strategies to nation IDs
# Nation 0 = neutral control (no strategy directive)
# Nations 1-4 = strategic LLM agents
NATION_STRATEGIES = {
    0: "neutral",
    1: "expansionist",
    2: "scientific",
    3: "mercantile",
    4: "diplomatic",
}
