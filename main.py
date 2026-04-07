import sys
from typing import Optional
from core.game_state import GameState
from engine.actions import ActionHandler
from ai.agent import AIAgent
from ai.config import (
    AI_MODE, OLLAMA_URL, OLLAMA_MODEL,
    LLM_BACKENDS, NATION_BACKEND_MAP,
)
from ai.llm_client import OllamaClient, create_llm_client
from ai.llm_agent import LLMAgent

class CivilizationGame:
    def __init__(self, num_players: int = 4):
        self.state = GameState(num_players)
        self.handler = ActionHandler(self.state)
        self.agents = {}

        if AI_MODE == "llm":
            if LLM_BACKENDS:
                self._init_multi_backend(num_players)
            else:
                self._init_single_backend(num_players)
        else:
            self.agents = {i: AIAgent(i) for i in range(num_players)}

    def _init_single_backend(self, num_players: int):
        """Legacy path: one Ollama server for all nations."""
        client = OllamaClient(base_url=OLLAMA_URL, model=OLLAMA_MODEL)
        if client.is_available():
            print(f"[INIT] Ollama available ({OLLAMA_MODEL}) — using LLM agents")
            for i in range(num_players):
                fallback = AIAgent(i)
                self.agents[i] = LLMAgent(i, client, fallback,
                                          backend_id="ollama",
                                          model_name=OLLAMA_MODEL)
        else:
            print("[INIT] Ollama unavailable — falling back to rule-based agents")
            self.agents = {i: AIAgent(i) for i in range(num_players)}

    def _init_multi_backend(self, num_players: int):
        """Per-nation routing: each nation maps to a backend pool via NATION_BACKEND_MAP."""
        pool_clients = {}
        pool_available = {}

        for pool_id, cfg in LLM_BACKENDS.items():
            client = create_llm_client(cfg["provider"], cfg["base_url"], cfg["model"])
            pool_clients[pool_id] = client
            avail = client.is_available()
            pool_available[pool_id] = avail
            status = "OK" if avail else "UNAVAILABLE"
            print(f"[INIT] Backend '{pool_id}' ({cfg['model']}) @ {cfg['base_url']} — {status}")

        for i in range(num_players):
            fallback = AIAgent(i)
            pool_id = NATION_BACKEND_MAP.get(i)

            if pool_id and pool_id in pool_clients and pool_available.get(pool_id):
                client = pool_clients[pool_id]
                cfg = LLM_BACKENDS[pool_id]
                self.agents[i] = LLMAgent(i, client, fallback,
                                          backend_id=pool_id,
                                          model_name=cfg["model"])
                print(f"[INIT] Nation {i} -> {pool_id} ({cfg['model']})")
            else:
                self.agents[i] = fallback
                reason = "no mapping" if not pool_id else "backend unavailable"
                print(f"[INIT] Nation {i} -> rule-based fallback ({reason})")

    def process_command(self, command: str) -> bool:
        """Process an abstract command"""
        if not command: return True

        parts = command.strip().split()
        cmd = parts[0].upper()

        try:
            if cmd == "HARVEST" and len(parts) == 2:
                return self.handler.harvest(parts[1])
            elif cmd == "RESEARCH" and len(parts) >= 2:
                return self.handler.research(" ".join(parts[1:]))
            elif cmd == "PURSUE_CIVIC" and len(parts) >= 2:
                return self.handler.pursue_civic(" ".join(parts[1:]))
            elif cmd == "PROPOSE_ALLIANCE" and len(parts) == 2:
                return self.handler.propose_alliance(int(parts[1]))
            elif cmd == "ACCEPT_ALLIANCE" and len(parts) == 2:
                return self.handler.accept_alliance(int(parts[1]))
            elif cmd == "DECLARE_WAR" and len(parts) == 2:
                return self.handler.declare_war(int(parts[1]))
            elif cmd == "MILITARY_STRIKE" and len(parts) == 2:
                return self.handler.military_strike(int(parts[1]))
            elif cmd == "NEXT_TURN":
                return self.handler.end_turn()
            else:
                return False
        except Exception as e:
            return False

    def check_winner(self) -> Optional[int]:
        return self.state.check_winner()

if __name__ == "__main__":
    game = CivilizationGame(num_players=4)
    print("Strategy Simulator Core Engine Initialized.")
