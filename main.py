import sys
from typing import Optional
from core.game_state import GameState
from engine.actions import ActionHandler
from ai.agent import AIAgent

class CivilizationGame:
    def __init__(self, num_players: int = 4):
        self.state = GameState(num_players)
        self.handler = ActionHandler(self.state)
        self.agents = {
            i: AIAgent(i) for i in range(num_players)
        }
    
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