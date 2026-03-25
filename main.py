import sys
from typing import Optional, Tuple
from core.game_state import GameState
from engine.actions import ActionHandler
from ai.agent import AIAgent

class CivilizationGame:
    def __init__(self, width: int = 10, height: int = 10, num_players: int = 2):
        self.state = GameState(width, height, num_players)
        self.handler = ActionHandler(self.state)
        self.agents = {
            i: AIAgent(i) for i in range(num_players)
        }
        self.running = True
    
    def process_command(self, command: str) -> bool:
        """Process a text command"""
        parts = command.strip().split()
        if not parts:
            return True
        
        cmd = parts[0].upper()
        
        try:
            if cmd == "MOVE" and len(parts) == 4:
                return self.handler.move_unit(parts[1], int(parts[2]), int(parts[3]))
            
            elif cmd == "ATTACK" and len(parts) == 3:
                return self.handler.attack_unit(parts[1], parts[2])
            
            elif cmd == "FOUND_CITY" and len(parts) == 4:
                return self.handler.found_city(parts[1], int(parts[2]), int(parts[3]))
            
            elif cmd == "BUILD" and len(parts) == 3:
                return self.handler.build_item(parts[1], parts[2])
            
            elif cmd == "RESEARCH" and len(parts) == 2:
                return self.handler.research_tech(parts[1])
            
            elif cmd == "BUY" and len(parts) == 4:
                return self.handler.buy_tile(parts[1], int(parts[2]), int(parts[3]))
            
            elif cmd == "NEXT_TURN":
                return self.handler.end_turn()
            
            elif cmd == "QUIT":
                self.running = False
                return True
            
            elif cmd == "HELP":
                self._show_help()
                return True
            
        except Exception as e:
            print(f"Error executing command: {e}")
            return False
        
        return False
    
    def _show_help(self):
        """Show available commands"""
        print("\nAvailable Commands:")
        print("  MOVE [unit_id] [x] [y]")
        print("  ATTACK [attacker_id] [defender_id]")
        print("  FOUND_CITY [settler_id] [x] [y]")
        print("  BUILD [city_id] [item] (items: Settler, Warrior, Granary, Campus, IndustrialZone)")
        print("  RESEARCH [tech_name]")
        print("  BUY [city_id] [x] [y]")
        print("  NEXT_TURN")
        print("  QUIT")
        print("  HELP\n")
    
    def run(self, interactive: bool = True):
        """Main game loop"""
        print("=== Minimal Civilization VI ===")
        print("Type HELP for commands\n")
        
        while self.running:
            # Render current state
            print(self.state.render())
            
            # Check win condition
            winner = self._check_winner()
            if winner is not None:
                print(f"\n=== PLAYER {winner} WINS! ===")
                break
            
            # Get action
            if interactive and self.state.current_player == 0:
                # Human player (player 0)
                command = input("\n> ").strip()
                if not self.process_command(command):
                    print("Invalid command!")
            else:
                # AI player
                agent = self.agents[self.state.current_player]
                command = agent.decide_action(self.state, self.handler)
                print(f"\nAI Player {self.state.current_player} chooses: {command}")
                self.process_command(command)
    
    def _check_winner(self) -> Optional[int]:
        """Check if there's a winner"""
        # Check if any player has lost all cities
        active_players = set()
        for city in self.state.cities.values():
            active_players.add(city.player_id)
        
        # Check domination win
        if len(active_players) == 1:
            return list(active_players)[0]
        
        # Check score victory at turn 100
        if self.state.turn >= 100:
            scores = {}
            for player_id, player in self.state.players.items():
                score = sum(self.state.cities[city_id].population for city_id in player.cities) * 2
                score += len([t for row in self.state.map.tiles for t in row if t.district != District.NONE])
                scores[player_id] = score
            
            return max(scores, key=scores.get)
        
        return None

if __name__ == "__main__":
    game = CivilizationGame(width=10, height=10, num_players=2)
    game.run(interactive=True)