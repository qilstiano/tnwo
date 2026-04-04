import json
import http.server
import socketserver
from typing import Any
from urllib.parse import urlparse

# Import the game
from main import CivilizationGame
from core.constants import Resource, DiplomaticState, NationAction, Tech, Civic

class GameEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (Resource, DiplomaticState, NationAction, Tech, Civic)):
            return obj.value
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)

def make_game(num_players=5):
    try:
        return CivilizationGame(num_players=num_players)
    except Exception:
        return CivilizationGame(num_players=5)

game = make_game()
ai_players = [0, 1, 2, 3, 4]  # Default to EvE (all 5 nations are AI)

class GameRequestHandler(http.server.SimpleHTTPRequestHandler):

    def _cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == '/api/state':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self._cors()
            self.end_headers()

            # Translate diplomacy to a JSON friendly format
            diplomacy_dict = {}
            for nid, relations in game.state.diplomacy.items():
                diplomacy_dict[nid] = {str(oid): state.value for oid, state in relations.items()}

            nations_dict = {nid: nation.to_dict() for nid, nation in game.state.nations.items()}

            winner_id = game.check_winner()
            winner_name = None
            victory_type = None
            if winner_id is not None:
                if isinstance(winner_id, list):
                    # Peace victory: multiple survivors
                    winner_name = ", ".join(game.state.nations[wid].name for wid in winner_id)
                    victory_type = "PEACE"
                else:
                    winner_name = game.state.nations[winner_id].name
                    active_count = sum(1 for n in game.state.nations.values() if not n.is_defeated)
                    victory_type = "DOMINATION" if active_count <= 1 else "SCORE"

            from ai.config import NATION_STRATEGIES

            backends_info = {}
            for nid in game.state.nations:
                agent = game.agents.get(nid)
                if hasattr(agent, 'backend_id') and agent.backend_id:
                    backends_info[nid] = {
                        "backend_id": agent.backend_id,
                        "model": agent.model_name,
                    }
                else:
                    backends_info[nid] = {
                        "backend_id": "rulebased",
                        "model": None,
                    }

            state_dict = {
                "turn": game.state.turn,
                "current_player": game.state.current_player,
                "winner": winner_id,
                "winner_name": winner_name,
                "victory_type": victory_type,
                "ai_players": ai_players,
                "nations": nations_dict,
                "diplomacy": diplomacy_dict,
                "strategies": NATION_STRATEGIES,
                "backends": backends_info,
            }

            response = json.dumps(state_dict, cls=GameEncoder)
            self.wfile.write(response.encode('utf-8'))
            return

        elif parsed.path == '/api/export':
            self.send_response(200)
            self.send_header('Content-type', 'application/jsonl')
            self.send_header('Content-Disposition', 'attachment; filename="game_export.jsonl"')
            self._cors()
            self.end_headers()
            try:
                with open("game_export.jsonl", "rb") as f:
                    self.wfile.write(f.read())
            except FileNotFoundError:
                self.wfile.write(b"") # return empty file safely instead of crashing
            return

        elif parsed.path == '/':
            self.path = '/index.html'

        return super().do_GET()

    def do_POST(self):
        global game
        parsed = urlparse(self.path)
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length) if content_length else b'{}'

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self._cors()
        self.end_headers()

        if parsed.path == '/api/reset':
            global game, ai_players
            try:
                data = json.loads(post_data.decode('utf-8'))
            except:
                data = {}
            mode = data.get("mode", "PvE")
            num_players = int(data.get("num_players", 5))

            game = make_game(num_players)

            # Clear export log on reset
            open("game_export.jsonl", "w").close()

            if mode == "PvP":
                ai_players = []
            elif mode == "EvE":
                ai_players = list(range(num_players))
            else:
                # PvE: Player 0 is human, rest are AI
                ai_players = list(range(1, num_players))

            self.wfile.write(json.dumps({"success": True, "message": "Game reset!"}).encode())
            return

        if parsed.path == '/api/command':
            try:
                data = json.loads(post_data.decode('utf-8'))
                command = data.get('command', '').strip()
                upper_cmd = command.upper()

                logs = []
                success = False
                msg = "Command received"

                if upper_cmd == "SUBMIT_TURN":
                    # Block further processing once game is over
                    if game.check_winner() is not None:
                        logs = ["⏹️ The game has concluded. No further turns may be processed."]
                        success = True
                        self.wfile.write(json.dumps({"success": success, "message": "Game Over", "logs": logs, "turn": game.state.turn}, cls=GameEncoder).encode())
                        return

                    intent_logs = []

                    # Let AIs queue their actions (skip defeated)
                    for ai in ai_players:
                        if ai in game.state.nations and not game.state.nations[ai].is_defeated:
                            agent = game.agents[ai]
                            cmds = agent.decide_actions(game.state, game.handler)
                            for c in cmds:
                                game.handler.queue_action(ai, c)

                            # Capture reasoning logs from LLM agents
                            if hasattr(agent, 'last_reasoning') and agent.last_reasoning:
                                nation_name = game.state.nations[ai].name
                                intent_logs.append(f"[INTENT] {nation_name}: {agent.last_reasoning}")

                    # Record state + actions + backend metadata before resolving
                    state_snapshot = {
                        "turn": game.state.turn,
                        "agents": {}
                    }
                    for nid, n in game.state.nations.items():
                        agent = game.agents.get(nid)
                        agent_meta = {
                            "state": game.state.get_symbolic_state(nid),
                            "queued_actions": list(n.queued_actions),
                        }
                        if hasattr(agent, 'backend_id') and agent.backend_id:
                            agent_meta["backend_id"] = agent.backend_id
                            agent_meta["model"] = agent.model_name
                        state_snapshot["agents"][nid] = agent_meta

                    with open("game_export.jsonl", "a") as f:
                        f.write(json.dumps(state_snapshot, cls=GameEncoder) + "\n")

                    # Resolve turn
                    resolved_logs = game.handler.resolve_simultaneous_turn()
                    logs = intent_logs + resolved_logs

                    success = True
                    msg = "Turn Resolved"

                elif upper_cmd.startswith("QUEUE"):
                    actual_cmd = command[6:].strip()  # strip "QUEUE "
                    success = game.handler.queue_action(game.state.current_player, actual_cmd)
                    msg = "Action Queued" if success else "Invalid Action or Not Enough AP"

                elif upper_cmd == "CANCEL_LAST":
                    success = game.handler.cancel_last_action(game.state.current_player)
                    msg = "Action Cancelled" if success else "Queue Empty"

                else:
                    msg = "Invalid endpoint command. Use QUEUE <cmd>, CANCEL_LAST, or SUBMIT_TURN."
                    success = False

                response = {"success": success, "message": msg, "logs": logs, "turn": game.state.turn}
            except Exception as e:
                import traceback
                response = {"success": False, "message": str(e) + "\n" + traceback.format_exc()}

            self.wfile.write(json.dumps(response, cls=GameEncoder).encode())
            return


        self.wfile.write(json.dumps({"success": False, "message": "Unknown endpoint"}).encode())

    def log_message(self, format, *args):  # silence access logs
        pass

PORT = 8080
socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("", PORT), GameRequestHandler) as httpd:
    print(f"Serving on http://localhost:{PORT}")
    httpd.serve_forever()
