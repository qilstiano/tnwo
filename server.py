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

def make_game(num_players=4):
    try:
        return CivilizationGame(num_players=num_players)
    except Exception:
        return CivilizationGame(num_players=4)

game = make_game()
ai_players = [1, 2, 3]  # Default to PvE (Player 0 vs 3 CPUs)

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

            state_dict = {
                "turn": game.state.turn,
                "current_player": game.state.current_player,
                "winner": game.check_winner(),
                "ai_players": ai_players,
                "nations": nations_dict,
                "diplomacy": diplomacy_dict
            }

            response = json.dumps(state_dict, cls=GameEncoder)
            self.wfile.write(response.encode('utf-8'))
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

            game = make_game()

            if mode == "PvP":
                ai_players = []
            elif mode == "EvE":
                ai_players = [0, 1, 2, 3]
            else:
                ai_players = [1, 2, 3]

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
                    # Let AIs queue their actions
                    for ai in ai_players:
                        if ai in game.state.nations and not game.state.nations[ai].is_defeated:
                            agent = game.agents[ai]
                            cmds = agent.decide_actions(game.state, game.handler)
                            for c in cmds:
                                game.handler.queue_action(ai, c)
                    
                    # Resolve turn
                    logs = game.handler.resolve_simultaneous_turn()
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
with socketserver.TCPServer(("", PORT), GameRequestHandler) as httpd:
    httpd.allow_reuse_address = True
    print(f"Serving on http://localhost:{PORT}")
    httpd.serve_forever()
