# 🌍 The New World Order (TNWO) - Onboarding Guide

Welcome to the **TNWO Project**! 

TNWO is a high-stakes, **Simultaneous-Turn Grand Strategy Nation Simulator**. Beneath the hood, it pairs a headless Python simulation server with an interactive React/Vite dashboard, allowing autonomous AI Agent nations to dynamically trade, research, and wage war with distinct "Personalities" in a global arena!

## 🚀 Quickstart: Running the App (Docker)

To make it completely painless to spin up the local environment, the entire stack has been Dockerized.

**Prerequisites:**
- [Docker](https://www.docker.com/) & Docker Compose

**To Launch:**
1. Open your terminal at the root directory of this repository (`/tnwo`).
2. Run the compose environment:
   ```bash
   docker-compose up --build
   ```
3. The React Frontend will be available at [http://localhost:5173](http://localhost:5173).
4. The Python API will implicitly run on `localhost:8080`.
   - *Note: Hot-reloading is fully supported in Docker! Any changes you save to the React files or Python files will trigger an automatic restart within the container.*

---

## 🏛️ Architecture & File Structure

This repository is split sharply between the **Backend Simulation Engine** and the **Frontend React Dashboard**.

### **Backend (`/` Root)**
The backend runs a headless `http.server` via Python standard libraries. Action resolution is strictly **simultaneous** (all pending actions execute together at the end of the turn to avoid first-mover advantages).

- `server.py`: Spawns the HTTP endpoint routing UI requests (`/api/state`, `/api/command`) to the engine loop in `main.py`.
- `main.py`: Coordinates the grand `CivilizationGame` tick loop, waking up the AI logic every turn.
- `ai/agent.py`: Brains of the AI. Features the `AIAgent`, equipped with dynamic weighted algorithms that choose strategies directly based off of their active `AgentPersonality` (e.g., `WARMONGER`, `SCIENTIST`, `DIPLOMAT`).
- `engine/actions.py`: The heart of game mechanics! Processes combat resolution (Looting & Annexation), Diplomatic proposal queuing/resolving, and Economic resource harvesting.
- `core/game_state.py`: Manages the overarching state dictionary (the `GameState` singleton block), processing passive turn bonuses from `active_trade_agreements` or checking unlocks for the massive `Achievements` skill tracker matrix.
- `core/models.py`: All standard python Dataclasses representing core units (Nations, Achievements).
- `core/constants.py`: Defining global variables like Tech costs, Civic costs, Enums for standard tracking, etc.

### **Frontend (`/frontend`)**
A fast React + Vite single-page application focused deeply on visualization.

- `src/App.jsx`: The monolithic controller for the browser interface. It fetches data state from the Python server and renders:
  - The comprehensive **Global Strategy Network** SVG Map (which displays node links dynamically representing Wars, Alliances, Trade, and Research connections).
  - The "Command Center" dashboard used for queueing up user controls (or sitting back to watch EvE mode).
  - Tracking displays for complex internal yields (e.g. War Exhaustion penalties resulting from combat attrition).
- `src/index.css`: Houses a premium styling suite prioritizing neon badges, glass-morphism containers, and interactive hover mechanics.

---

## 🛠️ How to Contribute!

### Adding a New Diplomatic Feature
1. **Define it:** Open `core/models.py` and register the variable inside the `Nation` dataclass (e.g., `active_defense_pacts: List[int]`).
2. **Handle the UI:** Go to `App.jsx` and render a new visual indicator or SVG line connecting nations based on this state.
3. **Handle the Engine:** Open `engine/actions.py` to add listeners for new queued action strings (e.g., `"PROPOSE_DEFENSE_PACT"`).
4. **Teach the AI:** Head over to `ai/agent.py` and write the specific behavioral block that instructs the AI under which circumstances they should execute your new command!

Jump in, build the world, and have fun!
