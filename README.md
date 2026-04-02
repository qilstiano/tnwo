# TNWO: LLM Nation Competition Sandbox

`TNWO` is a simultaneous-turn grand strategy simulator where each nation can be controlled by either:

- a rule-based agent
- an LLM served through an OpenAI-compatible `vLLM` endpoint

This repository is currently set up for two main workflows:

- run the lightweight local HTTP game server in `server.py`
- run headless LLM-vs-LLM experiments with `run_experiment.py` or `run_5in1.sh`

## What The Project Does

Each turn, every AI nation observes a symbolic view of the world, sends one LLM request, receives a JSON action plan, queues its actions, and then all actions resolve simultaneously.

The current game includes:

- economy and harvesting
- research and civics
- diplomacy, alliances, trade, and research pacts
- warfare, sabotage, skirmishes, and annexation
- persistent per-nation event memory for LLM agents

## Repository Layout

- `server.py`: HTTP server for the browser-based game UI and API
- `main.py`: constructs the game state, action handler, and nation agents
- `run_experiment.py`: headless experiment runner for multi-model or multi-backend evaluation
- `run_5in1.sh`: one-click launcher for the "one model controls five nations" setup
- `ai/llm_agent.py`: prompt building, LLM calling, response validation, memory
- `ai/llm_client.py`: Ollama and vLLM client adapters
- `ai/config.py`: default model/backend/strategy settings
- `engine/actions.py`: simultaneous turn resolution and game mechanics
- `core/game_state.py`: core world state and scoring
- `outputs/`: exported game logs, summaries, and vLLM server logs

## Python Requirements

Base Python dependency:

- `pydantic`

Optional but required for the integrated GPU experiment launcher:

- `vllm`

Install the base dependencies with:

```bash
pip install -r requirements.txt
```

For LLM experiments, install `vllm` in a suitable GPU-enabled environment:

```bash
pip install vllm
```

## Recommended Environment

- Linux
- Python `3.12` recommended
- NVIDIA GPU for `vllm`
- `nvidia-smi` available if you want to run GPU-backed experiments

A typical setup looks like:

```bash
conda create -n tnwo-vllm python=3.12 -y
conda activate tnwo-vllm
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
pip install vllm
```

## Running The Project

### 1. Run the HTTP game server

This launches the local game server on port `8080`:

```bash
python server.py
```

Then open `http://localhost:8080`.

### 2. Run a headless experiment from Python

Generate a sample config:

```bash
python run_experiment.py --gen-config > experiment.json
```

Run it:

```bash
python run_experiment.py -c experiment.json
```

Useful flags:

```bash
python run_experiment.py -c experiment.json --seed 42 --max-turns 50 -v
```

### 3. Run the integrated single-model, five-nation experiment

The easiest path for the current repo is:

```bash
./run_5in1.sh
```

This script will:

- create an experiment config
- check whether a vLLM server is already running
- start `vllm` automatically if needed
- wait until the endpoint is ready
- run a 5-nation experiment where all nations share the same model
- write outputs into `outputs/`

## Configuring `run_5in1.sh`

Edit these variables near the top of the script:

- `MAX_TURNS`
- `SEED`
- `TEMPERATURE`
- `CUDA_VISIBLE_DEVICES`
- `VLLM_PORT`
- `TENSOR_PARALLEL_SIZE`
- `GPU_MEMORY_UTILIZATION`
- `MAX_MODEL_LEN`
- `DTYPE`
- `SERVED_MODEL_NAME`
- `MODEL_PATH`

By default, all 5 nations point to the same shared backend.

You can also change the strategy prompt used by each nation:

- `neutral`
- `expansionist`
- `scientific`
- `mercantile`
- `diplomatic`

## Outputs

Experiment artifacts are written to `outputs/`:

- `game_export_<timestamp>.jsonl`: per-turn snapshots before resolution
- `game_export_<timestamp>_summary.json`: final score summary
- `vllm_<timestamp>.log`: vLLM startup and request log

## Important Notes

- The first `25` turns block foreign actions such as war and diplomacy. If you want actual geopolitical competition, use `MAX_TURNS > 25`.
- The LLM does not produce a hidden chain-of-thought log. The system stores validated actions and a short self-reported reasoning string.
- `run_5in1.sh` assumes the `vllm` command is installed and available in your shell.

## Quick Sanity Checks

Check that the Python environment works:

```bash
python tests/test_symbolic.py
```

Check that `vllm` is installed:

```bash
vllm --help
```

Check GPU visibility:

```bash
nvidia-smi
```
