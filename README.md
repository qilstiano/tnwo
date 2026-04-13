# TNWO: A Grand-Strategy Sandbox for LLM and RL Agents

`TNWO` is a simultaneous-turn grand-strategy simulator for five nations whose
actions — economy, research, diplomacy, trade, war, sabotage, and annexation —
are all resolved at the end of each turn. Every nation can be controlled by a
rule-based agent, an LLM served through a vLLM endpoint, or a trained PPO
policy.

The repository supports three main workflows:

1. **Browser-based game server** for manual play and visualization
   (`server.py` + `index.html`).
2. **Headless LLM-vs-LLM experiments** using a shared vLLM backend
   (`run_experiment.py`, `run_5in1.sh`).
3. **A PPO reinforcement-learning study** — training a nation-level policy
   against rule-based and LLM opponents, evaluating it across four opponent
   suites, and producing publication-quality figures
   (`train_ppo.py`, `evaluate_ppo.py`, `experiments/`).

The core research artifact in this repo is the `exp_main` PPO experiment
matrix: **18 runs** across 5 reward variants × 4 opponent compositions, with
unified evaluation and strategy-fingerprint analysis. The paper draft lives in
`report/experiments.tex`.

## Repository Layout

### Game engine and agents
- `core/game_state.py`  — world state, scoring, per-nation data
- `engine/actions.py`   — simultaneous action resolution, diplomacy, war
- `ai/agent.py`         — rule-based strategy agents (balanced, aggressor,
  turtle, scientist, trader, diplomat)
- `ai/llm_agent.py`     — prompt building, JSON-schema validation, per-nation
  event memory
- `ai/llm_client.py`    — Ollama / vLLM client adapters
- `ai/symbolic.py`      — compact symbolic observation fed to LLMs

### Front-end and servers
- `server.py`, `index.html`  — local HTTP server + browser UI
- `main.py`                  — constructs the world, handlers, and agents
- `misc/portraits/`          — nation portraits used by the UI

### Reinforcement learning
- `rl/env.py`           — `NationEnv`, a Gymnasium environment wrapping the
  simulator for a single learner nation
- `rl/action_space.py`  — 120-action discrete catalog + action masking
- `rl/encoding.py`      — symbolic observation encoder (196-d vector)
- `rl/reward.py`        — configurable reward shaping
  (dense / terminal / annex-heavy / peace-heavy)

### Training and evaluation entry points
- `train_ppo.py`                — MaskablePPO training for one nation
- `evaluate_ppo.py`              — deterministic evaluation of a PPO checkpoint
- `run_arena.py`                 — mixed PPO / LLM / rule-based tournaments
  with auto-visualization
- `train_lora_sft.py`            — LoRA/QLoRA supervised fine-tuning of an
  LLM agent (optional; demo-only dataset under `train_dataset/`)
- `run_ppo_train.sh`             — single-run PPO launcher (with or without
  a managed vLLM backend)
- `run_ppo_eval_rulebased.sh`    — quick eval of a PPO model against
  rule-based opponents
- `run_ppo_eval_llm.sh`          — eval a PPO model against LLM opponents via
  vLLM
- `run_experiment.py`, `run_5in1.sh` — headless LLM-only experiments

### Experiment matrix (the main study)
- `experiments/matrix.yaml`           — the 18-run `exp_main` matrix
- `experiments/matrix_ext.yaml`       — extended ablations (grace turns,
  symmetric rewards, extreme weights) — partially run
- `experiments/run_matrix.py`         — parallel launcher that fans runs out
  across GPUs
- `experiments/eval_matrix.py`        — uniform evaluation across four
  opponent suites (Balanced×4, Aggressor×4, Diplomat×4, Diverse mix)
- `experiments/plot_matrix.py`        — training-curve / heatmap / fingerprint
  grouping of the raw matrix
- `experiments/plot_fingerprint_full.py` — per-policy × per-suite strategy
  fingerprint sweep (5 eps × 18 policies × 4 suites = 360 rollouts)
- `experiments/plot_training_curves_pub.py` — publication-quality training
  curves (mean + ±1 s.d. band, grouped by reward family)
- `experiments/plot_heatmap_pub.py`         — publication-quality eval
  win-rate heatmap
- `experiments/plot_fingerprint_pub.py`     — publication-quality fingerprint
  figures (2×2 grid + per-suite single figures)
- `experiments/_pub_style.py`               — shared matplotlib style,
  reward-family / action-category colors, run-name parsing helpers
- `run_all_train.sh`, `run_all_eval.sh`     — one-shot wrappers around
  `run_matrix.py` and `eval_matrix.py`

### Paper
- `report/experiments.tex`   — LaTeX Experiments section written against the
  `exp_main` results (uses `outputs/exp_main/figures/*_pub.pdf`)

## Install

The project needs Python 3.12+. A minimal environment for running the game and
PPO experiments is:

```bash
conda create -n tnwo python=3.12 -y
conda activate tnwo
pip install -U pip
pip install -r requirements.txt
```

`requirements.txt` covers:
- `pydantic` — core engine
- `gymnasium`, `stable-baselines3`, `sb3-contrib` — RL stack
- `transformers`, `datasets`, `peft`, `trl`, `accelerate`, `bitsandbytes` —
  LoRA/SFT stack (optional)
- `numpy`

For LLM-backed experiments you also need `vllm`, which is best installed
separately in a GPU-enabled environment:

```bash
pip install vllm
```

System expectations:
- Linux, Python 3.12 recommended
- An NVIDIA GPU with CUDA for any LLM or large-scale PPO work
- `nvidia-smi` available in `$PATH` if you use the GPU launchers

## 1. Browser game server

```bash
python server.py
# → http://localhost:8080
```

## 2. Headless LLM-vs-LLM experiments

Generate and run an experiment config:

```bash
python run_experiment.py --gen-config > experiment.json
python run_experiment.py -c experiment.json --seed 42 --max-turns 50 -v
```

For a one-click "one model controls all five nations" run against a locally
managed vLLM server:

```bash
./run_5in1.sh
```

Edit the variables near the top of `run_5in1.sh` (`CUDA_VISIBLE_DEVICES`,
`VLLM_PORT`, `MODEL_PATH`, `MAX_TURNS`, …) to match your hardware. The first
~25 turns block foreign wars and diplomacy, so use `MAX_TURNS > 25` for
competitive runs.

## 3. PPO training and evaluation

### Train a single PPO policy

```bash
python train_ppo.py \
    --output-dir outputs/my_run \
    --total-timesteps 200000 \
    --max-turns 100 \
    --opponent-mode rule \
    --opponent-strategies balanced balanced balanced balanced
```

Key CLI flags:
- `--opponent-mode {rule,llm}` — rule-based or LLM-backed opponents
- `--opponent-strategies` — four strategy names (one per opponent slot)
- `--dense-reward-scale`, `--terminal-win-reward`, `--annex-bonus`, … —
  reward shaping
- `--checkpoint-freq` — save every N timesteps into `checkpoints/`

Each run writes:
```
outputs/<run>/
  maskable_ppo_nation.zip   final policy
  checkpoints/*.zip         periodic checkpoints
  tb_logs/                  TensorBoard
  training_curve.csv        per-episode metrics (tracked in git)
  train_config.json         exact CLI / hyperparameter snapshot
  train.log                 stdout
```

### Evaluate a trained PPO policy

```bash
# vs rule-based opponents (fast)
./run_ppo_eval_rulebased.sh outputs/my_run/maskable_ppo_nation.zip

# vs LLM opponents (requires vLLM running)
./run_ppo_eval_llm.sh outputs/my_run/maskable_ppo_nation.zip

# mixed tournament with auto-visualization
python run_arena.py \
    --agents ppo:outputs/my_run/maskable_ppo_nation.zip rule rule rule rule \
    --max-turns 800
```

## 4. Running the `exp_main` experiment matrix

The main research study is defined in `experiments/matrix.yaml`: 18 runs
crossing five reward families (`base`, `annex`, `peace`, `dense_only`,
`terminal_only`) against four opponent compositions (`balanced`, `aggressor`,
`diverse`, `diplomat`), plus long-horizon (`_200t`) and seed (`_seed1`,
`_seed2`) replicates.

### Train the whole matrix

```bash
# Train everything, parallelized over all visible GPUs
./run_all_train.sh

# Only train a subset
./run_all_train.sh --only base_vs_diverse,annex_vs_diverse

# Resume / skip runs that already have a final checkpoint
./run_all_train.sh --skip-existing

# Background run with log file
./run_all_train.sh --background
```

### Evaluate + visualize

```bash
# Run the four evaluation suites on every finished run and produce figures
./run_all_eval.sh

# Re-plot without re-evaluating
./run_all_eval.sh --skip-eval
```

This produces the aggregate artifacts in `outputs/exp_main/`:

```
outputs/exp_main/
  <run_name>/                  one dir per run
    eval_results.json          per-suite mean rewards + win rates
    train_config.json
    training_curve.csv
  eval_matrix.json             all runs × all suites, flat
  matrix_summary.json          quick-read summary
  fingerprints.json            aggregate action-category shares
  fingerprints_full.json       per-(policy, suite) distribution
  figures/
    training_curves_pub.{png,pdf}
    eval_heatmap_pub.{png,pdf}
    fingerprint_grid_pub.{png,pdf}
    fingerprint_vs_{balanced,aggressor,diverse,diplomat}_pub.{png,pdf}
```

The `*_pub.{png,pdf}` figures are the ones referenced by
`report/experiments.tex`.

### Producing the paper-quality figures manually

```bash
# Publication-quality training curves
python experiments/plot_training_curves_pub.py outputs/exp_main

# Eval heatmap
python experiments/plot_heatmap_pub.py outputs/exp_main

# Strategy fingerprints (grid + per-suite), consumes fingerprints_full.json
python experiments/plot_fingerprint_full.py outputs/exp_main   # if not cached
python experiments/plot_fingerprint_pub.py  outputs/exp_main
```

## `outputs/` hygiene

Only the following artifacts under `outputs/` are tracked in git:
- `outputs/exp_main/<run>/{train_config.json,training_curve.csv,eval_results.json}`
- `outputs/exp_main/{eval_matrix.json,matrix_summary.json,fingerprints*.json}`
- `outputs/exp_main/figures/*.{png,pdf}`
- same pattern for `outputs/exp_ext/`

Everything else — `maskable_ppo_nation.zip`, `checkpoints/*.zip`, `tb_logs/`,
`train.log`, ad-hoc arena JSONL exports, vLLM logs, `nohup.out` — is ignored
via `.gitignore`. This keeps the repo lean while preserving everything a
reader needs to reproduce the figures and tables in the paper.

## Tests and sanity checks

```bash
python tests/test_symbolic.py     # symbolic view smoke test
vllm --help                       # confirm vllm is installed
nvidia-smi                        # confirm GPU visibility
```

## Notes

- The first 25 turns of every episode block foreign actions (war, diplomacy),
  so training and evaluation always use `max_turns >> 25`. The matrix uses
  `max_turns=100` by default and `max_turns=200` for the long-game ablations.
- LLM agents do not produce a hidden chain-of-thought log; the engine stores
  only validated actions plus a short self-reported reasoning string.
- Training curves are tracked in git (`training_curve.csv` is a few KB per
  run) so the paper figures can be regenerated without retraining anything.
