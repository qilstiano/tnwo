#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# ==============================================================
#  PPO EVALUATION VS RULE-BASED OPPONENTS
#  No vLLM startup required.
# ==============================================================

# ---- Evaluation parameters ----
MODEL_PATH="outputs/ppo_vs_llm_20260406_100555/maskable_ppo_nation.zip"
EPISODES=20
NUM_PLAYERS=5
LEARNER_ID=0
MAX_TURNS=100
SEED=42
DETERMINISTIC=true

# ---- Output ----
OUTPUT_DIR="outputs"
RUN_TS=$(date +"%Y%m%d_%H%M%S")
EVAL_JSON="${OUTPUT_DIR}/ppo_eval_vs_rulebased_${RUN_TS}.json"
EVAL_LOG="${OUTPUT_DIR}/ppo_eval_vs_rulebased_${RUN_TS}.log"

mkdir -p "${OUTPUT_DIR}"

DET_FLAG=""
if [ "${DETERMINISTIC}" = true ]; then
  DET_FLAG="--deterministic"
fi

echo "Running PPO evaluation vs rule-based opponents..."
echo "  model       : ${MODEL_PATH}"
echo "  episodes    : ${EPISODES}"
echo "  output json : ${EVAL_JSON}"
echo "  output log  : ${EVAL_LOG}"
echo ""

python evaluate_ppo.py \
  --model-path "${MODEL_PATH}" \
  --episodes "${EPISODES}" \
  --num-players "${NUM_PLAYERS}" \
  --learner-id "${LEARNER_ID}" \
  --max-turns "${MAX_TURNS}" \
  --seed "${SEED}" \
  ${DET_FLAG} \
  --opponents-mode rulebased \
  --output-json "${EVAL_JSON}" \
  2>&1 | tee "${EVAL_LOG}"

echo ""
echo "Done."
echo "  Result JSON : ${EVAL_JSON}"
echo "  Eval log    : ${EVAL_LOG}"
