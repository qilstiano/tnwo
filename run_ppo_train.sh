#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# ==============================================================
#  PPO TRAINING LAUNCHER
#  1. Start vLLM backend (or reuse existing)
#  2. Train PPO learner (nation 0) vs 4 LLM opponents
#  Everything runs under nohup with logs redirected.
#
#  Usage:
#    bash run_ppo_train.sh            # foreground
#    nohup bash run_ppo_train.sh &    # background
# ==============================================================

# ---- vLLM parameters ----
CUDA_VISIBLE_DEVICES=0
VLLM_PORT=8001
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.30
MAX_MODEL_LEN=4096
DTYPE="bfloat16"
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
SERVED_MODEL_NAME="qwen-7b"

# ---- PPO training parameters ----
TOTAL_TIMESTEPS=200000
NUM_PLAYERS=5
LEARNER_ID=0
MAX_TURNS=100
SEED=0
LEARNING_RATE=3e-4
N_STEPS=2048
BATCH_SIZE=256
POLICY_HIDDEN_SIZE=256
CHECKPOINT_FREQ=20000

# ---- Opponent config ----
OPPONENT_MODE="llm"
OPPONENT_PROVIDER="vllm"
OPPONENT_BASE_URL="http://localhost:${VLLM_PORT}"
OPPONENT_MODEL="${SERVED_MODEL_NAME}"

# ---- Output ----
RUN_TS=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="outputs/ppo_vs_llm_${RUN_TS}"
LOG_DIR="${OUTPUT_DIR}/logs"

# ==============================================================
#  Nothing below this line normally needs editing.
# ==============================================================

mkdir -p "$LOG_DIR"

VLLM_LOG="${LOG_DIR}/vllm.log"
TRAIN_LOG="${LOG_DIR}/train.log"
VLLM_PID=""
VLLM_STARTED_BY_US=false

cleanup() {
  if [ "$VLLM_STARTED_BY_US" = true ] && [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
    echo ""
    echo "[$(date)] Stopping vLLM server (pid=${VLLM_PID})..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

check_vllm_ready() {
  python - "$1" <<'PY'
import json, sys, urllib.request
url = sys.argv[1].rstrip("/") + "/v1/models"
try:
    with urllib.request.urlopen(url, timeout=2) as resp:
        json.loads(resp.read().decode("utf-8"))
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
}

# ---- Step 1: Start vLLM ----
echo "[$(date)] === PPO Training Launcher ==="
echo "  Output dir : ${OUTPUT_DIR}"
echo "  vLLM log   : ${VLLM_LOG}"
echo "  Train log  : ${TRAIN_LOG}"
echo ""

if check_vllm_ready "$OPPONENT_BASE_URL"; then
  echo "[$(date)] Reusing existing vLLM server -> ${OPPONENT_BASE_URL}"
else
  echo "[$(date)] Starting vLLM server..."
  echo "  Model  : ${MODEL_PATH}"
  echo "  Port   : ${VLLM_PORT}"
  echo "  GPU    : ${CUDA_VISIBLE_DEVICES}"

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    vllm serve "${MODEL_PATH}" \
      --host 0.0.0.0 \
      --port "${VLLM_PORT}" \
      --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --dtype "${DTYPE}" \
      --served-model-name "${SERVED_MODEL_NAME}" \
      > "${VLLM_LOG}" 2>&1 &
  VLLM_PID=$!
  VLLM_STARTED_BY_US=true

  echo "[$(date)] Waiting for vLLM to be ready (pid=${VLLM_PID})..."
  for i in $(seq 1 120); do
    if check_vllm_ready "$OPPONENT_BASE_URL"; then
      echo "[$(date)] vLLM is ready after ~$((i*2))s"
      break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "[$(date)] ERROR: vLLM exited unexpectedly. Check ${VLLM_LOG}"
      exit 1
    fi
    sleep 2
  done

  if ! check_vllm_ready "$OPPONENT_BASE_URL"; then
    echo "[$(date)] ERROR: vLLM timed out after 240s. Check ${VLLM_LOG}"
    exit 1
  fi
fi

# ---- Step 2: Train PPO ----
echo ""
echo "[$(date)] Starting PPO training..."
echo "  Timesteps  : ${TOTAL_TIMESTEPS}"
echo "  Opponents  : ${OPPONENT_MODE} (${OPPONENT_MODEL} @ ${OPPONENT_BASE_URL})"
echo ""

python train_ppo.py \
  --output-dir "${OUTPUT_DIR}" \
  --total-timesteps "${TOTAL_TIMESTEPS}" \
  --num-players "${NUM_PLAYERS}" \
  --learner-id "${LEARNER_ID}" \
  --max-turns "${MAX_TURNS}" \
  --seed "${SEED}" \
  --learning-rate "${LEARNING_RATE}" \
  --n-steps "${N_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --policy-hidden-size "${POLICY_HIDDEN_SIZE}" \
  --checkpoint-freq "${CHECKPOINT_FREQ}" \
  --opponent-mode "${OPPONENT_MODE}" \
  --opponent-provider "${OPPONENT_PROVIDER}" \
  --opponent-base-url "${OPPONENT_BASE_URL}" \
  --opponent-model "${OPPONENT_MODEL}" \
  2>&1 | tee "${TRAIN_LOG}"

echo ""
echo "[$(date)] Training complete. Results in ${OUTPUT_DIR}"
