#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# ==============================================================
#  SINGLE-MODEL 5-NATION LAUNCHER
#  One backend, one model, five nations all share it.
#  Edit the variables below, then run:  bash run_5in1.sh
# ==============================================================

# ---- Game parameters ----
NUM_PLAYERS=5
MAX_TURNS=100
SEED=42
TEMPERATURE=0.7
VERBOSE=true          # true = print every event per turn

# ---- Output ----
OUTPUT_DIR="outputs"
RUN_TS=$(date +"%Y%m%d_%H%M%S")
OUTPUT="${OUTPUT_DIR}/game_export_${RUN_TS}.jsonl"
VLLM_LOG="${OUTPUT_DIR}/vllm_${RUN_TS}.log"

# ---- Shared backend for all 5 nations ----
# This script can launch a local vLLM server automatically.
# The game then points all 5 nations at that one shared endpoint.
SHARED_PROVIDER="vllm"

# ---- vLLM launch parameters ----
CUDA_VISIBLE_DEVICES=0
VLLM_PORT=8001
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.9
MAX_MODEL_LEN=8192
DTYPE="bfloat16"
SERVED_MODEL_NAME="qwen-7b"
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"

SHARED_URL="http://localhost:${VLLM_PORT}"
SHARED_MODEL="${SERVED_MODEL_NAME}"

# ---- Nation strategies ----
# Options: neutral, expansionist, scientific, mercantile, diplomatic
# Keep these distinct if you want one model to play different roles.
# Set them all to the same value if you want the exact same prompt style.
NATION_0_STRATEGY="neutral"
NATION_1_STRATEGY="expansionist"
NATION_2_STRATEGY="scientific"
NATION_3_STRATEGY="mercantile"
NATION_4_STRATEGY="diplomatic"

# ==============================================================
#  Nothing below this line normally needs editing.
# ==============================================================

VERBOSE_FLAG=""
if [ "$VERBOSE" = true ]; then
  VERBOSE_FLAG="-v"
fi

mkdir -p "$OUTPUT_DIR"

CONFIG_FILE=$(mktemp /tmp/tnwo_config_XXXXXX.json)
VLLM_PID=""

cleanup() {
  rm -f "$CONFIG_FILE"
  if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
    echo ""
    echo "Stopping vLLM server (pid=${VLLM_PID})..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT

check_vllm_ready() {
  python - "$1" <<'PY'
import json
import sys
import urllib.request

url = sys.argv[1].rstrip("/") + "/v1/models"
try:
    with urllib.request.urlopen(url, timeout=2) as resp:
        json.loads(resp.read().decode("utf-8"))
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
}

start_vllm_if_needed() {
  if check_vllm_ready "$SHARED_URL"; then
    echo "Reusing existing vLLM server -> ${SHARED_URL}"
    return
  fi

  echo "Starting vLLM server..."
  echo "  GPU                 : ${CUDA_VISIBLE_DEVICES}"
  echo "  Port                : ${VLLM_PORT}"
  echo "  Tensor parallel     : ${TENSOR_PARALLEL_SIZE}"
  echo "  GPU memory util     : ${GPU_MEMORY_UTILIZATION}"
  echo "  Max model len       : ${MAX_MODEL_LEN}"
  echo "  Dtype               : ${DTYPE}"
  echo "  Served model name   : ${SERVED_MODEL_NAME}"
  echo "  Model path          : ${MODEL_PATH}"
  echo "  Log file            : ${VLLM_LOG}"

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

  for _ in $(seq 1 120); do
    if check_vllm_ready "$SHARED_URL"; then
      echo "vLLM is ready -> ${SHARED_URL}"
      return
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "vLLM exited unexpectedly. Check ${VLLM_LOG}"
      exit 1
    fi
    sleep 2
  done

  echo "Timed out waiting for vLLM to become ready. Check ${VLLM_LOG}"
  exit 1
}

cat > "$CONFIG_FILE" <<ENDCONFIG
{
  "num_players": ${NUM_PLAYERS},
  "max_turns": ${MAX_TURNS},
  "seed": ${SEED},
  "temperature": ${TEMPERATURE},
  "output": "${OUTPUT}",
  "backends": {
    "shared": {
      "provider": "${SHARED_PROVIDER}",
      "base_url": "${SHARED_URL}",
      "model": "${SHARED_MODEL}"
    }
  },
  "nation_backend_map": {
    "0": "shared",
    "1": "shared",
    "2": "shared",
    "3": "shared",
    "4": "shared"
  },
  "strategies": {
    "0": "${NATION_0_STRATEGY}",
    "1": "${NATION_1_STRATEGY}",
    "2": "${NATION_2_STRATEGY}",
    "3": "${NATION_3_STRATEGY}",
    "4": "${NATION_4_STRATEGY}"
  }
}
ENDCONFIG

echo "Config generated -> ${CONFIG_FILE}"
echo "Shared model     -> ${SHARED_MODEL}"
echo "Shared endpoint  -> ${SHARED_URL}"
echo ""

start_vllm_if_needed

python run_experiment.py -c "$CONFIG_FILE" $VERBOSE_FLAG
