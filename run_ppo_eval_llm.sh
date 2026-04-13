#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# ==============================================================
#  PPO EVALUATION VS SHARED LLM OPPONENTS
#  All opponent nations share ONE vLLM server.
# ==============================================================

# ---- Evaluation parameters ----
MODEL_PATH="outputs/ppo_vs_llm_20260406_100555/maskable_ppo_nation.zip"
EPISODES=20
NUM_PLAYERS=5
LEARNER_ID=0
MAX_TURNS=100
SEED=42
DETERMINISTIC=true
LLM_TEMPERATURE=0.95

# ---- Output ----
OUTPUT_DIR="outputs"
RUN_TS=$(date +"%Y%m%d_%H%M%S")
EVAL_JSON="${OUTPUT_DIR}/ppo_eval_vs_llm_${RUN_TS}.json"
EVAL_LOG="${OUTPUT_DIR}/ppo_eval_vs_llm_${RUN_TS}.log"

# ---- Shared vLLM server ----
VLLM_PORT=8001
VLLM_BASE_URL="http://localhost:${VLLM_PORT}"
VLLM_PROVIDER="vllm"
VLLM_MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
VLLM_SERVED_NAME="Qwen/Qwen2.5-7B-Instruct"
VLLM_CUDA_VISIBLE_DEVICES=0
VLLM_TP=1
VLLM_GPU_MEMORY_UTIL=0.6
VLLM_MAX_MODEL_LEN=4096
VLLM_DTYPE="bfloat16"
VLLM_LOG="${OUTPUT_DIR}/vllm_shared_${RUN_TS}.log"

mkdir -p "${OUTPUT_DIR}"
VLLM_PID=""

cleanup() {
  if [ -n "${VLLM_PID}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "Stopping started vLLM process pid=${VLLM_PID}"
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
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
  if check_vllm_ready "${VLLM_BASE_URL}"; then
    echo "Reusing existing shared vLLM -> ${VLLM_BASE_URL}"
    return
  fi

  echo "Starting shared vLLM..."
  echo "  URL               : ${VLLM_BASE_URL}"
  echo "  CUDA              : ${VLLM_CUDA_VISIBLE_DEVICES}"
  echo "  model_path        : ${VLLM_MODEL_PATH}"
  echo "  served_model_name : ${VLLM_SERVED_NAME}"
  echo "  gpu_mem_util      : ${VLLM_GPU_MEMORY_UTIL}"
  echo "  max_model_len     : ${VLLM_MAX_MODEL_LEN}"
  echo "  log               : ${VLLM_LOG}"

  CUDA_VISIBLE_DEVICES="${VLLM_CUDA_VISIBLE_DEVICES}" \
    vllm serve "${VLLM_MODEL_PATH}" \
      --host 0.0.0.0 \
      --port "${VLLM_PORT}" \
      --tensor-parallel-size "${VLLM_TP}" \
      --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTIL}" \
      --max-model-len "${VLLM_MAX_MODEL_LEN}" \
      --dtype "${VLLM_DTYPE}" \
      --served-model-name "${VLLM_SERVED_NAME}" \
      > "${VLLM_LOG}" 2>&1 &
  VLLM_PID=$!

  for _ in $(seq 1 120); do
    if check_vllm_ready "${VLLM_BASE_URL}"; then
      echo "shared vLLM is ready -> ${VLLM_BASE_URL}"
      return
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
      echo "shared vLLM exited unexpectedly. Check ${VLLM_LOG}"
      exit 1
    fi
    sleep 2
  done

  echo "Timed out waiting for shared vLLM. Check ${VLLM_LOG}"
  exit 1
}

start_vllm_if_needed

DET_FLAG=""
if [ "${DETERMINISTIC}" = true ]; then
  DET_FLAG="--deterministic"
fi

echo ""
echo "Running PPO evaluation vs shared-LLM opponents..."
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
  --opponents-mode llm \
  --llm-temperature "${LLM_TEMPERATURE}" \
  --shared-llm-provider "${VLLM_PROVIDER}" \
  --shared-llm-base-url "${VLLM_BASE_URL}" \
  --shared-llm-model "${VLLM_SERVED_NAME}" \
  --output-json "${EVAL_JSON}" \
  2>&1 | tee "${EVAL_LOG}"

echo ""
echo "Done."
echo "  Result JSON : ${EVAL_JSON}"
echo "  Eval log    : ${EVAL_LOG}"
echo "  vLLM log    : ${VLLM_LOG}"
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# ==============================================================
#  PPO EVALUATION VS LLM OPPONENTS
#  One-command flow:
#    1) Reuse/start required vLLM servers
#    2) Run evaluate_ppo.py with --opponents-mode llm
#    3) Save logs + summary JSON under outputs/
#
#  Run:
#    bash run_ppo_eval_llm.sh
# ==============================================================

# ---- Evaluation parameters ----
MODEL_PATH="outputs/ppo_vs_llm_20260406_100555/maskable_ppo_nation.zip"
EPISODES=20
NUM_PLAYERS=5
LEARNER_ID=0
MAX_TURNS=100
SEED=42
DETERMINISTIC=true
LLM_TEMPERATURE=0.95

# ---- Output ----
OUTPUT_DIR="outputs"
RUN_TS=$(date +"%Y%m%d_%H%M%S")
EVAL_JSON="${OUTPUT_DIR}/ppo_eval_vs_llm_${RUN_TS}.json"
EVAL_LOG="${OUTPUT_DIR}/ppo_eval_vs_llm_${RUN_TS}.log"

# ---- vLLM backend topology (2+2+1 nations) ----
# Nation mapping in ai/config.py:
#   0->server_1, 1->server_1, 2->server_2, 3->server_2, 4->server_3
SERVER_1_PORT=8001
SERVER_1_MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
SERVER_1_SERVED_NAME="Qwen/Qwen2.5-7B-Instruct"
SERVER_1_CUDA_VISIBLE_DEVICES=0
SERVER_1_TP=1
SERVER_1_GPU_MEMORY_UTIL=0.5
SERVER_1_MAX_MODEL_LEN=8192
SERVER_1_DTYPE="bfloat16"

SERVER_2_PORT=8002
SERVER_2_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
SERVER_2_SERVED_NAME="meta-llama/Llama-3.1-8B-Instruct"
SERVER_2_CUDA_VISIBLE_DEVICES=1
SERVER_2_TP=1
SERVER_2_GPU_MEMORY_UTIL=0.90
SERVER_2_MAX_MODEL_LEN=8192
SERVER_2_DTYPE="bfloat16"

SERVER_3_PORT=8003
SERVER_3_MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
SERVER_3_SERVED_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
SERVER_3_CUDA_VISIBLE_DEVICES=2
SERVER_3_TP=1
SERVER_3_GPU_MEMORY_UTIL=0.90
SERVER_3_MAX_MODEL_LEN=8192
SERVER_3_DTYPE="bfloat16"

VLLM_LOG_1="${OUTPUT_DIR}/vllm_server1_${RUN_TS}.log"
VLLM_LOG_2="${OUTPUT_DIR}/vllm_server2_${RUN_TS}.log"
VLLM_LOG_3="${OUTPUT_DIR}/vllm_server3_${RUN_TS}.log"

mkdir -p "${OUTPUT_DIR}"

VLLM_PID_1=""
VLLM_PID_2=""
VLLM_PID_3=""

cleanup() {
  for pid in "${VLLM_PID_1}" "${VLLM_PID_2}" "${VLLM_PID_3}"; do
    if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
      echo "Stopping started vLLM process pid=${pid}"
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
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

start_one_vllm_if_needed() {
  local server_name="$1"
  local url="$2"
  local model_path="$3"
  local served_name="$4"
  local port="$5"
  local cuda_devices="$6"
  local tp="$7"
  local gpu_mem="$8"
  local max_len="$9"
  local dtype="${10}"
  local log_file="${11}"
  local pid_var="${12}"

  if check_vllm_ready "${url}"; then
    echo "Reusing existing ${server_name} -> ${url}"
    return
  fi

  echo "Starting ${server_name} ..."
  echo "  URL               : ${url}"
  echo "  CUDA              : ${cuda_devices}"
  echo "  model_path        : ${model_path}"
  echo "  served_model_name : ${served_name}"
  echo "  log               : ${log_file}"

  CUDA_VISIBLE_DEVICES="${cuda_devices}" \
    vllm serve "${model_path}" \
      --host 0.0.0.0 \
      --port "${port}" \
      --tensor-parallel-size "${tp}" \
      --gpu-memory-utilization "${gpu_mem}" \
      --max-model-len "${max_len}" \
      --dtype "${dtype}" \
      --served-model-name "${served_name}" \
      > "${log_file}" 2>&1 &

  local pid=$!
  eval "${pid_var}=${pid}"

  for _ in $(seq 1 120); do
    if check_vllm_ready "${url}"; then
      echo "${server_name} is ready -> ${url}"
      return
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "${server_name} exited unexpectedly. Check ${log_file}"
      exit 1
    fi
    sleep 2
  done

  echo "Timed out waiting for ${server_name}. Check ${log_file}"
  exit 1
}

start_one_vllm_if_needed \
  "server_1" \
  "http://localhost:${SERVER_1_PORT}" \
  "${SERVER_1_MODEL_PATH}" \
  "${SERVER_1_SERVED_NAME}" \
  "${SERVER_1_PORT}" \
  "${SERVER_1_CUDA_VISIBLE_DEVICES}" \
  "${SERVER_1_TP}" \
  "${SERVER_1_GPU_MEMORY_UTIL}" \
  "${SERVER_1_MAX_MODEL_LEN}" \
  "${SERVER_1_DTYPE}" \
  "${VLLM_LOG_1}" \
  "VLLM_PID_1"

start_one_vllm_if_needed \
  "server_2" \
  "http://localhost:${SERVER_2_PORT}" \
  "${SERVER_2_MODEL_PATH}" \
  "${SERVER_2_SERVED_NAME}" \
  "${SERVER_2_PORT}" \
  "${SERVER_2_CUDA_VISIBLE_DEVICES}" \
  "${SERVER_2_TP}" \
  "${SERVER_2_GPU_MEMORY_UTIL}" \
  "${SERVER_2_MAX_MODEL_LEN}" \
  "${SERVER_2_DTYPE}" \
  "${VLLM_LOG_2}" \
  "VLLM_PID_2"

start_one_vllm_if_needed \
  "server_3" \
  "http://localhost:${SERVER_3_PORT}" \
  "${SERVER_3_MODEL_PATH}" \
  "${SERVER_3_SERVED_NAME}" \
  "${SERVER_3_PORT}" \
  "${SERVER_3_CUDA_VISIBLE_DEVICES}" \
  "${SERVER_3_TP}" \
  "${SERVER_3_GPU_MEMORY_UTIL}" \
  "${SERVER_3_MAX_MODEL_LEN}" \
  "${SERVER_3_DTYPE}" \
  "${VLLM_LOG_3}" \
  "VLLM_PID_3"

DET_FLAG=""
if [ "${DETERMINISTIC}" = true ]; then
  DET_FLAG="--deterministic"
fi

echo ""
echo "Running PPO evaluation vs LLM opponents..."
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
  --opponents-mode llm \
  --llm-temperature "${LLM_TEMPERATURE}" \
  --output-json "${EVAL_JSON}" \
  2>&1 | tee "${EVAL_LOG}"

echo ""
echo "Done."
echo "  Result JSON : ${EVAL_JSON}"
echo "  Eval log    : ${EVAL_LOG}"
echo "  vLLM logs   : ${VLLM_LOG_1}, ${VLLM_LOG_2}, ${VLLM_LOG_3}"
