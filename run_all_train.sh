#!/usr/bin/env bash
# =====================================================================
#  TNWO experiment matrix — TRAIN launcher
#
#  Wraps experiments/run_matrix.py. By default trains all 18 runs in
#  experiments/matrix.yaml on all detected GPUs.
#
#  Usage:
#    ./run_all_train.sh                                # train everything
#    ./run_all_train.sh --dry-run                      # just print plan
#    ./run_all_train.sh --skip-existing                # resume / skip done
#    ./run_all_train.sh --only base_vs_diverse        # train one run
#    ./run_all_train.sh --only base_vs_diverse,annex_vs_diverse,peace_vs_diverse
#    ./run_all_train.sh --gpus 0,1                     # only GPUs 0 and 1
#    ./run_all_train.sh --workers 2                    # 2 concurrent workers
#    ./run_all_train.sh --matrix experiments/matrix.yaml  # custom matrix
#    ./run_all_train.sh --background                   # nohup, log to file
#
#  Time / memory estimates (4 × H200 GPUs, 4 workers):
#    short run  (50k timesteps, max_turns 100) : ~15 min/run
#    long run   (80k timesteps, max_turns 200) : ~24 min/run
#    full matrix (16 short + 2 long)            : ~70-80 min wallclock
#    GPU memory per worker                      : <1 GB (fits dozens per H200)
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")"

# Cap per-worker BLAS / OMP threads so multiple PPO workers on the same box
# don't oversubscribe the 192 host threads. 4 threads * 16 workers = 64.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

MATRIX="experiments/matrix.yaml"
ONLY=""
GPUS=""
WORKERS=""
DRY_RUN=""
SKIP_EXISTING=""
BACKGROUND=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --matrix)        MATRIX="$2"; shift 2;;
    --only)          ONLY="$2"; shift 2;;
    --gpus)          GPUS="$2"; shift 2;;
    --workers)       WORKERS="$2"; shift 2;;
    --dry-run)       DRY_RUN="--dry-run"; shift;;
    --skip-existing) SKIP_EXISTING="--skip-existing"; shift;;
    --background)    BACKGROUND="1"; shift;;
    -h|--help)
      sed -n '2,30p' "$0"; exit 0;;
    *)
      echo "Unknown flag: $1"; exit 1;;
  esac
done

CMD=(python experiments/run_matrix.py "$MATRIX")
[[ -n "$ONLY"          ]] && CMD+=(--only "$ONLY")
[[ -n "$GPUS"          ]] && CMD+=(--gpus "$GPUS")
[[ -n "$WORKERS"       ]] && CMD+=(--workers "$WORKERS")
[[ -n "$DRY_RUN"       ]] && CMD+=("$DRY_RUN")
[[ -n "$SKIP_EXISTING" ]] && CMD+=("$SKIP_EXISTING")

mkdir -p outputs
TS="$(date +%Y%m%d_%H%M%S)"
LOG="outputs/run_all_train_${TS}.log"

echo "================================================================"
echo "  TNWO TRAIN LAUNCHER"
echo "================================================================"
echo "  matrix         : $MATRIX"
echo "  only           : ${ONLY:-<all>}"
echo "  gpus           : ${GPUS:-<auto>}"
echo "  workers        : ${WORKERS:-<auto>}"
echo "  dry-run        : ${DRY_RUN:-no}"
echo "  skip-existing  : ${SKIP_EXISTING:-no}"
echo "  background     : ${BACKGROUND:-no}"
echo "  log file       : $LOG"
echo "  command        : ${CMD[*]}"
echo "================================================================"
echo ""

if [[ -n "$BACKGROUND" ]]; then
  nohup "${CMD[@]}" > "$LOG" 2>&1 &
  PID=$!
  echo "  launched in background, pid=$PID"
  echo "  tail logs with:  tail -f $LOG"
else
  "${CMD[@]}" 2>&1 | tee "$LOG"
fi
