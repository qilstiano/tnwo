#!/usr/bin/env bash
# =====================================================================
#  TNWO experiment matrix — EVAL + VISUALIZATION launcher
#
#  Runs experiments/eval_matrix.py followed by experiments/plot_matrix.py
#  on a finished experiment directory. Produces:
#    outputs/<exp>/<run>/eval_results.json
#    outputs/<exp>/eval_matrix.json
#    outputs/<exp>/fingerprints.json
#    outputs/<exp>/figures/training_curves_grouped.png
#    outputs/<exp>/figures/eval_heatmap.png
#    outputs/<exp>/figures/strategy_fingerprint.png
#
#  Usage:
#    ./run_all_eval.sh                                      # full pipeline on outputs/exp_main
#    ./run_all_eval.sh --exp outputs/exp_main               # explicit experiment dir
#    ./run_all_eval.sh --episodes 30                        # more eval episodes per suite
#    ./run_all_eval.sh --only base_vs_diverse               # eval one run
#    ./run_all_eval.sh --only base_vs_diverse,annex_vs_diverse
#    ./run_all_eval.sh --skip-eval                          # only re-plot
#    ./run_all_eval.sh --skip-plots                         # only re-eval
#    ./run_all_eval.sh --skip-fingerprint                   # plots without fingerprint pass
#    ./run_all_eval.sh --skip-existing                      # reuse cached eval_results.json
#    ./run_all_eval.sh --fingerprint-episodes 3             # cheaper fingerprint pass
#
#  Time estimates (4 × H200, 18 runs):
#    eval (20 ep × 4 suites × 18 runs)   : ~20 min sequential
#    fingerprint (5 ep × 18 runs)        : ~3 min
#    plots                                : <30 sec
#    full pipeline                        : ~25 min
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")"

EXP="outputs/exp_main"
EPISODES=20
ONLY=""
SKIP_EVAL=""
SKIP_PLOTS=""
SKIP_FP=""
SKIP_EXISTING=""
FP_EPISODES=5

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp)                  EXP="$2"; shift 2;;
    --episodes)             EPISODES="$2"; shift 2;;
    --only)                 ONLY="$2"; shift 2;;
    --skip-eval)            SKIP_EVAL=1; shift;;
    --skip-plots)           SKIP_PLOTS=1; shift;;
    --skip-fingerprint)     SKIP_FP="--skip-fingerprint"; shift;;
    --skip-existing)        SKIP_EXISTING="--skip-existing"; shift;;
    --fingerprint-episodes) FP_EPISODES="$2"; shift 2;;
    -h|--help)
      sed -n '2,38p' "$0"; exit 0;;
    *)
      echo "Unknown flag: $1"; exit 1;;
  esac
done

if [[ ! -d "$EXP" ]]; then
  echo "ERROR: experiment directory $EXP does not exist"
  exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOG="${EXP}/run_all_eval_${TS}.log"

echo "================================================================"
echo "  TNWO EVAL + PLOT LAUNCHER"
echo "================================================================"
echo "  experiment           : $EXP"
echo "  episodes / suite     : $EPISODES"
echo "  only                 : ${ONLY:-<all>}"
echo "  skip-eval            : ${SKIP_EVAL:-no}"
echo "  skip-plots           : ${SKIP_PLOTS:-no}"
echo "  skip-fingerprint     : ${SKIP_FP:-no}"
echo "  skip-existing(eval)  : ${SKIP_EXISTING:-no}"
echo "  fingerprint episodes : $FP_EPISODES"
echo "  log file             : $LOG"
echo "================================================================"
echo ""

# Mirror everything into the log file as well as the terminal.
exec > >(tee -a "$LOG") 2>&1

if [[ -z "$SKIP_EVAL" ]]; then
  echo "[$(date)] === eval_matrix.py ==="
  EVAL_CMD=(python experiments/eval_matrix.py "$EXP" --episodes "$EPISODES")
  [[ -n "$ONLY"          ]] && EVAL_CMD+=(--only "$ONLY")
  [[ -n "$SKIP_EXISTING" ]] && EVAL_CMD+=("$SKIP_EXISTING")
  echo "    ${EVAL_CMD[*]}"
  "${EVAL_CMD[@]}"
  echo ""
fi

if [[ -z "$SKIP_PLOTS" ]]; then
  echo "[$(date)] === plot_matrix.py ==="
  PLOT_CMD=(python experiments/plot_matrix.py "$EXP" --fingerprint-episodes "$FP_EPISODES")
  [[ -n "$SKIP_FP" ]] && PLOT_CMD+=("$SKIP_FP")
  echo "    ${PLOT_CMD[*]}"
  "${PLOT_CMD[@]}"
  echo ""
fi

echo "[$(date)] === DONE ==="
echo "Figures: ${EXP}/figures/"
ls -la "${EXP}/figures/" 2>/dev/null || true
