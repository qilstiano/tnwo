#!/usr/bin/env python3
"""
Parallel launcher for the TNWO PPO experiment matrix.

Reads experiments/matrix.yaml, fans out one `train_ppo.py` subprocess per run,
and round-robins them across the available GPUs via CUDA_VISIBLE_DEVICES.

Usage:
    python experiments/run_matrix.py experiments/matrix.yaml
    python experiments/run_matrix.py experiments/matrix.yaml --workers 4
    python experiments/run_matrix.py experiments/matrix.yaml --gpus 0,1,2,3
    python experiments/run_matrix.py experiments/matrix.yaml --dry-run
    python experiments/run_matrix.py experiments/matrix.yaml --only base_vs_diverse,annex_vs_diverse
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent


def discover_gpus() -> List[int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
        )
        return [int(x) for x in out.strip().splitlines() if x.strip()]
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []


def merge_run(defaults: Dict[str, Any], run: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(defaults)
    merged.update(run)
    return merged


def build_command(run_cfg: Dict[str, Any], output_dir: Path) -> List[str]:
    cmd = [
        sys.executable, "train_ppo.py",
        "--output-dir", str(output_dir),
        "--total-timesteps", str(run_cfg["total_timesteps"]),
        "--num-players", str(run_cfg["num_players"]),
        "--learner-id", str(run_cfg["learner_id"]),
        "--max-turns", str(run_cfg["max_turns"]),
        "--seed", str(run_cfg["seed"]),
        "--n-steps", str(run_cfg["n_steps"]),
        "--batch-size", str(run_cfg["batch_size"]),
        "--policy-hidden-size", str(run_cfg["policy_hidden_size"]),
        "--checkpoint-freq", str(run_cfg["checkpoint_freq"]),
        "--dense-reward-scale", str(run_cfg["dense_reward_scale"]),
        "--terminal-win-reward", str(run_cfg["terminal_win_reward"]),
        "--terminal-peace-reward", str(run_cfg["terminal_peace_reward"]),
        "--terminal-loss-penalty", str(run_cfg["terminal_loss_penalty"]),
        "--annex-bonus", str(run_cfg["annex_bonus"]),
        "--device", str(run_cfg["device"]),
        "--opponent-mode", "rulebased",
    ]
    strats = run_cfg.get("opponent_strategies")
    if strats:
        cmd.append("--opponent-strategies")
        cmd.extend(strats)

    # Optional extended-axis fields — only add the flag if the matrix
    # file actually overrides them, so existing matrices (which don't
    # mention these) keep using train_ppo.py's defaults.
    if "grace_period_turns" in run_cfg:
        cmd.extend(["--grace-period-turns", str(run_cfg["grace_period_turns"])])
    if "starting_asymmetry" in run_cfg:
        cmd.extend(["--starting-asymmetry", str(run_cfg["starting_asymmetry"])])
    if "starting_action_points" in run_cfg:
        cmd.extend(["--starting-action-points", str(run_cfg["starting_action_points"])])

    return cmd


def run_one(run_cfg: Dict[str, Any], output_dir: Path, gpu_id: Optional[int]) -> Dict[str, Any]:
    """Worker function. Spawns the train_ppo.py subprocess and waits for it."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"

    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = build_command(run_cfg, output_dir)
    start = time.time()

    with open(log_path, "w") as logf:
        logf.write(f"# launched at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        logf.write(f"# CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES','')}\n")
        logf.write(f"# command: {' '.join(cmd)}\n\n")
        logf.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
        )

    elapsed = time.time() - start
    return {
        "name": run_cfg["name"],
        "returncode": proc.returncode,
        "elapsed_sec": round(elapsed, 1),
        "output_dir": str(output_dir),
        "log": str(log_path),
        "gpu": gpu_id,
    }


def main():
    parser = argparse.ArgumentParser(description="TNWO experiment matrix launcher")
    parser.add_argument("matrix_path", help="Path to YAML matrix file")
    parser.add_argument("--workers", type=int, default=None,
                        help="Concurrent training jobs. Defaults to len(gpus).")
    parser.add_argument("--gpus", default=None,
                        help="Comma-separated GPU ids to round-robin (default: auto-detect all)")
    parser.add_argument("--only", default=None,
                        help="Comma-separated subset of run names to launch")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip runs whose output dir already contains maskable_ppo_nation.zip")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the planned commands and exit")
    args = parser.parse_args()

    with open(args.matrix_path) as f:
        spec = yaml.safe_load(f)

    experiment_name = spec.get("experiment_name", "exp_main")
    defaults = spec.get("defaults", {})
    runs = spec.get("runs", [])

    if args.only:
        wanted = {n.strip() for n in args.only.split(",") if n.strip()}
        runs = [r for r in runs if r["name"] in wanted]

    if not runs:
        print("No runs to launch.")
        return

    exp_dir = REPO_ROOT / "outputs" / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    if args.gpus:
        gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    else:
        gpus = discover_gpus()
    workers = args.workers or (len(gpus) if gpus else 1)

    print(f"Experiment    : {experiment_name}")
    print(f"Output root   : {exp_dir}")
    print(f"Runs          : {len(runs)}")
    print(f"GPUs          : {gpus or 'cpu only'}")
    print(f"Workers       : {workers}")
    print()

    plan = []
    for idx, raw_run in enumerate(runs):
        cfg = merge_run(defaults, raw_run)
        run_dir = exp_dir / cfg["name"]
        gpu_id = gpus[idx % len(gpus)] if gpus else None

        if args.skip_existing and (run_dir / "maskable_ppo_nation.zip").exists():
            print(f"  [SKIP existing] {cfg['name']}")
            continue

        plan.append((cfg, run_dir, gpu_id))
        print(f"  -> {cfg['name']:32s}  gpu={gpu_id}  out={run_dir.relative_to(REPO_ROOT)}")

    if args.dry_run:
        print("\n(dry run, exiting)")
        return

    print()
    print(f"Launching {len(plan)} runs with {workers} concurrent workers...")
    print()

    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(run_one, cfg, run_dir, gpu_id): cfg["name"]
                   for cfg, run_dir, gpu_id in plan}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                res = fut.result()
                status = "OK" if res["returncode"] == 0 else f"FAIL ({res['returncode']})"
                print(f"  [{status}] {name:32s}  {res['elapsed_sec']:8.1f}s  gpu={res['gpu']}")
                results.append(res)
            except Exception as exc:
                print(f"  [ERROR] {name}: {exc}")
                results.append({"name": name, "returncode": -1, "error": str(exc)})

    summary_path = exp_dir / "matrix_summary.json"
    import json
    summary_path.write_text(json.dumps({
        "experiment_name": experiment_name,
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
    }, indent=2))
    print()
    print(f"Summary written to {summary_path}")
    fails = [r for r in results if r.get("returncode") != 0]
    print(f"  {len(results) - len(fails)} succeeded, {len(fails)} failed")


if __name__ == "__main__":
    main()
