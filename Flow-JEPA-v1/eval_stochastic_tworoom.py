"""Run the existing checkpoint on matched stochastic Two-Room scenarios."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
SCENARIOS = {
    "original_geometry": ["world.two_doors=false", "world.hidden_door=false"],
    "two_doors_open": ["world.hidden_door=false"],
    "hidden_door": [],
    "position_noise": ["world.hidden_door=false", "world.position_noise_std=2.0"],
    "sliding": ["world.hidden_door=false", "world.sliding=true"],
    "combined": ["world.position_noise_std=2.0", "world.sliding=true"],
}


def wilson_interval(successes, n):
    p, z = successes / n, 1.96
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [100 * (center - radius), 100 * (center + radius)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "stable-wm/checkpoints/flow-jepa-tworoom-paper-4gpu/weights_epoch_20.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "logs/eval_results/tworoom_stochastic_seed42")
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cem-steps", type=int, default=30)
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--random-policy", action="store_true", help="Smoke test without loading a checkpoint")
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    if not args.random_policy and not checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {checkpoint}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(STABLEWM_HOME=str(ROOT / "stable-wm"), LOCAL_DATASET_DIR=str(ROOT / "stable-wm"), PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    results = {}
    for name in args.scenarios:
        directory = output / name
        directory.mkdir(exist_ok=True)
        if (directory / "metrics.json").exists():
            parser.error(f"Output already contains a result: {directory}; use a new --output")
        command = [
            sys.executable, str(ROOT / "eval.py"), "--config-name=tworoom_stochastic",
            f"policy={'random' if args.random_policy else checkpoint}",
            f"seed={args.seed}", f"eval.num_eval={args.num_eval}",
            f"solver.n_steps={args.cem_steps}", f"eval.eval_budget={args.budget}",
            f"output.dir={directory}", *SCENARIOS[name],
        ]
        print(f"Running {name}; log: {directory / 'run.log'}", flush=True)
        with (directory / "run.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads((directory / "metrics.json").read_text())
        results[name] = result
        print(f"{name}: {result['metrics']['success_rate']:.1f}%", flush=True)

    baseline = results.get("two_doors_open")
    rows = []
    for name, result in results.items():
        metrics = result["metrics"]
        successes = sum(metrics["episode_successes"])
        low, high = wilson_interval(successes, args.num_eval)
        row = dict(scenario=name, successes=successes, num_eval=args.num_eval,
                   success_rate=metrics["success_rate"], ci95_low=low, ci95_high=high,
                   mean_final_distance=metrics["mean_final_distance"],
                   blocked_episodes=metrics["blocked_episodes"],
                   sliding_episodes=metrics["sliding_episodes"],
                   budget_infeasible_lower_bound_episodes=metrics["budget_infeasible_lower_bound_episodes"])
        if baseline:
            reference = baseline["metrics"]["episodes"]
            paired = metrics["episodes"]
            keys = ("episode_idx", "start_step", "environment_seed", "start", "goal")
            assert all(all(a[k] == b[k] for k in keys) for a, b in zip(reference, paired))
            row["delta_vs_open_pp"] = metrics["success_rate"] - baseline["metrics"]["success_rate"]
        rows.append(row)
    with (output / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest() if not args.random_policy else None,
        "seed": args.seed, "cross_room_only": True,
        "summary": rows, "results": results,
    }
    (output / "summary.json").write_text(json.dumps(manifest, indent=2))
    print(f"Summary: {output / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
