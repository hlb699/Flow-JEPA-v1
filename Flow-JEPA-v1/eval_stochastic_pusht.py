"""Evaluate a trained PushT checkpoint under isolated and combined interventions."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
from queue import Queue
import subprocess
import sys

from eval_stochastic_tworoom import wilson_interval


ROOT = Path(__file__).resolve().parent
SCENARIOS = {
    "clean": [],
    "friction_reference": ["world.contact_friction=true"],
    "mass_only": ["world.randomize_mass=true"],
    "friction_only": ["world.randomize_friction=true"],
    "mass_friction": ["world.randomize_mass=true", "world.randomize_friction=true"],
    "random_force": ["world.force_std=2000.0"],
    "contact_turn": ["world.contact_turn=true"],
    "combined": ["world.randomize_mass=true", "world.randomize_friction=true", "world.force_std=2000.0", "world.contact_turn=true"],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "stable-wm/checkpoints/flow-jepa-pusht-paper-1gpu-strict/weights_epoch_20.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "logs/eval_results/pusht_stochastic_seed42")
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cem-steps", type=int, default=10)
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--random-policy", action="store_true")
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    if not args.random_policy and not checkpoint.is_file():
        parser.error(f"Missing checkpoint: {checkpoint}")
    if args.num_eval <= 0 or args.budget < 25 or args.cem_steps <= 0:
        parser.error("num-eval and cem-steps must be positive; budget must be >= 25")
    if len(set(args.gpus)) != len(args.gpus):
        parser.error("GPU ids must be unique")
    if len(set(args.scenarios)) != len(args.scenarios):
        parser.error("Scenario names must be unique")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in args.scenarios:
        if (output / name / "metrics.json").exists():
            parser.error(f"Result exists for {name}; choose a new --output")
    available = Queue()
    for device in args.gpus:
        available.put(device)

    def evaluate(name):
        device = available.get()
        try:
            directory = output / name
            directory.mkdir(exist_ok=True)
            env = dict(os.environ)
            env.update(STABLEWM_HOME=str(ROOT / "stable-wm"), LOCAL_DATASET_DIR=str(ROOT / "stable-wm"), PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES=device, SDL_VIDEODRIVER="dummy")
            command = [sys.executable, str(ROOT / "eval.py"), "--config-name=pusht_stochastic",
                       f"policy={'random' if args.random_policy else checkpoint}",
                       f"seed={args.seed}", f"eval.num_eval={args.num_eval}",
                       f"solver.n_steps={args.cem_steps}", f"eval.eval_budget={args.budget}",
                       f"output.dir={directory}", *SCENARIOS[name]]
            print(f"Running {name} on GPU {device}; log: {directory / 'run.log'}", flush=True)
            with (directory / "run.log").open("w") as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            result = json.loads((directory / "metrics.json").read_text())
            result["execution_gpu"] = device
            print(f"{name}: {result['metrics']['success_rate']:.1f}%", flush=True)
            return name, result
        finally:
            available.put(device)

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        results = dict(executor.map(evaluate, args.scenarios))
    reference = results.get("clean", next(iter(results.values())))["metrics"]["episodes"]
    rows = []
    for name, result in results.items():
        m = result["metrics"]
        keys = ("episode_idx", "start_step", "environment_seed", "requested_start", "start", "goal")
        if any(any(a[k] != b[k] for k in keys) for a, b in zip(reference, m["episodes"])):
            raise RuntimeError(f"Unmatched initial tasks in {name}")
        count = sum(m["episode_successes"])
        low, high = wilson_interval(count, args.num_eval)
        row = dict(scenario=name, successes=count, num_eval=args.num_eval, success_rate=m["success_rate"],
                   ci95_low=low, ci95_high=high, mean_final_pose_error=m["mean_final_pose_error"],
                   mean_final_angle_error=m["mean_final_angle_error"], contact_episodes=m["contact_episodes"], turn_episodes=m["turn_episodes"])
        for baseline in ("clean", "friction_reference"):
            if baseline in results:
                row[f"delta_vs_{baseline}_pp"] = m["success_rate"] - results[baseline]["metrics"]["success_rate"]
        rows.append(row)
    with (output / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    manifest = {"checkpoint": str(checkpoint),
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest() if not args.random_policy else None,
                "seed": args.seed, "summary": rows, "results": results}
    (output / "summary.json").write_text(json.dumps(manifest, indent=2))
    print(f"Summary: {output / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
