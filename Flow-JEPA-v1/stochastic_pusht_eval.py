"""Evaluate PushT on matched dataset tasks without modifying the training dataset."""

from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image

from eval_utils import extract_init_goal


def state_errors(state, goal):
    pose_error = float(np.linalg.norm(state[:4] - goal[:4]))
    angle_error = float(abs(np.arctan2(np.sin(state[4] - goal[4]), np.cos(state[4] - goal[4]))))
    return pose_error, angle_error


def evaluate_stochastic_pusht(
    world, dataset, episodes_idx, start_steps, goal_offset, eval_budget,
    callables, video, cfg,
):
    init, goal, dataset_videos = extract_init_goal(dataset, episodes_idx, start_steps, goal_offset)
    n = world.num_envs
    seeds = [int(cfg.stochastic.environment_seed) + i for i in range(n)]
    if any(spec["method"] not in ("_set_state", "_set_goal_state") for spec in callables or []):
        raise ValueError("PushT state and goal setup is performed through reset options.")
    world.reset(seed=seeds, options=[
        {"state": init["state"][i], "goal_state": goal["goal_state"][i]}
        for i in range(n)
    ])
    envs = [wrapped.unwrapped for wrapped in world.envs.envs]
    # Use the actual reset observations, including any inherited settling displacement.
    initial_states = np.stack([env._get_obs() for env in envs])
    snapshot = {
        key: deepcopy(value) for key, value in world.infos.items()
        if key == "goal" or key.startswith("goal_")
    }
    output = Path(cfg.output.dir)
    Image.fromarray(world.infos["pixels"][0, 0]).save(output / "initial.png")
    Image.fromarray(world.infos["goal"][0, 0]).save(output / "goal.png")
    successes = np.zeros(n, dtype=bool)
    success_steps = np.full(n, -1, dtype=int)
    alive = np.ones(n, dtype=bool)
    states = [initial_states.copy()]
    block_velocities = [np.array([[*env.block.velocity, env.block.angular_velocity] for env in envs])]
    actions, forces, kicks = [], [], []
    frames = defaultdict(list) if video else None
    for step in range(eval_budget):
        # Physics parameters and intervention traces are never added to policy infos.
        action = world.policy.get_action(world.infos)
        _, world.rewards, world.terminateds, world.truncateds, world.infos = world.envs.step(action, mask=alive)
        world.infos.update(deepcopy(snapshot))
        success = alive & world.terminateds
        successes |= success
        success_steps[success] = step + 1
        states.append(np.stack([env._get_obs() for env in envs]))
        block_velocities.append(np.array([[*env.block.velocity, env.block.angular_velocity] for env in envs]))
        actions.append(np.asarray(action).copy())
        forces.append([env.last_force.copy() if alive[i] else np.full(2, np.nan) for i, env in enumerate(envs)])
        kicks.append([env.last_angular_kick if alive[i] else np.nan for i, env in enumerate(envs)])
        if frames is not None:
            for i in range(n):
                frames[i].append(world.infos["pixels"][i, 0].copy())
        alive &= ~(world.terminateds | world.truncateds)
        if not alive.any():
            break

    records = []
    for i, env in enumerate(envs):
        state = env._get_obs()
        pose_error, angle_error = state_errors(state, env.goal_state)
        records.append({
            "episode_idx": int(episodes_idx[i]), "start_step": int(start_steps[i]),
            "environment_seed": seeds[i], "requested_start": init["state"][i].tolist(),
            "start": initial_states[i].tolist(), "goal": env.goal_state.tolist(),
            "success": bool(successes[i]), "steps_to_success": int(success_steps[i]),
            "executed_steps": env.step_count, "contact_steps": env.contact_steps,
            "turn_steps": env.turn_steps, "mass_scale": env.mass_scale,
            "mass": float(env.block.mass), "moment": float(env.block.moment),
            "contact_friction": env.friction_coefficient,
            "block_shape_frictions": [float(shape.friction) for shape in env.block.shapes],
            "agent_shape_frictions": [float(shape.friction) for shape in env.agent.shapes],
            "final_pose_error": pose_error, "final_angle_error": angle_error,
        })
    np.savez_compressed(
        output / "trajectories.npz", states=np.asarray(states),
        block_velocities=np.asarray(block_velocities), actions=np.asarray(actions),
        forces=np.asarray(forces), angular_kicks=np.asarray(kicks),
        goals=np.stack([env.goal_state for env in envs]),
    )
    if frames:
        from stable_worldmodel.plot import save_panel_videos

        save_panel_videos(Path(video), {
            "agent": frames, "dataset": dataset_videos,
            "goal": [snapshot["goal"][i, 0] for i in range(n)],
        }, fps=15)
    return {
        "success_rate": float(successes.mean() * 100),
        "episode_successes": successes.tolist(), "episodes": records, "seeds": seeds,
        "mean_final_pose_error": float(np.mean([r["final_pose_error"] for r in records])),
        "mean_final_angle_error": float(np.mean([r["final_angle_error"] for r in records])),
        "contact_episodes": sum(env.contact_steps > 0 for env in envs),
        "turn_episodes": sum(env.turn_steps > 0 for env in envs),
    }
