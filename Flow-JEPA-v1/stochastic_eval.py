"""Paired dataset evaluation using live images for the changed geometry."""

from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np

from eval_utils import apply_callables, extract_init_goal


def cross_room_mask(dataset, valid_mask, goal_offset):
    positions = dataset.get_col_data("proprio")
    indices = np.flatnonzero(valid_mask)
    start_x = positions[indices, 0]
    goal_x = positions[indices + goal_offset, 0]
    # Keep endpoints outside the wall's collision band (112 +/- 12).
    opposite = ((start_x < 99.5) & (goal_x > 124.5)) | (
        (start_x > 124.5) & (goal_x < 99.5)
    )
    result = np.zeros_like(valid_mask)
    result[indices[opposite]] = True
    return result


def optimistic_min_steps(env, start, goal):
    """Necessary travel-time bound; not a proof a task fits in the budget."""
    if env.position_noise_std:
        return None  # Gaussian displacement has no finite speed bound.
    speed = float(env.variation_space["agent"]["speed"].value.item())
    if env.sliding:
        speed *= max(1.0, float(env.sliding_scales.max()))
    x_travel = abs(float(start[0]) - float(goal[0]))
    y_travel = []
    for i in range(env.num_doors):
        if i == env.closed_door:
            continue
        low = float(env.door_positions[i] - env.door_sizes[i]) - 1.75
        high = float(env.door_positions[i] + env.door_sizes[i]) + 1.75
        detour = max(low - max(start[1], goal[1]), min(start[1], goal[1]) - high, 0)
        y_travel.append(abs(float(start[1]) - float(goal[1])) + 2 * detour)
    same_room = (float(start[0]) < env.WALL_CENTER) == (float(goal[0]) < env.WALL_CENTER)
    vertical = abs(float(start[1]) - float(goal[1])) if same_room else min(y_travel)
    # Success needs only the radius-16 goal neighborhood, not the exact goal.
    return int(np.ceil(max(0.0, x_travel - 16.0, vertical - 16.0) / speed))


def evaluate_stochastic(
    world, dataset, episodes_idx, start_steps, goal_offset, eval_budget,
    callables, video, cfg,
):
    init, goal, dataset_videos = extract_init_goal(
        dataset, episodes_idx, start_steps, goal_offset
    )
    n = world.num_envs
    seeds = [int(cfg.stochastic.environment_seed) + i for i in range(n)]
    # Supplying state at reset makes the first rendered frame match the physics.
    options = [
        {"state": init["proprio"][i], "target_state": goal["goal_proprio"][i]}
        for i in range(n)
    ]
    world.reset(seed=seeds, options=options)
    for i, wrapped in enumerate(world.envs.envs):
        apply_callables(
            wrapped.unwrapped, callables,
            {key: value[i] for key, value in {**init, **goal}.items()},
        )
    envs = [wrapped.unwrapped for wrapped in world.envs.envs]
    # Dataset images contain one door. Render both current and goal anew.
    world.infos["pixels"] = np.stack([env.render() for env in envs])[:, None]
    world.infos["goal"] = np.stack([
        env._render_frame(agent_pos=env.target_position).numpy().transpose(1, 2, 0)
        for env in envs
    ])[:, None]
    world.infos["goal_proprio"] = goal["goal_proprio"][:, None].copy()
    snapshot = {
        key: deepcopy(value) for key, value in world.infos.items()
        if key == "goal" or key.startswith("goal_")
    }
    successes = np.zeros(n, dtype=bool)
    steps_to_success = np.full(n, -1, dtype=int)
    alive = np.ones(n, dtype=bool)
    frames = defaultdict(list) if video else None
    trajectory = [[env.agent_position.numpy().copy() for env in envs]]
    action_history, scale_history = [], []
    closed_doors = [env.closed_door for env in envs]
    minimum_steps = [
        optimistic_min_steps(env, init["proprio"][i], goal["goal_proprio"][i])
        for i, env in enumerate(envs)
    ]

    output_dir = Path(cfg.output.dir)
    from PIL import Image

    Image.fromarray(world.infos["pixels"][0, 0]).save(output_dir / "initial.png")
    for step in range(eval_budget):
        # Hidden door, applied gain and diagnostics stay outside policy infos.
        actions = world.policy.get_action(world.infos)
        _, world.rewards, world.terminateds, world.truncateds, world.infos = (
            world.envs.step(actions, mask=alive)
        )
        world.infos.update(deepcopy(snapshot))
        new_success = alive & world.terminateds
        steps_to_success[new_success] = step + 1
        successes |= new_success
        trajectory.append([env.agent_position.numpy().copy() for env in envs])
        action_history.append(np.asarray(actions).copy())
        scale_history.append([env.last_scale if alive[i] else np.nan for i, env in enumerate(envs)])
        if frames is not None:
            for i in range(n):
                frames[i].append(world.infos["pixels"][i, 0].copy())
        alive &= ~(world.terminateds | world.truncateds)
        if not alive.any():
            break

    distances = np.array([
        float((env.agent_position - env.target_position).norm()) for env in envs
    ])
    records = [
        {
            "episode_idx": int(episodes_idx[i]), "start_step": int(start_steps[i]),
            "environment_seed": seeds[i], "closed_door": closed_doors[i],
            "start": init["proprio"][i].tolist(),
            "goal": goal["goal_proprio"][i].tolist(),
            "success": bool(successes[i]), "steps_to_success": int(steps_to_success[i]),
            "final_distance": float(distances[i]), "blocked_steps": env.blocked_steps,
            "sliding_steps": env.sliding_steps, "executed_steps": env.step_count,
            "optimistic_min_steps": minimum_steps[i],
        }
        for i, env in enumerate(envs)
    ]
    np.savez_compressed(
        output_dir / "trajectories.npz", positions=np.asarray(trajectory),
        actions=np.asarray(action_history), scales=np.asarray(scale_history),
        goals=goal["goal_proprio"],
    )
    if frames:
        from stable_worldmodel.plot import save_panel_videos

        save_panel_videos(Path(video), {
            "agent": frames, "dataset": dataset_videos,
            "goal": [snapshot["goal"][i, 0] for i in range(n)],
        }, fps=15)
    return {
        "success_rate": float(successes.mean() * 100),
        "mean_final_distance": float(distances.mean()),
        "blocked_episodes": sum(env.blocked_steps > 0 for env in envs),
        "sliding_episodes": sum(env.sliding_steps > 0 for env in envs),
        "closed_door_counts": [closed_doors.count(0), closed_doors.count(1)],
        "budget_infeasible_lower_bound_episodes": sum(
            steps is not None and steps > eval_budget for steps in minimum_steps
        ),
        "episode_successes": successes.tolist(), "episodes": records,
        "seeds": seeds,
    }
