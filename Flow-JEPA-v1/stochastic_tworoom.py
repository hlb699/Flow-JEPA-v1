"""Two-Room dynamics interventions; hidden variables never enter observations."""

import gymnasium as gym
import numpy as np
import torch
from stable_worldmodel.envs.two_room.env import TwoRoomEnv


ENV_ID = "flow-jepa/StochasticTwoRoom-v0"


class StochasticTwoRoomEnv(TwoRoomEnv):
    def __init__(
        self,
        two_doors=True,
        hidden_door=False,
        forced_closed_door=None,
        position_noise_std=0.0,
        sliding=False,
        sliding_bounds=(28.0, 28.0, 98.0, 196.0),
        sliding_scales=(0.5, 1.5),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.two_doors = bool(two_doors)
        self.hidden_door = bool(hidden_door)
        self.forced_closed_door = forced_closed_door
        self.position_noise_std = float(position_noise_std)
        self.sliding = bool(sliding)
        self.sliding_bounds = np.asarray(sliding_bounds, dtype=float)
        self.sliding_scales = np.asarray(sliding_scales, dtype=float)
        if self.hidden_door and not self.two_doors:
            raise ValueError("Hidden-door intervention requires two doors.")
        if forced_closed_door not in (None, 0, 1):
            raise ValueError("forced_closed_door must be null, 0, or 1.")
        if not np.isfinite(self.position_noise_std) or self.position_noise_std < 0:
            raise ValueError("position_noise_std must be finite and non-negative.")
        if self.sliding_bounds.shape != (4,) or not np.isfinite(self.sliding_bounds).all():
            raise ValueError("sliding_bounds must be [xmin, ymin, xmax, ymax].")
        if np.any(self.sliding_bounds[:2] >= self.sliding_bounds[2:]):
            raise ValueError("Sliding region must have positive width and height.")
        if self.sliding_scales.shape != (2,) or not np.isfinite(self.sliding_scales).all():
            raise ValueError("Exactly two finite sliding scales are required.")
        if np.any(self.sliding_scales <= 0):
            raise ValueError("Sliding scales must be positive.")
        self.closed_door = None

    def _cache_params(self):
        super()._cache_params()
        if self.two_doors:
            self.num_doors = 2
            self.door_positions = torch.tensor([49.0, 175.0])
            self.door_sizes = torch.tensor([14.0, 14.0])

    def reset(self, seed=None, options=None):
        door_seed, noise_seed, sliding_seed = np.random.SeedSequence(seed).spawn(3)
        self._noise_rng = np.random.default_rng(noise_seed)
        self._sliding_rng = np.random.default_rng(sliding_seed)
        self.closed_door = None
        if self.hidden_door:
            self.closed_door = (
                int(np.random.default_rng(door_seed).integers(2))
                if self.forced_closed_door is None
                else int(self.forced_closed_door)
            )
        self.blocked_steps = 0
        self.sliding_steps = 0
        self.step_count = 0
        self.last_scale = 1.0
        return super().reset(seed=seed, options=options)

    def _in_any_door_1d(self, coord_1d, margin):
        for index in range(self.num_doors):
            if index == self.closed_door:
                continue
            center = float(self.door_positions[index])
            size = float(self.door_sizes[index])
            if center - size - margin <= coord_1d <= center + size + margin:
                return True
        return False

    def _apply_collisions(self, pos1, pos2):
        self.last_scale = 1.0
        position = pos1.detach().cpu().numpy()
        if self.sliding and np.all(position >= self.sliding_bounds[:2]) and np.all(
            position <= self.sliding_bounds[2:]
        ):
            self.last_scale = float(self._sliding_rng.choice(self.sliding_scales))
            self.sliding_steps += 1
        proposal = pos1 + (pos2 - pos1) * self.last_scale
        if self.position_noise_std:
            proposal = proposal + torch.tensor(
                self._noise_rng.normal(0, self.position_noise_std, 2),
                dtype=pos1.dtype,
            )
        result = super()._apply_collisions(pos1, proposal)
        if self.closed_door is not None:
            closed = self.closed_door
            # Compare physics with both doors open, without changing rendering.
            self.closed_door = None
            try:
                open_result = super()._apply_collisions(pos1, proposal)
            finally:
                self.closed_door = closed
            self.blocked_steps += int(not torch.equal(result, open_result))
        self.step_count += 1
        return result


def register_env():
    if ENV_ID not in gym.registry:
        gym.register(ENV_ID, entry_point=StochasticTwoRoomEnv)
