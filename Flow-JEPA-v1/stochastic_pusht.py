"""PushT interventions on physical parameters, external forces and contact torque."""

import gymnasium as gym
import numpy as np
import pymunk
from stable_worldmodel.envs.pusht.env import PushT


ENV_ID = "flow-jepa/StochasticPushT-v0"


class StochasticPushT(PushT):
    def __init__(
        self, randomize_mass=False, mass_scale_range=(0.5, 2.0),
        contact_friction=False, randomize_friction=False,
        friction_range=(0.1, 1.0), fixed_friction=0.55,
        force_std=0.0, contact_turn=False, angular_velocity_kick=5.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.randomize_mass = bool(randomize_mass)
        self.mass_scale_range = self._range(mass_scale_range, "mass_scale_range", positive=True)
        self.contact_friction = bool(contact_friction or randomize_friction)
        self.randomize_friction = bool(randomize_friction)
        self.friction_range = self._range(friction_range, "friction_range")
        self.fixed_friction = self._nonnegative(fixed_friction, "fixed_friction")
        self.force_std = self._nonnegative(force_std, "force_std")
        self.contact_turn = bool(contact_turn)
        self.angular_velocity_kick = self._nonnegative(angular_velocity_kick, "angular_velocity_kick")
        self._in_control_step = False
        self.last_force = np.zeros(2)

    @staticmethod
    def _nonnegative(value, name):
        value = float(value)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative.")
        return value

    @staticmethod
    def _range(value, name, positive=False):
        value = np.asarray(value, dtype=float)
        if value.shape != (2,) or not np.isfinite(value).all():
            raise ValueError(f"{name} must contain two finite numbers.")
        if value[0] > value[1] or value[0] < 0 or (positive and value[0] == 0):
            raise ValueError(f"Invalid {name} bounds.")
        return value

    def reset(self, seed=None, options=None):
        streams = np.random.SeedSequence(seed).spawn(4)
        mass_rng, friction_rng, self._force_rng, self._turn_rng = [
            np.random.default_rng(stream) for stream in streams
        ]
        self._in_control_step = False
        self.last_force = np.zeros(2)
        self.last_angular_kick = 0.0
        self.contact_steps = self.turn_steps = self.step_count = 0
        observation, info = super().reset(seed=seed, options=options)

        # Apply hidden parameters after reset's settling steps so all cases start alike.
        self.mass_scale = float(mass_rng.uniform(*self.mass_scale_range)) if self.randomize_mass else 1.0
        self.block.mass *= self.mass_scale
        self.block.moment *= self.mass_scale
        self.friction_coefficient = None
        if self.contact_friction:
            self.friction_coefficient = (
                float(friction_rng.uniform(*self.friction_range))
                if self.randomize_friction else self.fixed_friction
            )
            # Chipmunk multiplies the two SHAPE coefficients; Body.friction is unused.
            for shape in self.agent.shapes:
                shape.friction = 1.0
            for shape in self.block.shapes:
                shape.friction = self.friction_coefficient
        return observation, info

    def _setup(self):
        super()._setup()
        if self.force_std:
            self.block.velocity_func = self._update_block_velocity

    def _update_block_velocity(self, body, gravity, damping, dt):
        # Equivalent to F/m on this body, reapplied at all 10 physics substeps.
        force = self.last_force if self._in_control_step else np.zeros(2)
        acceleration = pymunk.Vec2d(*gravity) + pymunk.Vec2d(*force) / body.mass
        pymunk.Body.update_velocity(body, acceleration, damping, dt)

    def _handle_collision(self, arbiter, space, data):
        super()._handle_collision(arbiter, space, data)
        if not self._in_control_step:
            return
        bodies = {shape.body for shape in arbiter.shapes}
        if self.agent not in bodies or self.block not in bodies:
            return
        self._contact_this_step = True
        if self.contact_turn and not self._turn_applied:
            self.last_angular_kick = self._turn_sign * self.angular_velocity_kick
            self.block.angular_velocity += self.last_angular_kick
            self._turn_applied = True

    def step(self, action):
        self.last_force = (
            self._force_rng.normal(0.0, self.force_std, 2)
            if self.force_std else np.zeros(2)
        )
        self._turn_sign = float(self._turn_rng.choice((-1, 1))) if self.contact_turn else 0.0
        self.last_angular_kick = 0.0
        self._contact_this_step = self._turn_applied = False
        self._in_control_step = True
        try:
            result = super().step(np.asarray(action, dtype=np.float64))
        finally:
            self._in_control_step = False
        self.step_count += 1
        self.contact_steps += int(self._contact_this_step)
        self.turn_steps += int(self._turn_applied)
        return result


def register_env():
    if ENV_ID not in gym.registry:
        gym.register(ENV_ID, entry_point=StochasticPushT)
