"""Checks for effective physics changes and unchanged observable initial states."""

import unittest

import numpy as np
import pymunk
from stable_worldmodel.envs.pusht.env import PushT

from stochastic_pusht import StochasticPushT


FREE_STATE = np.array([80, 80, 256, 256, 0, 0, 0], dtype=float)
CONTACT_STATE = np.array([256, 240, 256, 256, 0, 0, 0], dtype=float)
GOAL = np.array([100, 100, 350, 350, 0.5, 0, 0], dtype=float)


class StochasticPushTTests(unittest.TestCase):
    def make_env(self, state=FREE_STATE, seed=42, **kwargs):
        env = StochasticPushT(**kwargs)
        self.addCleanup(env.close)
        env.reset(seed=seed, options={"state": state.copy(), "goal_state": GOAL.copy()})
        return env

    def test_disabled_interventions_match_base_physics(self):
        original = PushT()
        self.addCleanup(original.close)
        original.reset(seed=42, options={"state": CONTACT_STATE.copy(), "goal_state": GOAL.copy()})
        control = self.make_env(state=CONTACT_STATE)
        for action in np.random.default_rng(8).uniform(-0.2, 0.2, (20, 2)):
            original.step(action)
            control.step(action)
            np.testing.assert_array_equal(control._get_obs(), original._get_obs())
            self.assertEqual(control.block.velocity, original.block.velocity)
            self.assertEqual(control.block.angular_velocity, original.block.angular_velocity)
        np.testing.assert_array_equal(control.render(), original.render())

    def test_parameters_are_hidden_and_fixed_within_episode(self):
        clean = self.make_env()
        modified = self.make_env(randomize_mass=True, randomize_friction=True, force_std=2000, contact_turn=True)
        np.testing.assert_array_equal(clean.render(), modified.render())
        np.testing.assert_array_equal(clean._get_obs(), modified._get_obs())
        for key, value in clean._get_info().items():
            np.testing.assert_array_equal(value, modified._get_info()[key])
        mass, moment = modified.block.mass, modified.block.moment
        friction = modified.friction_coefficient
        for _ in range(8):
            modified.step([0, 0])
            self.assertEqual(modified.block.mass, mass)
            self.assertEqual(modified.block.moment, moment)
            self.assertEqual(modified.friction_coefficient, friction)

    def test_friction_updates_shapes_used_by_chipmunk(self):
        clean = self.make_env()
        modified = self.make_env(contact_friction=True, fixed_friction=0.7)
        self.assertTrue(all(s.friction == 0 for s in clean.block.shapes))
        self.assertTrue(all(s.friction == 0 for s in clean.agent.shapes))
        self.assertTrue(all(s.friction == 0.7 for s in modified.block.shapes))
        self.assertTrue(all(s.friction == 1 for s in modified.agent.shapes))

    def test_force_is_reapplied_each_substep_and_respects_mass(self):
        light = self.make_env(force_std=2000, randomize_mass=True, mass_scale_range=(1, 1))
        heavy = self.make_env(force_std=2000, randomize_mass=True, mass_scale_range=(2, 2))
        start = np.array(light.block.position)
        light.step([0, 0])
        heavy.step([0, 0])
        np.testing.assert_array_equal(light.last_force, heavy.last_force)
        self.assertGreater(np.linalg.norm(light.last_force), 0)
        n_substeps = int(1 / (light.dt * light.control_hz))
        expected = (n_substeps - 1) * light.dt ** 2 * light.last_force / light.block.mass
        np.testing.assert_allclose(np.array(light.block.position) - start, expected, atol=1e-10)
        np.testing.assert_allclose(np.array(heavy.block.position) - start, expected / 2, atol=1e-10)
        self.assertEqual(heavy.block.moment / light.block.moment, 2)

    def test_turn_kicks_require_agent_block_contact(self):
        free = self.make_env(contact_turn=True)
        free.step([0, 0])
        self.assertEqual(free.turn_steps, 0)
        signs = set()
        for seed in range(8):
            env = self.make_env(state=CONTACT_STATE, seed=seed, contact_turn=True)
            for _ in range(4):
                previous = env.turn_steps
                env.step([0, 0.1])
                self.assertLessEqual(env.turn_steps - previous, 1)
                if env.last_angular_kick:
                    self.assertEqual(abs(env.last_angular_kick), 5)
                    signs.add(np.sign(env.last_angular_kick))
            self.assertGreater(env.turn_steps, 0)
            self.assertLessEqual(env.turn_steps, env.contact_steps)
        self.assertEqual(signs, {-1, 1})

    def test_wall_contacts_do_not_trigger_turn_kicks(self):
        env = self.make_env(contact_turn=True)
        env._contact_this_step = env._turn_applied = False
        env._in_control_step = True
        class Arbiter:
            shapes = (next(iter(env.block.shapes)), pymunk.Segment(env.space.static_body, (0, 0), (1, 1), 1))
            contact_point_set = type("Points", (), {"points": [None]})()
        env._handle_collision(Arbiter(), env.space, {})
        env._in_control_step = False
        self.assertFalse(env._turn_applied)
        self.assertFalse(env._contact_this_step)

    def test_seed_reproduces_force_turn_and_state_trajectories(self):
        a, b = [self.make_env(state=CONTACT_STATE, randomize_mass=True, randomize_friction=True, force_std=2000, contact_turn=True) for _ in range(2)]
        for _ in range(12):
            a.step([0, 0.1])
            b.step([0, 0.1])
            np.testing.assert_array_equal(a._get_obs(), b._get_obs())
            np.testing.assert_array_equal(a.last_force, b.last_force)
            self.assertEqual(a.last_angular_kick, b.last_angular_kick)


if __name__ == "__main__":
    unittest.main()
