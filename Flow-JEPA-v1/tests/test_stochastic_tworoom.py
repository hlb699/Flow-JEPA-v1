"""Physics, hidden-state and reproducibility checks for stochastic Two-Room."""

import unittest

import numpy as np
import torch

from stochastic_tworoom import StochasticTwoRoomEnv
from stochastic_eval import optimistic_min_steps
from stable_worldmodel.envs.two_room.env import TwoRoomEnv


class StochasticTwoRoomTests(unittest.TestCase):
    def make_env(self, **kwargs):
        env = StochasticTwoRoomEnv(**kwargs)
        self.addCleanup(env.close)
        env.reset(seed=42, options={"state": [60, 49], "target_state": [164, 49]})
        return env

    def test_hidden_branches_have_identical_observations_and_different_futures(self):
        branches = [self.make_env(hidden_door=True, forced_closed_door=i) for i in (0, 1)]
        np.testing.assert_array_equal(branches[0].render(), branches[1].render())
        torch.testing.assert_close(branches[0]._get_obs(), branches[1]._get_obs())
        for key, value in branches[0]._get_info().items():
            np.testing.assert_array_equal(value, branches[1]._get_info()[key])
        self.assertNotIn("closed_door", branches[0]._get_info())
        for env in branches:
            env._set_state([99.5, 49])
            for _ in range(6):
                env.step(np.array([1, 0], dtype=np.float32))
        self.assertLess(float(branches[0].agent_position[0]), 100)
        self.assertGreater(float(branches[1].agent_position[0]), 124)
        self.assertEqual(branches[0].blocked_steps, 6)

    def test_exactly_one_door_remains_passable_and_closure_is_fixed(self):
        env = self.make_env(hidden_door=True)
        closed = env.closed_door
        for _ in range(10):
            env.step([0, 0])
            self.assertEqual(env.closed_door, closed)
        for index, y in enumerate((49, 175)):
            env._set_state([99.5, y])
            for _ in range(6):
                env.step([1, 0])
            self.assertEqual(float(env.agent_position[0]) > 124, index != closed)

    def test_noise_and_sliding_repeat_with_seed_and_preserve_walls(self):
        a, b = [self.make_env(hidden_door=True, position_noise_std=2, sliding=True) for _ in range(2)]
        for _ in range(30):
            a.step([0.7, 0.1])
            b.step([0.7, 0.1])
            torch.testing.assert_close(a.agent_position, b.agent_position)
            self.assertEqual(a.last_scale, b.last_scale)
        self.assertGreater(a.sliding_steps, 0)
        self.assertTrue(torch.all(a.agent_position >= 21))
        self.assertTrue(torch.all(a.agent_position <= 203))

    def test_sliding_gain_only_applies_inside_region(self):
        env = self.make_env(sliding=True)
        scales = set()
        for _ in range(40):
            env._set_state([60, 49])
            env.step([1, 0])
            scales.add(env.last_scale)
            self.assertAlmostEqual(float(env.agent_position[0]), 60 + 5 * env.last_scale)
        self.assertEqual(scales, {0.5, 1.5})
        env._set_state([160, 49])
        env.step([1, 0])
        self.assertEqual(env.last_scale, 1)
        self.assertEqual(float(env.agent_position[0]), 165)

    def test_position_noise_changes_true_state_and_image(self):
        clean = self.make_env()
        noisy = self.make_env(position_noise_std=2)
        clean.step([0, 0])
        noisy.step([0, 0])
        self.assertFalse(torch.equal(clean.agent_position, noisy.agent_position))
        np.testing.assert_array_equal(noisy._get_info()["proprio"], noisy.agent_position.numpy())
        self.assertFalse(np.array_equal(clean.render(), noisy.render()))

    def test_disabled_interventions_match_original_environment(self):
        original = TwoRoomEnv()
        self.addCleanup(original.close)
        original.reset(seed=42, options={"state": [60, 49], "target_state": [164, 49]})
        control = self.make_env(two_doors=False)
        np.testing.assert_array_equal(control.render(), original.render())
        for action in np.random.default_rng(1).uniform(-1, 1, (50, 2)):
            control.step(action)
            original.step(action)
            torch.testing.assert_close(control.agent_position, original.agent_position)

    def test_travel_bound_allows_goal_tolerance_and_omits_unbounded_noise(self):
        env = self.make_env()
        self.assertEqual(optimistic_min_steps(env, [60, 49], [70, 49]), 0)
        self.assertEqual(optimistic_min_steps(env, [60, 49], [80, 49]), 1)
        blocked = self.make_env(hidden_door=True, forced_closed_door=0)
        self.assertGreater(
            optimistic_min_steps(blocked, [60, 49], [164, 49]),
            optimistic_min_steps(env, [60, 49], [164, 49]),
        )
        noisy = self.make_env(position_noise_std=2)
        self.assertIsNone(optimistic_min_steps(noisy, [60, 49], [164, 49]))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
