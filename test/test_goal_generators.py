import unittest
import envs
import gymnasium

from envs.goal_generators import goal_generator_factory


class TestGoalGenerators(unittest.TestCase):
    _generator_config = {
        "random": {},
        "fixed": {"goal": (2, 2)},
        "choice": {"goals": [(2, 2), (2, 3), (3, 2)]},
        "categorical": {"goals": [(2, 2), (2, 3), (3, 2)], "logits": [0.5, 0.3, 0.2]}
    }

    def _test_returned_free_space(self, gen_mode: str, **kwargs):
        """
        Test that the goal returned by the generator is free space.

        :param gen_mode: generator mode as a string
        :return:
        """
        env = gymnasium.make("one-door-2-agents-v0")
        gen = goal_generator_factory(gen_mode, **kwargs)

        n_tries = 10
        for _ in range(n_tries):
            goal = gen(env)
            self.assertTrue(type(goal) == tuple and len(goal) == 2, f"goal must be a tuple of length 2, got{goal}")
            self.assertTrue(env.grid.get(*goal) is None, f"the goal {goal} is not free space")

    def test_generator_modes(self):
        for gen_mode, params in self._generator_config.items():
            self._test_returned_free_space(gen_mode, **params)

