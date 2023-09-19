import unittest



class TestEnv(unittest.TestCase):

    def test_env_api(self):
        import gymnasium as gym
        from gymnasium.utils.env_checker import check_env
        import envs

        env = gym.make("door-2-agents-v0")
        check_env(env)

        self.assertTrue(True, "test api failed")