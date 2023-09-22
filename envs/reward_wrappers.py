from abc import abstractmethod
from typing import Callable

import gymnasium


class RewardFn(Callable):

    @abstractmethod
    def __call__(self, obs, action, done, info=None, next_obs=None) -> float | list[float]:
        raise NotImplementedError


class RewardWrapper(gymnasium.Wrapper):
    """
    Wrappers that modify the reward using a transition-based reward function (ie. R: S x A x S -> R)

    Args:
        env (gym.Env): environment to wrap
        reward_fn (RewardFn): reward function
    """

    def __init__(self, env, reward_fn: RewardFn):
        super().__init__(env)
        self.reward_fn = reward_fn

        self.last_obs = None

    def reset(self, **kwargs):
        self.last_obs, info = self.env.reset(**kwargs)
        return self.last_obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        reward = self.reward_fn(self.last_obs, action, done, info, obs)
        self.last_obs = obs
        return obs, reward, done, truncated, info


class SparseRewardFn(RewardFn):

    def __call__(self, obs, action, done, info=None, next_obs=None) -> float | list[float]:
        obs = next_obs if next_obs is not None else obs # if given next obs, compute reward on it

        assert isinstance(obs, dict), "SparseRewardFn requires dict obs"
        assert all(["grid" in obs[agent_id] for agent_id in obs]), "SparseRewardFn requires 'grid' in obs"

        # for each agent, returns a sparse reward for reaching the goal
        rewards = [float(info[agent_id]["success"]) for agent_id in obs]

        return rewards

class AltruisticRewardFn(RewardFn):

    def __call__(self, obs, action, done, info=None, next_obs=None) -> float | list[float]:
        obs = next_obs if next_obs is not None else obs  # if given next obs, compute reward on it

        assert isinstance(obs, dict) and len(obs) >= 2, "AltruisticRewardFn requires at least two agents in dict obs"
        assert all(["choice" in obs[agent_id] for agent_id in obs]), "AltruisticRewardFn requires 'choice' in obs"

        # for each agent, returns the reward as sum of choices of others
        rewards = []
        for agent_id in obs:
            other_choices = [obs[other_agent]["choice"] for other_agent in obs if other_agent != agent_id]
            reward = sum(other_choices)
            rewards.append(reward)

        return rewards


class NegativeRewardFn(RewardFn):

    def __call__(self, obs, action, done, info=None, next_obs=None) -> float | list[float]:
        obs = next_obs if next_obs is not None else obs  # if given next obs, compute reward on it

        assert isinstance(obs, dict) and isinstance(info, dict), "NegativeDistanceReward requires dict obs and info"
        assert all(["pos" in info[agent_id] for agent_id in obs]), "NegativeDistanceReward requires 'pos' in info"
        assert all(["goal" in info[agent_id] for agent_id in obs]), "NegativeDistanceReward requires 'goal' in info"

        scale = 0.1

        # for each agent, returns the reward as sum of choices of others
        rewards = []
        for agent_id in obs:
            if info[agent_id]["goal"] is None:
                reward = 0
            else:
                manhattan_dist = sum(abs(info[agent_id]["pos"] - info[agent_id]["goal"]))
                reward = -manhattan_dist
            rewards.append(reward * scale)

        return rewards



