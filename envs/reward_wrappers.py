from abc import abstractmethod
from typing import Callable

import gymnasium
import numpy as np

from gym_multigrid.world_objects import DIR_TO_VEC


class RewardFn(Callable):

    def __init__(self, env: gymnasium.Env):
        self.env = env

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

    def __init__(self, env, build_reward: Callable[[gymnasium.Env], RewardFn]):
        super().__init__(env)
        self.reward_fn = build_reward(env)

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

    def __init__(self, env: gymnasium.Env):
        super().__init__(env)

        self.num_agents = env.num_agents

        # precompute choices matrices for open doors
        self.choices_grid = self._precompute_choices_grid()
        self.choices_scale = 1 / np.max(self.choices_grid)  # used for scaling the reward in (0, 1)

    def _precompute_choices_grid(self):
        # iterate over all possible pose (x, y, dir) and env conditions (carrying, open_door)
        #   discard positions that are not valid (e.g. walls)
        #   count the possible actions that can be taken from that position
        choices = np.zeros(
            (self.env.width, self.env.height, 4, 2, 2), dtype=np.int32
        )

        doors = [
            obj for obj in self.env.grid.grid if obj is not None and obj.type == "door"
        ]
        assert len(doors) == 1, "Only one door is supported for now"

        # state validity
        is_valid = (
            lambda x, y, dir, carrying, open_door: self.env.grid.get(x, y) is None
            or self.env.grid.get(x, y).can_overlap()
            or self.env.grid.get(x, y).type == "agent"
        )

        for iod, open_door in enumerate([True, False]):
            for x in range(self.env.width):
                for y in range(self.env.height):
                    for dir in range(4):
                        for ic, carrying in enumerate([False, True]):
                            state = (x, y, dir, carrying, open_door)
                            if not is_valid(*state):
                                continue
                            for action in range(len(self.env.actions.available)):
                                if self._check_action(state, action):
                                    choices[x, y, dir, ic, iod] += 1

        return choices

    def _check_action(self, state: tuple, action):
        """
        Check if in the given pose (x, y, dir) the given action is valid.

        Note: this is implemented for MediumActions and do not consider "drop", "done", ...
        """
        x, y, dir, carrying, open_door = state  # unpack state
        fwd_pos = np.array((x, y)) + DIR_TO_VEC[dir]
        fwd_cell = self.env.grid.get(*fwd_pos)

        if action in [self.env.actions.still, self.env.actions.left, self.env.actions.right]:
            return True
        if (
            action == self.env.actions.forward
            and fwd_cell
            and (fwd_cell.can_overlap() or (fwd_cell.type == "door" and open_door))
        ):
            return True
        if (
            not carrying
            and action == self.env.actions.pickup
            and fwd_cell
            and fwd_cell.can_pickup()
        ):
            return True
        if (
            carrying
            and action == self.env.actions.toggle
            and fwd_cell
            and fwd_cell.type == "door"
        ):
            return True

        return False

    def _get_choices(self, obs) -> dict[str, int]:
        # compute choice as in "Learning Altruistic Behaviors"
        choices = - np.ones(self.num_agents, dtype=np.float32)
        for i, agent in enumerate(self.env.agents):
            doors = [
                obj for obj in self.env.grid.grid if obj is not None and obj.type == "door"
            ]
            assert len(doors) == 1, "Only one door is supported for now"
            # for multi-doors, we shold consider the closest one
            # just check if fwd_cell is a door?
            is_open = doors[0].is_open if len(doors) > 0 else True
            x, y, dir, carrying = agent.pos[0], agent.pos[1], agent.dir, agent.carrying is not None
            choices[i] = self.choices_grid[
                x, y, dir, int(carrying), int(is_open)
            ]

        # then only things to consider is if any other agent is in front, then we should put choices -1
        for i, agent in enumerate(self.env.agents):
            front_pos = self.env.agents[i].front_pos
            if self.env.grid.get(*front_pos) and self.env.grid.get(*front_pos).type == "agent":
                choices[i] -= 1

        # convert it to dict agent_id -> choice
        choices = {agent_id: choice for agent_id, choice in zip(obs.keys(), choices)}
        return choices

    def __call__(self, obs, action, done, info=None, next_obs=None) -> float | list[float]:
        obs = next_obs if next_obs is not None else obs  # if given next obs, compute reward on it

        assert isinstance(obs, dict) and len(obs) >= 2, "AltruisticRewardFn requires at least two agents in dict obs"
        choices = self._get_choices(obs)

        # for each agent, returns the reward as sum of choices of others
        rewards = []
        for agent_id in obs:
            other_choices = [choices[other_agent] for other_agent in obs if other_agent != agent_id]
            reward = sum(other_choices) * self.choices_scale
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

def reward_fn_factory(reward: str) -> RewardFn:
    if reward == "sparse":
        return SparseRewardFn()
    elif reward == "altruistic":
        return AltruisticRewardFn()
    elif reward == "neg_distance":
        return NegativeRewardFn()
    else:
        raise NotImplementedError(f"Reward function {reward} not implemented")


