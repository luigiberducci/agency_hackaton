from __future__ import annotations

from abc import abstractmethod

import gymnasium

from gym_multigrid.multigrid import (
    MultiGridEnv,
    World,
    Agent,
    Goal,
    COLOR_NAMES,
    Key,
    Door,
    Wall,
    Grid,
    Ball,
)


class SimpleEnv(MultiGridEnv):
    def __init__(
        self,
        width: int = 15,
        height: int = 7,
        num_agents: int = 2,
        view_size: int = 7,
        max_steps: int = 1000,
        render_mode: str = None,
        **kwargs,
    ):
        self.num_agents = num_agents
        self.goals = []

        self.world = World

        agents = []
        for i in range(num_agents):
            agent = Agent(self.world, i, view_size=view_size)
            agents.append(agent)

        super().__init__(
            grid_size=None,
            width=width,
            height=height,
            max_steps=max_steps,
            see_through_walls=False,  # Set this to True for maximum speed
            agents=agents,
            agent_view_size=view_size,
            render_mode=render_mode,
        )
        self.carrying = None

        original_action_space = self.action_space
        self.action_space = gymnasium.spaces.Dict(
            {f"agent_{i}": original_action_space for i in range(num_agents)}
        )

        original_observation_space = self.observation_space
        self.observation_space = gymnasium.spaces.Dict(
            {f"agent_{i}": original_observation_space for i in range(num_agents)}
        )

        self.render_mode = render_mode

    def step(self, actions):
        # convert from dict to list
        actions = [actions[f"agent_{i}"] for i in range(self.num_agents)]

        obs, reward, done, truncated, info = super().step(actions)

        # convert from list to dict
        obs = {f"agent_{i}": obs[i] for i in range(self.num_agents)}

        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        # convert from list to dict
        obs = {f"agent_{i}": obs[i] for i in range(self.num_agents)}

        return obs, info

    @abstractmethod
    def _gen_grid(self, width, height):
        raise NotImplementedError
