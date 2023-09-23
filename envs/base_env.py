from __future__ import annotations

from abc import abstractmethod

import gymnasium
import numpy as np

from gym_multigrid.multigrid import MultiGridEnv, World
from gym_multigrid.world_objects import Agent, DIR_TO_VEC, TILE_PIXELS


class SimpleEnv(MultiGridEnv):
    def __init__(
        self,
        width: int = 15,
        height: int = 7,
        num_agents: int = 2,
        view_size: int = 7,
        max_steps: int = 1000,
        tile_size: int = 25,
        render_mode: str = None,
        render_fps: int = None,
        goal_terminates: bool = True,
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
            tile_size=tile_size,
            goal_terminates=goal_terminates,
        )
        self.carrying = None

        original_action_space = self.action_space
        self.action_space = gymnasium.spaces.Dict(
            {f"agent_{i}": original_action_space for i in range(num_agents)}
        )

        original_observation_space = self.observation_space
        original_obs_as_dict = gymnasium.spaces.Dict(
            {
                "grid": original_observation_space,
            }
        )
        self.observation_space = gymnasium.spaces.Dict(
            {f"agent_{i}": original_obs_as_dict for i in range(num_agents)}
        )

        if render_fps is not None:
            self.metadata["render_fps"] = render_fps

    def step(self, actions):
        # convert from dict to list
        actions = [actions[f"agent_{i}"] for i in range(self.num_agents)]

        obs, reward, done, truncated, info = super().step(actions)

        # convert from list to dict
        obs = {f"agent_{i}": {"grid": obs[i]} for i in range(self.num_agents)}

        # expand info, setting success if the agent has reached the goal (reward > 0)
        info.update(
            {
                f"agent_{i}": {
                    "success": reward[i] > 0,
                    "goal": self.goals[i],
                    "pos": self.agents[i].pos,
                }
                for i in range(self.num_agents)
            }
        )

        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        # convert from list to dict
        obs = {f"agent_{i}": {"grid": obs[i]} for i in range(self.num_agents)}

        # initialize info with success to False
        info.update(
            {
                f"agent_{i}": {
                    "success": False,
                    "goal": self.goals[i],
                    "pos": self.agents[i].pos,
                }
                for i in range(self.num_agents)
            }
        )

        return obs, info

    @abstractmethod
    def _gen_grid(self, width, height):
        raise NotImplementedError
