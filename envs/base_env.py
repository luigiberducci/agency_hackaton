from __future__ import annotations

from abc import abstractmethod

import gymnasium
import numpy as np

from gym_multigrid.multigrid import MultiGridEnv, World
from gym_multigrid.world_objects import Agent, DIR_TO_VEC


class SimpleEnv(MultiGridEnv):
    def __init__(
        self,
        width: int = 15,
        height: int = 7,
        num_agents: int = 2,
        view_size: int = 7,
        max_steps: int = 1000,
        render_mode: str = None,
        render_fps: int = None,
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

        # precompute choices matrices for open doors
        self.choices_grid = self._precompute_choices_grid()
        self.choices_grid = self.choices_grid / np.max(self.choices_grid)   # normalize

        original_action_space = self.action_space
        self.action_space = gymnasium.spaces.Dict(
            {f"agent_{i}": original_action_space for i in range(num_agents)}
        )

        original_observation_space = self.observation_space
        original_obs_as_dict = gymnasium.spaces.Dict(
            {
                "grid": original_observation_space,
                "choice": gymnasium.spaces.Box(
                    low=0, high=100, shape=(1,), dtype=np.int32
                ),
            }
        )
        self.observation_space = gymnasium.spaces.Dict(
            {f"agent_{i}": original_obs_as_dict for i in range(num_agents)}
        )

        if render_fps is not None:
            self.metadata["render_fps"] = render_fps

    def _precompute_choices_grid(self):
        # iterate over all possible pose (x, y, dir)
        # discard positions that are not valid (e.g. walls)
        # iterate over all possible actions and check if the action is valid
        # if the action is valid, count 1
        choices = np.zeros(
            (self.width, self.height, 4, 2, 2), dtype=np.int32
        )  # open door

        doors = [
            obj for obj in self.grid.grid if obj is not None and obj.type == "door"
        ]
        assert len(doors) == 1, "Only one door is supported for now"

        # state validity
        is_valid = (
            lambda x, y, dir, carrying, open_door: self.grid.get(x, y) is None
            or self.grid.get(x, y).can_overlap()
            or self.grid.get(x, y).type == "agent"
        )

        for iod, open_door in enumerate([True, False]):
            for x in range(self.width):
                for y in range(self.height):
                    for dir in range(4):
                        for ic, carrying in enumerate([False, True]):
                            state = (x, y, dir, carrying, open_door)
                            if not is_valid(*state):
                                continue
                            for action in range(len(self.actions.available)):
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
        fwd_cell = self.grid.get(*fwd_pos)

        if action in [self.actions.still, self.actions.left, self.actions.right]:
            return True
        if (
            action == self.actions.forward
            and fwd_cell
            and (fwd_cell.can_overlap() or (fwd_cell.type == "door" and open_door))
        ):
            return True
        if (
            not carrying
            and action == self.actions.pickup
            and fwd_cell
            and fwd_cell.can_pickup()
        ):
            return True
        if (
            carrying
            and action == self.actions.toggle
            and fwd_cell
            and fwd_cell.type == "door"
        ):
            return True

        return False

    def step(self, actions):
        # convert from dict to list
        actions = [actions[f"agent_{i}"] for i in range(self.num_agents)]

        obs, reward, done, truncated, info = super().step(actions)

        # compute choice as in "Learning Altruistic Behaviors"
        choices = np.zeros(self.num_agents, dtype=np.int32)
        for i, agent in enumerate(self.agents):
            doors = [obj for obj in self.grid.grid if obj is not None and obj.type == "door"]
            is_open = doors[0].is_open if len(doors) > 0 else True
            x, y, dir, carrying = agent.pos[0], agent.pos[1], agent.dir, agent.carrying
            choices[i] = self.choices_grid[x, y, dir, int(carrying is None), int(is_open)]

        # then only things to consider is if any other agent is in front, then we should put choices -1
        for i, agent in enumerate(self.agents):
            front_pos = self.agents[i].front_pos
            if self.grid.get(*front_pos) and self.grid.get(*front_pos).type == "agent":
                choices[i] = -1

        # convert from list to dict
        obs = {
            f"agent_{i}": {"grid": obs[i], "choice": choices[i]}
            for i in range(self.num_agents)
        }

        # expand info, setting success if the agent has reached the goal (reward > 0)
        info.update(
            {f"agent_{i}": {"success": reward[i] > 0} for i in range(self.num_agents)}
        )

        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        # compute choice as in "Learning Altruistic Behaviors"
        choices = np.zeros(self.num_agents, dtype=np.int32)

        # convert from list to dict
        obs = {
            f"agent_{i}": {"grid": obs[i], "choice": choices[i][None]}
            for i in range(self.num_agents)
        }

        # initialize info with success to False
        info.update({f"agent_{i}": {"success": False} for i in range(self.num_agents)})

        return obs, info

    @abstractmethod
    def _gen_grid(self, width, height):
        raise NotImplementedError
