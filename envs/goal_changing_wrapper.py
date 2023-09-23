from typing import Any, SupportsFloat
import gymnasium
from gymnasium.core import Env
import numpy as np
from envs.goal_generators import goal_generator_factory
from gym_multigrid.world_objects import Goal


class GoalChangingWrapper(gymnasium.Wrapper):
    def __init__(
        self, env: Env, goal_generator: str = None, p_change: float = 0.1, **kwargs
    ):
        if goal_generator is not None:
            self.goal_generator = goal_generator_factory(goal_generator, **kwargs)
        else:
            self.goal_generator = None
        self.p_change = p_change
        super().__init__(env)

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        if np.random.random() < self.p_change:
            self.env.goals = []
            self.env.grid.grid = [
                obj if not isinstance(obj, Goal) else None for obj in self.env.grid.grid
            ]
            for i in range(self.num_agents):
                if i == 0:
                    goal_pos = None
                elif self.goal_generator is not None:
                    goal_pos = self.goal_generator(self, agent_id=f"agent_{i}")
                    self.put_obj(Goal(self.world, i), *goal_pos)
                else:
                    goal_pos = self.place_obj(Goal(self.world, i), max_tries=100)
                self.env.goals.append(goal_pos)

        return super().step(action)
