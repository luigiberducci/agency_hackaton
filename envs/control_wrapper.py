import gymnasium
import numpy as np
from gymnasium.core import ActType


class Planner:
    actions = ["still", "left", "right", "forward", "pickup", "drop", "toggle", "done"]

    def plan(self, pos, dir, goal):
        """
        TODO implement planner to navigate towards the goal.
        :param pos: agent position, tuple (x, y)
        :param dir: agent direction, int 0-3
        :param goal: goal position, tuple (x, y)
        :return: action, as int
        """
        return 0


class AutoControlWrapper(gymnasium.ActionWrapper):
    """
    Converts single-agent actions to multi-agent actions, where each non-controllable agent
    is controlled by a simple heuristic to navigate towards its goal.
    """

    def __init__(self, env):
        super().__init__(env)
        self.planner = Planner()

    def action(self, action) -> ActType:
        actions = [action]
        for i in range(1, len(self.env.agents)):
            pos, dir = self.env.agents[i].pos, self.env.agents[i].dir
            goal = self.env.goals[i]
            action = self.planner.plan(pos, dir, goal)
            actions.append(action)
        return actions
