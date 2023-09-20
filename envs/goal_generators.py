from abc import abstractmethod

import numpy as np

from envs.base_env import SimpleEnv


class GoalGenerator:
    @abstractmethod
    def __call__(self, env: SimpleEnv, agent_id: str = None):
        raise NotImplementedError


class RandomGoalGenerator(GoalGenerator):
    """
    Randomly pick a free position in the grid.
    """

    def __call__(self, env: SimpleEnv, agent_id: str = None, max_tries: int = 100):
        while max_tries > 0:  # avoid to sample a position on the wall
            x = np.random.random_integers(1, env.width - 2)
            y = np.random.random_integers(1, env.height - 2)
            if env.grid.get(x, y) is None:
                return x, y
            max_tries -= 1
        # if we cannot find a free position, return top left corner
        return 1, 1


class FixedGoalGenerator(GoalGenerator):
    """
    Always return the same goal position.
    """

    def __init__(self, goal: tuple[int, int]):
        self.x, self.y = goal

    def __call__(self, env: SimpleEnv, agent_id: str = None, max_tries: int = 100):
        assert (
            self.x < env.width and self.y < env.height
        ), "Goal position is outside the grid."
        return self.x, self.y


class ChoiceGoalGenerator(GoalGenerator):
    def __init__(self, goals: list[tuple[int, int]]):
        self.goals = goals

    def __call__(self, env: SimpleEnv, agent_id: str = None, max_tries: int = 20):
        while max_tries > 0:
            idx = np.random.choice(len(self.goals))
            goal = self.goals[idx]
            if env.grid.get(*goal) is None:
                return goal
            max_tries -= 1
        # return first goal, even if it is not free
        return self.goals[0]


def goal_generator_factory(mode: str, **kwargs) -> GoalGenerator:
    if mode == "random":
        return RandomGoalGenerator()
    elif mode == "fixed":
        assert "goal" in kwargs, "Missing goal position."
        return FixedGoalGenerator(goal=kwargs["goal"])
    elif mode == "choice":
        assert "goals" in kwargs, "Missing goals."
        return ChoiceGoalGenerator(goals=kwargs["goals"])
    else:
        raise ValueError(f"Unknown goal generator mode: {mode}.")
