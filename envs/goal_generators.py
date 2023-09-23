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

def softmax(x: np.ndarray):
    """
    Compute softmax values for each sets of scores in x.
    From https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
class CategoricalGoalGenerator(GoalGenerator):

    def __init__(self, goals: list[tuple[int, int]], logits: list[float]):
        self.goals = goals
        self.logits = np.array(logits)
        self.p = softmax(self.logits)

    def __call__(self, env: SimpleEnv, agent_id: str = None, max_tries: int = 20):
        idx = np.random.choice(len(self.goals), p=self.p)
        goal = self.goals[idx]
        return goal


def goal_generator_factory(mode: str, **kwargs) -> GoalGenerator:
    if mode == "random":
        return RandomGoalGenerator()
    elif mode == "fixed":
        assert "goal" in kwargs, "Missing goal position."
        return FixedGoalGenerator(goal=kwargs["goal"])
    elif mode == "choice":
        assert "goals" in kwargs, "Missing goals."
        return ChoiceGoalGenerator(goals=kwargs["goals"])
    elif mode == "categorical":
        assert "goals" in kwargs, "Missing goals."
        assert "logits" in kwargs, "Missing logits."
        return CategoricalGoalGenerator(goals=kwargs["goals"], logits=kwargs["logits"])
    else:
        raise ValueError(f"Unknown goal generator mode: {mode}.")
