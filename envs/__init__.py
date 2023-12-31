import gymnasium as gym
from .door_env import DoorEnv
from .two_doors_env import TwoDoorsEnv

# used in one-door-2-agents-v0 to sample goals beyond the door, in the right-hand side of the room
goals_beyond_door = [(x, y) for x in range(5, 9) for y in range(1, 5)]

gym.register(
    "one-door-2-agents-v0",
    entry_point="envs.door_env:DoorEnv",
    kwargs={
        "width": 12,
        "height": 6,
        "num_agents": 2,
        "max_steps": 1000,
        "goal_generator": "choice",
        "goals": goals_beyond_door,
    },
)

goal_include_corridor = [(x, y) for x in list(range(1, 11))+[10]*10 for y in range(1,5)]

gym.register(
    "one-door-2-agents-goal-change-v0",
    entry_point="envs.door_env:DoorEnv",
    kwargs={"width": 12, "height": 6, "num_agents": 2, "max_steps": 1000,
            "goal_generator": "choice", "goals": goal_include_corridor,"goal_terminates":False},
)

gym.register(
    "two-door-2-agents-goal-change-v0",
    entry_point="envs.two_doors_env:TwoDoorsEnv",
    kwargs={"width": 12, "height": 6, "num_agents": 2, "max_steps": 1000,
            "goal_generator": "choice", "goals": goal_include_corridor,"goal_terminates":False},
)

gym.register(
    "one-door-3-agents-v0",
    entry_point="envs.door_env:DoorEnv",
    kwargs={"width": 15, "height": 7, "num_agents": 3, "max_steps": 1000},
)

# used in two-doors env, to sample goals behind each door
goal_top_bottom_rows = [(x, y) for x in range(5, 9) for y in [1, 4]]

gym.register(
    "two-doors-2-agents-v0",
    entry_point="envs.two_doors_env:TwoDoorsEnv",
    kwargs={"width": 12, "height": 6, "num_agents": 2, "max_steps": 1000,
            "goal_generator": "choice", "goals": goal_top_bottom_rows},
)


goal_top_skewed_distr = [1.0 if y == 4 else 5.0 for x in range(5, 9) for y in [1, 4]]
goal_bottom_skewed_distr = [1.0 if y == 1 else 5.0 for x in range(5, 9) for y in [1, 4]]

gym.register(
    "two-doors-2-agents-skewed-v0",
    entry_point="envs.two_doors_env:TwoDoorsEnv",
    kwargs={"width": 12, "height": 6, "num_agents": 2, "max_steps": 1000,
            "goal_generator": "categorical", "goals": goal_top_bottom_rows, "logits": goal_top_skewed_distr},
)

gym.register(
    "two-doors-2-agents-skewed-v1",
    entry_point="envs.two_doors_env:TwoDoorsEnv",
    kwargs={"width": 12, "height": 6, "num_agents": 2, "max_steps": 1000,
            "goal_generator": "categorical", "goals": goal_top_bottom_rows, "logits": goal_bottom_skewed_distr},
)
