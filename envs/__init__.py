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

gym.register(
    "one-door-3-agents-v0",
    entry_point="envs.door_env:DoorEnv",
    kwargs={"width": 15, "height": 7, "num_agents": 3, "max_steps": 1000},
)

gym.register(
    "two-doors-v0",
    entry_point="envs.two_doors_env:TwoDoorsEnv",
    kwargs={"width": 15, "height": 7, "num_agents": 2, "max_steps": 1000},
)
