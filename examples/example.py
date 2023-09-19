from time import sleep

import numpy as np

from envs.control_wrapper import AutoControlWrapper
from envs.multi_agent_env import SimpleEnv

env = SimpleEnv(render_mode="human", num_agents=2)
env = AutoControlWrapper(env)

seed = 42
n_episodes = 10

for i in range(n_episodes):
    env.reset(seed=seed + i)
    env.render()

    print(env.observation_space)

    done, truncated = False, False
    while not done and not truncated:
        actions = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(actions)
        env.render()
        sleep(0.1)
        print("reward: {}, done: {}".format(reward, done))