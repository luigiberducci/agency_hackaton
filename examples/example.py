from time import sleep

import gymnasium as gym

import envs  # keep it, otherwise gym.make() won't work on custom envs
from envs.control_wrapper import AutoControlWrapper


env = gym.make("door-2-agents-v0", render_mode="human")
env = AutoControlWrapper(env)

seed = 42
n_episodes = 3

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
