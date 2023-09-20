from time import sleep

import gymnasium as gym

import envs  # keep it, otherwise gym.make() won't work on custom envs
from envs.control_wrapper import AutoControlWrapper


env = gym.make("two-doors-v0",
               goal_generator="choice", goals=[(10, 1), (10, 5)],
               render_mode="human")
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
