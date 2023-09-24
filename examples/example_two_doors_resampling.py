from time import sleep

import gymnasium as gym

import envs  # keep it, otherwise gym.make() won't work on custom envs
from envs.control_wrapper import AutoControlWrapper
from envs.resampling_wrapper import CorrectedResamplingWrapper

env = gym.make(
    "two-doors-2-agents-skewed-v0",
    render_mode="human",
    render_fps=250,
    max_episode_steps=10,
)
env = CorrectedResamplingWrapper(env)
env = AutoControlWrapper(env)

print(env.action_space)
print(env.observation_space)

seed = 42
n_episodes = 100


for i in range(n_episodes):
    env.reset(seed=seed + i)
    env.render()

    print(f"Episode {i}")

    done, truncated = False, False
    while not done and not truncated:
        actions = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(actions)
        env.render()
        #print("reward: {}, done: {}".format(reward, done))
    print()

env.close()
