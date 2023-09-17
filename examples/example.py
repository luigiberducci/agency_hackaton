from envs.control_wrapper import AutoControlWrapper
from envs.multi_agent_env import SimpleEnv

env = SimpleEnv(render_mode="human")
env = AutoControlWrapper(env)

env.reset()
env.render()

print(env.observation_space)

done, truncated = False, False
while not done and not truncated:
    actions = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(actions)
    env.render()
    print("reward: {}, done: {}".format(reward, done))