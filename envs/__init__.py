import gymnasium as gym

gym.register("door-2-agents-v0", entry_point="envs.multi_agent_env:SimpleEnv",
             kwargs={"width": 15, "height": 7, "num_agents": 2, "max_steps": 1000})

gym.register("door-3-agents-v0", entry_point="envs.multi_agent_env:SimpleEnv",
                kwargs={"width": 15, "height": 7, "num_agents": 3, "max_steps": 1000})