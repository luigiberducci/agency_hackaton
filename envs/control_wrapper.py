import gymnasium
import numpy as np
from gymnasium.core import ActType


class Planner:
    actions = ["still", "left", "right", "forward", "pickup", "drop", "toggle", "done"]
    move_forward_x = [1, 0, -1, 0]
    move_forward_y = [0, 1, 0, -1]

    def __init__(self, env):
        self._env = env
        self._doors_pos = [cell.cur_pos for cell in self._env.grid.grid if cell and cell.type == "door"]
        # CEM parameters
        self.horizon = 2  # planning horizon
        self.n = 30  # number of samples
        self.k = 10  # number of top samples to keep
        self.iterations = 2  # number of iterations

        assert self.k <= self.n and self.n % self.k == 0

    def forward(self, pos, dir, action):
        """
        Step forward according to the action.
        """
        if self.actions[action] == "still":
            pass
        elif self.actions[action] == "left":
            dir = (dir - 1) % 4
        elif self.actions[action] == "right":
            dir = (dir + 1) % 4
        elif self.actions[action] == "forward":
            next_pos = pos + np.array(
                [self.move_forward_x[dir], self.move_forward_y[dir]]
            )
            if (
                self._env.grid.get(*next_pos) is None
                or self._env.grid.get(*next_pos).can_overlap()
            ):
                pos = next_pos
        else:
            raise ValueError("Invalid action.")
        return pos, dir

    def cost(self, pos, dir, goal):
        """
        Compute the cost of a given position, as manhattan distance to the goal.
        """
        distance = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        door_bonus = 0.5 * sum([abs(pos[0] - door[0]) + abs(pos[1] - door[1]) for door in self._doors_pos]) / len(self._doors_pos)
        return distance + door_bonus

    def plan(self, pos, dir, goal):
        """
        Planning step with CEM algorithm.
        """
        action_seqs = np.random.random_integers(0, 3, size=(self.n, self.horizon))
        best_action_seq = action_seqs[0]
        for i in range(self.iterations):
            # add noise
            noise = np.random.random_integers(-1, 1, size=(self.n, self.horizon))
            action_seqs = np.clip(action_seqs + noise, 0, 3).astype(np.int32)

            costs = []
            traj = [pos]
            for action_seq in action_seqs:
                pos_ = pos
                dir_ = dir
                cost = 0
                for action in action_seq:
                    pos_, dir_ = self.forward(pos_, dir_, action)
                    traj.append(pos_)
                cost += self.cost(pos_, dir_, goal)
                costs.append(cost)

            costs = np.array(costs)
            topk = np.argsort(costs)[: self.k]
            action_seqs = action_seqs[topk]
            best_action_seq = action_seqs[0]

            # resample
            action_seqs = np.repeat(action_seqs, self.n // self.k, axis=0)

        return int(best_action_seq[0])


class AutoControlWrapper(gymnasium.Wrapper):
    """
    Converts single-agent actions to multi-agent actions, where each non-controllable agent
    is controlled by a simple heuristic to navigate towards its goal.
    """

    def __init__(self, env, n_auto_agents=1, action_noise=0.25):
        super().__init__(env)
        self.planner = Planner(env=env)
        self.n_auto_agents = n_auto_agents
        self.action_noise = action_noise

        # get the ids of the agents, separate them into controllable and non-controllable
        agents_ids = self.env.action_space.spaces.keys()
        self.contr_agent_ids = list(agents_ids)[: len(agents_ids) - n_auto_agents]
        self.auto_agent_ids = list(agents_ids)[len(agents_ids) - n_auto_agents :]

        # restrict action space to controllable agents
        original_action_space = self.action_space  # dict
        self.action_space = gymnasium.spaces.Dict(
            {k: original_action_space[k] for k in self.contr_agent_ids}
        )

        # restrict observation space to controllable agents
        original_observation_space = self.observation_space  # dict
        self.observation_space = gymnasium.spaces.Dict(
            {k: original_observation_space[k] for k in self.contr_agent_ids}
        )

    def step(self, actions) -> ActType:
        assert isinstance(actions, dict)

        # extend single-agent action with auto-controlled agents
        n_agents = len(self.env.agents)
        for i in range(n_agents - self.n_auto_agents, n_agents):
            iauto = i - (n_agents - self.n_auto_agents)
            agent_id = self.auto_agent_ids[iauto]

            pos, dir = self.env.agents[i].pos, self.env.agents[i].dir
            goal = self.env.goals[i]
            act = self.planner.plan(pos, dir, goal)

            # add noise
            if np.random.random() < self.action_noise:
                act = self.env.action_space[agent_id].sample()

            actions[agent_id] = act

        # step env
        obs, reward, done, truncated, info = super().step(actions)

        # filter out observations of auto-controlled agents
        obs = {k: obs[k] for k in self.contr_agent_ids}
        reward = sum(reward[-len(self.auto_agent_ids) :]) * np.ones(
            len(self.contr_agent_ids)
        )

        return obs, reward, done, truncated, info


class UnwrapSingleAgentDictWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        assert (
            type(env.action_space) == gymnasium.spaces.Dict
        ), "this wrapper is intended for Dict action spaces"
        assert (
            type(env.observation_space) == gymnasium.spaces.Dict
        ), "this wrapper is intended for Dict observation spaces"

        agent_ids = list(env.action_space.spaces.keys())
        assert len(agent_ids) == 1, "this wrapper is intended for single-agent envs"

        original_action_space = self.action_space
        original_obs_space = self.observation_space

        self.agent_id = agent_ids[0]
        self.action_space = original_action_space.spaces[self.agent_id]
        self.observation_space = original_obs_space.spaces["image"]

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        return obs["image"], info

    def step(self, action):
        action = {self.agent_id: action}
        obs, reward, done, truncated, info = super().step(action)
        reward = float(reward)
        return obs["image"], reward, done, truncated, info
