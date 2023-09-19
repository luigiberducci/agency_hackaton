import gymnasium
import numpy as np
from gymnasium.core import ActType


class Planner:
    actions = ["still", "left", "right", "forward", "pickup", "drop", "toggle", "done"]
    move_forward_x = [1, 0, -1, 0]
    move_forward_y = [0, 1, 0, -1]

    def __init__(self, env):
        self._env = env
        # CEM parameters
        self.horizon = 3    # planning horizon
        self.n = 100    # number of samples
        self.k = 10    # number of top samples to keep
        self.iterations = 3 # number of iterations

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
            next_pos = pos + np.array([self.move_forward_x[dir], self.move_forward_y[dir]])
            if self._env.grid.get(*next_pos) is None or self._env.grid.get(*next_pos).can_overlap():
                pos = next_pos
        else:
            raise ValueError("Invalid action.")
        return pos, dir

    def cost(self, pos, dir, goal):
        """
        Compute the cost of a given position, as manhattan distance to the goal.
        """
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])


    def plan(self, pos, dir, goal):
        """
        Planning step with CEM algorithm.
        """
        action_seqs = np.random.random_integers(0, 3, size=(self.n, self.horizon))
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

                # debug - visualize the action sequence and trajectory
                #import matplotlib.pyplot as plt
                #plt.figure()
                #plt.plot([p[0] for p in traj], [p[1] for p in traj], color="red")
                #plt.scatter(traj[0][0], traj[0][1], color="green")
                #plt.gca().invert_yaxis()
                #plt.show()

            costs = np.array(costs)
            topk = np.argsort(costs)[:self.k]
            action_seqs = action_seqs[topk]

            # resample
            action_seqs = np.repeat(action_seqs, self.n // self.k, axis=0)

        return int(action_seqs[0][0])


class AutoControlWrapper(gymnasium.ActionWrapper):
    """
    Converts single-agent actions to multi-agent actions, where each non-controllable agent
    is controlled by a simple heuristic to navigate towards its goal.
    """

    def __init__(self, env, prob_slip=0.25):
        super().__init__(env)
        self.planner = Planner(env=env)
        self.prob_slip = prob_slip

    def action(self, action) -> ActType:
        actions = [action]
        for i in range(1, len(self.env.agents)):
            pos, dir = self.env.agents[i].pos, self.env.agents[i].dir
            goal = self.env.goals[i]
            action = self.planner.plan(pos, dir, goal)

            # add noise
            if np.random.random() < self.prob_slip:
                action = np.random.randint(1, 4)

            actions.append(action)
        return actions
    