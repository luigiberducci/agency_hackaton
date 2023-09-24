from abc import abstractmethod
from collections import deque
from typing import Callable

import gym
import numpy as np

from envs.base_env import SimpleEnv
from envs.goal_generators import softmax
from sklearn.decomposition import PCA


PRETRAIN_SAMPLES = 1000

class Encoder:
    @abstractmethod
    def __call__(self, x):
        raise NotImplementedError

    @abstractmethod
    def update(self, x):
        raise NotImplementedError


class PCAEncoder(Encoder):
    def __init__(self, n_components=1):
        self.pca = PCA(n_components=n_components)
        self.X = None

    def __call__(self, x):
        x = np.array(x).reshape(1, -1)
        #self._plot()
        return self.pca.transform(x)

    def anomaly_score(self, buffer):
        all_goals = [x[1] for x in buffer]
        clean_goals = [goal[1] for goal in all_goals]
        X = np.array(clean_goals).reshape(-1, 2)
        y = self.pca.transform(X)
        X_hat = self.pca.inverse_transform(y)
        return np.linalg.norm(X - X_hat, axis=1)

    def update(self, buffer):
        all_goals = [x[1] for x in buffer]
        clean_goals = [goal[1] for goal in all_goals]
        self.X = np.array(clean_goals).reshape(-1, 2)
        self.pca.fit(self.X)

    def _plot(self):
        raise NotImplementedError
        import matplotlib.pyplot as plt
        noisy_X = self.X + np.random.normal(0, 0.1, size=self.X.shape)
        anomaly_scores = [self.anomaly_score(x) for x in noisy_X]

        anomaly_colors = anomaly_scores / np.max(anomaly_scores)
        # create rgb colors based on anomaly scores
        anomaly_colors = np.array([np.array([1, 0, 0]) * (1 - c) +
                                   np.array([0, 1, 0]) * c for c in anomaly_colors])


        plt.scatter(noisy_X[:, 0], noisy_X[:, 1], alpha=0.3, label="samples", color=anomaly_colors)
        for i, (comp, var) in enumerate(zip(self.pca.components_, self.pca.explained_variance_)):
            comp = comp * var  # scale component by its variance explanation power
            plt.plot(
                [0, comp[0]],
                [0, comp[1]],
                label=f"Component {i}",
                linewidth=5,
            )
        plt.gca().set(
            aspect="equal",
            title="2-dimensional dataset with principal components",
            xlabel="first feature",
            ylabel="second feature",
        )
        plt.legend()
        plt.show()

class CorrectedResamplingWrapper(gym.Wrapper):
    """
    This wrapper modifies the generation of initial conditions to correct bias in the distribution.

    At each reset, it keeps track of the initial conditions in a buffer (here, the initial pose and goal coordinates)
    and their frequency. In practice, it keeps a visiting count N(h) and a total count N for any hash h of the initial
    conditions. The hash function is provided by the user and could be any function or neural network.

    Then, it corrects the bias by resampling from the buffer according to a corrected distribution:
        1) compute a bonus b(s, g) = 1 / sqrt(N(s, g)/N + 0.01)
        2) sample from the distribution p(s, g) = softmax(b(s, g))
    """

    def __init__(
        self,
        env: SimpleEnv,
        encoder: Callable = None,
        max_buffer_size: int = 10000,
        warmup_resets: int = 250,
        p_resample: float = 0.5,
    ):
        super().__init__(env)

        self.buffer = deque(maxlen=max_buffer_size)
        self.hash_to_visit = {}  # visit table: hash -> count
        self.hash_to_cond = {}  # reconstruct condition from hash
        self.total_count = 0

        self.encoder = encoder if encoder is not None else PCAEncoder()
        self.max_buffer_size = max_buffer_size
        self.warmup_resets = warmup_resets
        self.p_resample = p_resample

    def reset(self, **kwargs):
        if self.total_count == 0:
            # collect few episodes to inizialize encoder
            for _ in range(self.warmup_resets):
                obs, info = super().reset(**kwargs)
                initial_poses, goals = self._get_initial_conditions(obs, info)
                self.buffer.append((initial_poses, goals))
        elif np.random.rand() < self.p_resample:
            print("resampling")
            # otherwise, resample from the buffer
            initial_poses, goals = self._resampling()

            if "options" in kwargs and kwargs["options"] is not None:
                options = kwargs["options"]
            else:
                options = {}

            other_kwargs = {k: v for k, v in kwargs.items() if k != "options"}
            options["initial_poses"] = [np.array(pose) for pose in initial_poses]
            options["goals"] = [np.array(goal) if goal else None for goal in goals]
            obs, info = super().reset(options=options, **other_kwargs)
        else:
            obs, info = super().reset(**kwargs)
            initial_poses, goals = self._get_initial_conditions(obs, info)

        # update
        self.buffer.append((initial_poses, goals))
        self.encoder.update(self.buffer)
        self.total_count += 1

        print("buffer size: {}".format(len(self.hash_to_visit)))
        #print("buffer", self.hash_to_visit)
        return obs, info

    def _resampling(self):
        scores = self.encoder.anomaly_score(self.buffer)
        assert len(scores) == len(self.buffer)
        probs = softmax(scores)

        # sample from distribution
        indices = np.arange(len(self.buffer))
        idx = np.random.choice(indices, p=probs)
        initial_poses, goals = self.buffer[idx]

        return initial_poses, goals

    def _get_initial_conditions(self, obs, info):
        initial_poses = tuple(
            [
                (
                    info[agent_id]["pos"][0],
                    info[agent_id]["pos"][1],
                    info[agent_id]["dir"],
                )
                for agent_id in obs
            ]
        )
        goals = tuple([info[agent_id]["goal"] for agent_id in obs])
        return initial_poses, goals

def g_hashing_fn(initial_conditions: tuple) -> str:
    """
    Projection over the goals, discarding the initial poses.

    :param initial_conditions: tuple of initial poses and goals
    :return: string hash
    """
    _, goals = initial_conditions
    h = hash(tuple(g[1] for g in goals if g is not None))
    return str(h)
