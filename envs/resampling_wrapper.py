from abc import abstractmethod
from typing import Callable

import gym
import numpy as np

from envs.base_env import SimpleEnv
from envs.goal_generators import softmax
from sklearn.decomposition import PCA

class Encoder:
    @abstractmethod
    def __call__(self, x):
        raise NotImplementedError

    @abstractmethod
    def update(self, X):
        raise NotImplementedError


class PCAEncoder(Encoder):
    def __init__(self, n_components=2):
        self.pca = PCA(n_components=n_components)

    def __call__(self, x):
        return self.pca.transform(x)

    def update(self, X):
        self.pca.fit(X)

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
        max_buffer_size: int = 1000,
        warmup_episodes: int = 2,
    ):
        super().__init__(env)

        self.hash_to_visit = {}  # visit table: hash -> count
        self.hash_to_cond = {}  # reconstruct condition from hash
        self.total_count = 0

        self.encoder = encoder if encoder is not None else g_hashing_fn
        self.max_buffer_size = max_buffer_size
        self.warmup_episodes = warmup_episodes

    def reset(self, **kwargs):
        if len(self.hash_to_visit) < self.warmup_episodes:
            # if we are still in the warmup phase, do not resample
            obs, info = super().reset(**kwargs)
        else:
            # otherwise, resample from buffer
            # compute probabilities
            all_conditions = list(self.hash_to_visit.keys())
            logits = np.array(
                [
                    1 / np.sqrt(self.hash_to_visit[cond] / self.total_count + 0.01)
                    for cond in all_conditions
                ]
            )
            probs = softmax(logits)

            # sample from distribution
            idx = np.random.choice(len(all_conditions), p=probs)
            hash_key = all_conditions[idx]
            initial_poses, goals = self.hash_to_cond[hash_key]

            options = (
                kwargs["options"]
                if "options" in kwargs and kwargs["options"] is not None
                else {}
            )
            other_kwargs = {k: v for k, v in kwargs.items() if k != "options"}
            options["initial_poses"] = initial_poses
            options["goals"] = goals
            obs, info = super().reset(options=options, **other_kwargs)

        # extract initial conditions
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

        # update buffer
        initial_conditions = (initial_poses, goals)
        hash_key = self.encoder(initial_conditions)
        if hash_key not in self.hash_to_visit:
            # if buffer is full, discard current initial conditions with prob 1/len(buffer)
            # otherwise, remove a random entry from the buffer
            if len(self.hash_to_visit) < self.max_buffer_size:
                self.hash_to_visit[hash_key] = 1
                self.hash_to_cond[hash_key] = initial_conditions
            else:
                # if max capacity, remove with equal probability one of the entries (including the current one)
                if np.random.random() < 1 / (len(self.hash_to_visit) + 1):
                    pass
                else:
                    idx = np.random.choice(len(self.hash_to_visit))
                    del self.hash_to_visit[idx]
                    del self.hash_to_cond[idx]
                self.hash_to_visit[hash_key] = 1
                self.hash_to_cond[hash_key] = initial_conditions
        else:
            self.hash_to_visit[hash_key] += 1
            self.hash_to_cond[hash_key] = initial_conditions

        self.total_count += 1

        print("buffer size: {}".format(len(self.hash_to_visit)))
        print("buffer", self.hash_to_visit)
        return obs, info


def g_hashing_fn(initial_conditions: tuple) -> str:
    """
    Projection over the goals, discarding the initial poses.

    :param initial_conditions: tuple of initial poses and goals
    :return: string hash
    """
    _, goals = initial_conditions
    h = hash(tuple(g[1] for g in goals if g is not None))
    return str(h)
