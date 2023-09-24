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
        # self._plot()
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


class CorrectedResamplingWrapper(gym.Wrapper):
    """
    This wrapper modifies the generation of initial conditions to correct bias in the distribution.

    It corrects the bias by resampling from the buffer according to a corrected distribution:
        1) compute a bonus b(s, g) = anomaly_score(s, g)
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

        # print(f"buffer size: {len(self.buffer)}")
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
