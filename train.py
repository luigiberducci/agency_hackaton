import datetime
import json
import pathlib
from typing import Callable

from gymnasium import Env
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import stable_baselines3
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import envs
import gymnasium as gym
from envs.control_wrapper import AutoControlWrapper
from envs.control_wrapper import UnwrapSingleAgentDictWrapper
from envs.observation_wrapper import RGBImgObsWrapper
from envs.reward_wrappers import SparseRewardFn, RewardWrapper, AltruisticRewardFn, NegativeRewardFn, reward_factory, \
    RewardFn
from helpers.callbacks import VideoRecorderCallback
from models import MinigridFeaturesExtractor

trainer_fns = {
    "ppo": stable_baselines3.PPO,
}

configs = {
    "ppo": {
        "train_params": {
            "n_envs": 8,
            "total_timesteps": 1e6,
        },
        "trainer_params": {
            "policy": "CnnPolicy",
            "n_steps": 512,
            "batch_size": 64,
            "gae_lambda": 0.95,
            "gamma": 0.99,
            "n_epochs": 10,
            "ent_coef": 0.0,
            "learning_rate": 2.5e-4,
            "clip_range": 0.2,
            "policy_kwargs": {
                "features_extractor_class": MinigridFeaturesExtractor,
                "features_extractor_kwargs": dict(features_dim=128),
            },
        },
    },
}


def make_env(env_id: str, rank: int, reward_fn: Callable[[Env], RewardFn] | None = None, seed: int = 42):
    goal_generator = "choice"
    goals_beyond_door = [
        (5, 1),
        (6, 1),
        (7, 1),
        (5, 2),
        (6, 2),
        (7, 2),
        (5, 3),
        (6, 3),
        (7, 3),
    ]  # choice of goals

    def make() -> gym.Env:
        env = gym.make(
            env_id,
            render_mode="rgb_array",
            goal_generator=goal_generator,
            goals=goals_beyond_door,
        )

        if reward_fn is not None:
            env = RewardWrapper(env, build_reward=reward_fn)

        env = AutoControlWrapper(env, n_auto_agents=1)
        env = RGBImgObsWrapper(env)
        env = UnwrapSingleAgentDictWrapper(env)
        env = Monitor(env)  # keep it here otherwise issue with vec env termination
        env.reset(seed=seed + rank)
        #check_env(env)
        return env

    set_random_seed(seed)
    return make


def make_trainer(algo: str):
    assert algo in trainer_fns.keys(), f"Unknown algorithm {algo}"
    return trainer_fns[algo], configs[algo]


def main(args):
    env_id = args.env_id
    reward_id = args.reward
    total_timesteps = args.total_timesteps
    n_envs = args.num_envs
    algo = args.algo
    seed = args.seed
    eval_freq = args.eval_freq
    n_eval_episodes = args.n_eval_episodes
    debug = args.debug

    # set seed
    if seed is None:
        seed = np.random.randint(0, 1e6)

    # setup logdir
    logdir = args.log_dir
    modeldir, evaldir = None, None
    if not debug:
        assert logdir is not None and isinstance(logdir, str), "logdir must be specified for non-debug mode"
        date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        logdir = f"{logdir}/{algo}-{env_id}-{reward_id}-{date_str}-{seed}"
        modeldir = f"{logdir}/models"
        evaldir = f"{logdir}/eval"
        for dir in [logdir, modeldir, evaldir]:
            pathlib.Path(dir).mkdir(parents=True, exist_ok=True)

        # copy arg params to logdir
        with open(f"{logdir}/args.json", "w") as f:
            json.dump(vars(args), f)

    # create trainer
    trainer_fn, trainer_config = make_trainer(algo)
    training_params = trainer_config["train_params"]
    trainer_params = trainer_config["trainer_params"]

    # process params
    n_envs: int = n_envs if n_envs is not None else training_params["n_envs"]
    total_timesteps: int = (
        total_timesteps
        if total_timesteps is not None
        else training_params["total_timesteps"]
    )

    # create training environment
    video_freq = eval_freq // n_envs // 1000    # from freq in steps to freq in episodes, assumes 1000 steps x episode
    train_reward = reward_factory(reward=reward_id)
    train_env = SubprocVecEnv([make_env(env_id, i, seed=seed, reward_fn=train_reward) for i in range(n_envs)])

    # create evaluation environment
    eval_reward = reward_factory(reward="sparse")
    eval_env = make_env(env_id=env_id, rank=0, seed=42, reward_fn=eval_reward)()

    # create model trainer
    model = trainer_fn(
        env=train_env,
        tensorboard_log=logdir,
        seed=seed,
        verbose=2,
        **trainer_params,
    )

    # setup callabacks
    callbacks = [
        EvalCallback(
            eval_env,
            log_path=evaldir,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            best_model_save_path=evaldir,
        ),
    ]
    if modeldir is not None:
        checkpoint_cb = CheckpointCallback(save_freq=eval_freq, save_path=modeldir)
        callbacks.append(checkpoint_cb)
    if not debug:
        videorec_cb = VideoRecorderCallback(eval_env, render_freq=eval_freq)
        callbacks.append(videorec_cb)

    # train model
    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    # evaluate trained model
    rewards, lengths = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=10,
        render=True,
        return_episode_rewards=True,
    )

    print(f"Reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"Length: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=str, default="logs", help="Log directory")
    parser.add_argument(
        "--env-id",
        type=str,
        default="one-door-2-agents-v0",
        help="Env ID as registered in Gymnasium",
    )
    parser.add_argument("--reward", type=str, default="sparse", help="Reward function")
    parser.add_argument(
        "--num-envs", type=int, default=None, help="Number of vectorized environments"
    )
    parser.add_argument("--algo", type=str, default="ppo", help="Algorithm to use")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Total number of timesteps to train",
    )
    parser.add_argument(
        "--eval-freq", type=int, default=5000, help="Evaluation frequency in steps of vectorized envs"
    )
    parser.add_argument(
        "--n-eval-episodes", type=int, default=5, help="Number of evaluation episodes"
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode, no log stored")
    args = parser.parse_args()

    main(args)
