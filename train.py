import datetime
import json
import pathlib
from distutils.util import strtobool
from typing import Callable

from gymnasium.wrappers import GrayScaleObservation
from gymnasium import Env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import stable_baselines3
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecFrameStack

import envs
import gymnasium as gym
from gymnasium.wrappers.frame_stack import FrameStack
from envs.control_wrapper import AutoControlWrapper
from envs.control_wrapper import UnwrapSingleAgentDictWrapper
from envs.observation_wrapper import RGBImgObsWrapper
from envs.reward_wrappers import RewardWrapper, reward_factory, RewardFn
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


def make_env(
    env_id: str,
    rank: int,
    reward_fn: Callable[[Env], RewardFn] | None = None,
    obj_to_hide: list[str] | None = None,
    stack_frames: int | None = None,
    seed: int = 42,
):
    def make() -> gym.Env:
        # base env
        env = gym.make(env_id, render_mode="rgb_array")

        # reward wrapper
        if reward_fn is not None:
            env = RewardWrapper(env, build_reward=reward_fn)

        # control wrapper for other agents
        env = AutoControlWrapper(env, n_auto_agents=1)

        # observation wrapper
        env = RGBImgObsWrapper(env, hide_obj_types=obj_to_hide)
        env = UnwrapSingleAgentDictWrapper(env)

        if stack_frames is not None:
            env = GrayScaleObservation(env)
            env = FrameStack(env, num_stack=stack_frames)

        # monitor, to consistently record episode stats
        env = Monitor(env)  # keep it here, otherwise issue with vec env termination
        env.reset(seed=seed + rank)

        return env

    set_random_seed(seed)
    return make


def make_trainer(algo: str):
    assert algo in trainer_fns.keys(), f"Unknown algorithm {algo}"
    return trainer_fns[algo], configs[algo]


def main(args):
    env_id = args.env_id
    reward_id = args.reward
    obj_to_hide = ["goal"] if args.hide_goals else []
    total_timesteps = args.total_timesteps
    n_envs = args.num_envs
    algo = args.algo
    stack_frames = args.stack_frames
    seed = args.seed
    eval_freq = args.eval_freq
    n_eval_episodes = args.num_eval_episodes
    debug = args.debug

    # set seed
    if seed is None:
        seed = np.random.randint(0, 1e6)

    # setup logging
    dirs = {dname: None for dname in ["logdir", "modeldir", "videodir"]}
    if not debug:
        dirs["logdir"] = args.log_dir
        date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        goal_str = "hide-goals" if args.hide_goals else "show-goals"
        dirs[
            "logdir"
        ] = f"{dirs['logdir']}/{algo}-{env_id}-{goal_str}-{reward_id}-{date_str}-{seed}"

        for dir, name in zip(dirs, ["log", "models", "videos"]):
            if dirs[dir] is None:
                dirs[dir] = f"{dirs['logdir']}/{name}"
            pathlib.Path(dirs[dir]).mkdir(parents=True, exist_ok=True)

        # copy arg params to logdir
        with open(f"{dirs['logdir']}/args.json", "w") as f:
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
    train_reward = reward_factory(reward=reward_id)
    vec_cls = (
        SubprocVecEnv if n_envs > 1 else DummyVecEnv
    )  # subproc introduces overhead, not worth it for 1 env
    train_env = vec_cls(
        [
            make_env(
                env_id, i, seed=seed, reward_fn=train_reward, obj_to_hide=obj_to_hide, stack_frames=stack_frames
            )
            for i in range(n_envs)
        ]
    )

    # create evaluation environment
    eval_reward = reward_factory(reward="sparse")
    eval_env = make_env(
        env_id=env_id, rank=0, seed=42, reward_fn=eval_reward, obj_to_hide=obj_to_hide, stack_frames=stack_frames
    )()

    # create model trainer
    model = trainer_fn(
        env=train_env,
        tensorboard_log=dirs["logdir"],
        seed=seed,
        verbose=2,
        **trainer_params,
    )

    # setup callabacks
    callbacks = [
        EvalCallback(
            eval_env,
            log_path=dirs["logdir"],
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            best_model_save_path=dirs["modeldir"],
        ),
    ]
    if dirs["modeldir"] is not None:
        checkpoint_cb = CheckpointCallback(
            save_freq=eval_freq, save_path=dirs["modeldir"]
        )
        callbacks.append(checkpoint_cb)
    if dirs["videodir"] is not None and not debug:
        videorec_cb = VideoRecorderCallback(
            eval_env, video_folder=dirs["videodir"], render_freq=eval_freq
        )
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

    # env params
    parser.add_argument(
        "--env-id",
        type=str,
        default="one-door-2-agents-v0",
        help="EnvID as registered in Gymnasium",
    )
    parser.add_argument(
        "--reward",
        type=str,
        default="sparse",
        help="Reward function to use during training, during evaluation 'sparse' reward is always used",
    )
    parser.add_argument(
        "--hide-goals",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggle partial observability by hiding goals from agents",
    )

    # algorithm params
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Number of vectorized environments run in parallel to collect rollouts",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        help="RL algorithm to use for training the agent",
    )

    parser.add_argument(
        "--stack-frames", type=int, default=None, help="Number of frames to stack"
    )

    # training params
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility, if None a random seed is used",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Total number of timesteps to train",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=5000,
        help="Evaluation frequency in vectorized envs steps (eg., 5000 means evaluate every 5000 * <num_envs> steps)",
    )
    parser.add_argument(
        "--num-eval-episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes to collect in the evaluation phase",
    )

    # logging params
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Path to log dir where to save results",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Toggle debug mode and disable logging on disk",
    )
    args = parser.parse_args()

    main(args)
