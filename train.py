import datetime
import json
import pathlib

from gymnasium.wrappers import FilterObservation, RecordVideo

from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import stable_baselines3
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv

import envs
import gymnasium as gym
from envs.control_wrapper import AutoControlWrapper
from envs.control_wrapper import UnwrapSingleAgentDictWrapper
from envs.observation_wrapper import RGBImgObsWrapper

trainer_fns = {
    "ppo": stable_baselines3.PPO,
    "sac": stable_baselines3.SAC,
}


def make_env(env_id: str, rank: int, seed: int = 42, log_dir: str = None):
    def make() -> gym.Env:
        if rank == 0 and log_dir is not None:
            env = gym.make(env_id, render_mode="rgb_array")
            env = RecordVideo(env, video_folder=f"{log_dir}/videos")
        else:
            env = gym.make(env_id, render_mode="rgb_array")
        env = AutoControlWrapper(env, n_auto_agents=1)
        env = RGBImgObsWrapper(env)
        env = FilterObservation(env, filter_keys=["image"])
        env = UnwrapSingleAgentDictWrapper(env)
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return make


def make_trainer(algo: str):
    assert algo in trainer_fns.keys(), f"Unknown algorithm {algo}"
    return trainer_fns[algo]


def main(args):
    env_id = args.env_id
    num_envs = args.num_envs
    algo = args.algo
    seed = args.seed
    total_timesteps = args.total_timesteps
    eval_freq = args.eval_freq
    n_eval_episodes = args.n_eval_episodes
    track = args.track

    # set seed
    if seed is None:
        seed = np.random.randint(0, 1e6)

    if track:
        import wandb

        logdir = modeldir = None
        run = wandb.init(
            project="agency_hackathon",
            config=vars(args),
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            # save_code=True,  # optional
        )
    else:
        # setup logdir
        date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        logdir = f"logs/{algo}-{env_id}-{date_str}-{seed}"
        modeldir = f"{logdir}/models"
        pathlib.Path(logdir).mkdir(parents=True, exist_ok=True)

        # copy arg params to logdir
        with open(f"{logdir}/args.json", "w") as f:
            json.dump(vars(args), f)

    # create environments and trainer
    train_env = SubprocVecEnv([make_env(env_id, i, log_dir=logdir) for i in range(num_envs)])
    trainer_fn = make_trainer(algo)

    # create evaluation environment
    eval_env = make_env(env_id=env_id, rank=0, seed=42)()
    eval_env = Monitor(eval_env, logdir)

    # create model trainer
    model = trainer_fn("CnnPolicy", train_env, tensorboard_log=logdir, seed=seed, verbose=1)

    # train model
    callbacks = [
        EvalCallback(eval_env, log_path=logdir, eval_freq=eval_freq, n_eval_episodes=n_eval_episodes,
                     best_model_save_path=logdir),
        CheckpointCallback(save_freq=eval_freq, save_path=modeldir)
    ]
    if track:
        from wandb.integration.sb3 import WandbCallback

        wandcb = WandbCallback(model_save_path=f"models/{run.id}", verbose=2)
        callbacks.append(wandcb)
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

    if track:
        run.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id",
        type=str,
        default="door-2-agents-v0",
        help="Env ID as registered in Gymnasium",
    )
    parser.add_argument(
        "--num-envs", type=int, default=2, help="Number of vectorized environments"
    )
    parser.add_argument("--algo", type=str, default="ppo", help="Algorithm to use")
    parser.add_argument(
        "--track", action="store_true", help="Whether to track the training on wandb"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=100000,
        help="Total number of timesteps to train",
    )
    parser.add_argument(
        "--eval-freq", type=int, default=5000, help="Evaluation frequency in steps"
    )
    parser.add_argument(
        "--n-eval-episodes", type=int, default=25, help="Number of evaluation episodes"
    )
    args = parser.parse_args()

    main(args)
