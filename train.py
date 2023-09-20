import datetime
import pathlib
import yaml

from stable_baselines3.common.callbacks import EvalCallback
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

trainer_fns = {
    "ppo": stable_baselines3.PPO,
    "sac": stable_baselines3.SAC,
}


def make_env(env_id: str, rank: int, seed: int = 42):
    def make() -> gym.Env:
        env = gym.make(env_id)
        env = AutoControlWrapper(env, n_auto_agents=1)
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
    outdir = args.outdir

    # set seed
    if seed is None:
        seed = np.random.randint(0, 1e6)

    # create environments and trainer
    train_env = SubprocVecEnv([make_env(env_id, i) for i in range(num_envs)])
    trainer_fn = make_trainer(algo)

    # create evaluation environment
    eval_env = make_env(env_id=env_id, rank=0, seed=42)()
    eval_env = Monitor(eval_env)

    # setup logdir
    date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logdir = f"{outdir}/{algo}-{env_id}-{date_str}-{seed}"
    pathlib.Path(logdir).mkdir(parents=True, exist_ok=True)

    # copy arg params to logdir
    with open(f"{logdir}/args.yaml", "w") as f:
        yaml.dump(vars(args), f)

    # create model trainer
    model = trainer_fn(
        "MlpPolicy", train_env, seed=seed, tensorboard_log=logdir, verbose=1
    )

    # train model
    eval_callback = EvalCallback(eval_env, log_path=logdir, eval_freq=eval_freq)
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    # evaluate trained model
    rewards, lengths = evaluate_policy(
        model, eval_env, n_eval_episodes=10, render=True, return_episode_rewards=True
    )

    print(f"Reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"Length: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")


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
        "--outdir", type=str, default="logs/", help="Directory to store logs"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=100000,
        help="Total number of timesteps to train",
    )
    parser.add_argument(
        "--eval-freq", type=int, default=1000, help="Evaluation frequency in stepst"
    )
    args = parser.parse_args()

    main(args)
