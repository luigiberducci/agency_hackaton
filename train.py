import datetime

from stable_baselines3.common.callbacks import EvalCallback
import stable_baselines3
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy

trainer_fns = {
    "ppo": stable_baselines3.PPO,
    "sac": stable_baselines3.SAC,
}


def make_env(env_id: str):
    import envs
    import gymnasium as gym
    from envs.control_wrapper import AutoControlWrapper
    from envs.control_wrapper import FlattenDiscreteAction
    from gymnasium.wrappers import FlattenObservation

    env = gym.make(env_id)
    env = AutoControlWrapper(env)
    env = FlattenDiscreteAction(env)
    env = FlattenObservation(env)

    return env


def make_trainer(algo: str):
    assert algo in trainer_fns.keys(), f"Unknown algorithm {algo}"
    return trainer_fns[algo]


def main(args):
    env_id = "door-2-agents-v0"
    algo = "ppo"
    outdir = "logs/"
    seed = 42
    total_timesteps = 100000
    eval_freq = 1000

    # create environments and trainer
    train_env = make_env(env_id=env_id)
    eval_env = make_env(env_id=env_id)
    trainer_fn = make_trainer(algo)

    # setup logdir and evaluation callbacks
    date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    logdir = f"{outdir}/{algo}-{env_id}-{date_str}"
    eval_callback = EvalCallback(eval_env, log_path=logdir, eval_freq=eval_freq)

    model = trainer_fn(
        "MlpPolicy", train_env, seed=seed, tensorboard_log=logdir, verbose=1
    )

    # train model
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
    args = parser.parse_args()

    main(args)