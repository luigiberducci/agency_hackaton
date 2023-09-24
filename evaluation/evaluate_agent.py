import csv
import os
import warnings
from datetime import datetime
from distutils.util import strtobool
from typing import Union, Tuple, Optional, Callable, Dict, Any, List

import imageio
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv, is_vecenv_wrapped, VecMonitor, DummyVecEnv

from envs.reward_wrappers import reward_factory
from train import make_env
import gymnasium as gym


def custom_evaluate_policy(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward, episode length and frames per episode.

    From sb3 implementation.
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])

    is_monitor_wrapped = (
        is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]
    )

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []
    all_episode_frames = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array(
        [(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int"
    )

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    episode_frames = []
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        new_observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    print(f"Done episode {episode_counts}")
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        observations = new_observations

        if render:
            frame = env.render()
            episode_frames.append(frame)

            if dones[0]:
                all_episode_frames.append(episode_frames)
                episode_frames = []

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, (
            "Mean reward below threshold: "
            f"{mean_reward:.2f} < {reward_threshold:.2f}"
        )
    if return_episode_rewards:
        return episode_rewards, episode_lengths, all_episode_frames
    return mean_reward, std_reward


def main(args):
    checkpoints = args.checkpoint if args.checkpoint is not None else []
    env_id = args.env_id
    goal_changes = args.goal_changes
    obj_to_hide = ["goal"] if args.hide_goals else []
    stack_frames = args.stack_frames
    num_eval_episodes = args.num_eval_episodes
    seed = args.seed if args.seed is not None else np.random.randint(0, 1000000)
    save = args.save
    save_video = args.save_video

    # eval env
    eval_reward = reward_factory(reward="sparse")
    eval_env = make_env(
        env_id=env_id,
        rank=0,
        seed=seed,
        reward_fn=eval_reward,
        goal_changes=goal_changes,
        obj_to_hide=obj_to_hide,
        stack_frames=stack_frames,
    )()

    # video dir
    if save:
        videodir = f"evaluations-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        os.makedirs(videodir, exist_ok=True)
        with open(os.path.join(videodir, "args.txt"), "w") as f:
            f.write(str(args) + "\n")

    # load trained model
    assert len(checkpoints) > 0 and all(
        [c.endswith("zip") for c in checkpoints]
    ), "invalid checkpoints"
    for checkpoint in checkpoints:
        model = PPO.load(checkpoint)

        rewards, lengths, frames = custom_evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=num_eval_episodes,
            render=save,
            return_episode_rewards=True,
        )
        rewards = np.array(rewards)
        lengths = np.array(lengths)
        frames = [np.array(ep_frames) for ep_frames in frames]

        print(
            f"Model: {checkpoint}, eval env: {env_id}, num eval episodes: {num_eval_episodes}"
        )
        print(f"Mean reward: {rewards.mean():.2f} +/- {rewards.std():.2f}")
        print(f"Mean length: {lengths.mean():.2f} +/- {lengths.std():.2f}")
        print(f"Collected frames: {[f.shape for f in frames]}")
        print()

        # saving video
        if save:
            if save_video:
                for i, ep_frames in enumerate(frames):
                    print(f"Saving video for episode {i}...")
                    imageio.mimwrite(
                        os.path.join(videodir, f"ep_{i}.mp4"),
                        ep_frames,
                        fps=10,
                        quality=10,
                    )

            # save rewards and lengths into csv
            with open(os.path.join(videodir, "rewards.csv"), "w") as f:
                writer = csv.writer(f)
                writer.writerow(rewards)

            with open(os.path.join(videodir, "lengths.csv"), "w") as f:
                writer = csv.writer(f)
                writer.writerow(lengths)




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, nargs="+", default=None)
    parser.add_argument("--num-eval-episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--save-video", action="store_true")

    parser.add_argument("--env-id", type=str, default="one-door-2-agents-v0")
    parser.add_argument(
        "--goal-changes",
        type=float,
        default=0.0,
        help="Probability of changing the goal",
    )
    parser.add_argument(
        "--hide-goals",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggle partial observability by hiding goals from agents",
    )
    parser.add_argument(
        "--stack-frames", type=int, default=None, help="Number of frames to stack"
    )

    args = parser.parse_args()
    main(args)
