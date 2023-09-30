import datetime
import pathlib

import gymnasium
import matplotlib.pyplot as plt
import pygame

import envs
from envs.control_wrapper import AutoControlWrapper, UnwrapSingleAgentDictWrapper
from envs.manual_control import ManualControl
from envs.observation_wrapper import RGBImgObsWrapper


def main():
    seed = 0
    cmds = "ddwawwpaawwdwwwwwldwwddddx"
    tile_size = 32
    frame_skip = 1
    save = True
    render_mode = "rgb_array" if save else "human"

    time_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # for saving frames
    outdir = pathlib.Path(f"frames_{time_id}")
    outdir.mkdir(exist_ok=True)

    env = gymnasium.make(
        "two-doors-2-agents-v0",
        render_mode=render_mode,
        tile_size=tile_size,
        render_fps=250,
        goal_generator="fixed",
        goal=(6, 4),
    )
    env = AutoControlWrapper(env)
    env = RGBImgObsWrapper(env)
    env = UnwrapSingleAgentDictWrapper(env)
    env = ManualControl(env)

    pygame.init()
    i = 0
    env.reset(seed=seed)
    done = False
    frame = env.render()

    # first frame
    if save:
        plt.imshow(frame)
        plt.box(False)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        for ext in ["png", "pdf"]:
            plt.savefig(f"{outdir}/frame_{time_id}_{i}.{ext}", dpi=300)

    keys = {
        "w": "up",
        "a": "left",
        "s": "down",
        "d": "right",
        "p": "page up",
        "l": "page down",
        "x": "escape",
        "r": "backspace",
    }
    while not done and i < len(cmds):
        # capture key events
        #key = input("Press a key: ")
        key = cmds[i]
        if key not in keys:
            continue

        event = pygame.event.Event(pygame.KEYDOWN, {"key": keys[key]})
        done = not env.key_handler(event)
        i += 1
        frame = env.render()

        if save and i % frame_skip == 0:
            plt.imshow(frame)
            plt.box(False)
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            for ext in ["png", "pdf"]:
                plt.savefig(f"{outdir}/frame_{time_id}_{i}.{ext}", dpi=300)

    # last frame
    if save:
        plt.imshow(frame)
        plt.box(False)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        for ext in ["png", "pdf"]:
            plt.savefig(f"{outdir}/frame_{time_id}_{i}.{ext}", dpi=300)


if __name__ == "__main__":
    main()
