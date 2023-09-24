import pygame
from gym_multigrid.actions import Actions

from envs.multi_agent_env import SimpleEnv


class ManualControl:
    def __init__(
        self,
        env: SimpleEnv,
        seed=None,
    ) -> None:
        self.env = env
        self.seed = seed
        self.closed = False

    def start(self):
        """Start the window display with blocking event loop"""
        self.reset(self.seed)
        self.render()

        while not self.closed:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.env.close()
                    break
                if event.type == pygame.KEYDOWN:
                    event.key = pygame.key.name(int(event.key))
                    self.key_handler(event)
                    self.render()

    def step(self, action: Actions):
        _, reward, terminated, truncated, _ = self.env.step(action)
        print(f"step={self.env.step_count}, reward={reward:.2f}")

        if terminated:
            print("terminated!")
            self.reset(self.seed)
        elif truncated:
            print("truncated!")
            self.reset(self.seed)

    def reset(self, seed=None):
        self.env.reset(seed=seed)

    def key_handler(self, event):
        key: str = event.key
        print("pressed", key)

        if key == "escape":
            self.env.close()
            return False
        if key == "backspace":
            self.reset()
            return True

        key_to_action = {
            "left": Actions.left,
            "right": Actions.right,
            "up": Actions.forward,
            "space": Actions.toggle,
            "page up": Actions.pickup,
            "page down": Actions.drop,
            "tab": Actions.pickup,
            "left shift": Actions.drop,
            "enter": Actions.done,
        }
        if key in key_to_action.keys():
            action = key_to_action[key]
            try:
                self.step(action)
            except Exception as e:
                import warnings
                warnings.warn(f"Exception: {e}")
        else:
            print(key)

        return True

    def render(self):
        return self.env.render()