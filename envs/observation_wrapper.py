from gymnasium import spaces
import gymnasium as gym

class RGBImgObsWrapper(gym.core.ObservationWrapper):
    """
    Wrapper to use fully observable RGB image as observation,
    This can be used to have the agent to solve the gridworld in pixel space.

    Example:
        >>> import gymnasium as gym
        >>> import matplotlib.pyplot as plt
        >>> from minigrid.wrappers import RGBImgObsWrapper
        >>> env = gym.make("MiniGrid-Empty-5x5-v0")
        >>> obs, _ = env.reset()
        >>> plt.imshow(obs['image'])  # doctest: +SKIP
        ![NoWrapper](../figures/lavacrossing_NoWrapper.png)
        >>> env = RGBImgObsWrapper(env)
        >>> obs, _ = env.reset()
        >>> plt.imshow(obs['image'])  # doctest: +SKIP
        ![RGBImgObsWrapper](../figures/lavacrossing_RGBImgObsWrapper.png)
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size

        new_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(
                self.unwrapped.height * tile_size,
                self.unwrapped.width * tile_size,
                3,
            ),
            dtype="uint8",
        )

        for agent in self.observation_space:
            self.observation_space[agent] = spaces.Dict(
                {**self.observation_space[agent].spaces, "image": new_image_space}
            )

    def observation(self, obs):
        rgb_img = self.get_frame(
            highlight=self.unwrapped.highlight, tile_size=self.tile_size
        )

        for agent in obs:
            obs[agent] = {**obs[agent], "image": rgb_img}

        return obs