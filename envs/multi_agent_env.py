from __future__ import annotations


from gym_multigrid.multigrid import (
    MultiGridEnv,
    World,
    Agent,
    Goal,
    COLOR_NAMES,
    Key,
    Door,
    Wall,
    Grid,
    Ball,
)


class SimpleEnv(MultiGridEnv):
    def __init__(
        self,
        width: int = 15,
        height: int = 7,
        num_agents: int = 2,
        max_steps: int = 1000,
        **kwargs,
    ):
        self.num_agents = num_agents
        self.goals = []

        self.world = World

        agents = []
        for i in range(num_agents):
            agent = Agent(self.world, i)
            agents.append(agent)

        super().__init__(
            grid_size=None,
            width=width,
            height=height,
            max_steps=max_steps,
            see_through_walls=False,  # Set this to True for maximum speed
            agents=agents,
            partial_obs=False,
        )
        self.carrying = None

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(self.world, 0, 0)
        self.grid.horz_wall(self.world, 0, height - 1)
        self.grid.vert_wall(self.world, 0, 0)
        self.grid.vert_wall(self.world, width - 1, 0)

        # Generate vertical separation wall
        self.grid.vert_wall(self.world, 6, 0)

        # Generate second vertical separation wall (almost complete)
        for i in range(0, height - 2):
            self.grid.set(12, i, Wall(self.world))

        # Place the door and key
        self.grid.set(6, 4, Door(self.world, COLOR_NAMES[0], is_locked=True))
        self.grid.set(13, 1, Key(self.world, COLOR_NAMES[0]))

        self.goals = []
        
        # Randomize the player start position, orientation and goal
        for i in range(self.num_agents):
            # agent
            top = size = None
            if i == 0:
                # the first agent (altruistic) must be on the right side of the wall
                top = (7, 1)
                size = (7, 5)
            else:
                top = (1, 1)
                size = (5, 5)

            self.place_agent(self.agents[i], top=top, size=size, max_tries=100)

            # goal (for all but the altruistic agent)
            if i == 0:
                goal_pos = None
            else:
                goal_pos = self.place_obj(Goal(self.world, i), max_tries=100)
            self.goals.append(goal_pos)


def main():
    env = SimpleEnv(render_mode="human")

    env.reset()
    env.render()

    print(env.observation_space)

    done, truncated = False, False
    while not done and not truncated:
        actions = [env.action_space.sample()]
        obs, reward, done, truncated, info = env.step(actions)
        env.render()
        print("reward: {}, done: {}".format(reward, done))


if __name__ == "__main__":
    main()
