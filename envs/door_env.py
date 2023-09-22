from __future__ import annotations

import gymnasium

from envs.base_env import SimpleEnv
from envs.goal_generators import goal_generator_factory
from gym_multigrid.world_objects import *


class DoorEnv(SimpleEnv):
    def __init__(self, goal_generator: str = None, **kwargs):
        if goal_generator is not None:
            self.goal_generator = goal_generator_factory(goal_generator, **kwargs)
        else:
            self.goal_generator = None

        super().__init__(**kwargs)

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(self.world, 0, 0)
        self.grid.horz_wall(self.world, 0, height - 1)
        self.grid.vert_wall(self.world, 0, 0)
        self.grid.vert_wall(self.world, width - 1, 0)

        # Generate vertical separation wall
        wall_w = int(self.width * 1/3)
        self.grid.vert_wall(self.world, wall_w, 0)

        # Generate second vertical separation wall (almost complete)
        for i in range(0, height - 2):
            self.grid.set(self.width - 3, i, Wall(self.world))

        # Place the door and key
        door = Door(self.world, COLOR_NAMES[0], is_locked=True, is_open=False)
        door.init_pos = door.cur_pos = wall_w, self.height - 2
        self.grid.set(*door.init_pos, door)

        key = Key(self.world, COLOR_NAMES[0])
        key.init_pos = key.cur_pos = (self.width - 2, 1)
        self.grid.set(*key.init_pos, key)

        self.goals = []

        # Randomize the player start position, orientation and goal
        for i in range(self.num_agents):
            # agent
            top = size = None
            if i == 0:
                # the first agent (altruistic) must be on the right side of the wall
                top = (wall_w + 1, 1)
                size = (wall_w + 1, 5)
            else:
                top = (1, 1)
                size = (wall_w - 1, 5)

            self.place_agent(self.agents[i], top=top, size=size, max_tries=100)

            # goal (for all but the altruistic agent)
            if i == 0:
                goal_pos = None
            elif self.goal_generator is not None:
                goal = Goal(self.world, i)
                goal_pos = self.goal_generator(self, agent_id=f"agent_{i}")
                self.put_obj(goal, *goal_pos)
            else:
                goal_pos = self.place_obj(Goal(self.world, i), max_tries=100)
            self.goals.append(goal_pos)


def main():
    env = DoorEnv(render_mode="human")

    env.reset()
    env.render()

    print(env.observation_space)

    done, truncated = False, False
    while not done and not truncated:
        actions = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(actions)
        env.render()
        print("reward: {}, done: {}".format(reward, done))


if __name__ == "__main__":
    main()
