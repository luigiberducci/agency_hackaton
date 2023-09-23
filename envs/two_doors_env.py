from __future__ import annotations

import gymnasium

from envs.base_env import SimpleEnv
from envs.goal_generators import goal_generator_factory
from gym_multigrid.world_objects import Grid, Wall, Door, COLOR_NAMES, Key, Goal


class TwoDoorsEnv(SimpleEnv):
    def __init__(self, goal_generator: str = None, **kwargs):
        if goal_generator is not None:
            self.goal_generator = goal_generator_factory(goal_generator, **kwargs)
        else:
            self.goal_generator = None

        # for resampling approach
        self.initial_poses = None
        self.initial_goals = None

        super().__init__(**kwargs)

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(self.world, 0, 0)
        self.grid.horz_wall(self.world, 0, height - 1)
        self.grid.vert_wall(self.world, 0, 0)
        self.grid.vert_wall(self.world, width - 1, 0)

        # Generate vertical separation wall
        wall_x = int(self.width * 1 / 3)
        self.grid.vert_wall(self.world, wall_x, 0)

        # Generate second vertical separation wall (almost complete)
        for i in range(0, height - 2):
            self.grid.set(self.width - 3, i, Wall(self.world))

        # Place the two doors
        door1 = Door(self.world, COLOR_NAMES[0], is_locked=True, is_open=False)
        door1.init_pos = door1.cur_pos = wall_x, self.height - 2
        self.grid.set(*door1.cur_pos, door1)

        door2 = Door(self.world, COLOR_NAMES[0], is_locked=True, is_open=False)
        door2.init_pos = door2.cur_pos = wall_x, 1
        self.grid.set(*door2.cur_pos, door2)

        # Place the key
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
                top = (7, 1)
                size = (7, 5)
            else:
                top = (1, 1)
                size = (5, 5)

            if self.initial_poses is not None:
                init_pose = self.initial_poses[i]
                init_pos, init_dir = init_pose[:2], init_pose[2]

                self.agents[i].pos = init_pos
                self.agents[i].init_pos = init_pos
                self.agents[i].dir = init_dir
                self.agents[i].init_dir = init_dir

                self.grid.set(*init_pos, self.agents[i])
            else:
                self.place_agent(self.agents[i], top=top, size=size, max_tries=100)

            # goal (for all but the altruistic agent)
            if self.initial_goals is not None:
                goal_pos = self.initial_goals[i]
                if goal_pos is not None:
                    goal = Goal(self.world, i)
                    self.put_obj(goal, *goal_pos)
            else:
                if i == 0:
                    goal_pos = None
                elif self.goal_generator is not None:
                    goal = Goal(self.world, i)
                    goal_pos = self.goal_generator(self, agent_id=f"agent_{i}")
                    self.put_obj(goal, *goal_pos)
                else:
                    goal_pos = self.place_obj(Goal(self.world, i), max_tries=100)
            self.goals.append(goal_pos)

    def reset(self, seed=None, options=None):
        self.initial_poses = None
        self.initial_goals = None

        # check if initial_poses and goals are specified in options
        if options is not None:
            if "initial_poses" in options:
                self.initial_poses = options["initial_poses"]
                assert len(self.initial_poses) == self.num_agents
                assert all([len(pos) == 3 for pos in self.initial_poses])

            if "goals" in options:
                self.initial_goals = options["goals"]
                assert len(self.initial_goals) == self.num_agents
                assert all([len(goal) == 2 for goal in self.initial_goals if goal is not None])

        return super().reset(seed=seed, options=options)

def main():
    env = TwoDoorsEnv(render_mode="human")

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
