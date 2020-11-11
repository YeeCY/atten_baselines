import numpy as np
import gym
import time
from gym import error, spaces
from gym import core, spaces
from gym.envs.registration import register
import random
from attn_toy.env.rendering import *
from copy import copy


class Fourrooms(object):
    # metadata = {'render.modes':['human']}
    # state :   number of state, counted from row and col
    # cell : (i,j)
    # observation : resultList[state]
    # small : 104 large 461
    def __init__(self, max_epilen=100):
        self.layout = """\
1111111111111
1     1     1
1     1     1
1           1
1     1     1
1     1     1
11 1111     1
1     111 111
1     1     1
1     1     1
1           1
1     1     1
1111111111111
"""
        self.block_size = 8
        self.occupancy = np.array(
            [list(map(lambda c: 1 if c == '1' else 0, line)) for line in self.layout.splitlines()])
        self.num_pos = int(np.sum(self.occupancy == 0))
        self.obs_height = self.block_size * len(self.occupancy)
        self.obs_width = self.block_size * len(self.occupancy[0])
        # From any state the agent can perform one of four actions, up, down, left or right
        self.action_space = spaces.Discrete(4)
        # self.observation_space = spaces.Discrete(np.sum(self.occupancy == 0))
        self.observation_space = spaces.Discrete(self.num_pos)

        self.directions = [np.array((-1, 0)), np.array((1, 0)), np.array((0, -1)), np.array((0, 1))]
        # self.rng = np.random.RandomState(1234)

        self.rand_color = np.random.randint(0, 255, (200, 3))
        self.tostate = {}
        self.semantics = dict()
        statenum = 0
        # print("Here", len(self.occupancy), len(self.occupancy[0]))
        for i in range(len(self.occupancy)):
            for j in range(len(self.occupancy[0])):
                if self.occupancy[i, j] == 0:
                    self.tostate[(i, j)] = statenum
                    statenum += 1
        self.tocell = {v: k for k, v in self.tostate.items()}

        self.goal = 62

        self.init_states = list(range(self.observation_space.n))
        self.init_states.remove(self.goal)
        # random encode
        self.mapping = np.arange(self.num_pos)
        self.dict = np.zeros((self.observation_space.n, 3))
        self.Row = np.shape(self.occupancy)[0]
        self.Col = np.shape(self.occupancy)[1]
        self.current_steps = 0
        self.max_epilen = max_epilen
        self.viewer = Viewer(self.block_size * len(self.occupancy), self.block_size * len(self.occupancy[0]))
        self.blocks = self.make_blocks()
        self.get_dict()
        self.currentcell = (-1, -1)
        self.reward_range = (0, 1)
        self.metadata = None
        self.done = False
        self.allow_early_resets = True
        self.unwrapped = self
        self.state = -1

    def make_blocks(self):
        blocks = []
        size = self.block_size
        for i, row in enumerate(self.occupancy):
            for j, o in enumerate(row):
                if o == 1:
                    v = [[i * size, j * size], [i * size, (j + 1) * size], [(i + 1) * size, (j + 1) * size],
                         [(i + 1) * size, (j) * size]]
                    geom = make_polygon(v, filled=True)
                    geom.set_color(0, 0, 0)
                    blocks.append(geom)
                    self.viewer.add_geom(geom)
        return blocks

    def check_obs(self, obs, info="None"):
        # print([ob for ob in obs if ob not in self.mapping])
        assert all([int(ob) in self.mapping for ob in obs]), "what happened? " + info

    def empty_around(self, cell):
        avail = []
        for action in range(self.action_space.n):
            nextcell = tuple(cell + self.directions[action])
            if not self.occupancy[nextcell]:
                avail.append(nextcell)
        return avail

    def reset(self, state=-1):
        # state = self.rng.choice(self.init_states)
        # self.viewer.close()
        if state < 0:
            state = np.random.choice(self.init_states)
        self.currentcell = self.tocell[state]
        self.done = False
        self.current_steps = 0
        self.state = state
        return np.array(self.mapping[state])

    def step(self, action):
        """
        The agent can perform one of four actions,
        up, down, left or right, which have a stochastic effect. With probability 2/3, the actions
        cause the agent to move one cell in the corresponding direction, and with probability 1/3,
        the agent moves instead in one of the other three directions, each with 1/9 probability. In
        either case, if the movement would take the agent into a wall then the agent remains in the
        same cell.

        We consider a case in which rewards are zero on all state transitions.
        """

        # print(self.currentcell, self.directions, action)
        try:
            nextcell = tuple(self.currentcell + self.directions[action])
        except TypeError:
            nextcell = tuple(self.currentcell + self.directions[action[0]])

        if not self.occupancy[nextcell]:
            self.currentcell = nextcell
            if np.random.uniform() < 0.:
                # if self.rng.uniform() < 1/3.:
                empty_cells = self.empty_around(self.currentcell)
                # self.currentcell = empty_cells[self.rng.randint(len(empty_cells))]
                self.currentcell = empty_cells[np.random.randint(len(empty_cells))]

        state = self.tostate[self.currentcell]

        self.current_steps += 1
        self.done = state == self.goal or self.current_steps >= self.max_epilen
        # if self.current_steps >= self.max_epilen:
        #     self.done = True
        info = {}
        if self.done:
            # print(self.current_step, state == self.goal, self.max_epilen)
            info = {'episode': {'r': 100 - self.current_steps if state == self.goal else -self.current_steps,
                                'l': self.current_steps}}
        # print(self.currentcell)
        self.state = state
        if state == self.goal:
            reward = 100
        else:
            reward = -1

        return np.array(self.mapping[state]), reward, self.done, info

    def get_dict(self):
        count = 0
        for i in range(self.Row):
            for j in range(self.Col):
                if self.occupancy[i, j] == 0:
                    # code
                    self.dict[count, 0] = self.mapping[count]
                    # i,j
                    self.dict[count, 1] = i
                    self.dict[count, 2] = j

                    self.semantics[self.mapping[count]] = str(i) + '_' + str(j)
                    count += 1

        # print(self.semantics)
        return self.semantics

    def add_block(self, x, y, color):
        size = self.block_size
        v = [[x * size, y * size], [x * size, (y + 1) * size], [(x + 1) * size, (y + 1) * size],
             [(x + 1) * size, y * size]]
        geom = make_polygon(v, filled=True)
        r, g, b = color
        geom.set_color(r, g, b)
        self.viewer.add_onetime(geom)

    def render(self, mode=0):

        if self.currentcell[0] > 0:
            x, y = self.currentcell
            # state = self.tostate[self.currentcell]
            # self.add_block(x, y, tuple(self.rand_color[state]/255))
            self.add_block(x, y, (0, 0, 1))

        x, y = self.tocell[self.goal]
        self.add_block(x, y, (1, 0, 0))
        # self.viewer.
        arr = self.viewer.render(return_rgb_array=True)

        return arr

    def seed(self, seed):
        pass

    def close(self):
        pass

    def all_states(self):
        return self.mapping


# register(
#     id='Fourrooms-v0',
#     entry_point='fourrooms:Fourrooms',
#     timestep_limit=20000,
#     reward_threshold=1,
# )


class ImageInputWarpper(gym.Wrapper):

    def __init__(self, env, max_steps=100):
        gym.Wrapper.__init__(self, env)
        screen_height = self.env.obs_height
        screen_width = self.env.obs_width
        self.observation_space = spaces.Box(low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8)
        # self.num_steps = 0
        self.max_steps = max_steps
        # self.state_space_capacity = self.env.state_space_capacity
        self.mean_obs = None

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        # self.num_steps += 1
        if self.num_steps >= self.max_steps:
            done = True
        obs = self.env.render(state)
        # print("step reporting",done)
        # if self.mean_obs is None:
        #     self.mean_obs = np.mean(obs)
        #     print("what is wrong?",self.mean_obs)
        # obs = obs - 0.5871700112336601
        # info['ori_obs'] = ori_obs
        info['s_tp1'] = state
        return obs, reward, done, info

    def reset(self, state=-1):
        if state < 0:
            state = np.random.randint(0, self.state_space_capacity)
        self.env.reset(state)
        # self.num_steps = self.env.num_steps
        obs = self.env.render(state)
        # print("reset reporting")
        # if self.mean_obs is None:
        #     self.mean_obs = np.mean(obs)
        # print("what is wrong? reset",self.mean_obs)
        # obs = obs - 0.5871700112336601
        # info['ori_obs'] = ori_obs
        return obs.astype(np.uint8)


class FourroomsDynamicNoise(Fourrooms):  # noise type = dynamic relevant
    def __init__(self, max_epilen=100, obs_size=128, seed=0):
        np.random.seed(seed)
        super(FourroomsDynamicNoise, self).__init__(max_epilen)
        self.obs_size = obs_size
        self.obs_height = obs_size
        self.obs_width = obs_size
        self.background = np.random.randint(0, 255, (10, 1, 1, 3))
        self.background[:, :, :, 2] = 0
        self.background = np.tile(self.background, (1, obs_size, obs_size, 1))
        self.seed = seed
        self.color = np.random.randint(100, 255, (200, 3))
        self.color[:, 2] = 100
        self.num_steps = 0
        self.observation_space = spaces.Discrete(self.num_pos * 3)
        self.state_space_capacity = self.observation_space.n

    def render(self, state=-1):
        which_background = state // self.num_pos
        # obs = np.zeros((self.obs_size, self.obs_size, 3))
        # obs[:12, :12, :] = self.color[state + 1]

        # obs = np.random.randint(0, 255, (self.obs_size, self.obs_size, 3))
        obs = np.tile(self.color[which_background][np.newaxis, np.newaxis, :], (self.obs_size, self.obs_size, 1))
        # obs = (state+100) * np.ones((self.obs_size,self.obs_size))

        arr = super(FourroomsDynamicNoise, self).render(state)
        padding_height, padding_width = (obs.shape[0] - arr.shape[0]) // 2, (obs.shape[1] - arr.shape[1]) // 2
        obs[padding_height:padding_height + arr.shape[0], padding_width:padding_width + arr.shape[1], :] = arr
        return obs.astype(np.uint8)

    def step(self, action):
        state, reward, done, info = super(FourroomsDynamicNoise, self).step(action)
        self.num_steps += 1
        state += self.num_pos * (self.num_steps % 3)
        return state, reward, done, info

    def reset(self, state=-1):
        self.num_steps = state % 3
        obs = super(FourroomsDynamicNoise, self).reset(state % self.num_pos)
        return state


class FourroomsDynamicNoise2(Fourrooms):  # noise type = state relevant
    def __init__(self, max_epilen=100, obs_size=128, seed=0):
        np.random.seed(seed)
        super(FourroomsDynamicNoise2, self).__init__(max_epilen)
        self.obs_size = obs_size
        self.obs_height = obs_size
        self.obs_width = obs_size
        self.background = np.random.randint(0, 255, (10, 1, 1, 3))
        self.background[:, :, :, 2] = 0
        self.background = np.tile(self.background, (1, obs_size, obs_size, 1))
        self.seed = seed
        self.color = np.random.randint(100, 255, (200, 3))
        self.color[:, 2] = 100
        self.num_steps = 0
        self.observation_space = spaces.Discrete(self.num_pos * max_epilen)
        self.state_space_capacity = self.num_pos * max_epilen
        self.last_action = -1

    def step(self, action):
        state, reward, done, info = super(FourroomsDynamicNoise2, self).step(action)
        self.num_steps += 1
        state += self.num_pos * self.num_steps
        return state, reward, done, info

    def reset(self, state=-1):
        self.num_steps = state // self.num_pos
        self.state = state
        obs = super(FourroomsDynamicNoise2, self).reset(state % self.num_pos)
        return state

    def render(self, state=-1):
        # which_background = self.num_steps % 3
        # obs = np.zeros((self.obs_size, self.obs_size, 3))
        # obs[:12, :12, :] = self.color[state + 1]
        obs = np.tile(self.color[self.num_steps + 1][np.newaxis, np.newaxis, :], (self.obs_size, self.obs_size, 1))
        # obs = np.random.randint(0, 255, (self.obs_size, self.obs_size, 3))
        # obs = np.tile(self.color[which_background][np.newaxis, np.newaxis, :], (self.obs_size, self.obs_size, 1))
        # obs = (state+100) * np.ones((self.obs_size,self.obs_size))

        arr = super(FourroomsDynamicNoise2, self).render(state % self.num_pos)
        padding_height, padding_width = (obs.shape[0] - arr.shape[0]) // 2, (obs.shape[1] - arr.shape[1]) // 2
        obs[padding_height:padding_height + arr.shape[0], padding_width:padding_width + arr.shape[1], :] = arr
        return obs.astype(np.uint8)


class FourroomsDynamicNoise3(Fourrooms):  # noise type = action relevant
    def __init__(self, max_epilen=100, obs_size=128, seed=0):
        np.random.seed(seed)
        super(FourroomsDynamicNoise3, self).__init__(max_epilen)
        self.obs_size = obs_size
        self.obs_height = obs_size
        self.obs_width = obs_size
        self.background = np.random.randint(0, 255, (10, 1, 1, 3))
        self.background[:, :, :, 2] = 0
        self.background = np.tile(self.background, (1, obs_size, obs_size, 1))
        self.seed = seed
        self.color = np.random.randint(100, 255, (200, 3))
        self.color[:, 2] = 100
        self.num_steps = 0
        self.observation_space = spaces.Discrete(self.num_pos * self.action_space.n)
        self.state_space_capacity = self.observation_space.n

    def render(self, state=-1):
        which_background = state // self.num_pos
        # obs = np.zeros((self.obs_size, self.obs_size, 3))
        # obs[:12, :12, :] = self.color[state + 1]
        # print(which_background, self.color[which_background])
        # obs = np.random.randint(0, 255, (self.obs_size, self.obs_size, 3))
        obs = np.tile(self.color[which_background][np.newaxis, np.newaxis, :], (self.obs_size, self.obs_size, 1))
        # obs = (state+100) * np.ones((self.obs_size,self.obs_size))

        arr = super(FourroomsDynamicNoise3, self).render(state)
        padding_height, padding_width = (obs.shape[0] - arr.shape[0]) // 2, (obs.shape[1] - arr.shape[1]) // 2
        obs[padding_height:padding_height + arr.shape[0], padding_width:padding_width + arr.shape[1], :] = arr
        return obs.astype(np.uint8)

    def step(self, action):
        state, reward, done, info = super(FourroomsDynamicNoise3, self).step(action)
        self.num_steps += 1
        state += self.num_pos * action
        # print("state in step",state)
        return state, reward, done, info

    def reset(self, state=-1):
        self.num_steps = state // self.num_pos
        obs = super(FourroomsDynamicNoise3, self).reset(state % self.num_pos)

        return state


class FourroomsRandomNoise(Fourrooms):  # noise type = random
    def __init__(self, max_epilen=100, obs_size=128, seed=0):
        np.random.seed(seed)
        super(FourroomsRandomNoise, self).__init__(max_epilen)
        self.obs_size = obs_size
        self.obs_height = obs_size
        self.obs_width = obs_size
        self.background = np.random.randint(0, 255, (10, 1, 1, 3))
        self.background[:, :, :, 2] = 0
        self.background = np.tile(self.background, (1, obs_size, obs_size, 1))
        self.seed = seed
        self.color = np.random.randint(100, 255, (200, 3))
        self.color[:, 2] = 100
        self.num_steps = 0
        self.rand_range = 4
        self.observation_space = spaces.Discrete(self.num_pos * self.rand_range)
        self.state_space_capacity = self.observation_space.n

    def render(self, state=-1):
        which_background = np.random.randint(0, self.rand_range)
        # obs = np.zeros((self.obs_size, self.obs_size, 3))
        # obs[:12, :12, :] = self.color[state + 1]

        # obs = np.random.randint(0, 255, (self.obs_size, self.obs_size, 3))
        obs = np.tile(self.color[which_background][np.newaxis, np.newaxis, :], (self.obs_size, self.obs_size, 1))
        # obs = (state+100) * np.ones((self.obs_size,self.obs_size))

        arr = super(FourroomsRandomNoise, self).render(state)
        padding_height, padding_width = (obs.shape[0] - arr.shape[0]) // 2, (obs.shape[1] - arr.shape[1]) // 2
        obs[padding_height:padding_height + arr.shape[0], padding_width:padding_width + arr.shape[1], :] = arr
        return obs.astype(np.uint8)

    def step(self, action):
        state, reward, done, info = super(FourroomsRandomNoise, self).step(action)
        self.num_steps += 1
        state += self.num_pos * action
        return state, reward, done, info

    def reset(self, state=-1):
        self.num_steps = state % 3
        obs = super(FourroomsRandomNoise, self).reset(state % self.num_pos)

        return state
