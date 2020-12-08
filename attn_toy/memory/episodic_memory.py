import numpy as np
import matplotlib.pyplot as plt


def np_hash(array):
    array = np.array(array).squeeze()
    return hash(array.data.tobytes())


# All the same as replay buffer except that it will not restore the same state more than once
class EpisodicMemory(object):
    def __init__(self, capacity, obs_shape, num_actions, hash_func=np_hash, gamma=1):
        self.hash_func = hash_func
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.gamma = gamma

        self.obs = np.empty((capacity,) + obs_shape, dtype=np.uint8)
        self.rewards = np.zeros((capacity, num_actions))
        self.dones = np.zeros((capacity, num_actions), dtype=np.bool)
        self.returns = -np.inf * np.ones((capacity, num_actions))
        self.next_id = -1 * np.ones((capacity, num_actions))
        self.curr_capacity = 0
        self.pointer = 0
        self.state_dict = {}

    def get_index(self, obs):
        obs_hash = self.hash_func(obs)
        index = self.state_dict.get(obs_hash, -1)
        if index < 0:
            index = self.pointer
            if self.pointer < self.curr_capacity:
                # not first time
                self.next_id[self.pointer] = -1 * np.ones(self.num_actions)
                self.rewards[self.pointer] = np.zeros(self.num_actions)
                self.dones[self.pointer] = np.zeros(self.num_actions)
                self.returns[self.pointer] = -np.inf * np.ones(self.num_actions)
                self.pointer = (self.pointer + 1) % self.capacity

                origin_hash = self.hash_func(self.obs[index])
                if origin_hash is self.state_dict.keys():
                    self.state_dict.pop(origin_hash)
            else:
                self.curr_capacity = min(self.curr_capacity + 1, self.capacity)
                self.pointer = (self.pointer + 1) % self.capacity

            self.state_dict[obs_hash] = index
        return index

    @staticmethod
    def switch_first_half(obs, obs_next, batch_size):
        half_size = int(batch_size / 2)
        tmp = obs[:half_size, ...]
        obs[:half_size, ...] = obs_next[:half_size, ...]
        obs_next[:half_size, ...] = tmp
        return obs, obs_next

    def store(self, obs, action, reward, done, rtn, obs_tp1=None):

        # done = reward == 100  # an ad-hoc solution
        index = self.get_index(obs)
        if obs_tp1 is not None:
            index_tp1 = self.get_index(obs_tp1)
        else:
            index_tp1 = -1

        self.rewards[index, action] = reward
        self.obs[index, ...] = obs
        self.dones[index, action] = done
        if not np.isnan(rtn):
            self.returns[index, action] = max(rtn, self.returns[index, action])
        if index_tp1 >= 0:
            self.next_id[index, action] = index_tp1
            # max_next = reward + self.gamma * (1. - float(done)) * np.max(self.returns[index_tp1])
            # # print(done,max_next)
            # if not np.isnan(max_next):
            #     self.returns[index, action] = max(max_next,
            #                                       self.returns[index, action])

    def add_batch(self, obs, actions, rewards, returns, dones):
        for ob, action, r, Rtn, done, obs_tp1 in zip(obs, actions, rewards, returns, dones, list(obs[1:]) + [None]):
            if len(ob.shape) >= 4:
                ob = np.squeeze(obs, axis=0)
            self.store(ob, action, r, done, Rtn, obs_tp1)

    def sample(self, sample_size, neg_num=1, priority='uniform'):

        sample_size = min(self.curr_capacity, sample_size)
        if sample_size % 2 == 1:
            sample_size -= 1
        if sample_size < 2:
            return None, None, None, None, None, None, None
        indexes = []
        positives = []
        negatives = []
        returns = []
        dones = []
        actions = []
        rewards = []

        iters = 0
        while len(indexes) < sample_size:
            iters += 1
            if iters > sample_size * 100:
                break
            if priority == 'uniform':
                ind = int(np.random.randint(0, self.curr_capacity, 1))
            elif priority == 'value_l2':
                value_variance = np.nan_to_num((self.returns[:self.curr_capacity] - 0) ** 2)
                probs = value_variance / np.sum(value_variance)
                ind = int(np.random.choice(np.arange(0, self.curr_capacity), p=probs))
            else:
                ind = int(np.random.randint(0, self.curr_capacity, 1))
            if ind in indexes:
                continue
            next_id = [(a, self.next_id[ind][a]) for a in range(self.num_actions) if self.next_id[ind][a] > 0]
            if len(next_id) == 0:
                continue

            random_next_id = np.random.randint(0, len(next_id))
            action, positive = next_id[random_next_id]

            indexes.append(ind)
            positives.append(int(positive))
            actions.append(action)
            rewards.append(self.rewards[ind, action])
            returns.append(self.returns[ind, :])
            dones.append(self.dones[ind, action])

        iters = 0
        while len(negatives) < len(indexes) * neg_num:
            ind = indexes[len(negatives) // neg_num]
            pos_ind = positives[len(negatives) // neg_num]

            neg_ind = int(np.random.randint(0, self.curr_capacity, 1))

            neg_ind_next = [self.next_id[neg_ind][a] for a in
                            range(self.num_actions)]
            ind_next = [self.next_id[ind][a] for a in range(self.num_actions)]
            pos_ind_next = [self.next_id[pos_ind][a] for a in
                            range(self.num_actions)]
            conflict_with_tar = ind in neg_ind_next or neg_ind in ind_next or neg_ind == ind
            conflict_with_pos = pos_ind in neg_ind_next or neg_ind in pos_ind_next or neg_ind == pos_ind

            iters += 1
            if iters % 1000000 == 999999:
                print("negative", iters)
                break
            if conflict_with_tar or conflict_with_pos:
                continue
            negatives.append(neg_ind)

        obs_target = self.obs[indexes]
        obs_positive = self.obs[positives]
        obs_negative = self.obs[negatives]
        rewards = np.array(rewards)
        dones = np.array(dones)
        returns = np.array(returns)

        return obs_target, actions, rewards, obs_positive, dones, obs_negative, returns

    # for debugging
    def percentage(self):
        return 1. - np.sum(np.isinf(self.returns[:self.curr_capacity])) / self.num_actions / self.curr_capacity

    def empty(self):
        self.obs = np.empty((self.capacity,) + self.obs_shape, dtype=np.uint8)
        self.rewards = np.zeros((self.capacity, self.num_actions))
        self.dones = np.zeros((self.capacity, self.num_actions), dtype=np.bool)
        self.returns = -np.inf * np.ones((self.capacity, self.num_actions))
        self.next_id = -1 * np.ones((self.capacity, self.num_actions))
        self.curr_capacity = 0
        self.pointer = 0
        self.state_dict = {}
