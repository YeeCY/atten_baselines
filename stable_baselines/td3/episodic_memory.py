import numpy as np
from sklearn.neighbors import BallTree, KDTree
import os
import gc
import pickle as pkl
import copy


def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


class EpisodicMemory(object):
    def __init__(self, buffer_size, state_dim, action_dim, action_shape, obs_shape, gamma=0.99,w_q=0.1):
        self.ec_buffer = []
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.capacity = buffer_size
        self.curr_capacity = 0

        self.latent_buffer = np.zeros((buffer_size, state_dim + action_dim))
        self.q_values = -np.inf * np.ones(buffer_size + 1)
        self.replay_buffer = np.empty((buffer_size,) + obs_shape, np.float32)
        self.action_buffer = np.empty((buffer_size,) + action_shape, np.float32)
        self.reward_buffer = np.empty((buffer_size,), np.float32)
        self.steps = np.empty((buffer_size,), np.int)
        self.done_buffer = np.empty((buffer_size,), np.bool)
        self.next_id = -1 * np.ones(buffer_size)
        self.prev_id = [[] for _ in range(buffer_size)]
        self.ddpg_q_values = -np.inf * np.ones(buffer_size)
        self.lru = np.zeros(buffer_size)
        self.time = 0
        self.gamma = gamma
        self.hashes = dict()
        self.reward_mean = None
        self.kd_tree = None
        self.state_kd_tree = None
        self.build_tree = False
        self.build_tree_times = 0
        self.w_q = w_q

    def save(self, filedir):
        pkl.dump(self, open(os.path.join(filedir, "episodic_memory.pkl"), "wb"))

    # def capacity(self):
    #     return [buffer.curr_capacity for buffer in self.ec_buffer]

    def add(self, obs, action, state, encoded_action, sampled_return, next_id=-1):
        if state is not None and encoded_action is not None:
            state, encoded_action = np.squeeze(state), np.squeeze(encoded_action)
            if len(encoded_action.shape) == 0:
                encoded_action = encoded_action[np.newaxis, ...]
        if self.curr_capacity >= self.capacity:
            # find the LRU entry
            # priority = self.w_q *(self.q_values[:self.capacity]) + self.lru
            # priority = self.q_values[:self.capacity]
            priority = self.lru
            index = int(np.argmin(priority))
            # index = int(np.argmin(self.q_values))
            self.prev_id[index] = []
            self.next_id[index] = -1
            self.q_values[index] = -np.inf
            old_key = tuple(
                np.squeeze(np.concatenate([self.replay_buffer[index], self.action_buffer[index]])).astype('float32'))
            self.hashes.pop(old_key, None)

        else:
            index = self.curr_capacity
            self.curr_capacity += 1
        self.replay_buffer[index] = obs
        self.action_buffer[index] = action
        if state is not None and encoded_action is not None:
            self.latent_buffer[index] = np.concatenate([state, encoded_action])
        self.q_values[index] = sampled_return
        self.lru[index] = self.time
        new_key = tuple(np.squeeze(np.concatenate([obs, action])).astype('float32'))
        self.hashes[new_key] = index

        if next_id >= 0:
            self.next_id[index] = next_id
            if index not in self.prev_id[next_id]:
                self.prev_id.append(index)
        self.time += 0.01
        return index

    def peek(self, action, state, value_decay, modify, next_id=-1, allow_decrease=False, knn=1):
        state = np.squeeze(state)
        if len(state.shape) == 1:
            state = state[np.newaxis, ...]
        action = np.squeeze(action)
        if len(action.shape) == 0:
            action = action[np.newaxis, ...]
        if len(action.shape) == 1:
            action = action[np.newaxis, ...]
        h = np.concatenate([state, action], axis=1)
        if self.curr_capacity == 0 or self.build_tree == False:
            return None, None
        # print(h.shape)
        dist, ind = self.kd_tree.query(h, k=knn)
        dist, ind = dist[0], ind[0]
        if dist[0] < 1e-5:
            self.lru[ind] = self.time
            self.time += 0.01
            if modify:
                # if value_decay > self.q_values[ind]:
                if allow_decrease:
                    self.q_values[ind] = value_decay
                else:
                    self.q_values[ind] = max(value_decay, self.q_values[ind])
                if next_id >= 0:
                    self.next_id[ind] = next_id
                    if ind not in self.prev_id[next_id]:
                        self.prev_id[next_id].append(ind)

            return self.q_values[ind], ind
        else:
            queried_q_value = 0.0
            dist = dist / (1e-12 + np.max(dist)) + 1e-13
            coeff = -np.log(dist)
            coeff = coeff / np.sum(coeff)
            for i, index in enumerate(ind):
                queried_q_value += self.q_values[index] * coeff[i]

            return queried_q_value, None

    # def sample(self, batch_size, num_neg=1):
    #     capacity = sum([buffer.curr_capacity for buffer in self.ec_buffer])
    #     assert 0 < batch_size < capacity, "can't sample that much!"
    #     anchor_idxes = []
    #     pos_idxes = []
    #     neg_idxes = []
    #     loop_count = 0
    #     while len(anchor_idxes) < batch_size:
    #         loop_count += 1
    #         if loop_count > self.curr_capacity * 100:
    #             return None
    #         rand_idx = np.random.randint(0, self.curr_capacity)
    #         if self.next_id[rand_idx] > 0:
    #             anchor_idxes.append(rand_idx)
    #             rand_pos = np.random.randint(0, len(self.next_id[rand_idx]))
    #             pos_idxes.append(self.next_id[rand_idx][rand_pos])
    #             prev_place = self.prev_id[rand_idx]
    #             next_place = [self.next_id[rand_idx]]
    #             neg_action, neg_idx = self.sample_neg_keys(
    #                 [rand_idx] + prev_place + next_place, num_neg)
    #             neg_idxes.append(neg_idx)
    #     neg_idxes = np.array(neg_idxes).reshape(-1)
    #     # anchor_obses = [self.ec_buffer[action].obses[id] for id, action in zip(anchor_idxes, anchor_actions)]
    #     # anchor_keys = [self.ec_buffer[action].hashes[id] for id, action in zip(anchor_idxes, anchor_actions)]
    #     # pos_keys = [self.ec_buffer[action].hashes[id] for id, action in zip(pos_idxes, pos_actions)]
    #     # neg_keys = [[self.ec_buffer[action].hashes[id] for id, action in zip(neg_idxes[i], neg_actions[i])] for i in
    #     #             range(len(neg_idxes))]
    #
    #     anchor_obs = [self.replay_buffer[ind] for ind in anchor_idxes]
    #     pos_obs = [self.replay_buffer[ind] for ind in pos_idxes]
    #     neg_obs = [self.replay_buffer[ind] for ind in neg_idxes]
    #     anchor_values = [self.q_values[index] for index in anchor_idxes]
    #     anchor_actions = [self.action_buffer[index] for index in anchor_idxes]
    #     return anchor_obs, pos_obs, neg_obs, anchor_values, anchor_actions

    def sample_neg_keys(self, avoids, batch_size):
        # sample negative keys
        assert batch_size + len(
            avoids) <= self.capacity, "can't sample that much neg samples from episodic memory!"
        places = []
        while len(places) < batch_size:
            ind = np.random.randint(0, self.curr_capacity)
            if ind not in places:
                places.append(ind)
        return places

    def update(self, idxes, reprs):
        self.latent_buffer[idxes] = reprs

    def state_knn_max_value(self, state, knn=4):
        knn = min(self.curr_capacity, knn)
        state = np.squeeze(state)
        if len(state.shape) <= 1:
            state = state[np.newaxis, ...]
        if not self.build_tree:
            return -np.inf
        dist, inds = self.state_kd_tree.query(state, k=knn)
        inds = inds[0]
        q_values = [self.q_values[i] for i in inds]
        return np.max(q_values)

    def state_knn_action(self, state_key, knn):
        knn = min(self.curr_capacity, knn)
        if self.curr_capacity < 2000 or self.build_tree == False:
            return None, None
        key = np.squeeze(state_key)
        while len(key.shape) <= 1:
            key = key[np.newaxis, ...]
        dist, ind = self.state_kd_tree.query(key, k=knn)
        ind = ind[0]
        q_values = self.q_values[ind]
        # temperature = 0.001
        # coeff = np.sum(np.square(self.latent_buffer[ind, :self.state_dim] - state_key), axis=1)
        # coeff = coeff / np.max(coeff)
        # print(coeff.shape,knn)
        # coeff = np.exp(-coeff / temperature)
        # coeff = coeff / np.sum(coeff)
        # q_values = q_values * coeff
        ind_max = ind[np.argmax(q_values)]
        max_action = self.action_buffer[ind_max]
        max_q = self.q_values[ind_max]
        return max_action, max_q

    def knn_value(self, state_key, action_key, knn):
        knn = min(self.curr_capacity, knn)
        if self.curr_capacity == 0 or self.build_tree == False:
            return None
        combined_key = np.concatenate((state_key, action_key), axis=1)
        key = np.squeeze(combined_key)
        if len(key.shape) <= 1:
            key = key[np.newaxis, ...]
        dist, ind = self.kd_tree.query(key, k=knn)
        # dist, ind = dist[0], ind[0]
        # dist = dist / (1e-12 + np.max(dist, axis=1, keepdims=True)) + 1e-13
        dist = dist + 1e-13
        # coeff = -np.log(dist)
        # coeff = np.exp(-dist / 0.1)
        coeff = 1. / dist ** 2
        coeff = coeff / np.sum(coeff, axis=1, keepdims=True)
        # value = 0.0
        queried_q_value = []
        # print("here",ind.shape,coeff.shape)
        for i in range(len(ind)):
            queried_q_value.append(0.0)
            for j, index in enumerate(ind[i]):
                queried_q_value[i] += self.q_values[index] * coeff[i, j]
                # self.lru[index] = self.time
                # self.time += 0.01

        return np.array(queried_q_value)[:, np.newaxis]

    def hash_value(self, state, action):
        if not self.build_tree:
            return None
        index = []
        for i in range(len(state)):
            key = tuple(np.squeeze(np.concatenate([state[i], action[i]])).astype('float32'))
            ind = self.hashes.get(key, self.capacity)
            # if ind == self.capacity:
            #     print("hash failure in episodic memory",self.curr_capacity)
            # else:
            #     print("success hash",self.capacity)
            index.append(ind)
        return (self.q_values[index]).reshape(-1)

    def knn(self, state_key, action_key, knn):
        combined_key = np.concatenate((state_key, action_key), axis=1)
        knn = min(self.curr_capacity, knn)
        if self.curr_capacity == 0 or self.build_tree == False:
            return None
        key = np.squeeze(combined_key)
        if len(key.shape) == 1:
            key = key[np.newaxis, ...]
        dist, ind = self.kd_tree.query(key, k=knn)
        # value = 0.0
        queried_q_values = []
        queried_obs = []
        queried_actions = []
        for i in range(len(ind)):
            queried_q_values.append([])
            queried_obs.append([])
            queried_actions.append([])
            for j, index in enumerate(ind[0]):
                queried_q_values[i].append(self.q_values[index])
                queried_obs[i].append(self.latent_buffer[index, :self.state_dim])
                queried_actions[i].append(self.latent_buffer[index, self.state_dim:])
                # self.lru[index] = self.time

        # self.time += 0.01

        return np.array(queried_obs), np.array(queried_actions), np.array(queried_q_values)

    def knn_with_intrinsic(self, state_key, action_key, knn):
        combined_key = np.concatenate((state_key, action_key), axis=1)
        knn = min(self.curr_capacity, knn)
        if self.curr_capacity == 0 or self.build_tree == False:
            return None
        key = np.squeeze(combined_key)
        if len(key.shape) == 1:
            key = key[np.newaxis, ...]
        dist, ind = self.kd_tree.query(key, k=knn)
        # value = 0.0
        queried_q_values = []
        queried_obs = []
        queried_actions = []
        for i in range(len(ind)):
            queried_q_values.append([])
            queried_obs.append([])
            queried_actions.append([])
            for j, index in enumerate(ind[0]):
                queried_q_values[i].append(self.q_values[index])
                queried_obs[i].append(self.latent_buffer[index, :self.state_dim])
                queried_actions[i].append(self.latent_buffer[index, self.state_dim:])
                self.lru[index] = self.time

        self.time += 0.01

        return np.array(queried_obs), np.array(queried_actions), np.array(queried_q_values)

    def update_sequence_horizon(self, sequence, horizen=100):
        # print(sequence)
        if self.reward_mean is None:
            reward_sum = 0
            for _, _, _, _, r, _, _ in sequence:
                reward_sum += r
            self.reward_mean = reward_sum / (len(sequence) - 1)
        next_id = -1
        Rtd = 0
        time_step = 0
        returns = []
        for obs, a, z, encoded_a, r, q_tp1, done in reversed(sequence):
            time_step += 1
            Rtd = self.gamma * Rtd + r
            returns.append(Rtd)
            # if self.gamma == 1:
            #     corrected_Rtd = Rtd + self.reward_mean * time_step
            # else:
            #     corrected_Rtd = Rtd + self.reward_mean * self.gamma * (1 - self.gamma ** time_step) / (1 - self.gamma)
            if time_step > horizen:
                corrected_Rtd = Rtd - self.gamma ** horizen * returns[time_step - 100]
                qd, current_id = self.peek(encoded_a, z, corrected_Rtd, True)
                if current_id is None:  # new action
                    current_id = self.add(obs, a, z, encoded_a, corrected_Rtd, next_id)

                self.replay_buffer[current_id] = obs
                self.reward_buffer[current_id] = r
            else:
                current_id = -1
            next_id = current_id
        self.reward_mean = np.mean(self.reward_buffer[:self.curr_capacity])
        return

    def update_sequence_corrected(self, sequence):
        # print(sequence)
        next_id = -1
        Rtd = 0
        for obs, a, z, encoded_a, r, q_tp1, done in reversed(sequence):
            # print(np.mean(z))
            if done and q_tp1 is None:
                continue
            if done and q_tp1 is not None:
                Rtd = float(np.squeeze(q_tp1))
            else:
                Rtd = self.gamma * Rtd + r

            # qd, current_id = self.peek(encoded_a, z, Rtd, True)
            # if current_id is None:  # new action
            current_id = self.add(obs, a, z, encoded_a, Rtd, next_id)

            self.replay_buffer[current_id] = obs
            self.reward_buffer[current_id] = r
            next_id = int(current_id)
        return

    def update_sequence_with_critic(self, sequence, q_values):
        next_id = -1
        Rtd = 0
        steps = len(q_values)
        for q, experience in zip(reversed(q_values), reversed(sequence)):
            obs, a, z, encoded_a, r, done = experience
            Rtd = self.gamma * Rtd + r
            steps -= 1
            qd, current_id = self.peek(encoded_a, z, q, True)
            if current_id is None:
                current_id = self.add(obs, a, z, encoded_a, q, next_id)

            self.replay_buffer[current_id] = obs
            self.reward_buffer[current_id] = r
            self.done_buffer[current_id] = done
            # self.ddpg_q_values[current_id] = q
            self.steps[current_id] = steps
            next_id = current_id

        return

    def update_sequence(self, sequence):
        # print(sequence)
        next_id = -1
        Rtd = 0
        for obs, a, z, encoded_a, r, done in reversed(sequence):
            # print(np.mean(z))

            Rtd = self.gamma * Rtd + r

            qd, current_id = self.peek(encoded_a, z, Rtd, True)
            if current_id is None:  # new action
                current_id = self.add(obs, a, z, encoded_a, Rtd, next_id)

            self.replay_buffer[current_id] = obs
            self.reward_buffer[current_id] = r
            self.done_buffer[current_id] = done

            next_id = current_id
        return

    def update_sequence_iterate(self, sequence, knn):
        # print(sequence)
        next_id = -1

        delta = np.inf
        last_value, last_key = None, None
        while delta > 1e6:
            Rtd = 0
            for obs, a, z, encoded_a, r, optimal_encoded_a, done in reversed(sequence):
                # print(np.mean(z))
                if done:
                    last_key = (z, encoded_a, knn)
                    Rtd = self.knn_value(z, encoded_a, knn=knn)

                    if Rtd is None:
                        Rtd = 0
                    else:
                        Rtd = float(Rtd)
                    last_value = Rtd
                    continue
                else:
                    Rtd = self.gamma * Rtd + r
                estimated_Rtd = -np.inf
                # estimated_Rtd = self.state_knn_max_value(obs, 4)
                # estimated_Rtd = -np.inf if estimated_Rtd is None else estimated_Rtd
                Rtd = max(estimated_Rtd, Rtd)
                qd, current_id = self.peek(encoded_a, z, Rtd, True, allow_decrease=done)
                if current_id is None:  # new action
                    current_id = self.add(obs, a, z, encoded_a, Rtd, next_id)

                self.replay_buffer[current_id] = obs
                self.reward_buffer[current_id] = r
                next_id = current_id

            self.update_kdtree()
            Rtd_knn = float(self.knn_value(*last_key))
            delta = abs(last_value - Rtd_knn)
            print(delta)
        return

    def update_kdtree(self, use_repr=True):
        if self.build_tree:
            del self.kd_tree
            del self.state_kd_tree
            # del self.hash_tree
        if self.curr_capacity <= 0:
            return
        # print("build tree", self.curr_capacity)
        # self.tree = KDTree(self.states[:self.curr_capacity])
        self.kd_tree = KDTree(self.latent_buffer[:self.curr_capacity])
        if use_repr:
            self.state_kd_tree = KDTree(self.latent_buffer[:self.curr_capacity, :self.state_dim])
        else:
            self.state_kd_tree = KDTree(self.replay_buffer[:self.curr_capacity])
        # self.hash_tree = KDTree(self.hashes[:self.curr_capacity])
        self.build_tree = True
        self.build_tree_times += 1
        if self.build_tree_times == 50:
            self.build_tree_times = 0
            gc.collect()

    def sample_negative(self, batch_size, batch_idxs, batch_idxs_next, batch_idx_pre):
        neg_batch_idxs = []
        i = 0
        while i < batch_size:
            neg_idx = np.random.randint(0, self.curr_capacity - 2)
            if neg_idx != batch_idxs[i] and neg_idx != batch_idxs_next[i] and neg_idx not in batch_idx_pre[i]:
                neg_batch_idxs.append(neg_idx)
                i += 1
        neg_batch_idxs = np.array(neg_batch_idxs)
        return self.replay_buffer[neg_batch_idxs]

    def switch_first_half(self, obs0, obs1, batch_size):
        tmp = copy.copy(obs0[:batch_size // 2, ...])
        obs0[:batch_size // 2, ...] = obs1[:batch_size // 2, ...]
        obs1[:batch_size // 2, ...] = tmp
        return obs0, obs1

    def sample(self, batch_size, mix=False):
        # Draw such that we always have a proceeding element
        if self.curr_capacity + 2 < batch_size:
            return None
        batch_idxs = []
        batch_idxs_next = []
        while len(batch_idxs) < batch_size:
            rnd_idx = np.random.randint(0, self.curr_capacity)
            if self.next_id[rnd_idx] == -1:
                continue
            batch_idxs.append(rnd_idx)
            batch_idxs_next.append(self.next_id[rnd_idx])

        batch_idxs = np.array(batch_idxs).astype(np.int)
        batch_idxs_next = np.array(batch_idxs_next).astype(np.int)
        batch_idx_pre = [self.prev_id[id] for id in batch_idxs]

        obs0_batch = self.replay_buffer[batch_idxs]
        obs1_batch = self.replay_buffer[batch_idxs_next]
        obs2_batch = self.sample_negative(batch_size, batch_idxs, batch_idxs_next, batch_idx_pre)
        action_batch = self.action_buffer[batch_idxs]
        reward_batch = self.reward_buffer[batch_idxs]
        terminal1_batch = self.done_buffer[batch_idxs]
        q_batch = self.q_values[batch_idxs]

        if mix:
            obs0_batch, obs1_batch = self.switch_first_half(obs0_batch, obs1_batch, batch_size)
        result = {
            'obs0': array_min2d(obs0_batch),
            'obs1': array_min2d(obs1_batch),
            'obs2': array_min2d(obs2_batch),
            'rewards': array_min2d(reward_batch),
            'actions': array_min2d(action_batch),
            'terminals1': array_min2d(terminal1_batch),
            'return': array_min2d(q_batch),
        }
        return result
