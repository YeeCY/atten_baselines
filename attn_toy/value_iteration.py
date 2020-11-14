import numpy as np
from attn_toy.memory.episodic_memory import EpisodicMemory
import pickle
import os


def value_iteration(env, gamma=0.99, buffer_size=2000, filedir=None):
    # print(env.color)
    if filedir is not None:
        filename = os.path.join(filedir, "replay_buffer.pkl")
        if os.path.exists(filename):
            with open(filename, "rb") as file:
                print("direct loading")
                return pickle.load(file)
    num_state = env.state_space_capacity
    values = np.zeros(num_state)
    transition = np.zeros((num_state, env.action_space.n))
    rewards = np.zeros((num_state, env.action_space.n))
    dones = np.zeros((num_state, env.action_space.n))
    obses = []
    for s in range(num_state):
        obs = None
        for a in range(env.action_space.n):
            obs = env.reset(s)
            # env.set_state(s)
            state_tp1, reward, done, info = env.step(a)
            if not isinstance(state_tp1, int):
                state_tp1 = info["s_tp1"]
            # print(state_tp1,s,a)
            # transition[s, a] = np.argmax(state_tp1).astype(np.int)
            transition[s, a] = state_tp1
            rewards[s, a] = reward
            dones[s, a] = done
        obses.append(obs)
    Q = np.zeros((num_state, env.action_space.n))
    for _ in range(len(values)):
        for s in range(len(values)):
            # q = np.zeros(env.action_space.n)
            for a in range(env.action_space.n):
                Q[s, a] = rewards[s, a]
                if not dones[s, a]:
                    Q[s, a] += gamma * values[int(transition[s, a])]
            values[s] = np.max(Q[s])
    # print(rewards)
    # print(transition)
    replay_buffer = EpisodicMemory(buffer_size, env.observation_space.shape, env.action_space.n)
    for s in range(num_state):
        for a in range(env.action_space.n):
            replay_buffer.store(obses[s], a, rewards[s, a], dones[s, a], 0.99 ** (100 - Q[s, a]),
                                env.reset(int(transition[s, a])))
    print(values)
    print("value iteration finished")
    # print(env.color)
    # assert len(obses) == len(values)
    # value_dict = {obs.astype(np.uint8).data.tobytes(): 0.99 ** (100 - value) for obs, value in zip(obses, values)}
    if filedir is not None:
        file = os.path.join(filedir, "replay_buffer.pkl")
        if not os.path.exists(file):
            if not os.path.exists(filedir):
                os.makedirs(filedir, exist_ok=True)
            with open(os.path.join(filedir, "replay_buffer.pkl"), "wb") as file:
                print("saving replaybuffer")
                pickle.dump(replay_buffer, file)

    return replay_buffer
