import numpy as np


def value_iteration(env, gamma=0.99):
    num_state = env.state_space_capacity
    values = np.zeros(num_state)
    transition = np.zeros((num_state, env.action_space.n))
    rewards = np.zeros((num_state, env.action_space.n))
    dones = np.zeros((num_state, env.action_space.n))
    obses = []
    for s in range(num_state):
        for a in range(env.action_space.n):
            obs = env.reset(s)
            # env.set_state(s)
            state_tp1, reward, done, info = env.step(a)
            if not isinstance(state_tp1,int):
                state_tp1 = info["s_tp1"]
            # print(state_tp1,s,a)
            # transition[s, a] = np.argmax(state_tp1).astype(np.int)
            transition[s, a] = state_tp1
            rewards[s, a] = reward
            dones[s, a] = done
            obses.append(obs)

    for _ in range(len(values)):
        for s in range(len(values)):
            q = np.zeros(env.action_space.n)
            for a in range(env.action_space.n):
                q[a] = rewards[s, a]
                if not dones[s, a]:
                    q[a] += gamma * values[int(transition[s, a])]
            values[s] = np.max(q)
    # print(rewards)
    # print(transition)
    print(values)
    print("value iteration finished")
    value_dict = {obs.data.tobytes(): value for obs, value in zip(obses, values)}
    return value_dict
