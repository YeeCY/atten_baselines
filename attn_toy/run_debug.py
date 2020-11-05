import matplotlib

# matplotlib.use('Agg')
# from pyvirtualdisplay import Display
# disp = Display(visible=1,size=(640,480)).start()

from stable_baselines import PPO2, PPO2Repr, logger
from stable_baselines.common.cmd_util import make_atari_env, atari_arg_parser
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy, MlpPolicy
from attn_toy.env.fourrooms import FourroomsDynamicNoise3, FourroomsDynamicNoise2, FourroomsDynamicNoise, \
    ImageInputWarpper, FourroomsRandomNoise
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from attn_toy.policies.attn_policy import AttentionPolicy
from attn_toy.agent.human_agent import HumanAgent
from attn_toy.agent.hybrid_agent import HybridAgent, HybridAgent2
import os
import numpy as np
import matplotlib.pyplot as plt


def train(env, load_path, num_timesteps, policy, nminibatches=4, n_steps=128):
    """
    Train PPO2 model for atari environment, for testing purposes

    :param env_id: (str) the environment id string
    :param num_timesteps: (int) the number of timesteps to run
    :param seed: (int) Used to seed the random generator.
    :param policy: (Object) The policy model to use (MLP, CNN, LSTM, ...)
    :param n_envs: (int) Number of parallel environments
    :param nminibatches: (int) Number of training minibatches per update. For recurrent policies,
        the number of environments run in parallel should be a multiple of nminibatches.
    :param n_steps: (int) The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    """

    policy = {'cnn': CnnPolicy, 'lstm': CnnLstmPolicy, 'lnlstm': CnnLnLstmPolicy, 'mlp': MlpPolicy,
              'attention': AttentionPolicy}[policy]
    # print(test_env)
    # model = PPO2Repr(policy=policy, env=env, test_env=env, n_steps=n_steps, nminibatches=nminibatches,
    #                  lam=0.95, gamma=0.99, noptepochs=4, ent_coef=.01,
    #                  learning_rate=lambda f: f * 2.5e-4, cliprange=lambda f: f * 0.1, verbose=1)
    model = PPO2Repr.load(load_path=load_path)

    # magic_num = tf.get_variable("magic")
    model.test_magic_num()

    human_agent = HumanAgent({"3": 3, "1": 1, "0": 0, "2": 2})
    hybrid_agent = HybridAgent2(human_agent, model)
    obs = env.reset()
    obses = []
    x = model.params[8].eval(session=model.sess)
    x = np.mean(x, axis=1)
    x = np.reshape(x, (17, 17, 64))
    x = np.mean(x, axis=2)
    plt.imshow(x, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.show()
    for step in range(num_timesteps):
        action = hybrid_agent.act(obs[np.newaxis, ...], is_train=False)
        obs, reward, done, info = env.step(action)
        obses.append(obs)
        hybrid_agent.observe(obs[np.newaxis, ...], reward, done, info, train=False)
        feature_map = model.sess.run(model.act_model.feature_map,
                                     feed_dict={model.act_model.obs_ph: obs[np.newaxis, ...]})
        feature_map = feature_map.reshape(17,17,64)
        feature_map = np.mean(np.abs(feature_map),axis=2)
        plt.imshow(feature_map, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.show()
        print("reward:", reward)
        if done:
            obs = env.reset()
    del model


def make_gridworld(noise_type=1, seed=0):
    envs = {1: FourroomsDynamicNoise, 2: FourroomsDynamicNoise2, 3: FourroomsDynamicNoise3,
            4: FourroomsRandomNoise}
    env = envs.get(noise_type, FourroomsDynamicNoise)

    def env_fn():
        return ImageInputWarpper(env(seed=seed))

    return env_fn


def main():
    """
    Runs the test
    """
    parser = atari_arg_parser()
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm', 'mlp', 'attention'],
                        default='attention')
    parser.add_argument('--load_path', help='Policy architecture', type=str, default=None)
    # parser.add_argument('--policy', help='Policy architecture', type=str, default=None)
    args = parser.parse_args()
    logger.configure()
    env = make_gridworld(noise_type=3, seed=args.seed)()

    train(env, args.load_path, num_timesteps=args.num_timesteps, policy=args.policy)


if __name__ == '__main__':
    main()
