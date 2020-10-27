from stable_baselines import PPO2, PPO2Repr, logger
from stable_baselines.common.cmd_util import make_atari_env, atari_arg_parser
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy, MlpPolicy
from attn_toy.env.fourrooms import FourroomsDynamicNoise3, FourroomsDynamicNoise2, FourroomsDynamicNoise, \
    ImageInputWarpper,FourroomsRandomNoise
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from attn_toy.policies.attn_policy import AttentionPolicy
import os


def train(train_env, test_env, finetune_num_timesteps, num_timesteps, policy, nminibatches=4, n_steps=128,
          test_interval=2048):
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
    model = PPO2Repr(policy=policy, env=train_env, test_env=test_env, n_steps=n_steps, nminibatches=nminibatches,
                     lam=0.95, gamma=0.99, noptepochs=4, ent_coef=.01,
                     learning_rate=lambda f: f * 2.5e-4, cliprange=lambda f: f * 0.1, verbose=1)

    # model = PPO2(policy=policy, env=train_env, n_steps=n_steps, nminibatches=nminibatches,
    #              lam=0.95, gamma=0.99, noptepochs=4, ent_coef=.01,
    #              learning_rate=lambda f: f * 2.5e-4, cliprange=lambda f: f * 0.1, verbose=1)
    for epoch in range(num_timesteps // test_interval):
        model.learn(total_timesteps=test_interval, reset_num_timesteps=epoch == 0)
        print(model.num_timesteps)
        model.eval(print_attention_map=True, filedir=os.getenv('OPENAI_LOGDIR'))
        print(model.num_timesteps)
    # model.learn(total_timesteps=num_timesteps)
    # finetune
    print("begin finetuning")
    model.learn(finetune_num_timesteps, finetune=True, reset_num_timesteps=True)
    train_env.close()
    test_env.close()
    # Free memory
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
    parser.add_argument('--n_env', help='Policy architecture', type=int, default=8)
    parser.add_argument('--finetune_num_timesteps', help='Policy architecture', type=int, default=100000)
    args = parser.parse_args()
    logger.configure()
    env = SubprocVecEnv([make_gridworld(noise_type=3, seed=args.seed) for _ in range(args.n_env)])
    # env = VecFrameStack(make_atari_env(args.env, args.n_envs, args.seed), 4)
    test_env = SubprocVecEnv([make_gridworld(noise_type=4, seed=args.seed) for _ in range(args.n_env)])
    # print(test_env)
    train(env, test_env, finetune_num_timesteps=args.finetune_num_timesteps, num_timesteps=args.num_timesteps,
          policy=args.policy)


if __name__ == '__main__':
    main()
