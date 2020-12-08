import matplotlib

# matplotlib.use('Agg')
# from pyvirtualdisplay import Display
# disp = Display(visible=1,size=(640,480)).start()

from stable_baselines import PPO2, PPO2Repr, logger
from stable_baselines.common.cmd_util import make_atari_env, atari_arg_parser
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy, MlpPolicy
from attn_toy.env.noisy_fourrooms import FourroomsDynamicNoise3, FourroomsDynamicNoise2, FourroomsDynamicNoise, \
    ImageInputWarpper, FourroomsRandomNoise, FourroomsOptimalNoise, FourroomsMyNoise
from attn_toy.env.fourrooms_multicoin import FourroomsMultiCoinRandomNoise
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from attn_toy.policies.attn_policy import AttentionPolicy
from attn_toy.value_iteration import value_iteration
import os
import numpy as np
from stable_baselines.common.cmd_util import make_atari, Monitor, wrap_deepmind, set_global_seeds, DummyVecEnv, \
    SubprocVecEnv, atari_arg_parser
from stable_baselines.a2c.rlgan_warpper import AtariRescale42x42, AtariNoisyBackground


def make_atari_env(env_id, num_env, seed, wrapper_kwargs=None,
                   start_index=0, allow_early_resets=True,
                   start_method=None, use_subprocess=False, variation="constant_rectangle"):
    """
    Create a wrapped, monitored VecEnv for Atari.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param wrapper_kwargs: (dict) the parameters for wrap_deepmind function
    :param start_index: (int) start rank index
    :param allow_early_resets: (bool) allows early reset of the environment
    :param start_method: (str) method used to start the subprocesses.
        See SubprocVecEnv doc for more information
    :param use_subprocess: (bool) Whether to use `SubprocVecEnv` or `DummyVecEnv` when
        `num_env` > 1, `DummyVecEnv` is usually faster. Default: False
    :return: (VecEnv) The atari environment
    """
    if wrapper_kwargs is None:
        wrapper_kwargs = {}

    def make_env(rank):
        def _thunk():
            env = make_atari(env_id)
            env.seed(seed + rank)
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
                          allow_early_resets=allow_early_resets)
            # env = AtariNoisyBackground(env)
            env = AtariRescale42x42(env, variation)
            return wrap_deepmind(env, **wrapper_kwargs)

        return _thunk

    set_global_seeds(seed)

    # When using one environment, no need to start subprocesses
    if num_env == 1 or not use_subprocess:
        return DummyVecEnv([make_env(i + start_index) for i in range(num_env)])

    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)],
                         start_method=start_method)


def train(train_env, test_env, finetune_num_timesteps, num_timesteps, policy, nminibatches=4, n_steps=128, repr_coef=0.,
          use_attention=True,
          test_interval=2048, replay_buffer=None):
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
    # model = PPO2(policy=policy, env=train_env, n_steps=n_steps, nminibatches=nminibatches,
    #              lam=0.95, gamma=0.99, noptepochs=4, ent_coef=.01,
    #              learning_rate=lambda f: f * 2.5e-4, cliprange=lambda f: f * 0.1, verbose=1)
    #
    # model.learn(total_timesteps=num_timesteps)
    model = PPO2Repr(policy=policy, env=train_env, test_env=test_env, n_steps=n_steps, nminibatches=nminibatches,
                     lam=0.95, gamma=0.99, noptepochs=10, ent_coef=.01,
                     learning_rate=lambda f: f * 2.5e-4, cliprange=lambda f: f * 0.1, verbose=1, repr_coef=repr_coef,
                     use_attention=use_attention,
                     replay_buffer=replay_buffer)

    for epoch in range(num_timesteps // test_interval):
        model.learn(total_timesteps=test_interval, reset_num_timesteps=epoch == 0, begin_eval=epoch == 0,
                    print_attention_map=epoch % 10 == 0)
        print(model.num_timesteps)
        model.eval(print_attention_map=True, filedir=os.getenv('OPENAI_LOGDIR'))
        print(model.num_timesteps)
        save_path = os.path.join(os.getenv('OPENAI_LOGDIR'), "save")
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        model.save(os.path.join(save_path, "model.pkl"))

    # finetune
    print("begin finetuning")
    model.learn(finetune_num_timesteps, finetune=True, reset_num_timesteps=True, begin_eval=True)
    train_env.close()
    test_env.close()
    # Free memory
    del model


def main():
    """
    Runs the test
    """
    parser = atari_arg_parser()
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm', 'mlp', 'attention'],
                        default='attention')
    parser.add_argument('--n_env', help='Policy architecture', type=int, default=8)
    parser.add_argument('--repr_coef', help='reprenstation loss coefficient', type=float, default=1.)
    parser.add_argument('--use-attention', help='whether or not add attention architecture in network', type=bool,
                        default=True)
    parser.add_argument('--variation',
                        choices=['standard', 'moving-square', 'constant-rectangle', 'green-lines', 'diagonals'],
                        default='constant-rectangle', help='Env variation')
    parser.add_argument('--finetune_num_timesteps', help='Policy architecture', type=int, default=131072)
    parser.add_argument('--env-id', help='env_id', type=str, default="Breakout")
    args = parser.parse_args()
    logger.configure()
    print("seed:", args.seed)
    # replay_buffer = value_iteration(make_gridworld(noise_type=3, seed=args.seed)(), gamma=1, filedir="/home/hh/attn/")
    # optimal_action = np.argmax(replay_buffer.returns[:replay_buffer.curr_capacity], axis=1)
    env = VecFrameStack(make_atari_env(args.env, args.n_env, args.seed, variation="standard"), 4)
    # env = VecFrameStack(make_atari_env(args.env, args.n_envs, args.seed), 4)
    test_env = VecFrameStack(make_atari_env(args.env, args.n_env, args.seed, variation=args.variation), 4)
    # [make_gridworld(noise_type=4, seed=args.seed, optimal_action=optimal_action) for _ in range(args.n_env)])
    # print(test_env)
    train(env, test_env, finetune_num_timesteps=args.finetune_num_timesteps, num_timesteps=args.num_timesteps,
          policy=args.policy, replay_buffer=None, repr_coef=args.repr_coef, use_attention=args.use_attention)


if __name__ == '__main__':
    main()

# disp.stop()
