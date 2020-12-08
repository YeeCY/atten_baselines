import time

import gym
import numpy as np
import tensorflow as tf

from stable_baselines import logger
from stable_baselines.common import explained_variance, tf_util, ActorCriticRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.policies import ActorCriticPolicy, RecurrentActorCriticPolicy
from stable_baselines.common.runners import AbstractEnvRunner
from stable_baselines.common.schedules import Scheduler
from stable_baselines.common.tf_util import mse, total_episode_reward_logger
from stable_baselines.common.math_util import safe_mean
import os
import cv2
from attn_toy.memory.episodic_memory import EpisodicMemory
from stable_baselines.ppo2.dqn_utils import get_true_return
from stable_baselines.common.policies import deconv_decoder, linear


def discount_with_dones(rewards, dones, gamma):
    """
    Apply the discount value to the reward, where the environment is not done

    :param rewards: ([float]) The rewards
    :param dones: ([bool]) Whether an environment is done or not
    :param gamma: (float) The discount value
    :return: ([float]) The discounted rewards
    """
    discounted = []
    ret = 0  # Return: discounted reward
    for reward, done in zip(rewards[::-1], dones[::-1]):
        ret = reward + gamma * ret * (1. - done)  # fixed off by one bug
        discounted.append(ret)
    return discounted[::-1]


class A2CRepr(ActorCriticRLModel):
    """
    The A2C (Advantage Actor Critic) model class, https://arxiv.org/abs/1602.01783

    :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) Discount factor
    :param n_steps: (int) The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param vf_coef: (float) Value function coefficient for the loss calculation
    :param ent_coef: (float) Entropy coefficient for the loss calculation
    :param max_grad_norm: (float) The maximum value for the gradient clipping
    :param learning_rate: (float) The learning rate
    :param alpha: (float)  RMSProp decay parameter (default: 0.99)
    :param momentum: (float) RMSProp momentum parameter (default: 0.0)
    :param epsilon: (float) RMSProp epsilon (stabilizes square root computation in denominator of RMSProp update)
        (default: 1e-5)
    :param lr_schedule: (str) The type of scheduler for the learning rate update ('linear', 'constant',
                              'double_linear_con', 'middle_drop' or 'double_middle_drop')
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
                              (used only for loading)
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    :param n_cpu_tf_sess: (int) The number of threads for TensorFlow operations
        If None, the number of cpu of the current machine will be used.
    """

    def __init__(self, policy, env, test_env=None, gamma=0.99, n_steps=5, vf_coef=0.25, ent_coef=0.01,
                 max_grad_norm=0.5,
                 learning_rate=7e-4, alpha=0.99, momentum=0.0, epsilon=1e-5, lr_schedule='constant',
                 repr_coef=1., contra_coef=1., atten_encoder_coef= 5 * 1. / 256, atten_decoder_coef=1.,
                 regularize_coef=1e-4, use_attention=True,
                 verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None, c_loss_type="origin",
                 full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None):

        self.n_steps = n_steps
        self.gamma = gamma
        self.vf_coef = vf_coef
        self.repr_coef = repr_coef
        self.repr_coef_ph = None
        self.contra_coef = contra_coef
        self.atten_encoder_coef = atten_encoder_coef
        self.atten_decoder_coef = atten_decoder_coef
        self.regularize_coef = regularize_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.alpha = alpha
        self.momentum = momentum
        self.epsilon = epsilon
        self.lr_schedule = lr_schedule
        self.learning_rate = learning_rate
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log
        self.use_attention = use_attention
        self.learning_rate_ph = None
        self.n_batch = None
        self.actions_ph = None
        self.advs_ph = None
        self.rewards_ph = None
        self.mem_return_ph = None
        self.pg_loss = None
        self.vf_loss = None
        self.entropy = None
        self.apply_backprop = None
        self.apply_backprop_repr = None
        self.train_model = None
        self.step_model = None
        self.proba_step = None
        self.value = None
        self.initial_state = None
        self.learning_rate_schedule = None
        self.summary = None
        self.num_actions = env.action_space.n
        self._test_runner = None
        self.c_loss_type = c_loss_type
        self.replay_buffer = EpisodicMemory(capacity=1000000, obs_shape=env.observation_space.shape,
                                            num_actions=env.action_space.n)
        self.recon_coef = 0.
        super(A2CRepr, self).__init__(policy=policy, env=env, verbose=verbose, requires_vec_env=True,
                                      _init_setup_model=_init_setup_model, policy_kwargs=policy_kwargs,
                                      seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)

        # if we are loading, it is possible the environment is not known, however the obs and action space are known
        self.policy_kwargs["add_attention"] = self.use_attention
        self.test_env = test_env
        if _init_setup_model:
            self.setup_model()

    def _make_runner(self) -> AbstractEnvRunner:
        return A2CRunner(self.env, self, n_steps=self.n_steps, gamma=self.gamma)

    def _make_test_runner(self):
        return A2CRunner(self.test_env, self, n_steps=self.n_steps, gamma=self.gamma)

    def _get_pretrain_placeholders(self):
        policy = self.train_model
        if isinstance(self.action_space, gym.spaces.Discrete):
            return policy.obs_ph, self.actions_ph, policy.policy
        return policy.obs_ph, self.actions_ph, policy.deterministic_action

    def setup_model(self):
        with SetVerbosity(self.verbose):

            assert issubclass(self.policy, ActorCriticPolicy), "Error: the input policy for the A2C model must be an " \
                                                               "instance of common.policies.ActorCriticPolicy."

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.set_random_seed(self.seed)
                self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)

                self.n_batch = self.n_envs * self.n_steps

                n_batch_step = None
                n_batch_train = None
                if issubclass(self.policy, RecurrentActorCriticPolicy):
                    n_batch_step = self.n_envs
                    n_batch_train = self.n_envs * self.n_steps

                step_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                                         n_batch_step, reuse=False, **self.policy_kwargs)

                with tf.variable_scope("loss", reuse=False):
                    self.target_actions_ph = step_model.pdtype.sample_placeholder([None], name="action_ph")
                with tf.variable_scope("train_model", reuse=True,
                                       custom_getter=tf_util.outer_scope_getter("train_model")):
                    train_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs,
                                              self.n_steps, n_batch_train, reuse=True, **self.policy_kwargs)
                    positive_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs,
                                                 self.n_steps, n_batch_train, reuse=True, **self.policy_kwargs)
                    negative_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs,
                                                 self.n_steps, n_batch_train, reuse=True, **self.policy_kwargs)
                    target_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs,
                                               self.n_steps, n_batch_train, reuse=True, **self.policy_kwargs)
                    attention_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs,
                                                  self.n_steps, n_batch_train, reuse=True, **self.policy_kwargs)
                    with tf.variable_scope("model/feature/predict") as scope:
                        # print(self.actions_ph.shape)
                        flat_residue_feature_map = tf.layers.flatten(target_model.residue_feature_map)
                        self.predicted_feature_map = linear(
                            tf.concat([flat_residue_feature_map,
                                       tf.one_hot(self.target_actions_ph, depth=self.num_actions)], axis=1),
                            scope=scope,
                            n_hidden=flat_residue_feature_map.shape[-1],
                            reuse=tf.AUTO_REUSE)

                    img_size = self.env.observation_space.shape[1]
                    with tf.variable_scope("model/feature/recon",reuse=tf.AUTO_REUSE):
                        self.decoded_image_target = deconv_decoder(
                            tf.reshape(target_model.feature_map, (-1, 10, 10, 16)), input_size=img_size // 8,
                            img_size=img_size,reuse=False)

                        combined_feature_map = target_model.residue_feature_map + self.predicted_feature_map

                        self.decoded_image_predict = deconv_decoder(
                            tf.reshape(combined_feature_map, (-1, 10, 10, 16)), input_size=img_size // 8,
                            img_size=img_size, reuse=True) / 84

                with tf.variable_scope("loss", reuse=False):
                    self.actions_ph = train_model.pdtype.sample_placeholder([None], name="action_ph")
                    self.advs_ph = tf.placeholder(tf.float32, [None], name="advs_ph")
                    self.rewards_ph = tf.placeholder(tf.float32, [None], name="rewards_ph")
                    self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")
                    self.mem_return_ph = tf.placeholder(tf.float32, [None, self.num_actions], name="mem_return_ph")
                    self.repr_coef_ph = tf.placeholder(tf.float32, [None], name="repr_coef_ph")
                    neglogpac = train_model.proba_distribution.neglogp(self.actions_ph)
                    self.entropy = tf.reduce_mean(train_model.proba_distribution.entropy())
                    self.pg_loss = tf.reduce_mean(self.advs_ph * neglogpac)
                    self.vf_loss = mse(tf.squeeze(train_model.value_flat), self.rewards_ph)

                    emb_cur = target_model.contra_repr
                    emb_next = positive_model.contra_repr
                    emb_neq = negative_model.contra_repr

                    self.contrastive_loss = self.contra_coef * \
                                            (self.contrastive_loss_fc(emb_cur, emb_next, emb_neq,
                                                                      c_type=self.c_loss_type) +
                                             self.contrastive_loss_fc(emb_next, emb_cur, emb_neq,
                                                                      c_type=self.c_loss_type))

                    self.encoder_loss = self.atten_encoder_coef * tf.reduce_mean(
                        tf.norm(attention_model.hard_attention, ord=1, axis=1))
                    # self.weight_loss = tf.norm(self.train_model.weighted_w,ord=1)
                    self.decoder_loss = self.atten_decoder_coef * tf.reduce_mean(
                        tf.square(attention_model.mem_value_fn - self.mem_return_ph))


                    # print(target_model.feature_map_raw.shape)

                    self.recon_origin_loss = tf.keras.losses.BinaryCrossentropy()(target_model.processed_obs,
                                                                                  self.decoded_image_target)
                    self.predict_loss = tf.keras.losses.BinaryCrossentropy()(positive_model.processed_obs,
                                                                             self.decoded_image_predict)

                    self.reconstruct_loss = 1/84*(self.recon_origin_loss + self.predict_loss)
                    self.repr_loss = self.contrastive_loss + self.encoder_loss + self.decoder_loss + self.recon_coef * self.reconstruct_loss
                    # https://arxiv.org/pdf/1708.04782.pdf#page=9, https://arxiv.org/pdf/1602.01783.pdf#page=4
                    # and https://github.com/dennybritz/reinforcement-learning/issues/34
                    # suggest to add an entropy component in order to improve exploration.

                    a2c_loss = self.pg_loss - self.entropy * self.ent_coef + self.vf_loss * self.vf_coef
                    loss = a2c_loss + self.repr_coef * self.repr_loss
                    tf.summary.scalar('entropy_loss', self.entropy)
                    tf.summary.scalar('policy_gradient_loss', self.pg_loss)
                    tf.summary.scalar('value_function_loss', self.vf_loss)
                    tf.summary.scalar('loss', loss)

                    self.params = tf_util.get_trainable_vars("model")
                    self.repr_params = tf_util.get_trainable_vars("model/feature")
                    print(self.params)
                    grads = tf.gradients(a2c_loss, self.params)
                    if self.max_grad_norm is not None:
                        grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
                    grads = list(zip(grads, self.params))

                    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.repr_params]) * self.regularize_coef

                    repr_grads = tf.gradients(self.repr_coef_ph * (self.repr_loss + l2_loss), self.repr_params)
                    repr_grads = list(zip(repr_grads, self.repr_params))

                    trainer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate_ph, decay=self.alpha,
                                                        epsilon=self.epsilon, momentum=self.momentum)
                    # trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph, epsilon=1e-5)
                    repr_trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph, epsilon=1e-5)
                    self.apply_backprop = trainer.apply_gradients(grads)
                    self.apply_backprop_repr = repr_trainer.apply_gradients(repr_grads)

                    with tf.variable_scope("input_info", reuse=False):
                        tf.summary.scalar('discounted_rewards', tf.reduce_mean(self.rewards_ph))
                    tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))
                    tf.summary.scalar('advantage', tf.reduce_mean(self.advs_ph))
                    if self.full_tensorboard_log:
                        tf.summary.histogram('discounted_rewards', self.rewards_ph)
                    tf.summary.histogram('learning_rate', self.learning_rate_ph)
                    tf.summary.histogram('advantage', self.advs_ph)
                    if tf_util.is_image(self.observation_space):
                        tf.summary.image('observation', train_model.obs_ph)
                    else:
                        tf.summary.histogram('observation', train_model.obs_ph)

                    self.train_model = train_model
                    self.step_model = step_model
                    self.target_model = target_model
                    self.positive_model = positive_model
                    self.negative_model = negative_model
                    self.attention_model = attention_model
                    self.step = step_model.step
                    self.proba_step = step_model.proba_step
                    self.value = step_model.value
                    self.initial_state = step_model.initial_state
                    tf.global_variables_initializer().run(session=self.sess)

                    self.summary = tf.summary.merge_all()

    def _train_step(self, repr_coef, action_target, obs_target, obs_pos, obs_neg, obs_value, true_returns_target,
                    obs, states, rewards, masks, actions, values, update, writer=None):
        """
        applies a training step to the model

        :param obs: ([float]) The input observations
        :param states: ([float]) The states (used for recurrent policies)
        :param rewards: ([float]) The rewards from the environment
        :param masks: ([bool]) Whether or not the episode is over (used for recurrent policies)
        :param actions: ([float]) The actions taken
        :param values: ([float]) The logits values
        :param update: (int) the current step iteration
        :param writer: (TensorFlow Summary.writer) the writer for tensorboard
        :return: (float, float, float) policy loss, value loss, policy entropy
        """
        advs = rewards - values
        cur_lr = None
        for _ in range(len(obs)):
            cur_lr = self.learning_rate_schedule.value()
        assert cur_lr is not None, "Error: the observation input array cannon be empty"

        td_map = {self.train_model.obs_ph: obs, self.actions_ph: actions, self.advs_ph: advs,
                  self.rewards_ph: rewards, self.learning_rate_ph: cur_lr,
                  self.target_model.obs_ph: obs_target,
                  self.attention_model.obs_ph: obs_value,
                  self.positive_model.obs_ph: obs_pos,
                  self.negative_model.obs_ph: obs_neg,
                  self.mem_return_ph: true_returns_target,
                  self.repr_coef_ph: repr_coef,
                  self.target_actions_ph: action_target
                  }
        if states is not None:
            td_map[self.train_model.states_ph] = states
            td_map[self.train_model.dones_ph] = masks

        if writer is not None:
            # run loss backprop with summary, but once every 10 runs save the metadata (memory, compute time, ...)
            if self.full_tensorboard_log and (1 + update) % 10 == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, reconstruct_loss, repr_loss, contrastive_loss, atten_encoder_loss, atten_decoder_loss, policy_loss, value_loss, policy_entropy, _, _ = self.sess.run(
                    [self.summary, self.reconstruct_loss, self.repr_loss, self.contrastive_loss, self.encoder_loss,
                     self.decoder_loss,
                     self.pg_loss, self.vf_loss, self.entropy, self.apply_backprop, self.apply_backprop_repr],
                    td_map, options=run_options, run_metadata=run_metadata)
                writer.add_run_metadata(run_metadata, 'step%d' % (update * self.n_batch))
            else:
                summary, reconstruct_loss, repr_loss, contrastive_loss, atten_encoder_loss, atten_decoder_loss, policy_loss, value_loss, policy_entropy, _, _ = self.sess.run(
                    [self.summary, self.reconstruct_loss, self.repr_loss, self.contrastive_loss, self.encoder_loss,
                     self.decoder_loss,
                     self.pg_loss, self.vf_loss, self.entropy, self.apply_backprop, self.apply_backprop_repr], td_map)
            writer.add_summary(summary, update * self.n_batch)

        else:
            reconstruct_loss, repr_loss, contrastive_loss, atten_encoder_loss, atten_decoder_loss, policy_loss, value_loss, policy_entropy, _, _ = self.sess.run(
                [self.reconstruct_loss, self.repr_loss, self.contrastive_loss, self.encoder_loss, self.decoder_loss,
                 self.pg_loss, self.vf_loss, self.entropy, self.apply_backprop, self.apply_backprop_repr], td_map)

        return reconstruct_loss, repr_loss, contrastive_loss, atten_encoder_loss, atten_decoder_loss, policy_loss, value_loss, policy_entropy

    def restore_map(self, flatten_map, obs_shape):
        length = int(np.sqrt(np.size(flatten_map)))
        flatten_map = flatten_map.reshape(length, length)
        flatten_map = (flatten_map - np.min(flatten_map)) / (np.max(flatten_map) - np.min(flatten_map) + 1e-12)
        flatten_map = cv2.resize(flatten_map, (obs_shape[0], obs_shape[1]))

        flatten_map = np.repeat(flatten_map[..., np.newaxis], 3, axis=2)
        return flatten_map

    def save_attention(self, attention, obs, recon, feature_map, subdir, step, num):
        # subdir = os.path.join(filedir, "./attention")
        # print(attention.squeeze())

        attention = self.restore_map(attention, obs.shape)
        feature_map = self.restore_map(feature_map, obs.shape)
        image = np.array(obs)[..., :3]

        recon = np.array(recon)[0, ..., :3]
        # print(image.shape)
        # print(recon.shape)
        attentioned_image = image * attention
        if not os.path.isdir(subdir):
            os.makedirs(os.path.join(subdir, "./mask/"))
            os.makedirs(os.path.join(subdir, "./masked_image/"))
            os.makedirs(os.path.join(subdir, "./image/"))
            os.makedirs(os.path.join(subdir, "./feature_map/"))
            os.makedirs(os.path.join(subdir, "./reconstruction/"))
        # print(attention.shape)
        cv2.imwrite(os.path.join(subdir, "./masked_image/", "masked_image_{}_{}.png".format(step, num)),
                    attentioned_image)
        # attentioned_image)
        cv2.imwrite(os.path.join(subdir, "./mask/", "attention_{}_{}.png".format(step, num)),
                    # attention * 255)
                    attention * 255)

        cv2.imwrite(os.path.join(subdir, "./feature_map/", "feature_map_{}_{}.png".format(step, num)),
                    # attention * 255)
                    feature_map * 255)

        cv2.imwrite(os.path.join(subdir, "./image/", "obs_{}_{}.png".format(step, num)),
                    # image * 255)
                    image)
        cv2.imwrite(os.path.join(subdir, "./reconstruction/", "recon_{}_{}.png".format(step, num)),
                    # image * 255)
                    recon * 255)

    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="A2C",
              reset_num_timesteps=True, begin_eval=False, print_attention_map=True, filedir=None, repr_coef=[1.]):
        print(repr_coef)
        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        callback = self._init_callback(callback)
        if begin_eval:
            self.replay_buffer.empty()
        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn()
            self.learning_rate_schedule = Scheduler(initial_value=self.learning_rate, n_values=total_timesteps,
                                                    schedule=self.lr_schedule)

            t_start = time.time()
            callback.on_training_start(locals(), globals())

            for update in range(1, total_timesteps // self.n_batch + 1):

                callback.on_rollout_start()
                # true_reward is the reward without discount
                rollout = self.runner.run(callback)
                # unpack
                obs, states, rewards, masks, actions, values, ep_infos, true_reward, attention, feature_map = rollout
                self.ep_info_buf.extend(ep_infos)
                true_returns = get_true_return(true_reward, masks, self.n_envs)
                self.replay_buffer.add_batch(obs, actions, true_reward, true_returns, masks)
                callback.update_locals(locals())
                callback.on_rollout_end()

                # Early stopping due to the callback
                if not self.runner.continue_training:
                    break

                obs_t_batch, act_t_batch, rew_t_batch, obs_tp1_batch, done_mask_batch, obs_neg_batch, _ = self.replay_buffer.sample(
                    len(obs))
                obs_value_batch, actions_target, _, _, _, _, return_t_batch = self.replay_buffer.sample(
                    len(obs))

                # deal with Nan/Inf values
                predict_returns = self.sess.run(self.attention_model.mem_value_fn,
                                                {self.attention_model.obs_ph: obs_value_batch})
                predict_returns = predict_returns.reshape(-1)

                return_t_batch = np.array(return_t_batch)
                shape = return_t_batch.shape
                return_t_batch = return_t_batch.reshape(-1)
                return_t_batch[np.isinf(return_t_batch)] = predict_returns[np.isinf(return_t_batch)]
                return_t_batch = return_t_batch.reshape(shape)

                reconstruct_loss, repr_loss, contrastive_loss, atten_encoder_loss, atten_decoder_loss, policy_loss, value_loss, policy_entropy = self._train_step(
                    repr_coef, actions_target,
                    obs_t_batch, obs_tp1_batch, obs_neg_batch,
                    obs_value_batch, return_t_batch,
                    obs, states, rewards, masks, actions, values,
                    self.num_timesteps // self.n_batch, writer)

                n_seconds = time.time() - t_start
                fps = int((update * self.n_batch) / n_seconds)
                # print("test tensorboard",self.tensorboard_log,writer)
                if writer is not None:
                    total_episode_reward_logger(self.episode_reward,
                                                true_reward.reshape((self.n_envs, self.n_steps)),
                                                masks.reshape((self.n_envs, self.n_steps)),
                                                writer, self.num_timesteps)

                if self.verbose >= 1 and (update % log_interval == 0 or update == 1):
                    explained_var = explained_variance(values, rewards)
                    logger.record_tabular("nupdates", update)
                    logger.record_tabular("total_timesteps", self.num_timesteps)
                    logger.record_tabular("fps", fps)
                    logger.record_tabular("policy_entropy", float(policy_entropy))
                    logger.record_tabular("value_loss", float(value_loss))
                    logger.record_tabular("policy_loss", float(policy_loss))
                    logger.record_tabular("contrastive_loss", float(contrastive_loss))
                    logger.record_tabular("atten_decoder_loss", float(atten_decoder_loss))
                    logger.record_tabular("atten_encoder_loss", float(atten_encoder_loss))
                    logger.record_tabular("reconstruct_loss", float(reconstruct_loss))
                    logger.record_tabular("explained_variance", float(explained_var))
                    if len(self.ep_info_buf) > 0 and len(self.ep_info_buf[0]) > 0:
                        logger.logkv('ep_reward_mean', safe_mean([ep_info['r'] for ep_info in self.ep_info_buf]))
                        logger.logkv('ep_len_mean', safe_mean([ep_info['l'] for ep_info in self.ep_info_buf]))
                    logger.dump_tabular()
                # print("saving image")
                # save attention image
                if print_attention_map and update == 1:
                    if filedir is None:
                        filedir = os.getenv('OPENAI_LOGDIR')
                        filedir = os.path.join(filedir, "attention_train")
                    rnd_indices = np.random.choice(len(obs), min(5, len(obs)), replace=False)
                    for i in range(len(rnd_indices)):
                        ind = rnd_indices[i]
                        recon = self.sess.run(self.decoded_image_target,
                                              {self.target_model.obs_ph: obs[np.newaxis, ind]})
                        self.save_attention(attention[ind], obs[ind], recon, feature_map[i], filedir,
                                            self.num_timesteps,
                                            i)
        callback.on_training_end()
        return self

    @property
    def test_runner(self) -> AbstractEnvRunner:
        if self._test_runner is None:
            self._test_runner = self._make_test_runner()
        return self._test_runner

    def eval(self, total_timesteps=100000, callback=None, log_interval=10, tb_log_name="A2C",
             print_attention_map=False, filedir=None):

        new_tb_log = self._init_num_timesteps(False)
        callback = self._init_callback(callback)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_test()

            t_start = time.time()
            callback.on_training_start(locals(), globals())
            for update in range(1, total_timesteps // self.n_batch + 1):

                # true_reward is the reward without discount
                rollout = self.test_runner.run(callback)
                # print(self.test_env, self.test_runner)
                # unpack
                obs, states, rewards, masks, actions, values, ep_infos, true_reward, attention, feature_map = rollout
                # print(masks)
                self.ep_info_buf_test.extend(ep_infos)

                n_seconds = time.time() - t_start
                fps = int(self.n_batch / n_seconds)
                # print("test tensorboard",self.tensorboard_log,writer)
                if writer is not None:
                    total_episode_reward_logger(self.episode_reward,
                                                true_reward.reshape((self.n_envs, self.n_steps)),
                                                masks.reshape((self.n_envs, self.n_steps)),
                                                writer, self.num_timesteps)

                if self.verbose >= 1 and (update % log_interval == 0 or update == 1):
                    explained_var = explained_variance(values, rewards)
                    logger.record_tabular("total_timesteps_eval", self.num_timesteps)
                    logger.record_tabular("fps_eval", fps)
                    logger.record_tabular("explained_variance_eval", float(explained_var))
                    if len(self.ep_info_buf_test) > 0 and len(self.ep_info_buf_test[0]) > 0:
                        logger.logkv('ep_reward_mean_eval',
                                     safe_mean([ep_info['r'] for ep_info in self.ep_info_buf_test]))
                        logger.logkv('ep_len_mean_eval', safe_mean([ep_info['l'] for ep_info in self.ep_info_buf_test]))

                    # print(self.ep_info_buf_test, ep_infos,len(obs))
                    logger.dump_tabular()

                # save attention image
                if print_attention_map and update == 1:
                    if filedir is None:
                        filedir = os.getenv('OPENAI_LOGDIR')
                        filedir = os.path.join(filedir, "attention_eval")
                    rnd_indices = np.random.choice(len(obs), min(5, len(obs)), replace=False)
                    for i in range(len(rnd_indices)):
                        ind = rnd_indices[i]
                        recon = self.sess.run(self.decoded_image_target,
                                              {self.target_model.obs_ph: obs[np.newaxis, ind]})
                        self.save_attention(attention[ind], obs[ind], recon, feature_map[i], filedir,
                                            self.num_timesteps,
                                            i)
            # print("saving image")
            # self.save_img(obs[np.random.randint(0, len(obs), 2)], numsteps=self.num_timesteps)
        callback.on_training_end()
        return self

    def save_img(self, obs, file_path=None, numsteps=0):
        if file_path is None:
            file_path = os.getenv("OPENAI_LOGDIR")
        if not os.path.isdir(os.path.join(file_path, "./image/")):
            os.makedirs(os.path.join(file_path, "./image/"))
        # print(file_path)
        # print(np.max(obs))
        for i in range(len(obs)):
            image = np.array(obs[i, :, :, :1])
            cv2.imwrite(os.path.join(file_path, "./image/", "obs_{}_{}.png".format(numsteps, i)),
                        image)
            # image)

    def save(self, save_path, cloudpickle=False):
        data = {
            "gamma": self.gamma,
            "n_steps": self.n_steps,
            "vf_coef": self.vf_coef,
            "ent_coef": self.ent_coef,
            "max_grad_norm": self.max_grad_norm,
            "learning_rate": self.learning_rate,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            "lr_schedule": self.lr_schedule,
            "verbose": self.verbose,
            "policy": self.policy,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "n_envs": self.n_envs,
            "n_cpu_tf_sess": self.n_cpu_tf_sess,
            "seed": self.seed,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs
        }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)


class A2CRunner(AbstractEnvRunner):
    def __init__(self, env, model, n_steps=5, gamma=0.99):
        """
        A runner to learn the policy of an environment for an a2c model

        :param env: (Gym environment) The environment to learn from
        :param model: (Model) The model to learn
        :param n_steps: (int) The number of steps to run for each environment
        :param gamma: (float) Discount factor
        """
        super(A2CRunner, self).__init__(env=env, model=model, n_steps=n_steps)
        self.gamma = gamma

    def _run(self):
        """
        Run a learning step of the model

        :return: ([float], [float], [float], [bool], [float], [float])
                 observations, states, rewards, masks, actions, values
        """
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [], []
        mb_attention, mb_featuremap = [], []
        mb_states = self.states
        ep_infos = []
        for _ in range(self.n_steps):

            actions, values, states, _, attention, feature_map = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            mb_attention.append(attention)
            mb_featuremap.append(feature_map)
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.env.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)
            obs, rewards, dones, infos = self.env.step(clipped_actions)

            self.model.num_timesteps += self.n_envs

            if self.callback is not None:
                # Abort training early
                self.callback.update_locals(locals())
                if self.callback.on_step() is False:
                    self.continue_training = False
                    # Return dummy values
                    return [None] * 8

            for info in infos:
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    ep_infos.append(maybe_ep_info)

            self.states = states
            self.dones = dones
            self.obs = obs
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(0, 1)
        mb_actions = np.asarray(mb_actions, dtype=self.env.action_space.dtype).swapaxes(0, 1)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(0, 1)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(0, 1)
        mb_attention = np.asarray(mb_attention, dtype=np.float32)
        mb_featuremap = np.asarray(mb_featuremap, dtype=np.float32)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        true_rewards = np.copy(mb_rewards)
        last_values = self.model.value(self.obs, self.states, self.dones).tolist()
        # discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards + [value], dones + [0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards

        # convert from [n_env, n_steps, ...] to [n_steps * n_env, ...]
        mb_rewards = mb_rewards.reshape(-1, *mb_rewards.shape[2:])
        mb_actions = mb_actions.reshape(-1, *mb_actions.shape[2:])
        mb_values = mb_values.reshape(-1, *mb_values.shape[2:])
        mb_masks = mb_masks.reshape(-1, *mb_masks.shape[2:])
        mb_attention = mb_attention.reshape(-1, *mb_attention.shape[2:])
        mb_featuremap = mb_featuremap.reshape(-1, *mb_featuremap.shape[2:])
        true_rewards = true_rewards.reshape(-1, *true_rewards.shape[2:])
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, ep_infos, true_rewards, mb_attention, mb_featuremap
