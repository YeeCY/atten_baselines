from stable_baselines.common.policies import *


class AttentionPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None, net_arch=None,
                 act_fun=tf.tanh, cnn_extractor=nature_cnn_exposed, feature_extraction="cnn", **kwargs):
        super(AttentionPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                              scale=(feature_extraction == "cnn"))

        self._kwargs_check(feature_extraction, kwargs)

        if layers is not None:
            warnings.warn("Usage of the `layers` parameter is deprecated! Use net_arch instead "
                          "(it has a different semantics though).", DeprecationWarning)
            if net_arch is not None:
                warnings.warn("The new `net_arch` parameter overrides the deprecated `layers` parameter!",
                              DeprecationWarning)

        if net_arch is None:
            if layers is None:
                layers = [64, 64]
            net_arch = [dict(vf=layers, pi=layers)]

        with tf.variable_scope("model", reuse=reuse):
            assert feature_extraction == "cnn", "Attention policy only support cnn extrator now"
            with tf.variable_scope("feature_pi", reuse=False):
                feature_map = cnn_extractor(self.processed_obs, **kwargs)
                attention, latent = attention_mask(feature_map)
            with tf.variable_scope("feature_value", reuse=False):
                feature_map_value = cnn_extractor(self.processed_obs, **kwargs)
                _, latent_value = attention_mask(feature_map_value)
            pi_latent = latent
            vf_latent = latent_value
            self.attention = attention
            self.pi_latent = pi_latent
            self.vf_latent = vf_latent
            self._value_fn = linear(vf_latent, 'vf', 1)
            self._stop_gd_value_fn = linear(tf.stop_gradient(vf_latent), 'vf', 1, reuse=True)

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

            self._stop_gd_proba_distribution, self._stop_gd_policy, self.stop_gd_q_value = \
                self.pdtype.proba_distribution_from_latent(tf.stop_gradient(pi_latent), tf.stop_gradient(vf_latent),
                                                           init_scale=0.01, reuse=True)

        self._setup_init()
        self.attention_saved = None

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp, attention_saved = self.sess.run(
                [self.deterministic_action, self.value_flat, self.neglogp, self.attention],
                {self.obs_ph: obs})
        else:
            action, value, neglogp, attention_saved = self.sess.run(
                [self.action, self.value_flat, self.neglogp, self.attention],
                {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp, attention_saved

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})
