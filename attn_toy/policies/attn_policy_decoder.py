from stable_baselines.common.policies import *
from attn_toy.policies.attn_policy import AttentionPolicy


class AttentionPolicyDecoder(AttentionPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None, net_arch=None,
                 act_fun=tf.tanh, cnn_extractor=attention_cnn_exposed, feature_extraction="cnn", num_actions=4,
                 add_attention=True,
                 **kwargs):
        ActorCriticPolicy.__init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
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
            with tf.variable_scope("feature", reuse=False):
                feature_map = cnn_extractor(self.processed_obs, **kwargs)
                attention, attentioned_feature_map = attention_mask(feature_map)
                reduced_feature_map = conv_to_fc(tf.reduce_mean(feature_map, axis=-1, keepdims=False))
                if add_attention:
                    used_feature_map = attentioned_feature_map
                else:
                    used_feature_map = feature_map
                self.feature_map_ph = tf.placeholder(tf.float32, used_feature_map.shape, name="feature_map_ph")
                self._mem_value_fn = linear(used_feature_map, 'mem_vf', num_actions, init_scale=np.sqrt(2))
                self.contra_repr = linear(used_feature_map, 'contra_repr', 32, init_scale=np.sqrt(2))
                # with tf.variable_scope("feature_value", reuse=False):
                #     feature_map_value = cnn_extractor(self.processed_obs, **kwargs)
                #     _, attentioned_feature_map_value = attention_mask(feature_map_value)
                img_size = ob_space.shape[1]
                print(img_size)
                self.decoded_image = deconv_decoder(used_feature_map, input_size=img_size // 8, img_size=img_size)
                self.decoded_image_from_feature_map = deconv_decoder(self.feature_map_ph, input_size=img_size // 8,
                                                                     img_size=img_size, reuse=True)
            with tf.variable_scope("last_layer", reuse=False):
                pi_latent = tf.nn.relu(
                    linear(used_feature_map, 'pi_latent', n_hidden=512, init_scale=np.sqrt(2)))
                # vf_latent = tf.nn.relu(linear(attentioned_feature_map_value, 'vi_latent', n_hidden=512, init_scale=np.sqrt(2)))
                vf_latent = pi_latent
                self.unattended_feature_map = feature_map
                self.reduced_feature_map = reduced_feature_map
                self.feature_map = attentioned_feature_map
                self.attention = attention
                self.pi_latent = pi_latent
                self.vf_latent = vf_latent
                self._value_fn = linear(vf_latent, 'vf', 1)
                # self._mem_value_fn = linear(pi_latent, 'mem_vf', 1)
                self._stop_gd_value_fn = linear(tf.stop_gradient(vf_latent), 'vf', 1, reuse=True)

                self._proba_distribution, self._policy, self.q_value = \
                    self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

                self._stop_gd_proba_distribution, self._stop_gd_policy, self.stop_gd_q_value = \
                    self.pdtype.proba_distribution_from_latent(tf.stop_gradient(pi_latent), tf.stop_gradient(vf_latent),
                                                               init_scale=0.01, reuse=True)

        self._setup_init()
        self.attention_saved = None
