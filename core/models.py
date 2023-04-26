import math

import keras
import tensorflow as tf
from keras import layers

from keras.mixed_precision.loss_scale_optimizer import BaseLossScaleOptimizer
from official.nlp.modeling import layers as nlp_layers


class RegressionHead(layers.Layer):
    def __init__(self, num_outputs=2, dim_ff=64, activation='relu', random_feature=False, **kwargs):
        super().__init__(**kwargs)
        self._dim_ff = dim_ff
        self._num_outputs = num_outputs
        self._activation = activation
        self._random_feature = random_feature

        self._config_dict = {'dim_ff': dim_ff, 'num_outputs': num_outputs, 'activation': activation,
                             'random_feature': random_feature}

    def build(self, input_shape):

        if not self._random_feature and self._dim_ff > 0:
            reg_layers = [layers.Dense(self._dim_ff, kernel_initializer='HeNormal'),
                          layers.Activation(self._activation),
                          layers.Dense(self._num_outputs, activation='linear', kernel_initializer='HeNormal')]
        else:
            reg_layers = [layers.Dense(self._num_outputs, activation='linear', kernel_initializer='HeNormal')]
        self.reg_head = keras.Sequential(reg_layers)

    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        x = inputs

        outs = self.reg_head(x, training=training)

        if self._random_feature:
            outs = tf.cast(outs, dtype=inputs.dtype)
        return outs

    def get_config(self):
        return self._config_dict


class Conv1DProjection(layers.Layer):
    def __init__(self, proj_dim=128, kernel_size=1, dilation_rate=1, stride=1, activation='linear', padding='valid',
                 **kwargs):
        super().__init__(**kwargs)
        self._kernel_size = kernel_size
        self._proj_dim = proj_dim
        self._activation = activation
        self._dilation_rate = dilation_rate
        self._padding = padding
        self._stride = stride

        self._config_dict = {"proj_dim": proj_dim, "kernel_size": kernel_size, "dilation_rate": dilation_rate,
                             "activation": activation, "padding": padding, 'stride': stride}

    def build(self, input_shape):
        self.proj = keras.Sequential(
            [layers.Conv1D(filters=self._proj_dim, kernel_size=self._kernel_size, dilation_rate=self._dilation_rate,
                           padding=self._padding, strides=self._stride),
             layers.Activation(activation=self._activation),
             layers.LayerNormalization(dtype=tf.float32)
             ])

    def call(self, inputs, *args, **kwargs):
        input_dtype = inputs.dtype
        out = self.proj(inputs)
        out = tf.cast(out, input_dtype)

        return out

    def get_config(self):
        return self._config_dict


class FeatureTransform(layers.Layer):
    def __init__(self, num_feats, n_heads=4, n_layers=4, activation='gelu', inner_dim=256, dropout_rate=0.,
                 attn_dropout_rate=0., inner_dropout=0., use_bias=False, norm_first=True, norm_epsilon=1e-6,
                 output_dim=None, use_pos=True, **kwargs):
        super().__init__(**kwargs)

        self._num_feats = num_feats
        self._n_heads = n_heads
        self._n_layers = n_layers
        self._inner_dim = inner_dim
        self._activation = activation
        self._dropout_rate = dropout_rate
        self._inner_dropout = inner_dropout
        self._attn_dropout_rate = attn_dropout_rate
        self._use_bias = use_bias
        self._norm_first = norm_first
        self._norm_epsilon = norm_epsilon
        self._output_dim = output_dim
        self._use_pos = use_pos

    def build(self, input_shape):
        if self._use_pos:
            self.position_embedding = nlp_layers.RelativePositionEmbedding(hidden_size=self._num_feats)
        self.encoder_layers = []
        for i in range(self._n_layers):
            self.encoder_layers.append(
                nlp_layers.TransformerEncoderBlock(num_attention_heads=self._n_heads, inner_dim=self._inner_dim,
                                                   inner_activation=self._activation, output_dropout=self._dropout_rate,
                                                   attention_dropout=self._attn_dropout_rate, use_bias=self._use_bias,
                                                   norm_first=self._norm_first, norm_epsilon=self._norm_epsilon,
                                                   inner_dropout=self._inner_dropout,
                                                   name=f"layer_{i}"))
        self.output_norm = layers.LayerNormalization(epsilon=self._norm_epsilon, dtype=tf.float32)
        if self._output_dim is not None:
            self.proj_head = layers.Dense(self._output_dim, kernel_initializer='HeNormal')
        super(FeatureTransform, self).build(input_shape)

    def call(self, inputs, training=None):
        """

        :param inputs:  [batch_size, seq_dim, input_last_dim]
        :param training:
        :return:
        """
        if training is None:
            training = tf.keras.backend.learning_phase()

        x = inputs

        # Transformer encoder
        if self._use_pos:
            pos_encoding = self.position_embedding(x, training=training)
            pos_encoding = tf.cast(pos_encoding, inputs.dtype)
            x = x + pos_encoding

        for layer_idx in range(self._n_layers):
            x = self.encoder_layers[layer_idx](x, training=training)
        if self._output_dim is not None:
            x = self.proj_head(x)
        x = self.output_norm(x, training=training)
        # End of Transformer encoder

        x = tf.reduce_mean(x, axis=1)
        x = tf.cast(x, inputs.dtype)

        return x

    def get_config(self):
        config = {"num_feats": self._num_feats, "n_heads": self._n_heads, "n_layers": self._n_layers,
                  "activation": self._activation, "inner_dim": self._inner_dim, "dropout_rate": self._dropout_rate,
                  "attn_dropout_rate": self._attn_dropout_rate, "inner_dropout": self._inner_dropout,
                  "use_bias": self._use_bias, "norm_first": self._norm_first, "norm_epsilon": self._norm_epsilon,
                  'output_dim': self._output_dim,
                  }
        return config


class EPiCModel(tf.keras.Model):
    def __init__(self, seq_len, num_stages=4, num_outputs=2, dim_ff=64, n_heads=4, n_layers=4, hid_multiplier=4,
                 random_feature=False, variant=1, n_feats=1024, **kwargs):
        inputs = layers.Input(shape=(seq_len, 8), name='phys')

        if variant == 1:
            # Normalize each feature over time-dimension
            # Input size: B x L x 8, convert to B x 8 x L, do Normalize, and convert back
            x = tf.einsum("...ij->...ji", inputs)
            x = layers.LayerNormalization(axis=-1)(x)
            x = tf.einsum("...ij->...ji", x)

            inner_dim = dim_ff * hid_multiplier

            out_stages = []
            for idx in range(num_stages):
                x = Conv1DProjection(proj_dim=dim_ff, kernel_size=3, dilation_rate=1, activation='relu',
                                     padding='causal',
                                     stride=2, name=f"stage_{idx + 1}_proj")(x)
                x1 = FeatureTransform(dim_ff, n_heads=n_heads, n_layers=n_layers, inner_dim=inner_dim,
                                      name=f"stage_{idx + 1}_feat")(x)
                out_stages.append(x1)

            x = tf.concat(out_stages, axis=-1)

            if random_feature:
                x = tf.cast(x, dtype=tf.float32)
                x = tf.keras.layers.experimental.RandomFourierFeatures(dim_ff * num_stages * hid_multiplier,
                                                                       kernel_initializer='gaussian',
                                                                       dtype=tf.float32)(x)

            x = RegressionHead(num_outputs=num_outputs, dim_ff=dim_ff * num_stages * hid_multiplier, activation='relu',
                               random_feature=True)(x)

        elif variant == 2:
            # Input size: B x L x 8, convert to (B x 8) x L
            x = tf.einsum("...ij->...ji", inputs)
            x = tf.reshape(x, (-1, seq_len))
            # Compute RandomFeature
            x = tf.cast(x, dtype=tf.float32)
            n_feats = dim_ff * num_stages
            x = tf.keras.layers.experimental.RandomFourierFeatures(n_feats,
                                                                   kernel_initializer='gaussian', dtype=tf.float32)(x)
            x = tf.reshape(x, (-1, 8, n_feats))

            out_stages = []
            for idx in range(num_stages):
                inner_dim = n_feats * hid_multiplier
                x1 = FeatureTransform(n_feats, n_heads=n_heads, n_layers=n_layers, inner_dim=inner_dim,
                                      name=f"stage_{idx + 1}_feat")(x)
                out_stages.append(x1)

            x = tf.concat(out_stages, axis=-1)
            x = RegressionHead(num_outputs=num_outputs, dim_ff=dim_ff * num_stages * hid_multiplier, activation='relu',
                               random_feature=True)(x)

        elif variant == 3:

            x = inputs
            inner_dim = dim_ff * hid_multiplier
            out_stages = []
            # x0 = self.compute_output(-1, x, seq_len, use_random_feature=True, down_sampling=3, n_feats=dim_ff,
            #                          n_heads=n_heads, n_layers=n_layers, inner_dim=inner_dim, output_dim=n_feats,
            #                          swap_tf_dim=False)
            # out_stages.append(x0)

            for idx in range(num_stages):
                # Transformer for fusion between signals with original features in time domain
                x1 = self.compute_output(2 * idx, x, seq_len, use_random_feature=False, down_sampling=idx,
                                         n_feats=n_feats, n_heads=n_heads, n_layers=n_layers, inner_dim=inner_dim,
                                         output_dim=n_feats, swap_tf_dim=True)
                # Transformer for fusion between signals with original features transformed to non-linear domain with Gaussian features
                x2 = self.compute_output(2 * idx + 1, x, seq_len, use_random_feature=True, down_sampling=idx,
                                         n_feats=n_feats, n_heads=n_heads, n_layers=n_layers, inner_dim=inner_dim,
                                         output_dim=n_feats, swap_tf_dim=True)

                out_stages.append(x1)
                out_stages.append(x2)

            # out_stages = tf.stack(out_stages, axis=1)
            # out_stages = FeatureTransform(num_feats=n_feats, n_heads=n_heads, n_layers=1,
            #                               inner_dim=n_feats * hid_multiplier)(out_stages)
            out_stages = tf.concat(out_stages, axis=-1)
            x = RegressionHead(num_outputs=num_outputs, dim_ff=dim_ff)(out_stages)

        else:
            raise ValueError(f'Unknown EPiCModel with variant = {variant}')

        output = tf.cast(x, dtype=inputs.dtype)
        super(EPiCModel, self).__init__(inputs=[inputs, ], outputs=[output], **kwargs)

        self._config_dict = {'seq_len': seq_len, 'num_stages': num_stages, 'num_outputs': num_outputs, "dim_ff": dim_ff,
                             "n_heads": n_heads, "n_layers": n_layers, "hid_multiplier": hid_multiplier,
                             "random_feature": random_feature, "variant": variant}

    @staticmethod
    def compute_output(block_idx, x, seq_len, use_random_feature=True, down_sampling=0, n_feats=2048, n_heads=4,
                       n_layers=4, inner_dim=4096, output_dim=1024, swap_tf_dim=False):
        if down_sampling > 0:
            seq_len = int(math.ceil(seq_len / (2 ** down_sampling)))
            x = tf.keras.layers.AveragePooling1D(pool_size=2 ** down_sampling, strides=2 ** down_sampling,
                                                 dtype=tf.float32)(x)
        # print(block_idx, x.shape)
        if swap_tf_dim:
            # Convert x from B x L x C to B x C x L
            n_input_feats = seq_len
            x = tf.einsum("...ij->...ji", x)
            seq_len = 8
            use_pos = False
        else:
            n_input_feats = 8
            use_pos = True
            # # Input size: B x L x 8, convert to B x 8 x L
            # x = tf.einsum("...ij->...ji", x)
            # # Do normalize
            # x = layers.LayerNormalization(axis=-1)(x)
            # #  Convert back to B x L x 8
            # x = tf.einsum("...ij->...ji", x)

        if use_random_feature:
            x = tf.reshape(x, (-1, n_input_feats))
            in_dtype = x.dtype
            x = tf.cast(x, dtype=tf.float32)
            x = tf.keras.layers.experimental.RandomFourierFeatures(n_feats,
                                                                   kernel_initializer='gaussian', dtype=tf.float32)(x)
            x = tf.reshape(x, (-1, seq_len, n_feats))
            x = tf.cast(x, in_dtype)
            n_input_feats = n_feats

        # print(block_idx, n_input_feats)
        x = FeatureTransform(n_input_feats, n_heads=n_heads, n_layers=n_layers, inner_dim=inner_dim, use_pos=use_pos,
                             output_dim=output_dim, name=f"stage_{block_idx + 1}_feat")(x)

        return x

    def get_outputs_call(self, input_data, training=None):
        input_dict = {'phys': input_data['phys'][:, :, 1:]}
        return self(input_dict, training=training)

    def train_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data

        y_anno = y['anno'][:, 1:]
        with tf.GradientTape() as tape:
            y_pred = self.get_outputs_call(x, training=True)

            loss = self.compiled_loss(y_anno, y_pred, sample_weight=sample_weight,
                                      regularization_losses=self.losses)

            if isinstance(self.optimizer, BaseLossScaleOptimizer):
                loss = self.optimizer.get_scaled_loss(loss)

        # Compute gradient
        tvars = self.trainable_variables
        gradients = tape.gradient(loss, tvars)
        if isinstance(self.optimizer, BaseLossScaleOptimizer):
            gradients = self.optimizer.get_unscaled_gradients(gradients)

        self.optimizer.apply_gradients(zip(gradients, tvars))

        # Update the metrics.
        # Metrics are configured in `compile()`.
        self.compiled_metrics.update_state(y_anno, y_pred)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data

        y_anno = y['anno'][:, 1:]
        y_pred = self.get_outputs_call(x, training=False)

        self.compiled_loss(y_anno, y_pred, regularization_losses=self.losses)
        # Update the metrics.
        # Metrics are configured in `compile()`.
        self.compiled_metrics.update_state(y_anno, y_pred)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data

        y_pred = self.get_outputs_call(x, training=False)
        # Updates the metrics tracking the loss
        # self.compiled_loss(y_anno, y_pred, regularization_losses=self.losses)

        return y['anno'], y['name'], y_pred

    def get_config(self):
        return self._config_dict
