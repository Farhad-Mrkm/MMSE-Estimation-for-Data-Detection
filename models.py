import os
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

import tensorflow_probability as tfp

import hyperparameters


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



def Coupling(input_shape):
    input = layers.Input(shape=input_shape)

    t_layer = input
    for _ in range(hyperparameters.N_ST_LAYER):
        t_layer = layers.Dense(
            hyperparameters.HIDDEN_DIM, activation=hyperparameters.ACT, kernel_regularizer=regularizers.l2(hyperparameters.L2)
        )(t_layer)
    t_layer = layers.Dense(
        input_shape, activation="linear", kernel_regularizer=regularizers.l2(hyperparameters.L2)
    )(t_layer)


    s_layer = input
    for _ in range(hyperparameters.N_ST_LAYER):
        s_layer = layers.Dense(
            hyperparameters.HIDDEN_DIM, activation=hyperparameters.ACT, kernel_regularizer=regularizers.l2(hyperparameters.L2)
        )(s_layer)
    s_layer = layers.Dense(
        input_shape, activation="tanh", kernel_regularizer=regularizers.l2(hyperparameters.L2)
    )(s_layer)

    return keras.Model(inputs=input, outputs=[s_layer, t_layer])


def checkerboard_mask(num_coupling_layers, n_variable=2, dtype=np.float32):

    checkerboard = [[((i % 2) + j) % 2 for j in range(n_variable)] for i in range(n_variable)]
    mask = np.array(checkerboard, dtype=dtype)

    mask = mask.reshape(n_variable, n_variable)
    mask = np.tile(mask, [num_coupling_layers // n_variable, 1])

    return mask


class RealNVP(keras.Model):
    def __init__(self, num_coupling_layers, n_variable=2):
        super(RealNVP, self).__init__()

        self.num_coupling_layers = num_coupling_layers
        self.n_variable = n_variable

        # Distribution of the latent space.
        self.distribution = tfp.distributions.MultivariateNormalDiag(
            loc=[0.0] * n_variable, scale_diag=[1.0] * n_variable
        )
        self.masks = checkerboard_mask(num_coupling_layers, n_variable)

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.layers_list = [Coupling(n_variable) for i in range(num_coupling_layers)]

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, x, training=True):
        log_det_inv = 0
        direction = 1
        if training:
            direction = -1
        for i in range(self.num_coupling_layers)[::direction]:
            x_masked = x * self.masks[i]
            reversed_mask = 1 - self.masks[i]
            s, t = self.layers_list[i](x_masked)
            s *= reversed_mask
            t *= reversed_mask
            gate = (direction - 1) / 2
            x = (
                reversed_mask
                * (x * tf.exp(direction * s) + direction * t * tf.exp(gate * s))
                + x_masked
            )
            log_det_inv += gate * tf.reduce_sum(s, [1])

        return x, log_det_inv

    def log_loss(self, x):
        y, logdet = self(x)
        log_likelihood = self.distribution.log_prob(y) + logdet
        return -tf.reduce_mean(log_likelihood)

    def train_step(self, data):
        with tf.GradientTape() as tape:

            loss = self.log_loss(data)

        g = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(g, self.trainable_variables))
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self.log_loss(data)
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}