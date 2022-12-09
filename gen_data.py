
import os
import numpy as np
import tensorflow as tf
from sklearn.datasets import make_moons
from sklearn.utils import shuffle
import tensorflow_probability as tfp

try:
    from tensorflow.keras.layers import Normalization
except:
    from tensorflow.keras.layers.experimental.preprocessing import Normalization

import matplotlib.pyplot as plt

from sklearn.utils import shuffle
import tensorflow_probability as tfp


#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf.config.list_physical_devices('GPU')

# def moon_distribution(n_sample, noise=0.05):

#     data = make_moons(3000, noise=0.05)[0].astype("float32")

#     norm = Normalization()
#     norm.adapt(data)
#     normalized_data = norm(data)

#     return normalized_data


def custom_Gaussian(n_samples,snr):
        data=[]
        distribution1= tfp.distributions.MultivariateNormalDiag(loc=[0.0],scale_diag=[1.0])
        distribution2= tfp.distributions.MultivariateNormalDiag(loc=[0.0],scale_diag=[1.0])
        data = distribution1.sample(n_samples)
        noise = distribution2.sample(n_samples)
        data_y = tf.math.sqrt(tf.cast(snr,tf.float32))*data + noise
        return data_y,data
def custom_laplace(n_samples,snr):
        data=[]
        distribution1 = tfp.distributions.Laplace(loc=[0.0],scale=[1.0])
        distribution2 = tfp.distributions.MultivariateNormalDiag(loc=[0.0],scale_diag=[0.25])

        data = distribution1.sample(n_samples)
        noise = distribution2.sample(n_samples)

        data_y = tf.sqrt(tf.cast(snr,tf.float32))*data + noise
        

        return data_y,data
def custom_BPSK(n_samples,snr):
        data=[]
        input_distribution=tfp.distributions.Bernoulli(probs=[0.5])
        noise_distribution=tfp.distributions.MultivariateNormalDiag(loc=[0.0],scale_diag=[1.0])
        noise=noise_distribution.sample(n_samples)
        data=input_distribution.sample(n_samples)
        data_y=tf.sqrt(tf.cast(snr,tf.float32))*tf.cast((2*data-1),tf.float32)+noise
        return tf.cast(data_y,tf.float32),tf.cast(data,tf.float32)