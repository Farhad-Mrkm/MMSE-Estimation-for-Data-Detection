import os
import numpy as np

from sklearn.datasets import make_moons
from sklearn.utils import shuffle
import tensorflow_probability as tfp

try:
    from tensorflow.keras.layers import Normalization
except:
    from tensorflow.keras.layers.experimental.preprocessing import Normalization


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



def moon_distribution(n_sample, noise=0.05):

    data = make_moons(3000, noise=0.05)[0].astype("float32")

    norm = Normalization()
    norm.adapt(data)
    normalized_data = norm(data)

    return normalized_data


def custom_distribution(n_sample):
    data = []

    distribution = tfp.distributions.MultivariateNormalDiag(loc=[-0.5, -0.5], scale_diag=[0.25, 2.0])
    data.append(distribution.sample(n_sample // 2))

    distribution = tfp.distributions.MultivariateNormalDiag(loc=[-0.5, -0.5], scale_diag=[2, 0.25])
    data.append(distribution.sample(n_sample // 2))

    data = np.concatenate(data, axis=0)

    norm = Normalization()
    norm.adapt(data)
    normalized_data = norm(data)

    return normalized_data


def custom_laplace(n_samples):
    data=[]
    distribution1 = tfp.distributions.Laplace(loc=[0.0, 0.0],scale=[1.0, 1.0])
    distribution2 = tfp.distributions.MultivariateNormalDiag(loc=[0.0, 0.0],scale_diag=[0.25, 0.25])

    data = distribution1.sample(n_samples)
    noise = distribution2.sample(n_samples)

    data = data + noise

    return data


def custom_communication_system(n_samples):
    data=[]
    distribution1 = tfp.distributions.MultivariateNormalDiag(loc=[0.0],scale_diag=[1.0])
    distribution2 = tfp.distributions.MultivariateNormalDiag(loc=[0.0],scale_diag=[0.316])

    X = distribution1.sample(n_samples)
    noise = distribution2.sample(n_samples)

    Y = X + noise
    X, Y = X.numpy(), Y.numpy()

    joint = np.concatenate([X, Y], axis=-1)
    marginal = np.concatenate([X, shuffle(Y)], axis=-1)

    return joint, marginal